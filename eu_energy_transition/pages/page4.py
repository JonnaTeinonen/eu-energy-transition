import dash
from dash import html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf, acf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from eu_energy_transition.data.data import DATA, products


#########################################
############ HELP FUNCTIONS #############
#########################################
def series_format(data: pd.DataFrame) -> pd.Series(int):
    """Returns Pandas Series object ('time' as an index) of the data for the chosen energy source"""
    data = data.groupby(['time'])['value'].sum().reset_index()

    series_data = data['value']
    series_data.index = data['time']

    # Analyses the observaton intervals within the time series and infers that the data is monthly
    series_data.index = pd.DatetimeIndex(series_data.index.values, freq=series_data.index.inferred_freq)

    return series_data

def data_for_certain_product(data: pd.DataFrame, prod: str):
    """Chooses the data only including certain product and changed the time format"""
    if prod == 'Overall':
        prod = 'Net electricity production'

    df = DATA[DATA['product'] == prod]
    df['time'] = pd.to_datetime(df['time'], format="%B %Y")

    return df


def hwes_model(data: pd.Series(int), prod: str, forecast_len=48) -> pd.Series(int):
    """Returns a tuple including the predictions of Triple HWES and the model fit"""
    trend_param = 'add'
    if prod == "Renewables":
        season_param = 'mul'
    else:
        season_param = 'add'

    model = ExponentialSmoothing(data,
                                 trend=trend_param,
                                 seasonal=season_param,
                                 initialization_method="estimated")
    model_fit = model.fit()
    model_pred = model_fit.forecast(forecast_len)
    model_AIC = model_fit.aic

    return model_pred, model_fit


def sarima_model(data: pd.Series(int), param: dict, prod, forecast_len=48) -> pd.Series(int):
    """Returns a Series object including the predictions of SARIMA"""
    model_param = param[prod]

    model = sm.tsa.statespace.SARIMAX(data,
                                      order=(model_param[0], model_param[1], model_param[2]),
                                      seasonal_order=(model_param[3], model_param[4], model_param[5], 12))
    model_fit = model.fit()
    model_pred = model_fit.forecast(forecast_len)
    model_AIC = model_fit.aic

    return model_pred, model_AIC


def stl(data: pd.Series(int), prod: str) -> statsmodels.tsa.seasonal.DecomposeResult:
    if prod == "Renewable":
        stl_model = 'multiplicative'
    else:
        stl_model = 'additive'

    stl_result = seasonal_decompose(data, model=stl_model)

    return stl_result


def model_performance_eval(model_predictions: pd.Series, test_data: pd.Series) -> tuple:
    """Returns MSE, RMSE, MAE and AIC of given model"""

    MSE = mean_squared_error(test_data, model_predictions)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(test_data, model_predictions)

    return (MSE, RMSE, MAE)


def get_corr_plot(data: pd.Series(int), partial_cor: bool = False):
    """The function returns either the ACF or PACF plot.
    The function is based on an answer by "nonameperson"
    from https://community.plotly.com/t/plot-pacf-plot-acf-autocorrelation-plot-and-lag-plot/24108/4
    """

    if partial_cor == False:
        fig_data = acf(data, alpha=0.05)
        fig_title = "Autocorrelation plot"
        y_title = "Autocorrelation"
    else:
        fig_data = pacf(data, alpha=0.05)
        fig_title = "Partial autocorrelation plot"
        y_title = "Partial autocorrelation"

    # Computes the upper and lower limits of y for the confidence intervals
    CI_lower_y = fig_data[1][:, 0] - fig_data[0]
    CI_upper_y = fig_data[1][:, 1] - fig_data[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(fig_data[0])),
                             y=fig_data[0],
                             mode='markers',
                             name='Autocorrelation plot',
                             marker_size=10,
                             line=dict(color='darkblue'),
                             hovertemplate="%{y}<extra></extra>"))

    for i in range(len(fig_data[0])):
        fig.add_trace(go.Scatter(x=(i, i),
                                 y=(0, fig_data[0][i]),
                                 mode='lines',
                                 hoverinfo='skip',
                                 line=dict(color='darkblue', dash='dash')))

    fig.add_trace(go.Scatter(x=np.arange(len(fig_data[0])),
                             y=CI_upper_y,
                             mode='lines',
                             line_color='rgba(255,255,255,0)',
                             hoverinfo='skip'))

    fig.add_trace(go.Scatter(x=np.arange(len(fig_data[0])),
                             y=CI_lower_y,
                             mode='lines',
                             fillcolor='rgba(32, 146, 230, 0.2)',
                             fill='tonexty',
                             line_color='rgba(255,255,255,0)',
                             hoverinfo='skip'))

    fig.add_hline(y=0, line_color="black")
    fig.update_traces(showlegend=False)

    fig.update_layout(title=dict(text=fig_title, xanchor='center', x=0.5),
                      xaxis_title="Lag",
                      yaxis_title=y_title,
    #                  width=550,
    #                  height=370,
                      margin=go.layout.Margin(t=30))

    return fig


#########################################
################# PAGE ##################
#########################################

# Defines the page
dash.register_page(__name__, name='TSA Predictions')

# CONTENT
sidebar = html.Div([

    dbc.Row([html.P('Choose the energy type:', style={'margin-top': '10vh', 'margin-left': '2vh'})],
            className='text-white'),

    dbc.Row([dcc.Dropdown(id='chosen-product-page4',
                          multi=False,
                          value='Overall',
                          options=['Fossil fuels', 'Nuclear', 'Renewables', 'Overall'],
                          style={'width': '20vh',
                                 'margin-left': '1vh',
                                 'margin-top': '0.5vh'})
                          ])
                    ])

content = html.Div([
    dbc.Row(style={'height': '2vh'}, className='bg-primary'),
    dbc.Row([
        html.P(children='The Predicted Electricity Production in European Countries in Total for Years  2023-2024',
               style={'font-weight': 'bold', 'font-size': 25, 'height': '3vh', 'textAlign': 'center'}
               ),
    ], className='bg-primary text-white font-weight-bold'
    ),

    dbc.Row([
        dbc.Col([dbc.Row([html.P(id='stl-title-page4')],
                         style={'font-weight': 'bold',
                                'textAlign': 'center',
                                'margin-top': '20px',
                                'font-size': 20}),

                 dbc.Row([dcc.Graph(id='stl-page4')])
                 ], md=6),

        dbc.Col([
            dbc.Row([html.P(id='table-title-page4',
                                 style={'margin-top': '20px',
                                        'font-weight': 'bold',
                                        'textAlign': 'center',
                                        'font-size': 20})
                          ]),

            dbc.Row([html.Div(id="table-page4")]),

            dbc.Row([html.P(id='TSA-params-page4',
                                 style={'margin-top': '5px',
                                        'font-style': 'italic',
                                        'font-size': 13})
                          ]),

            dbc.Row([
                dbc.Col([dcc.Graph(id='ACF-page4')], style={'margin-top': '15px'}, md=6),
                dbc.Col([dcc.Graph(id='PACF-page4')], style={'margin-top': '15px'}, md=6)
            ])
        ], md=6),
    ]),

    dbc.Row([html.P(id='linetitle-page4')],
            style={'font-weight': 'bold',
                   'textAlign': 'center',
                   'margin-top': '30px',
                   'font-size': 20}),

    dbc.Row([dcc.Checklist(id='chosen-method-page4',
                           value=['HWES'],
                           options=['HWES', 'SARIMA'],
                           style={'width': '200px', 'margin-left': '80px'}
                           )]),

    dbc.Row([html.Div([dcc.Graph(id='linechart-TSA-page4')])])

    ])


layout = dbc.Container(children=[
    dbc.Row([
        dbc.Col(sidebar, className='bg-primary', xs=2, sm=2, md=2, lg=2),
        dbc.Col(content, xs=10, sm=10, md=10, lg=10)
    ],
        style={'height': '95vh'}        # Viewport height set to 95%
    )
], fluid=True)                          # Additional margins removed


#########################################
############### CALLBACKS ###############
#########################################
@callback(Output("stl-page4", 'figure'),
          Output('stl-title-page4', 'children'),
          Input('chosen-product-page4', 'value'))
def stl_page4(product):

    stl_data = data_for_certain_product(DATA, product)
    stl_series = series_format(stl_data)
    stl_title = f"Results of the seasonal and trend composition for {product.lower()} energy production"
    stl_result = stl(stl_series, product)

    stl_fig = make_subplots(rows=3, cols=1, subplot_titles=('Trend', 'Seasonality', 'Residuals'))

    stl_fig.add_trace(go.Scatter(
        x=stl_series.index,
        y=stl_result.trend,
        name="Trend",
        line=dict(color='cornflowerblue'),
        hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                      "<br> <b>GWh:</b> %{y}"
    ), row=1, col=1)

    stl_fig.add_trace(go.Scatter(
        x=stl_series.index,
        y=stl_result.seasonal,
        line=dict(color='cornflowerblue'),
        hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                      "<br> <b>GWh:</b> %{y}"
    ), row=2, col=1)

    stl_fig.add_trace(go.Scatter(
        x=stl_series.index,
        y=stl_result.resid,
        line=dict(color='cornflowerblue'),
        hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                      "<br> <b>GWh:</b> %{y}",
        mode='markers'
    ), row=3, col=1)

    stl_fig.add_hline(y=0, line_width=2, line_color="black", row=3, col=1)

    stl_fig.update_layout(showlegend=False,
#                          width=1100,
#                          height=500,
                          margin=go.layout.Margin(t=30))

    return stl_fig, stl_title


@callback(Output('table-page4', 'children'),
          Output('table-title-page4', 'children'),
          Output('TSA-params-page4', 'children'),
          Input('chosen-product-page4', 'value'),
          Input('chosen-method-page4', 'value'))
def table_page4(product, methods):

    if product == "Overall":
        product = 'Net electricity production'

    table_data = data_for_certain_product(DATA, product)

    train_data = table_data[table_data['year'] < 2021]
    train_series = series_format(train_data)

    test_data = table_data[table_data['year'] >= 2021]
    test_series = series_format(test_data)

    # Computes the model predictions
    if "HWES" in methods:
        hwes_model_pred, hwes_model_fit = hwes_model(train_series, product, 24)
        hwes_aic = hwes_model_fit.aic
        hwes_model_pred_eval = model_performance_eval(hwes_model_pred, test_series)

        hwes_metrics = {'Method': 'Triple HWES',
                        'MSE': [round(hwes_model_pred_eval[0], 2)],
                        'RMSE': [round(hwes_model_pred_eval[1], 2)],
                        'MAE': [round(hwes_model_pred_eval[2], 2)],
                        'AIC': [round(hwes_aic, 2)]}
        hwes_metrics = pd.DataFrame.from_dict(hwes_metrics)


    if "SARIMA" in methods:
        # Dictionary containing the parameters of SARIMA models for each energy source type
        SARIMA_param = {'Net electricity production': [2, 1, 2, 1, 1, 1],
                        'Fossil fuels': [2, 1, 2, 1, 1, 1],
                        'Nuclear': [0, 1, 1, 1, 1, 1],
                        'Renewables': [0, 1, 1, 0, 1, 2]}

        SARIMA_model_pred, SARIMA_aic = sarima_model(train_series, SARIMA_param, product, 24)
        SARIMA_model_pred_eval = model_performance_eval(SARIMA_model_pred, test_series)

        SARIMA_metrics = {'Method': 'SARIMA',
                          'MSE': [round(SARIMA_model_pred_eval[0], 2)],
                          'RMSE': [round(SARIMA_model_pred_eval[1], 2)],
                          'MAE': [round(SARIMA_model_pred_eval[2], 2)],
                          'AIC': [round(SARIMA_aic, 2)]}
        SARIMA_metrics = pd.DataFrame.from_dict(SARIMA_metrics)

        sarima_model_param = SARIMA_param[product]

    # Output of the final table
    if len(methods) == 2:
        output_table_df = pd.concat([hwes_metrics, SARIMA_metrics], axis=0)
        table_title = (f"Performance metrics of {methods[0]} and {methods[1]} for predicting the total "
                       f"generated electricity")
        model_params = (rf"Triple HWES used the following parameters: "
                        f"α = {round(hwes_model_fit.params['smoothing_level'], 2)}, "
                        f"β = {round(hwes_model_fit.params['smoothing_trend'], 2)} and "
                        f"γ = {round(hwes_model_fit.params['smoothing_seasonal'], 2)}. "
                        f"SARIMA used the following parameters: p = {round(sarima_model_param[0], 2)}, "
                        f"d = {round(sarima_model_param[1], 2)}, q = {round(sarima_model_param[2], 2)}, "
                        f"P = {round(sarima_model_param[3], 2)}, D = {round(sarima_model_param[4], 2)}, "
                        f"Q = {round(sarima_model_param[5], 2)}. ")

    else:
        if "HWES" in methods:
            output_table_df = hwes_metrics
            model_params = (rf"Triple HWES used the following parameters: "
                            f"α = {round(hwes_model_fit.params['smoothing_level'], 2)}, "
                            f"β = {round(hwes_model_fit.params['smoothing_trend'], 2)} and "
                            f"γ = {round(hwes_model_fit.params['smoothing_seasonal'], 2)}.")

        if "SARIMA" in methods:
            output_table_df = SARIMA_metrics
            model_params = (f"SARIMA used the following parameters: p = {round(sarima_model_param[0], 2)}, "
                            f"d = {round(sarima_model_param[1], 2)}, q = {round(sarima_model_param[2], 2)}, "
                            f"P = {round(sarima_model_param[3], 2)}, D = {round(sarima_model_param[4], 2)}, "
                            f"Q = {round(sarima_model_param[5], 2)}. ")


        table_title = f"Performance metrics of {methods[0]} for predicting the total generated electricity"

    output_table = html.Div([dash_table.DataTable(data=output_table_df.to_dict('records'))])

    return output_table, table_title, model_params


@callback(Output('ACF-page4', 'figure'),
          Output('PACF-page4', 'figure'),
          Input('chosen-product-page4', 'value'))
def correlation_plots_page4(product):

    corr_data = data_for_certain_product(DATA, product)
    corr_series = series_format(corr_data)
    corr_series_diff = corr_series - corr_series.shift(1)
    corr_series_diff = corr_series_diff.dropna()
    corr_series_diff_seasonal = corr_series_diff - corr_series_diff.shift(12)
    corr_series_diff_seasonal = corr_series_diff_seasonal.dropna()

    # Calls the function that creates the plots
    acf_fig = get_corr_plot(corr_series_diff_seasonal)
    pacf_fig = get_corr_plot(corr_series_diff_seasonal, True)


    return acf_fig, pacf_fig


@callback(Output('linechart-TSA-page4', 'figure'),
          Output("linetitle-page4", 'children'),
          Input('chosen-product-page4', 'value'),
          Input('chosen-method-page4', 'value'))
def update_linechart_value_page4(product, methods):

    line_data = data_for_certain_product(DATA, product)
    train_data = line_data[line_data['year'] < 2021]
    train_series = series_format(train_data)
    test_data = line_data[line_data['year'] >= 2021]
    test_series = series_format(test_data)

    line_title = f"The predictions of the total generated electricity in {product.lower()} "

    if product == "Overall":
        product = 'Net electricity production'

    # Creating an empty figure with the correct timeline
    fig_linechart = go.Figure()
    fig_linechart.add_trace(go.Scatter(x=line_data['time'], y=pd.Series(dtype=object), mode='lines'))

    fig_linechart.add_trace(go.Scatter(x=train_series.index,
                                       y=train_series,
                                       name="Training data",
                                       line=dict(color='darkblue'),
                                       mode='lines',
                                       hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                    "<br> <b>GWh:</b> %{y}"
                                       ))

    fig_linechart.add_trace(go.Scatter(x=test_series.index,
                                       y=test_series,
                                       name="Test data",
                                       line=dict(color='cornflowerblue'),
                                       mode='lines',
                                       hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                    "<br> <b>GWh:</b> %{y}"
                                       ))

    if "HWES" in methods:
        hwes_model_pred = hwes_model(train_series, product)[0]

        fig_linechart.add_trace(go.Scatter(x=hwes_model_pred.index,
                                           y=hwes_model_pred,
                                           name="Triple HWES predictions",
                                           line=dict(color='darkred', dash='dash'),
                                           mode='lines',
                                           hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                         "<br> <b>GWh:</b> %{y}"
                                           ))

    if "SARIMA" in methods:
        # Dictionary containing the parameters of SARIMA models for each energy source type
        SARIMA_param = {'Net electricity production': [2, 1, 2, 1, 1, 1],
                        'Fossil fuels': [2, 1, 2, 1, 1, 1],
                        'Nuclear': [0, 1, 1, 1, 1, 1],
                        'Renewables': [0, 1, 1, 0, 1, 2]}

        SARIMA_model_pred = sarima_model(train_series, SARIMA_param, product)[0]

        fig_linechart.add_trace(go.Scatter(x=SARIMA_model_pred.index,
                                           y=SARIMA_model_pred,
                                           name="SARIMA predictions",
                                           line=dict(color='darkgreen', dash='dash'),
                                           mode='lines',
                                           hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                         "<br> <b>GWh:</b> %{y}"
                                           ))

    fig_linechart.add_vline(x='2022-12-01', line_width=2.5, line_color="black")
    fig_linechart.add_vrect(x0='2022-12-01', x1='2024-12-01', line_width=0, fillcolor="red", opacity=0.07)
    fig_linechart.update_layout(yaxis_title="GWh",
                                legend_title="Data",
                                margin=go.layout.Margin(t=30),
    #                            height=450,
    #                            width=2320,
                                autosize=True)
    fig_linechart.update_xaxes(dtick='M3', tickformat="%b\n%Y")

    return fig_linechart, line_title


