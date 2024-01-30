import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from eu_energy_transition.data.data import DATA, products

# HELP FUNCTIONS
def series_format(data: pd.DataFrame) -> pd.Series(int):
    """Returns Pandas Series object ('time' as an index) of the data for the chosen energy source"""
    data = data.groupby(['time'])['value'].sum().reset_index()

    series_data = data['value']
    series_data.index = data['time']

    # Analyses the observaton intervals within the time series and infers that the data is monthly
    series_data.index = pd.DatetimeIndex(series_data.index.values, freq=series_data.index.inferred_freq)

    return series_data


def hwes_model(data: pd.Series(int), prod: str) -> pd.Series(int):
    """Returns a Series object including the predictions of Triple HWES"""
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
    model_pred = model_fit.forecast(48)

    return model_pred


def sarima_model(data: pd.Series(int), param: dict, prod) -> pd.Series(int):
    """Returns a Series object including the predictions of SARIMA"""
    model_param = param[prod]

    model = sm.tsa.statespace.SARIMAX(data,
                                      order=(model_param[0], model_param[1], model_param[2]),
                                      seasonal_order=(model_param[3], model_param[4], model_param[5], 12))
    model_fit = model.fit()
    model_pred = model_fit.forecast(48)

    return model_pred


def stl(data: pd.Series(int), prod: str) -> statsmodels.tsa.seasonal.DecomposeResult:
    if prod == "Renewable":
        stl_model = 'multiplicative'
    else:
        stl_model = 'additive'

    stl_result = seasonal_decompose(data, model=stl_model)

    return stl_result


### PAGE ###

# Defines the page
dash.register_page(__name__, name='TSA Predictions')

# CONTENT
sidebar = html.Div([])
content = html.Div([
    dbc.Row(style={'height': '2vh'}, className='bg-primary'),
    dbc.Row([
        html.P(children='The Electricity Production in European Countries in 2010-2022',
               style={'font-weight': 'bold', 'font-size': 25, 'height': '3vh', 'textAlign': 'center'}
               ),
    ], className='bg-primary text-white font-weight-bold'
    ),

dbc.Row([dcc.Dropdown(id='chosen-product-linechart-page4',
                          multi=False,
                          value='Net electricity production',
                          options=['Fossil fuels', 'Nuclear', 'Renewables', 'Net electricity production'],
                          style={'width': '250px',
                                 'margin-left': '40px',
                                 'margin-top': '20px'})
                          ]),

    dbc.Row([
        dbc.Col([dbc.Row([html.P(id='stl-title-page4')], style={'font-weight': 'bold',
                                                   'textAlign': 'center',
                                                   'margin-top': '10px'})]),

                 dbc.Row([dcc.Graph(id='stl-page4')]),

                 ]),

        dbc.Col([
            dbc.Row([]),

            dbc.Row([]),

                 ]),

    dbc.Row([html.P(id='linetitle-page4')], style={'font-weight': 'bold',
                                                   'textAlign': 'center',
                                                   'margin-top': '30px'}),
    dbc.Row([dcc.Checklist(id='chosen-method-page4',
                           value=['HWSE'],
                           options=['HWSE', 'SAMIRA'],
                           style={'width': '200px', 'margin-left': '80px'}
                           )]),

    dbc.Row([dcc.Graph(id='linechart-TSA-page4')])

    ])


layout = dbc.Container(children=[
    dbc.Row([
        dbc.Col(sidebar, width=1, className='bg-primary'),
        dbc.Col(content, width=11)
    ],
        style={'height': '95vh'}        # Viewport height set to 95%
    )
], fluid=True)                          # Additional margins removed


#########################################
############### CALLBACKS ###############
#########################################
@callback(Output("stl-page4", 'figure'),
          Output('stl-title-page4', 'children'),
          Input('chosen-product-linechart-page4', 'value'))
def stl_page4(product):
    stl_data = DATA[DATA['product'] == product]
    stl_data['time'] = pd.to_datetime(stl_data['time'], format="%B %Y")
    stl_series = series_format(stl_data)
    stl_title = f"Results of the Seasonal and Trend Composition for {product.lower()}"
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
                          width=1200,
                          height=700,
                          margin=go.layout.Margin(t=30))

    return stl_fig, stl_title


@callback(Output('linechart-TSA-page4', 'figure'),
          Output("linetitle-page4", 'children'),
          Input('chosen-product-linechart-page4', 'value'),
          Input('chosen-method-page4', 'value'))
def update_linechart_value_page4(product, methods):

   # Dictionary containign the parameters of SARIMA models for each energy source type

    line_data = DATA[DATA['product'] == product]
    line_data['time'] = pd.to_datetime(line_data['time'], format="%B %Y")

    train_data = line_data[line_data['year'] < 2021]
    train_series = series_format(train_data)

    test_data = line_data[line_data['year'] >= 2021]
    test_series = series_format(test_data)

    line_title = f"The predictions of the total generated electricity for {product.lower()} "


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

    if "HWSE" in methods:
        hwes_model_pred = hwes_model(train_series, product)

        fig_linechart.add_trace(go.Scatter(x=hwes_model_pred.index,
                                           y=hwes_model_pred,
                                           name="Triple HWES predictions",
                                           line=dict(color='darkred', dash='dash'),
                                           mode='lines',
                                           hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                         "<br> <b>GWh:</b> %{y}"
                                           ))

    if "SAMIRA" in methods:
        SARIMA_param = {'Net electricity production': [2, 1, 2, 1, 1, 1],
                        'Fossil fuels': [2, 1, 2, 1, 1, 1],
                        'Nuclear': [0, 1, 1, 1, 1, 1],
                        'Renewables': [0, 1, 1, 0, 1, 2]}

        SARIMA_model_pred = sarima_model(train_series, SARIMA_param, product)

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
                                margin=go.layout.Margin(t=30))
    fig_linechart.update_xaxes(dtick='M3', tickformat="%b\n%Y")

    return fig_linechart, line_title


