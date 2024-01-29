import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing

from eu_energy_transition.data.data import DATA, products

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

    dbc.Row([html.P('Choose the time range:', style={'margin-top': '15px', 'margin-bottom': '10px'}),
             dcc.RangeSlider(min=2010,
                             max=2023,
                             step=1,
                             value=[2010, 2023],
                             id='year-range-slider-page4',
                             marks={i: '{}'.format(i) for i in range(2010, 2024)}
                             )
             ]),

    dbc.Row([dcc.Dropdown(id='chosen-product-linechart-page4',
                          multi=False,
                          value='Net electricity production',
                          options=['Fossil fuels', 'Nuclear', 'Renewables', 'Net electricity production'],
                          style={'width': '250px',
                                 'margin-left': '40px'}
                          ),
             dcc.Checklist(id='chosen-method-page4',
                          value=['HWSE'],
                          options=['HWSE', 'SAMIRA'],
                          style={'width': '200px', 'margin-left': '65px'}
                          )
             ]),
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
@callback(Output('linechart-TSA-page4', 'figure'),
          Input('chosen-product-linechart-page4', 'value'),
          Input('chosen-method-page4', 'value'))
def update_linechart_value_page4(product, methods):
    def series_format(data: pd.DataFrame) -> pd.Series:
        """Returns Pandas Series object ('time' as an index) of the data for the chosen energy source"""
        data = data.groupby(['time'])['value'].sum().reset_index()

        series_data = data['value']
        series_data.index = data['time']

        # Analyses the observaton intervals within the time series and infers that the data is monthly
        series_data.index = pd.DatetimeIndex(series_data.index.values, freq=series_data.index.inferred_freq)

        return series_data

   # Dictionary containign the parameters of SARIMA models for each energy source type
    SARIMA_param = {'Net electricity production': [2, 1, 2, 1, 1, 1],
                    'Fossil fuels' : [2, 1, 2, 1, 1, 1],
                    'Nuclear' : [0, 1, 1, 1, 1, 1],
                    'Renewables' : [0, 1, 1, 0, 1, 2]}

    line_data = DATA[DATA['product'] == product]
    line_data['time'] = pd.to_datetime(line_data['time'], format="%B %Y")

    train_data = line_data[line_data['year'] < 2021]
    train_series = series_format(train_data)

    test_data = line_data[line_data['year'] >= 2021]
    test_series = series_format(test_data)


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
        trend_param = 'add'
        if product == "Renewables":
            season_param = 'mul'
        else:
            season_param = 'add'


        hwes_model = ExponentialSmoothing(train_series,
                                          trend=trend_param,
                                          seasonal=season_param,
                                          initialization_method="estimated")
        hwes_model_fit = hwes_model.fit()
        hwes_model_pred = hwes_model_fit.forecast(48)

        fig_linechart.add_trace(go.Scatter(x=hwes_model_pred.index,
                                           y=hwes_model_pred,
                                           name="Triple HWES predictions",
                                           line=dict(color='darkred', dash='dash'),
                                           mode='lines',
                                           hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                         "<br> <b>GWh:</b> %{y}"
                                           ))

    if "SAMIRA" in methods:
        model_param = SARIMA_param[product]

        SARIMA_model = sm.tsa.statespace.SARIMAX(train_series,
                                                 order=(model_param[0], model_param[1], model_param[2]),
                                                 seasonal_order=(model_param[3], model_param[4], model_param[5], 12))
        SARIMA_model_fit = SARIMA_model.fit()
        SARIMA_model_pred = SARIMA_model_fit.forecast(48)

        fig_linechart.add_trace(go.Scatter(x=SARIMA_model_pred.index,
                                           y=SARIMA_model_pred,
                                           name="SARIMA predictions",
                                           line=dict(color='darkgreen', dash='dash'),
                                           mode='lines',
                                           hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                         "<br> <b>GWh:</b> %{y}"
                                           ))

    print(type(test_series.index))
    print(test_series.index)

    fig_linechart.add_vline(x='2022-12-01', line_width=2.5, line_color="black")
    fig_linechart.add_vrect(x0='2022-12-01', x1='2024-12-01', line_width=0, fillcolor="red", opacity=0.07)


    return fig_linechart
