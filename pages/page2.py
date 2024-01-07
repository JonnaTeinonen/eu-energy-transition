import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, callback
import dash_bootstrap_components as dbc
import matplotlib
from matplotlib.pyplot import cm
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Data
data = pd.read_csv('C:\\Users\\JonnaMS\\Desktop\\Itseopiskelu\\Portfolio\\Datasetit\\Monthly Electricity Production in GWh [2010-2022].zip')

# Changing the column names to lowercase
col_names = list(data.columns)
col_names_low = [col.lower() for col in col_names[0:9]]
col_names_low.extend(col_names[9:])
data.columns = col_names_low

# Choosing the EU-countries
eu_countries = ['Austria' 'Belgium', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
                'Hungary', 'Iceland,' 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Netherlands', 'Norway',
                'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Bulgaria',
                'Croatia', 'Cyprus', 'Malta', 'North Macedonia', 'Romania']
data = data[data['country'].isin(eu_countries)]

# Variables
country_var = data.country.unique()
year_var = data.year.unique()


# HELP FUNCTIONS

def products(category: bool = False):
    """Returns the names and colors of the wanted energy sources / products"""
    if category:
        prod_names = ['Fossil fuels', 'Nuclear', 'Renewables']
        prod_colors = {'Fossil fuels' : 'rgb(95, 70, 144)',
                       'Nuclear' : 'rgb(56, 166, 165)',
                       'Renewables' : 'rgb(15, 133, 84)',
                       '(?)': 'lightgrey'}
    else:
        prod_names = ['Coal', 'Hydro', 'Nuclear', 'Wind', 'Solar', 'Natural gas', 'Oil']
        prod_colors = {'Coal': 'rgb(95, 70, 144)',
                       'Hydro': 'rgb(29, 105, 150)',
                       'Nuclear': 'rgb(56, 166, 165)',
                       'Wind': 'rgb(15, 133, 84)',
                       'Solar': 'rgb(115, 175, 72)',
                       'Natural gas': 'rgb(237, 173, 8)',
                       'Oil': 'rgb(225, 124, 5)',
                       '(?)': 'lightgrey'}

    return prod_names, prod_colors


# APP

sidebar = html.Div([
    dbc.Row([html.P('Choose energy source grouping:',
                    style={'margin-top': '95px', 'margin-bottom': '5px'}
                    )],
            className='text-white'),

    dbc.Row(dcc.RadioItems(id='energy-grouping-page2',
                           options=[' Grouped', ' Granular'],
                           value=' Granular'),
            style={'padding': 10, 'flex': 1, 'padding-left': 10},
            className='text-white'),

    dbc.Row([
             html.P("Choose the country:",
                    style={'margin-top': '20px'})],
        className='text-white'),
    dbc.Row(dcc.Dropdown(id='chosen-countries-page2',
                         multi=True,
                         value='Netherlands',
                         options=[{'label': x, 'value': x} for x in country_var],
                         style={'width': '280px'})),
    dbc.Row(html.P()),

    dbc.Row(
        html.Button(id='apply-button-page2', n_clicks=0, children='apply',
                                style={'margin-top': '2px'},
                                className='bg-dark text-white'))
])

content = html.Div([
    dbc.Row(style={'height': '2vh'}, className='bg-primary'),
    dbc.Row([
        html.P('The Electricity Production in European Countries in 2010-2022',
               style={'font-weight': 'bold', 'font-size': 25, 'height': '5vh', 'textAlign': 'center'}),
    ], className='bg-primary text-white font-weight-bold'
    ),
    dbc.Row([html.P('Choose the time range:', style={'margin-top': '15px', 'margin-bottom': '10px'}),
             dcc.RangeSlider(2010,
                             2023,
                             1,
                             value=[2010, 2023],
                             id='year-range-slider-page2',
                             marks={i: '{}'.format(i) for i in range(2010, 2024)})]),


    dbc.Row([
        dbc.Col([dbc.Row([html.P(id='bar-title-page2', style={'margin-top':'15px', 'font-weight': 'bold', 'textAlign': 'center'})]),
                 dbc.Row([dcc.Graph(id='barchart-products-page2')])
                 ]),
        dbc.Col([dbc.Row([html.P(id='treemap-title-page2', style={'margin-top':'15px', 'height': '5vh', 'font-weight': 'bold', 'textAlign': 'center'})]),
                 dbc.Row([dcc.Graph(id='treemap-countries-page2')])
                 ])
    ]),

    dbc.Row([html.P(id='line-title-page2')], style={'font-weight': 'bold', 'textAlign': 'center'}),
    dbc.Row([dcc.Graph(id='linechart-value-page2')]),
    dbc.Row([dcc.Dropdown(id='chosen-product-page2',
                         multi=False,
                         value='Renewables',
                         options=['Fossil fuels', 'Nuclear', 'Renewables', 'Coal', 'Hydro', 'Nuclear', 'Wind',
                                  'Solar', 'Natural gas', 'Oil'],
                         style={'width': '280px'})])

    # dbc.Row([html.Label(id='line-title', style={'margin-top':'50px',
    #                                             'margin-left':'80px'}),
    #          dcc.Graph(id='linechart-value')])
])

# Creates a link
dash.register_page(__name__)

layout = dbc.Container([
    dbc.Row([
        dbc.Col(sidebar, width=2, className='bg-primary'),
        dbc.Col(content, width=10)],

    # Viewport height set to 100%
    style={'height': '100vh'})],
    # Additional margins removed
fluid=True)




#########################################
############### CALLBACKS ###############
#########################################

@callback(Output('barchart-products-page2', 'figure'),
              Output('bar-title-page2', 'children'),
              Input('apply-button-page2', 'n_clicks'),
              State('chosen-countries-page2', 'value'),
              State('year-range-slider-page2', 'value'),
              State('energy-grouping-page2', 'value'))
def update_barchart_products_page2(n_clicks, countries: list, years: list[int], grouping: str):
    """TODO: Change the decimal format to percentage"""

    def divide_by_country_total(df_row, lookup_df: pd.DataFrame):
        """Computes the average share for each product per country"""
        country = df_row['country']
        product_share = df_row.share
        lookup_df = lookup_df[lookup_df['country'].isin([country])]
        country_total = lookup_df['share']

        return product_share / country_total.values[0]

    if grouping == ' Granular':
        product_names, product_col = products(False)
    else:
        product_names, product_col = products(True)

    if years[0] == years[1]:
        chosen_years = [years[0]]
        bar_title = f"The average share of each energy source in the total energy generation per country in {years[0]}"
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        bar_title = f"The average share of each energy source in the total energy generation per country between {years[0]}-{years[1]}"

    if type(countries) != list:
        chosen_countries = [countries]
    else:
        chosen_countries = countries

    bar_data = data[data['year'].isin(chosen_years)]
    bar_data = bar_data[bar_data['country'].isin(chosen_countries)]

    bar_data_product = bar_data[bar_data['product'].isin(product_names)]

    barchart_data_sumproduct = bar_data_product.groupby(['country', 'product'])['share'].sum().reset_index()

    total_data = bar_data[bar_data['product'] == 'Net electricity production']
    barchart_data_sumtotal = total_data.groupby(['country'])['share'].sum().reset_index()

    barchart_data = barchart_data_sumproduct
    barchart_data['avg_share'] = barchart_data.apply(
        lambda row: divide_by_country_total(row, barchart_data_sumtotal), axis=1)
    barchart_data['avg_share'] = barchart_data['avg_share'] * 100

    fig_bar = px.bar(barchart_data,
                     x='avg_share',
                     y='country',
                     color='product',
                     color_discrete_map=product_col,
                     width=1000,
                     height=470,
                     orientation='h',
                     category_orders={"product": product_names})  # Product category order in the legend/graph

    fig_bar.update_traces(width=0.5, hovertemplate="<b>Share:</b> %{x}")

    fig_bar.update_layout(xaxis_title='Average share in percentage',
                          yaxis_title='',
                          showlegend=True,
                          legend_title="Energy Source",
                          margin=go.layout.Margin(t=25))

    return fig_bar, bar_title


@callback(Output('treemap-countries-page2', 'figure'),
              Output('treemap-title-page2', 'children'),
              Input('apply-button-page2', 'n_clicks'),
              State('chosen-countries-page2', 'value'),
              State('year-range-slider-page2', 'value'),
              State('energy-grouping-page2', 'value'))
def update_treemap_countries_page2(n_clicks, countries: list[str], years: list[int], grouping: str):

    if grouping == ' Granular':
        product_names, product_col = products(False)
    else:
        product_names, product_col = products(True)

    if years[0] == years[1]:
        chosen_years = [years[0]]
        treemap_title = f"The total amount of generated energy for each energy source per country {years[0]}"
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        treemap_title = f"The total amount of generated energy for each energy source per country between {years[0]}-{years[1]}"

    if type(countries) != list:
        chosen_countries = [countries]
    else:
        chosen_countries = countries

    figure_data = data[data['year'].isin(chosen_years)]
    figure_data = figure_data[figure_data['country'].isin(chosen_countries)]
    figure_data = figure_data[figure_data['product'].isin(product_names)]

    figure_data = figure_data.groupby(['country', 'product'])['value'].sum().reset_index()

    fig_treemap = px.treemap(figure_data,
                             path=['country', 'product'],
                             values='value',
                             color='product',
                             width=1000,
                             height=500,
                             color_discrete_map=product_col)

    fig_treemap.update_traces(root_color="white",
                              hovertemplate="<b>%{parent}</b> | %{label} <br> "
                                            "<b>Total energy produced (GWh):</b> %{value}")
    fig_treemap.update_layout(margin=go.layout.Margin(t=25))

    return fig_treemap, treemap_title


@callback(Output('linechart-value-page2', 'figure'),
              Output('line-title-page2', 'children'),
              Input('apply-button-page2', 'n_clicks'),
              State('chosen-countries-page2', 'value'),
              State('year-range-slider-page2', 'value'),
              State('energy-grouping-page2', 'value'),
              State('chosen-product-page2', 'value'))
def update_linechart_value_page2(n_clicks, countries, years, grouping, chosen_product):

    if years[0] == years[1]:
        chosen_years = [years[0]]
        line_title = (f"The average amount of generated electricity with {chosen_product.lower()} in GWh per country "
                      f"in {years[0]}")
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        line_title = (f"The average amount of generated electricity  with {chosen_product.lower()} in GWh per country "
                      f"between {years[0]}-{years[1]}")

    if type(countries) != list:
        chosen_countries = [countries]
    else:
        chosen_countries = countries

    line_data = data[data['product'] == chosen_product]
    line_data = line_data[line_data['country'].isin(chosen_countries)]
    line_data = line_data[line_data['year'].isin(chosen_years)]

    # Creating an empty figure with the correct timeline
    fig_linechart = go.Figure()
    fig_linechart.add_trace(go.Scatter(x=[line_data['year'], line_data['month']],
                                           y=pd.Series(dtype=object),
                                           mode='lines'))

    # Adding information of the value of the chosen product for all the chosen countries into the timeline
    for i in range(len(chosen_countries)):
        fig_data = line_data[line_data['country'] == chosen_countries[i]].sort_values(by=['year', 'month'],
                                                                                   ascending=True).reset_index(drop=True)
        fig_linechart.add_trace(go.Scatter(x=[fig_data['year'], fig_data['month']],
                                           y=fig_data['value'],
                                           name=chosen_countries[i],
                                           line=dict(color=px.colors.qualitative.Dark24[i]),
                                           mode='lines',
                                           hovertemplate="<b>Year:</b> %{x[0]} <br> <b>Month:</b> %{x[1]} "
                                                         "<br> <b>Value:</b> %{y}"))

    fig_linechart.update_layout(height=400,
                                yaxis_title="GWh",
                                legend_title="Country",
                                margin=go.layout.Margin(t=30))

    return fig_linechart, line_title
