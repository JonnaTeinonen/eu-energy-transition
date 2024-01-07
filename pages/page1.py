import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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



sidebar = html.Div([
    dbc.Row([html.P('Choose energy source grouping:',
                    style={'margin-top': '95px', 'margin-bottom': '5px'}
                    )],
            className='text-white'),

    dbc.Row(dcc.RadioItems(id='energy-grouping-page1',
                           options=[' Grouped', ' Granular'],
                           value=' Granular'),
            style={'padding': 10, 'flex': 1, 'padding-left': 10},
            className='text-white'),

    dbc.Row(html.P()),

    dbc.Row(
        html.Button(id='apply-button-page1', n_clicks=0, children='apply',
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

                             id='year-range-slider-page1',
                             marks={i: '{}'.format(i) for i in range(2010, 2024)})]),


    dbc.Row([
        dbc.Col([dbc.Row([html.P(id='bar-title-page1', style={'margin-top':'15px', 'font-weight': 'bold', 'textAlign': 'center'})]),
                 dbc.Row([dcc.Dropdown(id='chosen-val-barchart-page1',
                                       multi=False,
                                       value='Energy share',
                                       options=['Energy share', 'Generated energy'],
                                       style={'width': '200px',
                                              'margin-left': '40px'}),
                         dcc.Dropdown(id='chosen-grouping-barchart-page1',
                                     multi=False,
                                     value='Grouped',
                                     options=['Grouped', 'Granular'],
                                     style={'width': '200px',
                                            'margin-left': '35px'})]
                ),
                 dbc.Row([dcc.Graph(id='barchart-products-page1')])
                 ]),
        dbc.Col([dbc.Row([html.P(id='treemap-title-page1', style={'margin-top': '15px',
                                                            'height': '5vh',
                                                            'font-weight': 'bold',
                                                            'textAlign': 'center'})]),
                 dbc.Row([dcc.Dropdown(id='chosen-product-treemap-page1',
                                               multi=False,
                                               value='Fossil fuels',
                                               options=['Fossil fuels', 'Nuclear', 'Renewables', 'Coal', 'Hydro',
                                                        'Nuclear', 'Wind', 'Solar', 'Natural gas', 'Oil'],
                         style={'width': '280px',
                                'margin-left': '40px'})]),
                 dbc.Row([dcc.Graph(id='treemap-countries-page1')])
                 ])
    ]),

    dbc.Row([html.P(id='line-title-page1')], style={'font-weight': 'bold', 'textAlign': 'center'}),
    dbc.Row([
        dbc.Col(html.Div([dcc.Dropdown(id='chosen-grouping-linechart-page1',
                     multi=False,
                     value='Grouped',
                     options=['Grouped', 'Granular'],
                     style={'width': '200px',
                            'margin-left': '35px'})]), width=2),
        dbc.Col(html.Div([dcc.Checklist([" Monthly Data"], [" Monthly Data"], id="monthlydata-check-page1", inline=True),
                      dcc.Checklist([" Trendlines"], [], id="trendline-check-page1", inline=True)]))
                           ]),
    dbc.Row([dcc.Graph(id='linechart-value-page1')])
])

# Creates a link
dash.register_page(__name__, path='/')

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

@callback(Output('barchart-products-page1', 'figure'),
              Output('bar-title-page1', 'children'),
              Input('year-range-slider-page1', 'value'),
              Input('chosen-val-barchart-page1', 'value'),
              Input('chosen-grouping-barchart-page1', 'value'))
def update_barchart_products_page1(years: list[int], chosen_val: str, grouping: str):

    def divide_by_country_total(df_row, lookup_df: pd.DataFrame):
        """Computes the average share for each product per country"""
        country = df_row['country']
        product_share = df_row.share
        lookup_df = lookup_df[lookup_df['country'].isin([country])]
        country_total = lookup_df['share']

        return product_share / country_total.values[0]

    if grouping == 'Grouped':
        product_names, product_col = products(True)
        show_legend = False
    else:
        product_names = ['Coal', 'Natural gas', 'Oil', 'Nuclear', 'Hydro', 'Wind', 'Solar']
        product_col =  {'Coal': 'rgb(95, 70, 144)',
                        'Natural gas': 'rgb(237, 173, 8)',
                        'Oil': 'rgb(225, 124, 5)',
                        'Nuclear': 'rgb(56, 166, 165)',
                        'Hydro': 'rgb(29, 105, 150)',
                        'Wind': 'rgb(15, 133, 84)',
                        'Solar': 'rgb(115, 175, 72)'}
        show_legend = True

    if years[0] == years[1]:
        chosen_years = [years[0]]

        if chosen_val == "Energy share":
            bar_title = (f"The average share of each energy source in the total energy generation per country "
                         f"in {years[0]}")
        else:
            bar_title = f"The total generated energy in GWh for each energy source per country in {years[0]}"
    else:
        chosen_years = [x for x in range(years[0], years[1])]

        if chosen_val == "Energy share":
            bar_title = (f"The average share of each energy source in the total energy generation per country between "
                         f"{years[0]}-{years[1]}")
        else:
            bar_title = (f"The total generated energy in GWh for each energy source per country between "
                         f"{years[0]}-{years[1]}")


    bar_data = data[data['year'].isin(chosen_years)]

    if chosen_val == "Energy share":
        barchart_data_sumproduct = bar_data.groupby(['country', 'product'])['share'].sum().reset_index()

        total_data = bar_data[bar_data['product'] == 'Net electricity production']
        barchart_data_sumtotal = total_data.groupby(['country'])['share'].sum().reset_index()

        barchart_data = barchart_data_sumproduct
        barchart_data['avg'] = barchart_data.apply(
            lambda row: divide_by_country_total(row, barchart_data_sumtotal), axis=1)
        barchart_data['avg'] = barchart_data['avg'] * 100

        hovertext = "<b>Country:</b> %{y} <br> <b>Share:</b> %{x}"
        x_axis_title = "Average share in percentage"

    else:
        barchart_data_sumproduct = bar_data.groupby(['country', 'product'])['value'].sum().reset_index()
        barchart_data = barchart_data_sumproduct
        barchart_data['avg'] = barchart_data_sumproduct['value']

        hovertext = "<b>Country:</b> %{y} <br> <b>GWh:</b> %{x}"
        x_axis_title = "Total generated energy in GWh"

    fig_bar = make_subplots(rows=1, cols=3,
                            x_title= x_axis_title,
                            subplot_titles=('Fossil fuels', 'Nuclear', 'Renewables'))
    c = 0
    for i in range(3):

        if i == 0 and grouping == 'Granular':
            fig = go.Figure()

            for j in range(3):
                fig_data = barchart_data[barchart_data['product'] == product_names[j]]

                fig.add_trace(go.Bar(x=fig_data['avg'],
                              y=fig_data['country'],
                              name=product_names[j],
                              orientation='h',
                              marker_color=list(product_col.values())[j],
                              offsetgroup=0))

            c=+j

            for t in fig.data:
                fig_bar.add_trace(t, row=1, col=1)

        if i == 2 and grouping == 'Granular':
            fig = go.Figure()

            for j in range(3):
                fig_data = barchart_data[barchart_data['product'] == product_names[j+c]]

                fig.add_trace(go.Bar(x=fig_data['avg'],
                                     y=fig_data['country'],
                                     name=product_names[j+c],
                                     orientation='h',
                                     marker_color=list(product_col.values())[j+c],
                                     offsetgroup=0))

            for t in fig.data:
                fig_bar.add_trace(t, row=1, col=3)

        else:
            fig_data = barchart_data[barchart_data['product'] == product_names[c]]
            fig_bar.add_trace(go.Bar(x=fig_data['avg'],
                                 y=fig_data['country'],
                                 name=product_names[c],
                                 orientation='h',
                                 marker_color=list(product_col.values())[c]), row = 1, col =i+1)
            c+=1

    fig_bar.update_traces(hovertemplate=hovertext)

    fig_bar.update_layout(height=600,
                          width=1270,
                          showlegend=show_legend,
                          xaxis=dict(categoryorder='total descending'),
                          yaxis=dict(categoryorder='total ascending', tickfont=dict(size=8)),
                          yaxis2=dict(categoryorder='total ascending', tickfont=dict(size=8)),
                          yaxis3=dict(categoryorder='total ascending', tickfont=dict(size=8)))

    return fig_bar, bar_title


@callback(Output('treemap-countries-page1', 'figure'),
              Output('treemap-title-page1', 'children'),
              Input('year-range-slider-page1', 'value'),
              Input('chosen-product-treemap-page1', 'value'))
def update_treemap_countries_page1(years: list[int], chosen_product):

    if years[0] == years[1]:
        chosen_years = [years[0]]
        treemap_title = f"The total amount of generated energy with {chosen_product.lower()} per country in {years[0]} in EU"
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        treemap_title = f"The total amount of generated energy with {chosen_product.lower()} per country between {years[0]}-{years[1]} in EU"

    figure_data = data[data['year'].isin(chosen_years)]
    figure_data = figure_data[figure_data['product'] == chosen_product]

    figure_data = figure_data.groupby(['country'])['value'].sum().reset_index()
    figure_data = figure_data[figure_data['value'] != 0] # Removes sums from data that are 0
    figure_data['EU'] = 'EU' # Creates a single root node

    fig_treemap = px.treemap(figure_data,
                             path=['EU', 'country'],
                             values='value',
                             color='value',
                             width=750,
                             height=600,
                             color_continuous_scale='Blues'
    )

    fig_treemap.update_traces(hovertemplate="<b>%{label}</b> <br> "
                                            "<b>Total energy produced (GWh):</b> %{value}")
    fig_treemap.update_layout(margin=go.layout.Margin(t=25))

    return fig_treemap, treemap_title


@callback(Output('linechart-value-page1', 'figure'),
              Output('line-title-page1', 'children'),
              Input('year-range-slider-page1', 'value'),
              Input('chosen-grouping-linechart-page1', 'value'),
              Input('monthlydata-check-page1', 'value'),
              Input('trendline-check-page1', 'value'))
def update_linechart_value_page1(years, grouping, monthlydata, trendlinedata):

    if grouping == 'Granular':
        product_names, product_col = products(False)
    else:
        product_names, product_col = products(True)

    if years[0] == years[1]:
        chosen_years = [years[0]]
        line_title = (f"The total amount of generated electricity in GWh for each energhy source in EU in {years[0]}")
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        line_title = (f"The total amount of generated electricity in GWh for each energy source per in EU "
                      f"between {years[0]}-{years[1]}")

    line_data = data[data['product'].isin(product_names)]
    line_data = line_data[line_data['year'].isin(chosen_years)]
    line_data['time'] = pd.to_datetime(line_data['time'], format="%B %Y")

    # Creating an empty figure with the correct timeline
    fig_linechart = go.Figure()
    fig_linechart.add_trace(go.Scatter(x=line_data['time'],
                                           y=pd.Series(dtype=object),
                                           mode='lines'))

    # Adding information of the value of the chosen product for all the chosen countries into the timeline
    fig_linechart = go.Figure()

    for i in range(len(product_names)):
        fig_data = (line_data[line_data['product'] == product_names[i]].
                    sort_values(by=['time'], ascending=True).reset_index(drop=True))

        fig_data = fig_data.groupby(['time', 'product'])['value'].sum().reset_index()
        fig_data['product'] = f"{product_names[i]} trendline"

        if monthlydata == [" Monthly Data"]:

            fig_linechart.add_trace(go.Scatter(x=fig_data['time'],
                                               y=fig_data['value'],
                                               name=product_names[i],
                                               line=dict(color=list(product_col.values())[i]),
                                               mode='lines',
                                               hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                             "<br> <b>Value:</b> %{y}"))

        if trendlinedata == [" Trendlines"]:
            trendline_data = px.scatter(fig_data,
                                        x=fig_data['time'],
                                        y="value", color='product',
                                        labels={'Renewables': f"{product_names[i]} Trendline"},
                                        trendline="lowess",
                                        trendline_color_override= list(product_col.values())[i],
                                        trendline_options=dict(frac=0.2))
            trendline = trendline_data.data[1]

            fig_linechart.add_trace(trendline)
            fig_linechart.update_traces(showlegend=True,
                                        hovertemplate="<b>Year:</b> %{x|%Y} <br> <b>Month:</b> %{x|%B}"
                                                      "<br> <b>Value:</b> %{y}")


    fig_linechart.update_layout(height=400,
                                yaxis_title="GWh",
                                legend_title="Energy Source",
                                margin=go.layout.Margin(t=30))

    fig_linechart.update_xaxes(dtick='M3', tickformat="%b\n%Y")

    return fig_linechart, line_title

