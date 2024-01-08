import dash
from dash import html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from eu_energy_transition.data.data import DATA, UNIQUE_COUNTRIES, products

# Creates the page
dash.register_page(__name__, name='Country Comparison')

# APP
sidebar = html.Div([
    dbc.Row([html.P('Energy source grouping:', style={'margin-top': '110px', 'margin-bottom': '5px'})],
            className='text-white'
            ),

    dbc.Row(dcc.RadioItems(id='energy-grouping-page2', options=[' Grouped', ' Granular'], value=' Granular'),
            style={'padding': 10, 'flex': 1, 'padding-left': 10},
            className='text-white'
            ),

    dbc.Row([html.P("Choose the country:", style={'margin-top': '20px'})],
            className='text-white'
            ),

    dbc.Row(dcc.Dropdown(id='chosen-countries-page2',
                         multi=True,
                         value='Netherlands',
                         options=[{'label': x, 'value': x} for x in UNIQUE_COUNTRIES],
                         style={'width': '175px'})
            )
])

content = html.Div([
    dbc.Row(style={'height': '2vh'}, className='bg-primary'),

    dbc.Row([
        html.P('The Electricity Production in European Countries in 2010-2022',
               style={'font-weight': 'bold',
                      'font-size': 25,
                      'height': '3vh',
                      'textAlign': 'center'}
               ),
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
        dbc.Col([dbc.Row([html.P(id='bar-title-page2',
                                 style={'margin-top': '15px', 'font-weight': 'bold', 'textAlign': 'center'})
                          ]),

                 dbc.Row([dcc.Graph(id='barchart-products-page2')])
                 ]),

        dbc.Col([dbc.Row([html.P(id='treemap-title-page2',
                                 style={'margin-top': '15px',
                                        'height': '5vh',
                                        'font-weight': 'bold',
                                        'textAlign': 'center'})
                          ]),

                 dbc.Row([dcc.Graph(id='treemap-countries-page2')])
                 ])
    ]),

    dbc.Row([html.P(id='line-title-page2')], style={'font-weight': 'bold', 'textAlign': 'center'}),

    dbc.Row([dcc.Dropdown(id='chosen-product-page2',
                          multi=False,
                          value='Renewables',
                          options=['Fossil fuels',
                                   'Nuclear',
                                   'Renewables',
                                   'Coal',
                                   'Hydro',
                                   'Nuclear',
                                   'Wind',
                                   'Solar',
                                   'Natural gas',
                                   'Oil'],
                          style={'width': '280px',
                                 'margin-left': '40px'})
             ]),

    dbc.Row([dcc.Graph(id='linechart-value-page2')])
])

layout = dbc.Container([
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


@callback(Output('barchart-products-page2', 'figure'),
          Output('bar-title-page2', 'children'),
          Input('chosen-countries-page2', 'value'),
          Input('year-range-slider-page2', 'value'),
          Input('energy-grouping-page2', 'value'))
def update_barchart_products_page2(countries: list, years: list[int], grouping: str):

    def divide_by_country_total(df_row, lookup_df: pd.DataFrame) -> float:
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
        bar_title = (f"The average share of each energy source in the total energy generation per country "
                     f"in {years[0]}")
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        bar_title = (f"The average share of each energy source in the total energy generation per country between "
                     f"{years[0]}-{years[1]}")

    if type(countries) != list:
        chosen_countries = [countries]
    else:
        chosen_countries = countries

    bar_data = DATA[DATA['year'].isin(chosen_years)]
    bar_data = bar_data[bar_data['country'].isin(chosen_countries)]

    bar_data_product = bar_data[bar_data['product'].isin(product_names)]
    barchart_data_sumproduct = bar_data_product.groupby(['country', 'product'])['share'].sum().reset_index()

    total_data = bar_data[bar_data['product'] == 'Net electricity production']
    barchart_data_sumtotal = total_data.groupby(['country'])['share'].sum().reset_index()

    barchart_data = barchart_data_sumproduct

    # Computes the average GWh per product
    barchart_data['avg_share'] = barchart_data.apply(
        lambda row: divide_by_country_total(row, barchart_data_sumtotal), axis=1)

    # Changes the values from decimal format to percentages
    barchart_data['avg_share'] = barchart_data['avg_share'] * 10

    fig_bar = px.bar(barchart_data,
                     x='avg_share',
                     y='country',
                     color='product',
                     color_discrete_map=product_col,
                     width=1020,
                     height=550,
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
          Input('chosen-countries-page2', 'value'),
          Input('year-range-slider-page2', 'value'),
          Input('energy-grouping-page2', 'value'))
def update_treemap_countries_page2(countries: list[str], years: list[int], grouping: str):

    if grouping == ' Granular':
        product_names, product_col = products(False)
    else:
        product_names, product_col = products(True)

    if years[0] == years[1]:
        chosen_years = [years[0]]
        treemap_title = f"The total amount of generated energy for each energy source per country {years[0]}"
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        treemap_title = (f"The total amount of generated energy for each energy source per country between "
                         f"{years[0]}-{years[1]}")

    if type(countries) != list:
        chosen_countries = [countries]
    else:
        chosen_countries = countries

    figure_data = DATA[DATA['year'].isin(chosen_years)]
    figure_data = figure_data[figure_data['country'].isin(chosen_countries)]
    figure_data = figure_data[figure_data['product'].isin(product_names)]

    figure_data = figure_data.groupby(['country', 'product'])['value'].sum().reset_index()
    figure_data = figure_data[figure_data['value'] != 0]        # Removes sums from data that are 0
    figure_data['All'] = 'All'                                  # Creates a single root node

    fig_treemap = px.treemap(figure_data,
                             path=['All', 'country', 'product'],
                             values='value',
                             color='product',
                             width=1200,
                             height=550,
                             color_discrete_map=product_col)

    fig_treemap.update_traces(root_color="white",
                              hovertemplate="<b>%{parent}</b> | %{label} <br> "
                                            "<b>Total energy produced (GWh):</b> %{value}")
    fig_treemap.update_layout(margin=go.layout.Margin(t=25))

    return fig_treemap, treemap_title


@callback(Output('linechart-value-page2', 'figure'),
          Output('line-title-page2', 'children'),
          Input('chosen-countries-page2', 'value'),
          Input('year-range-slider-page2', 'value'),
          Input('chosen-product-page2', 'value'))
def update_linechart_value_page2(countries, years, chosen_product):

    if years[0] == years[1]:
        chosen_years = [years[0]]
        line_title = (f"The total amount of generated electricity with {chosen_product.lower()} in GWh per country "
                      f"in {years[0]}")
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        line_title = (f"The total amount of generated electricity  with {chosen_product.lower()} in GWh per country "
                      f"between {years[0]}-{years[1]}")

    if type(countries) != list:
        chosen_countries = [countries]
    else:
        chosen_countries = countries

    line_data = DATA[DATA['product'] == chosen_product]
    line_data = line_data[line_data['country'].isin(chosen_countries)]
    line_data = line_data[line_data['year'].isin(chosen_years)]
    line_data['time'] = pd.to_datetime(line_data['time'], format="%B %Y")

    # Creating an empty figure with the correct timeline
    fig_linechart = go.Figure()
    fig_linechart.add_trace(go.Scatter(x=line_data['time'], y=pd.Series(dtype=object), mode='lines'))

    # Adding information of the value of the chosen product for all the chosen countries into the timeline
    for i in range(len(chosen_countries)):
        fig_data = (line_data[line_data['country'] == chosen_countries[i]].
                    sort_values(by=['time'], ascending=True).reset_index(drop=True))

        fig_linechart.add_trace(go.Scatter(x=fig_data['time'],
                                           y=fig_data['value'],
                                           name=chosen_countries[i],
                                           line=dict(color=px.colors.qualitative.Dark24[i]),
                                           mode='lines',
                                           hovertemplate="<b>Year:</b> %{x|%Y} <br> "
                                                         "<b>Month:</b> %{x|%B}<br> "
                                                         "<b>GWh:</b> %{y}"))

    fig_linechart.update_layout(height=350,
                                yaxis_title="GWh",
                                legend_title="Country",
                                margin=go.layout.Margin(t=30))

    fig_linechart.update_xaxes(dtick='M3', tickformat="%b\n%Y")

    return fig_linechart, line_title
