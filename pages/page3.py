import dash
from dash import html, dcc, Output, Input, dash_table, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Creates the page
dash.register_page(__name__, name='Page 3: Individual Countries')

# DATA
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
    dbc.Row([html.P('Energy source grouping:', style={'margin-top': '110px', 'margin-bottom': '5px'})],
            className='text-white'
            ),

    dbc.Row(dcc.RadioItems(id='energy-grouping-page3', options=[' Grouped', ' Granular'], value=' Granular'),
            style={'padding': 10, 'flex': 1, 'padding-left': 10},
            className='text-white'
            ),

    dbc.Row([html.P("Choose the country:", style={'margin-top': '20px'})],
            className='text-white'
            ),

    dbc.Row(dcc.Dropdown(id='chosen-country-page3',
                         multi=False,
                         value='Netherlands',
                         options=[{'label': x, 'value': x} for x in country_var],
                         style={'width': '175px'}
                         )
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
               )
    ], className='bg-primary text-white font-weight-bold'
    ),

    dbc.Row([html.P('Choose the time range:', style={'margin-top': '15px', 'margin-bottom': '10px'}),
             dcc.RangeSlider(2010,
                             2023,
                             1,
                             value=[2010, 2023],
                             id='year-range-slider-page3',
                             marks={i: '{}'.format(i) for i in range(2010, 2024)})
             ]),

    dbc.Row([
        dbc.Col([dbc.Row([html.P(id='bar-title-page3',
                                 style={'margin-top': '15px', 'font-weight': 'bold', 'textAlign': 'center'})
                          ]),

                 dbc.Row([dcc.Graph(id='barchart-products-page3')])
                 ]),

        dbc.Col([dbc.Row([html.Label(id='table-title-page3',
                                     style={'margin-top': '15px',
                                            'margin-bottom': '15px',
                                            'font-weight': 'bold',
                                            'textAlign': 'center'})
                          ]),

                 dbc.Row([dcc.Dropdown(id='chosen-product-page3',
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
                                       style={'width': '200px', 'margin-bottom': '15px'}),

                          dcc.Dropdown(id='chosen-val-table-page3',
                                       multi=False,
                                       value='Energy share',
                                       options=['Energy share', 'Generated energy'],
                                       style={'width': '200px', 'margin-left': '15px', 'margin-bottom': '15px'})
                          ]),

                 dbc.Row([html.Div(id="product-table-page3", style={'margin-bottom': '15px'})]),

                 ])
    ]),

    dbc.Row([html.P(id='line-title-page3',
                    style={'margin-top': '50px', 'margin-left': '80px', 'font-weight': 'bold', 'textAlign': 'center'})
             ]),

    dbc.Row([dcc.Graph(id='linechart-value-page3')])
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

@callback(Output('barchart-products-page3', 'figure'),
          Output('bar-title-page3', 'children'),
          Input('chosen-country-page3', 'value'),
          Input('year-range-slider-page3', 'value'),
          Input('energy-grouping-page3', 'value'))
def update_barchart_products_page3(country: str, years: list[int], grouping: str):

    if grouping == ' Granular':
        product_names, product_col = products(False)
    else:
        product_names, product_col = products(True)

    if years[0] == years[1]:
        chosen_years = [years[0]]
        bar_title = (f"The average share of each energy source in the total energy generation in {country} "
                     f"in {years[0]}")
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        bar_title = (f"The average share of each energy source in the total energy generation in {country} between "
                     f"{years[0]}-{years[1]}")

    bar_data = data[data['country'] == country]
    bar_data = bar_data[bar_data['year'].isin(chosen_years)]
    bar_data_product = bar_data[bar_data['product'].isin(product_names)]

    barchart_data_sumproduct = bar_data_product.groupby(['product'])['share'].sum().reset_index()

    total_data = bar_data[bar_data['product'] == 'Net electricity production']
    barchart_data_sumtotal = total_data['share'].sum()

    barchart_data = barchart_data_sumproduct
    barchart_data['avg_share'] = (barchart_data_sumproduct['share'] / barchart_data_sumtotal) * 100

    fig_bar = px.bar(barchart_data,
                     x='product',
                     y=['avg_share'],
                     color='product',
                     color_discrete_sequence=list(product_col.values()),
                     width=900,
                     height=550,
                     category_orders={"product": product_names})

    fig_bar.update_traces(width=0.5, hovertemplate="<b>Share:</b> %{y}")

    fig_bar.update_layout(xaxis_title='Energy Source',
                          yaxis_title='Average share in percentage',
                          showlegend=True,
                          legend_title="",
                          xaxis_type='category',
                          xaxis=dict(tickmode='array',
                                     tickvals=['Hydro',
                                               'Wind',
                                               'Solar',
                                               'Coal',
                                               'Oil',
                                               'Natural gas',
                                               'Others',
                                               'Nuclear']),
                          margin=go.layout.Margin(t=27))

    return fig_bar, bar_title


@callback(Output('linechart-value-page3', 'figure'),
          Output('line-title-page3', 'children'),
          Input('chosen-country-page3', 'value'),
          Input('year-range-slider-page3', 'value'),
          Input('energy-grouping-page3', 'value'))
def update_linechart_value_page3(country, years, grouping):

    if grouping == ' Granular':
        product_names, product_col = products(False)
    else:
        product_names, product_col = products(True)

    if years[0] == years[1]:
        chosen_years = [years[0]]
        line_title = f"The average amount of generated electricity in GWh in {country} in {years[0]}"

    else:
        chosen_years = [x for x in range(years[0], years[1])]
        line_title = f"The average amount of generated electricity in GWh in {country} between {years[0]}-{years[1]}"

    line_data = data[data['product'].isin(product_names)]
    line_data = line_data[line_data['country'] == country]
    line_data = line_data[line_data['year'].isin(chosen_years)]
    line_data['time'] = pd.to_datetime(line_data['time'], format="%B %Y")

    # Creating an empty figure with the correct timeline
    fig_linechart = go.Figure()
    fig_linechart.add_trace(go.Scatter(x=line_data['time'],
                                       y=pd.Series(dtype=object),
                                       mode='lines'))

    for i in range(len(product_names)):
        fig_data = line_data[line_data['product'] ==
                             product_names[i]].sort_values(by=['time'], ascending=True).reset_index(drop=True)

        fig_linechart.add_trace(go.Scatter(x=fig_data['time'],
                                           y=fig_data['value'],
                                           name=product_names[i],
                                           line=dict(color=list(product_col.values())[i]),
                                           mode='lines',
                                           hovertemplate="Year: %{x[0]} <br> Month: %{x[1]} </br> Value: %{y}"))

    fig_linechart.update_layout(height=400,
                                yaxis_title="GWh",
                                legend_title="Energy source",
                                margin=go.layout.Margin(t=30))

    fig_linechart.update_xaxes(dtick='M3', tickformat="%b\n%Y")

    return fig_linechart, line_title


@callback(Output('product-table-page3', 'children'),
          Output('table-title-page3', 'children'),
          Input('chosen-product-page3', 'value'),
          Input('chosen-val-table-page3', 'value'),
          Input('chosen-country-page3', 'value'),
          Input('year-range-slider-page3', 'value'))
def update_product_table_page3(product, var, country, years):

    if years[0] == years[1]:
        chosen_years = [years[0]]

        if var == "Energy share":
            table_title = (f"The descriptive statistics for {var.lower()} for {product.lower()} "
                           f"in {country} in {years[0]}")
        else:
            table_title = (f"The descriptive statistics for {var.lower()} for {product.lower()} in GWh in "
                           f"{country} in {years[0]}")
    else:
        chosen_years = [x for x in range(years[0], years[1])]

        if var == "Energy share":
            table_title = (f"The descriptive statistics for {var.lower()} for {product.lower()} in {country} "
                           f"between {years[0]}-{years[1]}")
        else:
            table_title = (f"The descriptive statistics for {var.lower()} for {product.lower()} in GWh in"
                           f" {country} between {years[0]}-{years[1]}")

    df_data = data[data['country'] == country]
    df_data = df_data[df_data['product'] == product]
    df_data = df_data[df_data['year'].isin(chosen_years)]

    data_grouped = df_data.groupby(['year'])

    df_count_share = data_grouped['share'].count().reset_index()
    df_count_value = data_grouped['value'].count().reset_index()
    df_means = data_grouped.mean(['share', 'value']).reset_index()
    df_medians = data_grouped.median(['share', 'value']).reset_index()
    df_variances = data_grouped[['share', 'value']].agg('var').reset_index()
    df_stds = data_grouped[['share', 'value']].agg('std').reset_index()
    df_mins = data_grouped.min(['share', 'value']).reset_index()
    df_maxs = data_grouped.max(['share', 'value']).reset_index()

    data_share = {'Year': df_data['year'].unique(),
                  'Count': df_count_share.share,
                  'Mean': round(df_means.share * 100, 3),
                  'Median': round(df_medians.share * 100, 3),
                  'Variance': round(df_variances.share * 100, 3),
                  'SD': round(df_stds.share * 100, 3),
                  'Min': round(df_mins.share * 100, 3),
                  'Max': round(df_maxs.share * 100, 3)}

    df_share = pd.DataFrame.from_dict(data_share)

    data_value = {'Year': df_data['year'].unique(),
                  'Count': df_count_value.value,
                  'Mean': round(df_means.value, 3),
                  'Median': round(df_medians.value, 3),
                  'Variance': round(df_variances.value, 3),
                  'SD': round(df_stds.value, 3),
                  'Min': round(df_mins.value, 3),
                  'Max': round(df_maxs.value, 3)}

    df_value = pd.DataFrame.from_dict(data_value)

    output_share = html.Div([dash_table.DataTable(data=df_share.to_dict('records'))])
    output_value = html.Div([dash_table.DataTable(data=df_value.to_dict('records'))])

    if var == 'Energy share':
        return output_share, table_title
    else:
        return output_value, table_title
