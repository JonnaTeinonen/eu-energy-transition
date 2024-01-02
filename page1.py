import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
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

#Bar chart
# bar_products = ['Hydro', 'Wind', 'Solar', 'Coal', 'Oil', 'Natural gas', 'Others', 'Nuclear', 'Other renewables']
# bar_data = data[data['product'].isin(bar_products)]
# bar_data = bar_data[bar_data['country'] == 'Netherlands']
# bar_data = bar_data[bar_data['year'] == 2012]
# barchart_data = bar_data.groupby(['product'])['share'].mean().reset_index()
#
# fig_bar = get_barchart(barchart_data, x='product', y=['share'], color_var='product')




# APP
app = dash.Dash(external_stylesheets=[dbc.themes.SANDSTONE])

sidebar = html.Div([
    dbc.Row([
             html.P("Choose the country:",
                    style={'margin-top': '80px', 'margin-bottom': '10px'})],
            className='text-white'),
    dbc.Row(dcc.Dropdown(id='chosen-country',
                         multi=False,
                         value='Netherlands',
                         options=[{'label': x, 'value': x} for x in country_var],
                         style={'width': '280px'})),

    dbc.Row(
        html.Button(id='apply-button', n_clicks=0, children='apply',
                                style={'margin-top': '2px'},
                                className='bg-dark text-white'))
])

content = html.Div([
    dbc.Row(style={'height': '2vh'}, className='bg-primary'),
    dbc.Row([
        html.P('The Electricity Production in European Countries in 2010-2022',
               style={'font-weight': 'bold',
                      'font-size': 25}),
    ],
        style={'height': '5vh', 'textAlign': 'center'}, className='bg-primary text-white font-weight-bold'
    ),
    dbc.Row([html.P('Choose the time range:', style={'margin-top': '15px', 'margin-bottom': '10px'}),
             dcc.RangeSlider(2010,
                             2023,
                             1,
                             value=[2010, 2023],
                             id='year-range-slider',
                             marks={i: '{}'.format(i) for i in range(2010, 2024)})]),


    dbc.Row([
        dbc.Col([html.P(id='bar-title', style={'margin-top':'15px'}),
                 dcc.Graph(id='barchart-products')
                 ]),
        dbc.Col([html.Label(id='table-title', style={'margin-top':'15px',
                                                     'margin-bottom':'15px'}),
                 html.Div(id="product-table", style={'margin-bottom':'15px'}),
                 dbc.Row([dcc.Dropdown(id='chosen-product',
                                       multi=False,
                                       value='Renewables',
                                       options=['Renewables', 'Fossil fuels', 'Nuclear'],
                                       style={'width': '200px'}),

                          dcc.Dropdown(id='chosen-val-table',
                                      multi=False,
                                      value='Energy share',
                                      options=['Energy share', 'Generated energy'],
                                      style={'width': '200px',
                                             'margin-left': '15px'})]
                )
                 ])
    ], style={'height': '55vh'}),

    dbc.Row([html.Label(id='line-title', style={'margin-top':'50px',
                                                'margin-left':'80px'}),
             dcc.Graph(id='linechart-value')])
])


app.layout = dbc.Container([
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

@app.callback(Output('barchart-products', 'figure'),
              Output('bar-title', 'children'),
              Input('apply-button', 'n_clicks'),
              State('chosen-country', 'value'),
              State('year-range-slider', 'value'))
def update_barchart_products(n_clicks, country: str, years: list[int]):
    bar_products = ['Hydro', 'Wind', 'Solar', 'Coal', 'Oil', 'Natural gas', 'Others', 'Nuclear']

    if years[0] == years[1]:
        chosen_years = [years[0]]
        bar_title = f"The average share of each energy source in the total energy generation in {country} in {years[0]}"
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        bar_title = f"The average share of each energy source in the total energy generation in {country} between {years[0]}-{years[1]}"

    bar_data = data[data['product'].isin(bar_products)]
    bar_data = bar_data[bar_data['country'] == country]
    bar_data = bar_data[bar_data['year'].isin(chosen_years)]

    barchart_data = bar_data.groupby(['product'])['share'].mean().reset_index()

    fig_bar = px.bar(barchart_data,
                     x='product',
                     y=['share'],
                     color='product',
                     color_discrete_sequence=px.colors.qualitative.Prism,
                     width=800,
                     height=450)

    fig_bar.update_traces(width=0.5, hovertemplate="Share: %{y}")

    fig_bar.update_layout(xaxis_title='Energy Source',
                          yaxis_title='Average share in percentage',
                          showlegend=True,
                          legend_title="",
                          xaxis_type='category',
                          xaxis=dict(tickmode='array',
                                     tickvals=['Hydro', 'Wind', 'Solar', 'Coal', 'Oil', 'Natural gas', 'Others', 'Nuclear']),
                          margin=go.layout.Margin(t=27))

    return fig_bar, bar_title


@app.callback(Output('linechart-value', 'figure'),
              Output('line-title', 'children'),
              Input('apply-button', 'n_clicks'),
              State('chosen-country', 'value'),
              State('year-range-slider', 'value'))
def update_linechart_value(n_clicks, country, years):
    line_products = ['Renewables', 'Fossil fuels', 'Nuclear']

    if years[0] == years[1]:
        chosen_years = [years[0]]
        line_title = f"The average amount of generated electricity in GWh in {country} in {years[0]}"

    else:
        chosen_years = [x for x in range(years[0], years[1])]
        line_title = f"The average amount of generated electricity in GWh in {country} between {years[0]}-{years[1]}"

    line_data = data[data['product'].isin(line_products)]
    line_data = line_data[line_data['country'] == country]
    line_data = line_data[line_data['year'].isin(chosen_years)]
#    line_data_sorted = line_data.sort_values(by=['year', 'month'], ascending=True).reset_index(drop=True)


    fig_linechart = go.Figure()
    colors = px.colors.qualitative.Set1

    for i in range(len(line_products)):
        fig_data = line_data[line_data['product'] == line_products[i]].sort_values(by=['year', 'month'],
                                                                                   ascending=True).reset_index(drop=True)
        fig_linechart.add_trace(go.Scatter(x=[fig_data['year'], fig_data['month']],
                                           y=fig_data['value'],
                                           name=line_products[i],
                                           line=dict(color=colors[2-i]),
                                           mode='lines',
                                           hovertemplate="Year: %{x[0]} <br> Month: %{x[1]} </br> Value: %{y}"))

    fig_linechart.update_layout(height=400,
                                yaxis_title="GWh",
                                margin=go.layout.Margin(t=30))

    return fig_linechart, line_title


@app.callback(Output('product-table', 'children'),
              Output('table-title', 'children'),
              Input('apply-button', 'n_clicks'),
              Input('chosen-product', 'value'),
              Input('chosen-val-table', 'value'),
              State('chosen-country', 'value'),
              State('year-range-slider', 'value'))
def update_product_table(n_clicks, product, var, country, years):

    if years[0] == years[1]:
        chosen_years = [years[0]]
        table_title = f"The descriptive statistics for {var.lower()} for {product.lower()} in {country} in {years[0]}"
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        table_title = f"The descriptive statistics for {var.lower()} for {product.lower()} in {country} between {years[0]}-{years[1]}"

    df_data = data[data['country'] == country]
    df_data = df_data[df_data['product'] == product]
    df_data = df_data[df_data['year'].isin(chosen_years)]

    data_grouped = df_data.groupby(['year'])

    df_means = round(data_grouped.mean(['share', 'value']).reset_index(), 3)
    df_medians = round(data_grouped.median(['share', 'value']).reset_index(), 3)
    df_variances = round(data_grouped[['share', 'value']].agg('var'), 3).reset_index()
    df_stds = round(data_grouped[['share', 'value']].agg('std'), 3).reset_index()
    df_mins = round(data_grouped.min(['share', 'value']).reset_index(), 3)
    df_maxs = round(data_grouped.max(['share', 'value']).reset_index(), 3)

    data_share = {'Year': df_data['year'].unique(),
                  'Mean': df_means.share,
                  'Median': df_medians.share,
                  'Variance': df_variances.share,
                  'SD': df_stds.share,
                  'Min': df_mins.share,
                  'Max': df_maxs.share}

    df_share = pd.DataFrame.from_dict(data_share)

    data_value = {'Year': df_data['year'].unique(),
                  'Mean': df_means.value,
                  'Median': df_medians.value,
                  'Variance': df_variances.value,
                  'SD': df_stds.value,
                  'Min': df_mins.value,
                  'Max': df_maxs.value}

    df_value = pd.DataFrame.from_dict(data_value)

    output_share = html.Div([dash_table.DataTable(data=df_share.to_dict('records'))])
    output_value = html.Div([dash_table.DataTable(data=df_value.to_dict('records'))])

    if var == 'Energy share':
        return output_share, table_title
    else:
        return output_value, table_title


if __name__ == "__main__":
    app.run_server(debug=True)