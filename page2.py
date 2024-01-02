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
    dbc.Row(dcc.Dropdown(id='chosen-countries',
                         multi=True,
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
              State('chosen-countries', 'value'),
              State('year-range-slider', 'value'))
def update_barchart_products(n_clicks, countries: list, years: list[int]):

    def divide_by_country_total(df_row, lookup_df: pd.DataFrame):
        """Computes the average share for eac product per country"""
        country = df_row['country']
        product_share = df_row.share
        lookup_df = lookup_df[lookup_df['country'].isin([country])]
        country_total = lookup_df['share']

        return product_share / country_total.values[0]

    bar_products = ['Hydro', 'Wind', 'Solar', 'Coal', 'Oil', 'Natural gas', 'Nuclear']

    if years[0] == years[1]:
        chosen_years = [years[0]]
        bar_title = f"The average share of each energy source in the total energy generation in each country in {years[0]}"
    else:
        chosen_years = [x for x in range(years[0], years[1])]
        bar_title = f"The average share of each energy source in the total energy generation in each country between {years[0]}-{years[1]}"

    if type(countries) != list:
        chosen_countries = [countries]
    else:
        chosen_countries = countries

#    bar_data = data[data['product'].isin(bar_products)]
    bar_data = data[data['year'].isin(chosen_years)]
    bar_data = bar_data[bar_data['country'].isin(chosen_countries)]

    bar_data_product =  bar_data[bar_data['product'].isin(bar_products)]

    #barchart_data = bar_data.groupby(['country', 'product'])['share'].mean().reset_index()

    barchart_data_sumproduct = bar_data_product.groupby(['country', 'product'])['share'].sum().reset_index()
    print(barchart_data_sumproduct.head())

    total_data = bar_data[bar_data['product'] == 'Net electricity production']
    barchart_data_sumtotal = total_data.groupby(['country'])['share'].sum().reset_index()
    print(barchart_data_sumtotal.head())
    print(len(barchart_data_sumproduct.share))

    barchart_data = barchart_data_sumproduct
    barchart_data['avg_share'] = barchart_data.apply(
        lambda row: divide_by_country_total(row, barchart_data_sumtotal), axis=1)

    fig_bar = px.bar(barchart_data,
                     x='avg_share',
                     y='country',
                     color='product',
                     color_discrete_sequence=px.colors.qualitative.Prism,
                     width=800,
                     height=450,
                     orientation='h')

    fig_bar.update_traces(width=0.5, hovertemplate="Share: %{x}")

    fig_bar.update_layout(xaxis_title='Average share in percentage',
                          yaxis_title='',
                          showlegend=True,
                          legend_title="Energy Source",
                          margin=go.layout.Margin(t=27))

    return fig_bar, bar_title



if __name__ == "__main__":
    app.run_server(debug=True)