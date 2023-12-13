import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
import matplotlib
from matplotlib.pyplot import cm
import plotly.express as px
import pandas as pd
from functions import *

# Current version
# Data
# Setting so that all the variables can bee seen
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 12)
data = reading_csv('C:\\Users\\JonnaMS\\Desktop\\Itseopiskelu\\Portfolio\\Datasetit\\Monthly Electricity Production in GWh [2010-2022].zip')

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
        html.P('The Electricity Production in European Countries in 2010-2022'),
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
        dbc.Col([html.P(),
                 html.P(id='bar-title'),
                 dcc.Graph(id='barchart-products')
                 ]),
        dbc.Col([dcc.Dropdown(id='chosen-product',
                         multi=False,
                         value='Renewables',
                         options=['Renewables', 'Fossil fuels', 'Nuclear'],
                         style={'width': '200px'}),
                html.Div(id="product-table-share")])
    ], style={'height': '50vh'}),

    dbc.Row([html.P(id='line-title'),
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
                          legend_title=False,
                          xaxis_type='category',
                          xaxis=dict(tickmode='array',
                                     tickvals=['Hydro', 'Wind', 'Solar', 'Coal', 'Oil', 'Natural gas', 'Others', 'Nuclear']))

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
        fig_data = line_data[line_data['product'] == line_products[i]].sort_values(by=['year', 'month'], ascending=True).reset_index(drop=True)
        pd.options.display.max_rows = 999
        print(fig_data['year'])
        fig_linechart.add_trace(go.Scatter(x=[fig_data['year'], fig_data['month']],
                                           y=fig_data['value'],
                                           name=line_products[i],
#                                           name=line_data['product'].unique()[i],
                                           line=dict(color=colors[i]),
                                           mode='lines',
                                           hovertemplate="Year: %{x[0]} <br> Month: %{x[1]} </br> Value: %{y}"))

    fig_linechart.update_layout(height=400,
                                title=line_title,
                                yaxis_title="GWh")

    return fig_linechart, line_title


@app.callback(Output('product-table-share', 'children'),
              Input('apply-button', 'n_clicks'),
              Input('chosen-product', 'value'),
              State('chosen-country', 'value'),
              State('year-range-slider', 'value'))
def update_product_table(n_clicks, product, country, years):

    if years[0] == years[1]:
        chosen_years = [years[0]]
    else:
        chosen_years = [x for x in range(years[0], years[1])]

    df_data = data[data['country'] == country]
    df_data = df_data[df_data['product'] == product]
    df_data = df_data[df_data['year'].isin(chosen_years)]

    df_means = round(df_data.groupby(['product', 'year']).mean(['share', 'value']).reset_index(), 3)
    df_medians = round(df_data.groupby(['product', 'year']).median(['share', 'value']).reset_index(), 3)
    df_variances = round(df_data.groupby(['product', 'year'])[['share', 'value']].agg('var'), 3).reset_index()
    df_stds = round(df_data.groupby(['product', 'year'])[['share', 'value']].agg('std'), 3).reset_index()
    df_mins = round(df_data.groupby(['product', 'year']).min(['share', 'value']).reset_index(), 3)
    df_maxs = round(df_data.groupby(['product', 'year']).max(['share', 'value']).reset_index(), 3)

    data_share = {'Year': df_data['year'].unique(),
                  'Mean': df_means.share,
                  'Median': df_medians.share,
                  'Variance': df_variances.share,
                  'SD': df_stds.share,
                  'Min': df_mins.share,
                  'Max': df_maxs.share}

    #df_share = data_share.to_dict('records')
    df_share = pd.DataFrame.from_dict(data_share)
    #df_share_table = html.Div(dash_table.DataTable(dta=df_share,
    #                                               columns=[x for x in df_share.columns]))

    output = html.Div(
        [
            dash_table.DataTable(
                data=df_share.to_dict('records'),
            )
        ]
    )



    # fig_bar = get_barchart(barchart_data,
    #                        x='product',
    #                        y=['share'],
    #                        color_var='product',
    #                        colors=px.colors.qualitative.Prism,
    #                        width=900,
    #                        x_title='Energy Source',
    #                        y_title='Average share in percentage',
    #                        title=bar_title,
    #                        bar_width=1)

    return output



if __name__ == "__main__":
    app.run_server(debug=True)