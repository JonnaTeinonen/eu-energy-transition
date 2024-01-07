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


# APP
app = dash.Dash(external_stylesheets=[dbc.themes.SANDSTONE], use_pages=True)

#sidebar = html.Div([])

#content = html.Div([])


app.layout = html.Div([
    html.Div([
            dcc.Link(f"{page['name']} | ", href=page["path"])
         for page in dash.page_registry.values()
    ], className='bg-dark text-white font-weight-bold'),
    dash.page_container
])
    # dbc.Row([
    #     dbc.Col(sidebar, width=2, className='bg-primary'),
    #     dbc.Col(content, width=10)],
    #
    # # Viewport height set to 100%
    # style={'height': '100vh'})],
    # Additional margins removed)







if __name__ == "__main__":
    app.run_server(debug=True)