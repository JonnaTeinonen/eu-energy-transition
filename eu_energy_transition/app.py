import dash
from dash import html
import dash_bootstrap_components as dbc

# APP
app = dash.Dash(external_stylesheets=[dbc.themes.SANDSTONE],
                use_pages=True,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
                )


app.layout = html.Div([
    html.Div([
        dbc.Button(f"{page['name']}", href=page['path'], color='dark', style={'font-size': '15px'})
        for page in dash.page_registry.values()

    ], className='bg-dark text-white font-weight-bold'),

    dash.page_container,
])


if __name__ == "__main__":
    app.run_server(debug=True)