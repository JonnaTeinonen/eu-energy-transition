import pandas as pd

# DATA
import os; print(os.getcwd())
# data = pd.read_csv('./data/data.csv')
data = pd.read_csv('./data.csv')

# Changing the column names to lowercase
col_names = list(data.columns)
col_names_low = [col.lower() for col in col_names[0:9]]
col_names_low.extend(col_names[9:])
data.columns = col_names_low

# Choosing the EU-countries
europe_countries = ['Austria', 'Belgium', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
                    'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
                    'Netherlands', 'Norway', 'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden',
                    'Switzerland', 'Bulgaria', 'Croatia', 'Cyprus', 'Malta', 'Romania', 'United Kingdom',
                    'North Macedonia']
europe_countries.sort()

DATA: pd.DataFrame = data[data['country'].isin(europe_countries)]

# Variables
UNIQUE_COUNTRIES: list[str] = DATA.country.unique()
# YEAR_VAR: list[int] = data.year.unique()

# HELP FUNCTIONS
def products(category: bool = False) -> tuple[list[str], dict[str, str]]:
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
