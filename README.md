A Plotly / Dash / Flask web application to analyse and visualise the energy transition data in the European countries. The data is a subset of the International Energy Agency’s (IAE) monthly Electricity Statistics from [Kaggle](https://www.kaggle.com/datasets/ccanb23/iea-monthly-electricity-statistics), licenced under ODC-By v1.0

Please see the details of the file structure and information about each file below.

# Controls:
⋅⋅* All pages have a time series slider to select the time range to analyse. 
⋅⋅* Most pages also have controls for selecting between grouped (fossil fuel, renewables, nuclear) and granular data (individual energy sources)
⋅⋅* Most plots have additional controls for selecting e.g. energy sources and grouping
⋅⋅* Some plots have additional control to show the average share of the energy source in percentage ("Energy share") or the total generated energy in GWh ("Generated energy")
⋅⋅* The second page also has a selection for adding multiple countries to compare.
 
# Boilerplate
--`app.py` contains the code that creates and runs the multi-page app.
# Data
--`/data/` contains the data used in the app. This include the raw IEA Monthly Electricity Statistics data between 2010 and 2022 (`data.csv`) and `data.py`, which loads and prepares the data for consumption on each page of the app

# Dash Pages
--page1: The big picture of the energy transition within the European countries: stacked bar charts of energy mix, tree map of Europe, energy mix time series with LOWESS trendlines

--page2: Interactive comparisons of the energy transition between user-selected European countries

--page3: Analysis of the energy transition within one chosen European country and some descriptive statistics


