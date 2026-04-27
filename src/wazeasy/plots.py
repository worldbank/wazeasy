from datetime import datetime as dt
import itertools
import pandas as pd
import utils
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import plotly.graph_objects as go
import altair as alt

alt.data_transformers.enable('json')

def jams_per_day(data):
    '''
    Plot the number of unique traffic jams per day using Altair.

    Parameters:
    - data (DataFrame): A Dask/Pandas DataFrame containing 'date' and 'uuid' columns.

    Returns:
    - None: Displays the plot and optionally saves it to a file.
    '''
    # Data processing remains the same
    if utils.is_dask_dataframe(data):
        jams_per_day = data.groupby('date')['uuid'].nunique().compute().reset_index()
    else:
        jams_per_day = data.groupby('date')['uuid'].nunique().reset_index()
    jams_per_day.sort_values('date', inplace=True)
    
    # Create Altair chart
    chart = alt.Chart(jams_per_day).mark_line().add_selection(
        alt.selection_interval(bind='scales')
    ).encode(
        x=alt.X('date:T', 
                title='Date',
                axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('uuid:Q', 
                title='Number of Jams'),
        tooltip=['date:T', 'uuid:Q']
    ).properties(
        width=800,
        height=400,
        title='Number of Jams per day'
    ).interactive()
    
    # Display the chart
    chart.show()
    

def jams_per_day_rolling_avg(data, window=7):
    '''
    Plot the number of unique traffic jams per day with a rolling average.

    Parameters:
    - data (DataFrame): A Dask/Pandas DataFrame containing 'date' and 'uuid' columns.
    - window (int): Window size for rolling average. Default is 7 days.

    Returns:
    - None: Displays the plot and optionally saves it to a file.
    '''
    if utils.is_dask_dataframe(data):
        jams_per_day = data.groupby('date')['uuid'].nunique().compute().reset_index()
    else:
        jams_per_day = data.groupby('date')['uuid'].nunique().reset_index()
    jams_per_day.sort_values('date', inplace=True)
    jams_per_day['rolling_avg'] = jams_per_day['uuid'].rolling(window=window).mean()

    plt.figure(figsize=(15, 7))
    plt.plot(jams_per_day['date'], jams_per_day['uuid'], alpha=0.5, label='Daily')
    plt.plot(jams_per_day['date'], jams_per_day['rolling_avg'], label=f'{window}-day rolling average')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Number of Jams')
    plt.title('Number of Jams per day with Rolling Average')
    plt.legend()
    plt.show()
    plt.close()

def jams_per_day_per_level(data):
    '''
    Plot the number of unique traffic jams per day, grouped by congestion level.

    Parameters:
    - data (DataFrame): A Dask DataFrame containing 'date', 'level', and 'uuid' columns.

    Returns:
    - None: Displays the plot and optionally saves it to a file.
    '''
    if utils.is_dask_dataframe(data):
        jams_per_day_per_level = (data.groupby(['date', 'level'])['uuid'].nunique().compute()).reset_index()
    else:
        jams_per_day_per_level = (data.groupby(['date', 'level'])['uuid'].nunique()).reset_index()
    jams_per_day_per_level.sort_values('date', inplace = True)
    jams_per_day_per_level['date'] = pd.to_datetime(jams_per_day_per_level['date'])
    jams_per_day_per_level['level_str'] = jams_per_day_per_level['level'].astype(str)

    colors_by_level = {1: '#FFD700', 2: '#FFA500', 3: '#FF4500', 4: '#FF0000', 5: '#4DFF00'}
    domain = [str(k) for k in sorted(colors_by_level.keys())]
    range_ = [colors_by_level[k] for k in sorted(colors_by_level.keys())]

    chart = (
        alt.Chart(jams_per_day_per_level)
        .mark_line()
        .encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('uuid:Q', title='Number of Jams'),
            color=alt.Color(
                'level_str:N',
                title='Level',
                scale=alt.Scale(domain=domain, range=range_)
            )
        )
        .properties(
            title='Number of Jams per day per level',
            width=900,
            height=400
        )
        .interactive()
    )

    chart.show()

def jams_monthly_aggregated(data):
    '''
    Plot the number of unique traffic jams aggregated by month.

    Parameters:
    - data (DataFrame): A Dask DataFrame containing 'year', 'month', and 'uuid' columns.

    Returns:
    - None: Displays the plot and optionally saves it to a file.
    '''
    if utils.is_dask_dataframe(data):
        jams_per_month = data.groupby(['year', 'month'])['uuid'].nunique().compute()
    else:
        jams_per_month = data.groupby(['year', 'month'])['uuid'].nunique()

    jams_per_month = jams_per_month.reset_index()
    jams_per_month['month_with_year'] = jams_per_month.apply(lambda row: dt.strptime('{}-{}-{}'.format(row['year'], row['month'], '15'), '%Y-%m-%d'), axis = 1)
    jams_per_month.set_index('month_with_year', inplace = True)
    plt.figure(figsize=(15, 7))
    plt.bar(jams_per_month.index, jams_per_month.uuid, width=10)
    plt.xticks(jams_per_month.index, rotation=45)
    plt.xlabel('Month')
    plt.ylabel('Number of Jams')
    plt.title('Number of Jams per month')
    plt.show()
    plt.close()

def plot_tci_daily_spatial(df, agg_spatial, agg_column, agg_spatial_name, start_date = None, end_date = None, dow = None):
    '''
    Plot the daily regional Traffic Congestion Index (TCI), aggregated at the area of operation level.

    Parameters:
    - df (DataFrame): The DataFrame (Dask or Pandas) containing traffic jam data.
    - agg_spatial (str): Name of column used for spatial aggregation.
    - agg_column (str): The column to aggregate.
    - agg_spatial_name (str): Name of the spatial aggregation level (e.g. region name) for labeling the plot.

    Returns:
    - None: Displays the plot and optionally saves it to a file.
    '''

    tci = utils.tci_temporal_spatial(df, ['date'], agg_spatial, agg_column, start_date, end_date, dow)
    tci.reset_index(inplace = True)
    tci.sort_values('date', inplace = True)
    
    # Create Altair chart
    chart = alt.Chart(tci).mark_line().add_selection(
        alt.selection_interval(bind='scales')
    ).encode(
        x=alt.X('date:T',
                title='Date',
                axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('tci:Q', 
                title='TCI'),
        color=alt.Color(f'{agg_spatial}:N',
                       title=agg_spatial.capitalize()),
        tooltip=['date:T', 'tci:Q', f'{agg_spatial}:N']
    ).properties(
        width=800,
        height=400,
        title=f'Daily TCI - by {agg_spatial.capitalize()}'
    ).interactive()
    
    # Display the chart
    chart.show()


def hourly_tci_by_month (df, dow, group_name, start_date = None, end_date = None):
    '''
    Plot the hourly Traffic Congestion Index (TCI) for selected months.

    Parameters:
    - df (DataFrame): A Dask/Pandas DataFrame containing traffic data.
    - geog (str): The geographic column to group by.
    - combination_year_month (list of tuples): List of (year, month) pairs to plot.
    - dow (list): Days of the week to include (e.g. [0, 1, 2, 3, 4] for weekdays).
    - group_name (str): Label used in the plot title (e.g. region name).
    - start_date (str, optional): The start date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the minimum date in the data.
    - end_date (str, optional): The end date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the maximum date in the data.

    Returns:
    - None: Displays the interactive Plotly figure.
    '''

    monthly_hourly_tci = utils.monthly_hourly_tci(df, 'length', start_date = start_date, end_date = end_date, dow = dow).reset_index()
    
    selection = alt.selection_point(fields=['year_month'], bind='legend')

    chart = alt.Chart(monthly_hourly_tci).mark_line().encode(
        x=alt.X('hour:Q', title='Hour'),
        y=alt.Y('tci:Q', title='TCI'),
        color=alt.Color('year_month:N', title='Year-Month'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.05)),
        tooltip=['hour:Q', 'tci:N', 'year_month:N']
    ).properties(
        width=600,
        height=300,
        title='TCI by Hour and Year-Month for {}'.format(group_name)
    ).add_params(selection
                 ).interactive()

    chart.show()
    
def hourly_tci_by_geog(df, agg_spatial, agg_column, agg_spatial_name, group_name, start_date = None, end_date = None, dow = None):
    '''
    Plot the average hourly Traffic Congestion Index (TCI) by geography.

    Parameters:
    - df (DataFrame): A Dask/Pandas DataFrame containing traffic data.
    - agg_spatial (str): Name of column used for spatial aggregation.
    - agg_column (str): Name of column used for aggregation.
    - agg_spatial_name (str): Name of the spatial aggregation level (e.g. region name) for labeling the plot.
    - group_name (str): Label used in the plot title (e.g. region name).
    - start_date (str, optional): The start date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the minimum date in the data.
    - end_date (str, optional): The end date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the maximum date in the data.
    - dow (list, optional): Days of the week to include (e.g. [0, 1, 2, 3, 4] for weekdays). If None, all days are included.

    Returns:
    - None: Displays the interactive Plotly figure.
    '''
    hourly_tci_by_geography = utils.hourly_tci_by_geography(df, agg_spatial, agg_column, start_date = start_date, end_date = end_date, dow = dow)

    selection = alt.selection_point(fields=[f'{agg_spatial}'], bind='legend')

    chart = alt.Chart(hourly_tci_by_geography).mark_line().encode(
        x=alt.X('hour:Q', title='Hour'),
        y=alt.Y('tci:Q', title='TCI'),
        color=alt.Color(f'{agg_spatial}:N', title=f'{agg_spatial.capitalize()}'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.05)),
        tooltip=['hour:Q', 'tci:N', f'{agg_spatial}:N']
    ).properties(
        width=600,
        height=300,
        title='TCI by Hour and {} for {}'.format(agg_spatial.capitalize(), group_name)
    ).add_params(selection
                 ).interactive()

    chart.show()


def plot_year_to_year_tci(df, agg_column, start_date = None, end_date = None, dow = None):
    '''
    Generates a lineplot per year to compare tci during the same periods of time

    Parameter:
    tci_all: DataFrame, must have a column named tci and date

    Returns:
    Line plots
    '''
    month_total_tci = utils.tci_temporal_spatial(df, ['year', 'month'], 'region', agg_column,
                                                    start_date = start_date, end_date = end_date, 
                                                    dow = dow).reset_index()

    selection = alt.selection_point(fields=['year'], bind='legend')

    chart = alt.Chart(month_total_tci).mark_line().encode(
        x=alt.X('month:Q', title='Month'),
        y=alt.Y('tci:Q', title='TCI'),
        color=alt.Color('year:N', title='Year'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.05)),
        tooltip=['month:Q', 'tci:N', 'year:N']
    ).properties(
        width=600,
        height=300,
        title='TCI by Month and Year'
    ).add_params(selection
                 ).interactive()

    chart.show()


def map_tci(df, agg_spatial, agg_column, layer, start_date = None, end_date = None, dow = None):
    layer.set_index('Region', inplace = True)
    layer['TCI'] = utils.mean_daily_tci_geog(df, agg_spatial, agg_column, layer, start_date = start_date, end_date = end_date, dow = dow)
    layer = layer[layer['TCI']>0]
    layer.reset_index(inplace = True)
    return layer.explore(column = 'TCI', cmap = 'Spectral_r', tiles = 'CartoDB Positron', legend_kwds={'label': "TCI by Region", 'orientation': "horizontal"})