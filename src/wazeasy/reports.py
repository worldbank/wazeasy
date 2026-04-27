
import utils
import plots
import geopandas as gpd

def run_basic_report(df, start_date = None, end_date = None):

    dates_of_interest = utils.define_dates_of_interest(df, start_date, end_date)
    df = df[df['date'].isin(dates_of_interest)].copy()
    
    plots.jams_per_day(df)
    plots.jams_per_day_rolling_avg(df)
    plots.jams_monthly_aggregated(df)
    plots.jams_per_day_per_level(df)
    plots.hourly_tci_by_month(df, dow = [0,1,2,3,4], group_name = 'Weekdays')
    plots.hourly_tci_by_month(df, dow = [5,6], group_name = 'Weekends')
    plots.plot_year_to_year_tci(df, 'length', start_date, end_date)


def run_geog_report(df, geographies, start_date = None, end_date = None):

    dates_of_interest = utils.define_dates_of_interest(df, start_date, end_date)
    df = df[df['date'].isin(dates_of_interest)].copy()
    
    for geog, geog_data in geographies.items():
        region_name = geog_data['name']
        gdf_area = gpd.read_file(geog_data['path'])
        plot_by_geography = geog_data['plot_by_geography']

        print(f"Running report for {region_name}")
        m = plots.map_tci(df, geog, 'length', gdf_area, start_date = start_date, end_date = end_date, dow = None)
        display(m)
        if plot_by_geography:
            plots.plot_tci_daily_spatial(df, geog, 'length', region_name, start_date = start_date, end_date = end_date, dow = None)
            plots.hourly_tci_by_geog(df, geog, 'length', region_name, 'Weekdays', start_date = start_date, end_date = end_date, dow = [0,1,2,3,4])
            plots.hourly_tci_by_geog(df, geog, 'length', region_name, 'Weekends', start_date = start_date, end_date = end_date, dow = [5,6])