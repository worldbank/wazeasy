"""Pre-built reports composed from wazeasy plot primitives."""

import geopandas as gpd
from IPython.display import display

from wazeasy import plots, utils
import contextily as cx

def run_basic_report(df, agg_column = 'length' , start_date=None, end_date=None):
    dates_of_interest = utils.define_dates_of_interest(df, start_date, end_date)
    df_filt = df[df["date"].isin(dates_of_interest)]
    df_filt = df_filt.persist()

    display(plots.jams_per_day(df_filt))
    display(plots.jams_per_day_rolling_avg(df_filt))
    display(plots.jams_monthly_aggregated(df_filt))
    display(plots.jams_per_day_per_level(df_filt))
    display(plots.plot_tci_daily_spatial(df_filt, 'region', agg_column, 'AMM'))
    display(
        plots.hourly_tci_by_month(df_filt, dow=[0, 1, 2, 3, 4], agg_column = agg_column, group_name="Weekdays")
    )
    display(plots.hourly_tci_by_month(df_filt, dow=[5, 6], agg_column = agg_column, group_name="Weekends"))
    display(plots.plot_year_to_year_tci(df_filt, agg_column = agg_column, start_date = start_date, end_date = end_date))

def run_geog_report(df, geographies, agg_column, start_date=None, end_date=None, dow = None):
    dates_of_interest = utils.define_dates_of_interest(df, start_date, end_date, dow)
    df_filt = df[df["date"].isin(dates_of_interest)]
    df_filt = utils.assign_geography_to_jams(df_filt, geographies)
    df_filt = df_filt.persist()
    # import pdb; pdb.set_trace()
    for geog, geog_data in geographies.items():
        region_name = geog_data["name"]
        agg_spatial = geog_data['agg_spatial']
        gdf_area = gpd.read_file(geog_data["path"])
        plot_by_geography = geog_data["plot_by_geography"]

        print(f"Running report for {region_name}")
        layer = utils.add_mean_daily_tci_to_vector_layer(df_filt, 
                                                         agg_spatial, 
                                                         agg_column, 
                                                         gdf_area, 
                                                         start_date=start_date, 
                                                         end_date=end_date, 
                                                         dow=dow)
        layer = layer[layer['TCI'] > 0]
        display(plots.map_tci(layer, 'TCI', 'TCI by Region'))
        
        if plot_by_geography:
            display(
                plots.plot_tci_daily_spatial(
                    df_filt,
                    agg_spatial,
                    agg_column,
                    region_name,
                    start_date=start_date,
                    end_date=end_date,
                    dow=None,
                )
            )
            display(
                plots.hourly_tci_by_geog(
                    df_filt,
                    agg_spatial,
                    agg_column,
                    region_name,
                    "Weekdays",
                    start_date=start_date,
                    end_date=end_date,
                    dow=[0, 1, 2, 3, 4],
                )
            )
            display(
                plots.hourly_tci_by_geog(
                    df_filt,
                    agg_spatial,
                    agg_column,
                    region_name,
                    "Weekends",
                    start_date=start_date,
                    end_date=end_date,
                    dow=[5, 6],
                )
            )

def report_TCI_spatial_changes(df, geographies, agg_column, start_period, end_period, dow):

    dates_of_interest_start = utils.define_dates_of_interest(df, start_period[0], start_period[1], dow)
    dates_of_interest_end = utils.define_dates_of_interest(df, end_period[0], end_period[1], dow)
    
    df_filt_start = df[df["date"].isin(dates_of_interest_start)].copy()
    df_filt_end = df[df["date"].isin(dates_of_interest_end)].copy()
    
    df_filt_start = utils.assign_geography_to_jams(df_filt_start, geographies)
    df_filt_start = df_filt_start.persist()

    df_filt_end = utils.assign_geography_to_jams(df_filt_end, geographies)
    df_filt_end = df_filt_end.persist()


    for geog, geog_data in geographies.items():
        region_name = geog_data["name"]
        agg_spatial = geog_data['agg_spatial']
        gdf_area = gpd.read_file(geog_data["path"])

        layer_start = utils.add_mean_daily_tci_to_vector_layer(df_filt_start, 
                                                    agg_spatial, 
                                                    agg_column, 
                                                    gdf_area, 
                                                    start_date=start_period[0], 
                                                    end_date=start_period[1], 
                                                    dow=dow)
        layer_end = utils. add_mean_daily_tci_to_vector_layer(df_filt_end, 
                                                   agg_spatial, 
                                                   agg_column, 
                                                   gdf_area, 
                                                   start_date=end_period[0], 
                                                   end_date=end_period[1], 
                                                   dow=dow)

        layer = layer_start[['geometry']]
        layer['Change TCI'] = layer_end['TCI'] - layer_start['TCI']
        display(plots.map_tci(layer, 'Change TCI', 'Change in TCI'))

        