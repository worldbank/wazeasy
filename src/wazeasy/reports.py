"""Pre-built reports composed from wazeasy plot primitives."""

import geopandas as gpd
from IPython.display import display

from wazeasy import plots, utils


def run_basic_report(df, start_date=None, end_date=None):
    dates_of_interest = utils.define_dates_of_interest(df, start_date, end_date)
    df_filt = df[df["date"].isin(dates_of_interest)]
    df_filt = df_filt.persist()

    display(plots.jams_per_day(df_filt))
    display(plots.jams_per_day_rolling_avg(df_filt))
    display(plots.jams_monthly_aggregated(df_filt))
    display(plots.jams_per_day_per_level(df_filt))
    display(
        plots.hourly_tci_by_month(df_filt, dow=[0, 1, 2, 3, 4], group_name="Weekdays")
    )
    display(plots.hourly_tci_by_month(df_filt, dow=[5, 6], group_name="Weekends"))
    display(plots.plot_year_to_year_tci(df_filt, "length", start_date, end_date))


def run_geog_report(df, geographies, start_date=None, end_date=None):
    dict_geogs = {
        geog: gpd.read_file(geog_data["path"])
        for geog, geog_data in geographies.items()
    }
    dates_of_interest = utils.define_dates_of_interest(df, start_date, end_date)
    df_filt = df[df["date"].isin(dates_of_interest)]
    df_filt = utils.assign_geography_to_jams(df_filt, dict_geogs)
    df_filt = df_filt.persist()

    for geog, geog_data in geographies.items():
        region_name = geog_data["name"]
        gdf_area = gpd.read_file(geog_data["path"])
        plot_by_geography = geog_data["plot_by_geography"]

        print(f"Running report for {region_name}")
        display(
            plots.map_tci(
                df_filt,
                geog,
                "length",
                gdf_area,
                start_date=start_date,
                end_date=end_date,
                dow=None,
            )
        )
        if plot_by_geography:
            display(
                plots.plot_tci_daily_spatial(
                    df_filt,
                    geog,
                    "length",
                    region_name,
                    start_date=start_date,
                    end_date=end_date,
                    dow=None,
                )
            )
            display(
                plots.hourly_tci_by_geog(
                    df_filt,
                    geog,
                    "length",
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
                    geog,
                    "length",
                    region_name,
                    "Weekends",
                    start_date=start_date,
                    end_date=end_date,
                    dow=[5, 6],
                )
            )
