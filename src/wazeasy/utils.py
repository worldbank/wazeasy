import pandas as pd
import geopandas as gpd
from datetime import timedelta
from datetime import datetime as dt
import dask.dataframe as dd
import dask_geopandas
import itertools
from shapely import Point, Polygon
import h3


def is_dask_dataframe(df):
    """Check if DataFrame is a Dask DataFrame."""
    return isinstance(df, dd.DataFrame)

def load_data(main_path, year, month, storage_options = None, file_type = 'csv'):
    '''
    Load data from a specified path for a given year and month.

    Parameters:
    - main_path (str): The main directory path where data files are stored.
    - year (int): The year of the data to load.
    - month (int): The month of the data to load.
    - storage_options (dict, optional): Options for storage backends, e.g., for cloud storage.
    - file_type (str, optional): The type of file to load ('csv' or 'parquet'). Defaults to 'csv'.

    Returns:
    - DataFrame: A Dask DataFrame containing the loaded data.
    '''
    if file_type == 'csv':
        return load_data_csv(main_path, year, month, storage_options)
    elif file_type == 'parquet':
        return load_data_parquet(main_path, year, month, storage_options)

def load_data_csv(main_path, year, month, storage_options=None):
    '''
    Load CSV data from a specified path for a given year and month.

    Parameters:
    - main_path (str): The main directory path where CSV files are stored.
    - year (int): The year of the data to load.
    - month (int): The month of the data to load.
    - storage_options (dict, optional): Options for storage backends, e.g., for cloud storage.

    Returns:
    - DataFrame: A Dask DataFrame containing the loaded CSV data.
    '''
    path = main_path + 'year={}/month={}/*.csv'.format(year, month)
    df = dd.read_csv(path, storage_options=storage_options)
    return df

def load_data_parquet(main_path, year, month, storage_options):
    '''
    Load parquet data from a specified path for a given year and month.

    Parameters:
    - main_path (str): The main directory path where parquet files are stored.
    - year (int): The year of the data to load.
    - month (int): The month of the data to load.
    - storage_options (dict): Options for storage backends, e.g., for cloud storage.

    Returns:
    - DataFrame: A Dask DataFrame containing the loaded parquet data.
    '''
    path = main_path + 'year={}/month={}/*.parquet'.format(year, month)
    df = dd.read_parquet(path, storage_options=storage_options, engine = 'pyarrow')
    return df

def handle_time(df, utc_region):
    '''
    Handle time column to ensure it is in the correct UTC and calculate the following time-related attributes:
    - year: Year of the record (numeric).
    - month: Month of the record (numeric, 1–12).
    - date: Calendar date (YYYY-MM-DD).
    - hour: Hour of the day in 24-hour format.
    - local_time: Timestamp converted to the specified UTC region.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - utc_region (str): The UTC region to convert the time to.

    Returns:
    - None: Modifies the DataFrame in place.
    '''

    if is_dask_dataframe(df):
        df['ts'] = df.ts.dt.tz_localize('UTC')
    else:
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df['local_time'] = df['ts'].dt.tz_convert(utc_region)
    time_attributes(df)

def assign_geography_to_jams(df, geog_info = None):
    '''
    Assign a geography to each traffic jam. The geography is given based on the starting point of the jam. 
    Do not use this function for detailed geographies. In that case, refer to: 

    Parameters:
    - df (DataFrame): The Dask or Pandas DataFrame containing traffic jam data.
    - geog_info (dict): A dictionary containing geographical information for assignment. 
    The key is the name of the geography, and the value is the georreferenced data with the 
    geographic subdivisions. 

    Returns:
    - None: Modifies the DataFrame in place.
    '''
    df['region'] = 'region'

    if geog_info is not None:
        if is_dask_dataframe(df):
            gddf_points = create_dask_gdf_start_point(df)
            for region_name, gdf_area in geog_info.items():
                    gddf_points = sjoin_with_dask(gddf_points, gdf_area, polygon_id_col='Region')
                    gddf_points = gddf_points.rename(columns = {'Region': region_name})
            return gddf_points
        else:
            gdf_points = create_pandas_gdf_start_point(df)
            for region_name, gdf_area in geog_info.items():
                gdf_points = gpd.sjoin(gdf_points, gdf_area[['Region', 'geometry']], how='left', predicate='intersects')
                gdf_points.drop(columns=['index_right'], inplace=True)
                gdf_points = gdf_points.rename(columns = {'Region': region_name})
            return gdf_points

def remove_level5(ddf):
    '''
    Remove traffic jams with level 5 from the DataFrame as these jams are associated to road closures.

    Parameters:
    - ddf (DataFrame): The Dask DataFrame containing traffic jam data.

    Returns:
    - DataFrame: A DataFrame excluding level 5 jams.
    '''
    return ddf[ddf['level']!=5]

def time_attributes(df):
    '''
    Calculate year, month, date, and hour for each jam record.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.

    Returns:
    - None: Modifies the DataFrame in place.
    '''
    df['year'] = df['local_time'].dt.year
    df['month'] = df['local_time'].dt.month
    df['date'] = df['local_time'].dt.date
    df['hour'] = df['local_time'].dt.hour
    
    # Use appropriate datetime function based on DataFrame type
    if is_dask_dataframe(df):
        df['date'] = dd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime(df['date'])

def tci_temporal_spatial(df, agg_temporal, agg_spatial, agg_column, 
                            start_date = None, end_date = None, dow = None):
    '''
    Calculate the Traffic Congestion Index (TCI) by period and geography.

    Parameters:
    - df (DataFrame): The DataFrame (Dask or Pandas) containing traffic jam data.
    - agg_temporal (list): Name of columns used for temporal aggregation.
    - agg_spatial (str): Name of column used for spatial aggregation.
    - start_date (str, optional): The start date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the minimum date in the data.
    - end_date (str, optional): The end date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the maximum date in the data.
    - dow (list, optional): Days of the week to consider (0 = Monday, 6 = Sunday)
    - agg_column (str): The column to aggregate.

    Returns:
    - DataFrame: A DataFrame with the TCI calculated.
    '''    
    dates_of_interest = define_dates_of_interest(df, start_date, end_date, dow)
    df_filtered = df[df['date'].isin(dates_of_interest)].copy()
    if is_dask_dataframe(df_filtered):
        tci = df_filtered.groupby(agg_temporal + [agg_spatial])[[agg_column]].sum().compute()  
    else:
        tci = df_filtered.groupby(agg_temporal + [agg_spatial])[[agg_column]].sum()
    tci.rename(columns = {agg_column: 'tci'}, inplace = True)    
    return tci

def define_dates_of_interest(df, start_date = None, end_date = None, dow = None):
    if is_dask_dataframe(df):
        if start_date is None:
            start_date = df['date'].min().compute().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = df['date'].max().compute().strftime('%Y-%m-%d')
    else:
        if start_date is None:
            start_date = df['date'].min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = df['date'].max().strftime('%Y-%m-%d')
    
    if dow is None:
        dow = [0, 1, 2, 3, 4, 5, 6]

    date_range = pd.date_range(start_date, end_date)
    dates_of_interest = filter_date_range_by_dow(date_range, dow)
    return dates_of_interest

def mean_daily_tci_geog(df, agg_spatial, agg_column, layer, start_date = None, end_date = None, dow = None):
    '''
    Averages the Traffic Congestion Intensity Index (TCI) for each geography daily, for a period of time - if defined.

    Parameters:
    - df (DataFrame): The Dask/Pandas DataFrame containing traffic jam data.
    - start_date (str, optional): The start date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the minimum date in the data.
    - end_date (str, optional): The end date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the maximum date in the data.
    - dow (list, optional): Days of the week to consider (0 = Monday, 6 = Sunday)
    - agg_column (str): The column to aggregate for the TCI, generally length of jam.

    Returns:
    - DataFrame: A DataFrame with the mean TCI for each geography.
    '''
    #TODO: make another function that does the hourly version of this
    dates_of_interest = define_dates_of_interest(df, start_date, end_date, dow)
    df_filtered = df[df['date'].isin(dates_of_interest)].copy()

    tci = tci_temporal_spatial(df_filtered, ['date'], agg_spatial, agg_column)
    geog_ids = layer.index.unique()
    idxs = pd.MultiIndex.from_tuples(list(itertools.product(dates_of_interest, geog_ids)),
                                     names = ['date', agg_spatial])
    tci = tci.reindex(idxs, fill_value = 0)
    tci.reset_index(inplace = True)
    return tci.groupby(agg_spatial)['tci'].mean()

def filter_date_range_by_dow(date_range, dow):
    '''
    Filter a date range by days of the week.

    Parameters:
    - date_range (DatetimeIndex): The range of dates to filter.
    - dow (list): Days of the week to consider (0 = Monday, 6 = Sunday).

    Returns:
    - list: A list of dates that match the specified days of the week.
    '''
    filtered_dates = []
    for date in date_range:
        if date.weekday() in dow:
            filtered_dates.append(date)
    return filtered_dates

def monthly_hourly_tci(df, agg_column, start_date = None, end_date = None, dow = None):
    '''
    Calculate the monthly Traffic Congestion Intensity (TCI) Index, hourly distributed, for a time period.

    Parameters:
    - ddf (DataFrame): The Dask DataFrame containing traffic jam data.
    - agg_column (str): The column to aggregate in the TCI, normally length.
    - start_date (str, optional): The start date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the minimum date in the data.
    - end_date (str, optional): The end date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the maximum date in the data.
    - dow (list, optional): Days of the week to consider (0 = Monday, 6 = Sunday).

    Returns:
    - Series: A Series with the monthly TCI for each hour, month and year.
    '''
    dates_of_interest = define_dates_of_interest(df, start_date, end_date, dow)
    daily_hourly_tci = tci_temporal_spatial(df, ['date', 'hour'], 'region', agg_column, 
                                               start_date, end_date, dow)

    idxs = pd.MultiIndex.from_tuples(list(itertools.product(dates_of_interest, list(range(24)), ['region'])),
                                        names = ['date', 'hour', 'region'])

    daily_hourly_tci = daily_hourly_tci.reindex(idxs, fill_value = 0)
    daily_hourly_tci.reset_index(inplace = True)
    
    daily_hourly_tci['year'] = (daily_hourly_tci['date'].dt.year).astype(str)
    daily_hourly_tci['month'] = (daily_hourly_tci['date'].dt.month).astype(str)
    monthly_hourly_tci = daily_hourly_tci.groupby(['year', 'month', 'hour'])['tci'].mean()
    monthly_hourly_tci = monthly_hourly_tci.reset_index()
    monthly_hourly_tci['year_month'] = monthly_hourly_tci.apply(lambda row: f"{int(row['year'])}-{int(row['month']):02d}", axis=1)
    return monthly_hourly_tci

def hourly_tci_by_geography (df, agg_spatial, agg_column, start_date = None, end_date = None, dow = None):
    '''
    Calculate the hourly average Traffic Congestion Intensity (TCI) Index, for a time period.
    Parameters:
    - ddf (DataFrame): The Dask/Pandas DataFrame containing traffic jam data.
    - agg_spatial (str): Name of column used for spatial aggregation.
    - agg_column (str): The column to aggregate in the TCI, normally length.
    - start_date (str, optional): The start date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the minimum date in the data.
    - end_date (str, optional): The end date (YYYY-MM-DD) of the period to consider. 
        If None, it will use the maximum date in the data.
    - dow (list, optional): Days of the week to consider (0 = Monday, 6 = Sunday).

    Returns:
    - Series: A Series with the average TCI for each hour and geography.
    '''

    dates_of_interest = define_dates_of_interest(df, start_date, end_date, dow)
    daily_hourly_tci = tci_temporal_spatial(df, ['date', 'hour'], agg_spatial, agg_column)
    geographies = list(daily_hourly_tci.reset_index()[agg_spatial].unique())

    idxs = pd.MultiIndex.from_tuples(list(itertools.product(dates_of_interest, list(range(24)), geographies)),
                                        names = ['date', 'hour', agg_spatial])
    daily_hourly_tci = daily_hourly_tci.reindex(idxs, fill_value = 0)
    hourly_tci_by_geography = daily_hourly_tci.groupby(['hour', agg_spatial])['tci'].mean()
    return hourly_tci_by_geography.reset_index()
   
def create_gdf(ddf):
    '''
    Create a Dask-Geopandas GeoDataFrame from a Dask DataFrame.

    Parameters:
    - ddf (DataFrame): The Dask DataFrame containing geographical data.

    Returns:
    - GeoDataFrame: A GeoDataFrame with the geometry column set.
    '''
    ddf['geometry'] = dask_geopandas.from_wkt(ddf['geoWKT'], crs='epsg:4326')
    gddf = dask_geopandas.from_dask_dataframe(ddf, geometry='geometry')
    gddf = gddf.set_crs("EPSG:4326")
    return gddf


def obtain_hexagons_for_area(area, resolution):
    '''
    Create a georeferenced layer of H3 hexagons for a given Area of Operation.

    Parameters:
    - area (Polygon): The area of operation as a h3 LatLngPolygon.
    - resolution (int): The resolution of the H3 hexagons.

    Returns:
    - GeoDataFrame: A GeoDataFrame with H3 hexagons.
    '''
    hexagons = list(h3.h3shape_to_cells(area, resolution))
    hexagons_coords = [h3.cell_to_boundary(h) for h in hexagons]
    flipped_coords = [
        tuple((lon, lat) for lat, lon in hex_coords)
        for hex_coords in hexagons_coords
    ]
    hex_geometries = [Polygon(coords) for coords in flipped_coords]
    hex_ids = [h for h in hexagons]
    hex_gdf = gpd.GeoDataFrame({'hex_id': hex_ids, 'geometry': hex_geometries}, crs="EPSG:4326")
    hex_gdf.rename(columns={'hex_id': 'Region'}, inplace=True)
    return hex_gdf

def classify_jam_by_region(ddf, geogs, year, month, projected_crs, dow = None):
    '''It is important to filter the dataset as much as it can be filtered before the spatial operation'''
    start_date = dt(year, month, 1)
    if month == 12:
        end_date = dt(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = dt(year, month + 1, 1) - timedelta(days=1)
    date_range = [x.date() for x in pd.date_range(start_date, end_date)]
    dates_of_interest = filter_date_range_by_dow(date_range, dow)
    
    ddf_filtered = ddf[ddf['date'].isin(dates_of_interest)].copy()
    unique_jams_over_agg_geom = parallelized_overlay(ddf_filtered, geogs)
    jams_over_agg_geom = distribute_jams_over_aggregation_geom(unique_jams_over_agg_geom, ddf_filtered, projected_crs)    
    return jams_over_agg_geom

def create_dask_gdf_start_point(ddf):
    '''
    Create a Dask-Geopandas GeoDataFrame from a Dask DataFrame using the start
    point from the jam as the geometry

    Parameters:
    - ddf (DataFrame): The Dask DataFrame containing geographical data.

    Returns:
    - GeoDataFrame: A GeoDataFrame with the geometry column set.
    '''
    ddf['geoWKT_point'] = ddf['geoWKT'].map_partitions(process_geowkt_partition, meta=('geoWKT', 'object'))
    ddf['geometry'] = dask_geopandas.from_wkt(ddf['geoWKT_point'], crs='epsg:4326')
    gddf = dask_geopandas.from_dask_dataframe(ddf, geometry='geometry')
    gddf = gddf.set_crs("EPSG:4326")
    return gddf

def create_pandas_gdf_start_point(df):
    '''
    Create a GeoDataFrame from a DataFrame by using the last point from a LineString stored in WKT format as the geometry.

    Parameters:
    - df (DataFrame): The Pandascontaining WKT geometry in 'geoWKT' column.

    Returns:
    - GeoDataFrame: A GeoDataFrame with geometry set from the WKT column.
    '''
    lines_geometry = gpd.GeoSeries.from_wkt(df['geoWKT'], crs='epsg:4326')
    points = lines_geometry.apply(lambda x: Point(x.coords[-1]))
    df['geometry'] = points
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='epsg:4326')
    return gdf

def process_geowkt_partition(partition):
    """Process a partition using vectorized pandas operations"""
    return 'POINT (' + partition.str.split(', ').str[-1].str.replace(')', '', regex=False) + ')'

def sjoin_with_dask(gddf_points, gdf_polygons, polygon_id_col='Region'):
    '''
    Assign points to polygons. Dask does not have a function for left spatial join, so we need 
    to do it manually. 
    
    Parameters:
    - gddf_points (GeoDataFrame): Dask GeoDataFrame with Point geometries 
    - gdf_polygons (GeoDataFrame): GeoDataFrame with Polygon geometries
    - polygon_id_col (str): Column name in gdf_polygons to use as identifier
    
    Returns:
    - GeoDataFrame: Original points with h3_id and polygon assignment columns
    '''
    
    if gddf_points.crs != 'EPSG:4326':
            gddf_points = gddf_points.to_crs('EPSG:4326')
    if gdf_polygons.crs != 'EPSG:4326':
        gdf_polygons = gdf_polygons.to_crs('EPSG:4326')

    join = gddf_points.sjoin(gdf_polygons[[polygon_id_col, 'geometry']], how='inner', predicate='intersects')
    gddf_points[polygon_id_col] = join[polygon_id_col]
    return gddf_points


############ Old code ##############
# def filter_points_in_area_dask(gddf_points, gdf_area):
#     """
#     Filter dask geodataframe points that lie within geopandas area geometries.
    
#     Parameters:
#     - gddf_points: Dask GeoDataFrame with Point geometries
#     - gdf_area: Geopandas GeoDataFrame with area geometries (Polygon/MultiPolygon). The name of the geometry should be 
#     stored in the column 'Region'.
    
#     Returns:
#     - Filtered Dask GeoDataFrame with points inside the areas
#     """
#     if 'Region' not in gdf_area.columns:
#         raise ValueError("The area GeoDataFrame must have a 'Region' column to identify the areas.")
#     # Ensure both have the same CRS
#     if gddf_points.crs != gdf_area.crs:
#         gdf_area = gdf_area.to_crs(gddf_points.crs)
    
#     # Perform spatial join - keeps points that intersect with areas
#     result = dask_geopandas.sjoin(gddf_points, gdf_area, how='inner', predicate='within')
    
#     # Remove the extra columns from the area dataframe if not needed
#     original_columns = gddf_points.columns.tolist()
#     result = result[original_columns + ['Region']]
#     return result

# def get_summary_statistics_street(df, street_names, year, working_days):  
#     '''Not in used for now'''
#     streets = df[df['street'].isin(street_names)].copy()
#     table = (streets.groupby('street')['uuid']
#              .nunique()
#              .to_frame('number_of_jams')
#              .compute())
#     table['total_jam_length'] = (streets.groupby('street')['length']
#                                  .sum()
#                                  .compute()) / 1000

#     by_levels = (streets.groupby(['street', 'level'])[['length']]
#                  .sum()
#                  .compute()).unstack(level=1)

#     for level in range(1, 5):
#         table['total_jam_length_level_{}'.format(level)] = by_levels[('length', level)]
#     table['tci'] = mean_tci_geog(streets, 'date', 'street', 'length', working_days)
#     return table.add_suffix(year)


# def get_summary_statistics_city(ddf, year, working_days):
#     '''Not in used for now'''
#     table = (ddf.groupby('city')['uuid']
#              .nunique()
#              .to_frame('number_of_jams')
#              .compute())
#     table['total_jam_length'] = (ddf.groupby('city')['length']
#                                  .sum()
#                                  .compute()) / 1000
#     by_levels = (ddf.groupby(['city', 'level'])[['length']]
#                  .sum()
#                  .compute()).unstack(level=1)

#     for level in range(1, 5):
#         table['total_jam_length_level_{}'.format(level)] = by_levels[('length', level)]
#     table['tci'] = mean_tci_geog(ddf, 'date', 'city', 'length', working_days)

#     return table.add_suffix(year)

# def line_to_segments(x):
#     '''Not in used for now'''
#     '''Break linestrings into individual segments'''
#     l = x[11:-1].split(', ')
#     l1 = l[:-1]
#     l2 = l[1:]
#     points = list(zip(l1, l2))
#     return ['LineString('+', '.join(elem)+')' for elem in points]

# def get_jam_count_per_segment(df):
#     '''Not in used for now'''
#     '''Count how many jams occured in one segment'''
#     df['segments'] = df['geoWKT'].apply(lambda x: line_to_segments(x))
#     df_exp = df.explode('segments')
#     segment_count = df_exp.groupby('segments').size().reset_index()
#     segment_count.rename(columns={0: 'jam_count'}, inplace=True)
#     segment_count['geometry'] = segment_count['segments'].apply(wkt.loads)
#     segment_count_gdf = gpd.GeoDataFrame(segment_count, crs='epsg:4326', geometry=segment_count['geometry'])
#     return segment_count_gdf

# def remove_last_comma(name):
#     '''Not in used for now'''
#     if name[-2:] == ', ':
#         return name[:-2]
#     else:
#         return name
    
# def harmonize_data(table):
#     '''Not in used for now'''
#     table.reset_index(inplace=True)
#     table['city'] = table['city'].apply(lambda x: remove_last_comma(x))
#     table.set_index('city', inplace=True)




# def obtain_unique_jams_linestrings(ddf):
#     '''
#     Get unique jam linestrings to avoid overlaying the same linestring multiple times.

#     Parameters:
#     - ddf (DataFrame): The Dask DataFrame containing traffic jam data.

#     Returns:
#     - GeoDataFrame: A GeoDataFrame with unique jam linestrings.
#     '''
#     unique_geo = ddf[["geoWKT"]].drop_duplicates().reset_index(drop=True).reset_index()
#     unique_geo = create_gdf(unique_geo)
#     return unique_geo

# def overlay_group(group, hexagons):
#     '''
#     Perform an overlay between layers for delayed processes.

#     Parameters:
#     - group (GeoDataFrame): A GeoDataFrame group to overlay.
#     - hexagons (GeoDataFrame): A GeoDataFrame of hexagons to overlay with.

#     Returns:
#     - GeoDataFrame: The result of the overlay operation.
#     '''
#     result = gpd.overlay(group, hexagons, how = 'intersection')
#     return result

# def parallelized_overlay(ddf, aggregation_geog):
#     '''
#     Parallelize overlay by groups over some geometry.

#     Parameters:
#     - ddf (DataFrame): The Dask DataFrame containing traffic jam data.
#     - aggregation_geog (GeoDataFrame): The geographical areas for aggregation.

#     Returns:
#     - GeoDataFrame: The result of the parallelized overlay operation.
#     '''
#     unique_geo = obtain_unique_jams_linestrings(ddf).persist()
#     delayed_process_group = delayed(overlay_group)
#     groups = [unique_geo.get_partition(i) for i in range(unique_geo.npartitions)]
#     tasks = [delayed_process_group(group, aggregation_geog) for group in groups]
#     results = compute(*tasks)
#     final_result = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
#     return final_result

# def distribute_jams_over_aggregation_geom(gddf, ddf, projected_crs):
#     '''
#     Distribute jams over aggregation geometry.

#     Parameters:
#     - gddf (GeoDataFrame): The GeoDataFrame with jams and geometry.
#     - ddf (DataFrame): The Dask DataFrame containing traffic jam data.
#     - projected_crs (str): The coordinate reference system for projection.

#     Returns:
#     - DataFrame: A DataFrame with jams distributed over the aggregation geometry.
#     '''
#     gddf = gddf.to_crs(projected_crs)
#     gddf['length_in_geom'] = gddf['geometry'].length
#     df = dd.from_pandas(gddf)
#     merge = ddf.merge(df, left_on = 'geoWKT', right_on = 'geoWKT', how = 'left')   
#     return merge
