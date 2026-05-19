"""Visualization functions for Waze data, themed with the WBG style guide via attaviz."""

import json
from datetime import datetime as dt

import altair as alt
import altair_tiles as alt_tiles
import attaviz
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from wazeasy import utils

attaviz.enable(size="large")
alt.data_transformers.disable_max_rows()


def _collect(grouped):
    """Materialize a Dask groupby result if needed, otherwise return as-is."""
    return grouped.compute() if hasattr(grouped, "compute") else grouped


SEVERITY_COLORS = {str(i + 1): attaviz.SEQ_RED[i] for i in range(5)}


def jams_per_day(data):
    """
    Plot the number of unique traffic jams per day.

    Parameters:
    - data (DataFrame): A Dask/Pandas DataFrame containing 'date' and 'uuid' columns.

    Returns:
    - alt.Chart: Altair line chart.
    """
    jams = _collect(data.groupby("date")["uuid"].nunique()).reset_index()
    jams.sort_values("date", inplace=True)

    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["date"], empty=False
    )

    base = alt.Chart(jams).encode(
        x=alt.X("date:T", title="Date"),
    )

    lines = base.mark_line(color=attaviz.CATEGORICAL[0]).encode(
        y=alt.Y(
            "uuid:Q",
            title="Number of jams",
            axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
        ),
    )
    rules = (
        base.mark_rule(color=attaviz.GREY_300, strokeWidth=0.5)
        .encode(
            opacity=alt.when(nearest).then(alt.value(0.2)).otherwise(alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("uuid:Q", title="Number of jams", format=",.0f"),
            ],
        )
        .add_params(nearest)
    )

    chart = alt.layer(lines, rules).properties(title="Number of jams per day")
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def jams_per_day_rolling_avg(data, window=7):
    """
    Plot the number of unique traffic jams per day with a rolling average overlay.

    Parameters:
    - data (DataFrame): A Dask/Pandas DataFrame containing 'date' and 'uuid' columns.
    - window (int): Window size for rolling average. Default is 7 days.

    Returns:
    - alt.LayerChart: Altair layered chart (daily values + rolling mean).
    """
    jams = _collect(data.groupby("date")["uuid"].nunique()).reset_index()
    jams.sort_values("date", inplace=True)
    jams["rolling_avg"] = jams["uuid"].rolling(window=window).mean()

    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["date"], empty=False
    )

    base = alt.Chart(jams).encode(
        x=alt.X("date:T", title="Date"),
    )

    daily = base.mark_line(color=attaviz.GREY_300, opacity=0.6).encode(
        y=alt.Y(
            "uuid:Q",
            title="Number of jams",
            axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
        )
    )

    rolling = base.mark_line(color=attaviz.CATEGORICAL[0], strokeWidth=2.5).encode(
        y=alt.Y(
            "rolling_avg:Q", axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr())
        ),
    )

    rules = (
        base.mark_rule(color=attaviz.GREY_300, strokeWidth=0.5)
        .encode(
            opacity=alt.when(nearest).then(alt.value(0.2)).otherwise(alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("uuid:Q", title="Daily", format=",.0f"),
                alt.Tooltip("rolling_avg:Q", title=f"{window}-day avg", format=",.1f"),
            ],
        )
        .add_params(nearest)
    )

    chart = alt.layer(daily, rolling, rules).properties(
        title=f"Number of jams per day with {window}-day rolling average"
    )
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def jams_per_day_per_level(data):
    """
    Plot the number of unique traffic jams per day, grouped by congestion level.

    Parameters:
    - data (DataFrame): A Dask/Pandas DataFrame containing 'date', 'level', and 'uuid' columns.

    Returns:
    - alt.Chart: Altair multi-line chart, one line per severity level.
    """
    jpdpl = (
        _collect(data.groupby(["date", "level"])["uuid"].nunique())
        .reset_index()
        .sort_values("date")
        .assign(
            date=lambda df: pd.to_datetime(df["date"]),
            level_str=lambda df: df["level"].astype(str),
        )
    )

    domain = sorted(SEVERITY_COLORS.keys())
    range_ = [SEVERITY_COLORS[k] for k in domain]

    ALL = "All"
    input_dropdown = alt.binding_select(options=[ALL] + domain, name="Severity level: ")
    severity = alt.param(name="severity", value=ALL, bind=input_dropdown)
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["date"], empty=False
    )

    color_scale = alt.Scale(domain=domain, range=range_)
    is_highlighted = f"{severity.name} == '{ALL}' || datum.level_str == {severity.name}"

    base = alt.Chart(jpdpl).encode(x=alt.X("date:T", title="Date"))

    lines_dim = (
        base.mark_line(color=attaviz.REFERENCE)
        .transform_filter(f"!({is_highlighted})")
        .encode(
            y=alt.Y(
                "uuid:Q",
                title="Number of jams",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            opacity=alt.value(0.15),
            detail="level_str:N",
        )
    )

    lines_focus = (
        base.mark_line()
        .transform_filter(is_highlighted)
        .encode(
            y=alt.Y(
                "uuid:Q",
                title="Number of jams",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            color=alt.Color("level_str:N", title="Severity level", scale=color_scale),
            detail="level_str:N",
        )
    )

    rule_data = (
        jpdpl.pivot(index="date", columns="level_str", values="uuid")
        .reset_index()
        .fillna(0)
    )
    rule_tooltip = [alt.Tooltip("date:T", title="Date")] + [
        alt.Tooltip(f"{lvl}:Q", title=f"Level {lvl}", format=",.0f") for lvl in domain
    ]
    rule = (
        alt.Chart(rule_data)
        .mark_rule(color=attaviz.GREY_300, strokeWidth=0.5)
        .encode(
            x=alt.X("date:T"),
            opacity=alt.when(nearest).then(alt.value(0.3)).otherwise(alt.value(0)),
            tooltip=rule_tooltip,
        )
        .add_params(nearest)
    )

    chart = (
        alt.layer(lines_dim, lines_focus, rule)
        .properties(title="Number of jams per day per level")
        .add_params(severity)
    )
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def jams_monthly_aggregated(data):
    """
    Plot the number of unique traffic jams aggregated by month.

    Parameters:
    - data (DataFrame): A Dask/Pandas DataFrame containing 'year', 'month', and 'uuid' columns.

    Returns:
    - alt.Chart: Altair bar chart.
    """
    jpm = (
        _collect(data.groupby(["year", "month"])["uuid"].nunique())
        .reset_index()
        .assign(
            month_with_year=lambda df: df.apply(
                lambda row: dt.strptime(
                    "{}-{}-15".format(row["year"], row["month"]), "%Y-%m-%d"
                ),
                axis=1,
            )
        )
    )

    chart = (
        alt.Chart(jpm, title="Number of jams per month")
        .mark_bar(color=attaviz.CATEGORICAL[0])
        .encode(
            x=alt.X(
                "month_with_year:T",
                title="Month",
            ),
            y=alt.Y(
                "uuid:Q",
                title="Number of jams",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            tooltip=[
                alt.Tooltip("month_with_year:T", title="Month", format="%b %Y"),
                alt.Tooltip("uuid:Q", title="Jams", format=",.2f"),
            ],
        )
    )
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def plot_tci_daily_spatial(
    df,
    agg_spatial,
    agg_column,
    agg_spatial_name,
    start_date=None,
    end_date=None,
    dow=None,
):
    """
    Plot the daily regional Traffic Congestion Index (TCI), aggregated at the area-of-operation level.

    Parameters:
    - df (DataFrame): Dask/Pandas DataFrame containing traffic jam data.
    - agg_spatial (str): Name of column used for spatial aggregation.
    - agg_column (str): The column to aggregate.
    - agg_spatial_name (str): Name of the spatial aggregation level (e.g. region name) for labeling the plot.

    Returns:
    - alt.Chart: Altair line chart, one line per spatial unit.
    """
    tci = utils.tci_temporal_spatial(
        df, ["date"], agg_spatial, agg_column, start_date, end_date, dow
    ).reset_index()
    tci.sort_values("date", inplace=True)

    regions = sorted(tci[agg_spatial].dropna().unique().tolist())
    ALL = "All"
    input_dropdown = alt.binding_select(
        options=[ALL] + regions, name=f"{agg_spatial.capitalize()}: "
    )
    picked = alt.param(name=f"{agg_spatial}_pick", value=ALL, bind=input_dropdown)
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["date"], empty=False
    )

    is_highlighted = (
        f"{picked.name} == '{ALL}' || datum['{agg_spatial}'] == {picked.name}"
    )

    base = alt.Chart(tci).encode(x=alt.X("date:T", title="Date"))

    lines_dim = (
        base.mark_line(color=attaviz.REFERENCE)
        .transform_filter(f"!({is_highlighted})")
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            opacity=alt.value(0.15),
            detail=f"{agg_spatial}:N",
        )
    )

    lines_focus = (
        base.mark_line()
        .transform_filter(is_highlighted)
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            color=alt.Color(
                f"{agg_spatial}:N",
                title=agg_spatial.capitalize(),
                scale=alt.Scale(range=attaviz.CATEGORICAL),
                legend=None,
            ),
            detail=f"{agg_spatial}:N",
        )
    )

    rule_opacity = alt.when(nearest).then(alt.value(0.3)).otherwise(alt.value(0))
    rule_style = dict(color=attaviz.GREY_300, strokeWidth=0.5)

    rule_one = (
        alt.Chart(tci)
        .mark_rule(**rule_style)
        .transform_filter(
            f"{picked.name} == '{ALL}' || datum['{agg_spatial}'] == {picked.name}"
        )
        .encode(
            x=alt.X("date:T"),
            opacity=rule_opacity,
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("tci:Q", title="TCI", format=",.2f"),
            ],
        )
        .add_params(nearest)
    )

    chart = (
        alt.layer(lines_dim, lines_focus, rule_one)
        .properties(title=f"Daily TCI by {agg_spatial.capitalize()}")
        .add_params(picked)
    )
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def hourly_tci_by_month(df, dow, group_name, start_date=None, end_date=None):
    """
    Plot the hourly Traffic Congestion Index (TCI) for selected months.

    Parameters:
    - df (DataFrame): Dask/Pandas DataFrame containing traffic data.
    - dow (list): Days of the week to include (e.g. [0, 1, 2, 3, 4] for weekdays).
    - group_name (str): Label used in the plot title (e.g. region name).
    - start_date (str, optional): Period start date (YYYY-MM-DD). Defaults to the data minimum.
    - end_date (str, optional): Period end date (YYYY-MM-DD). Defaults to the data maximum.

    Returns:
    - alt.Chart: Altair line chart, one line per (year, month).
    """
    monthly = utils.monthly_hourly_tci(
        df, "length", start_date=start_date, end_date=end_date, dow=dow
    ).reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)

    year_months = sorted(monthly["year_month"].unique().tolist())

    ALL = "All"
    input_dropdown = alt.binding_select(
        options=[ALL] + year_months, name="Year-Month: "
    )
    picked = alt.param(name="year_month_pick", value=ALL, bind=input_dropdown)
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["hour"], empty=False
    )

    is_highlighted = f"{picked.name} == '{ALL}' || datum.year_month == {picked.name}"
    color_scale = alt.Scale(scheme="blues")

    base = alt.Chart(monthly).encode(
        x=alt.X(
            "hour:Q",
            title="Hour of day",
            axis=alt.Axis(values=list(range(0, 24, 4))),
        ),
    )

    lines_dim = (
        base.mark_line(color=attaviz.REFERENCE)
        .transform_filter(f"!({is_highlighted})")
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            opacity=alt.value(0.15),
            detail="year_month:N",
        )
    )

    lines_focus = (
        base.mark_line()
        .transform_filter(is_highlighted)
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            color=alt.Color(
                "year_month:N", title="Year-Month", scale=color_scale, legend=None
            ),
            detail="year_month:N",
        )
    )

    rule_opacity = alt.when(nearest).then(alt.value(0.3)).otherwise(alt.value(0))
    rule_style = dict(color=attaviz.GREY_300, strokeWidth=0.5)

    rule_one = (
        alt.Chart(monthly)
        .mark_rule(**rule_style)
        .transform_filter(
            f"{picked.name} != '{ALL}' && datum.year_month == {picked.name}"
        )
        .encode(
            x=alt.X("hour:Q"),
            opacity=rule_opacity,
            tooltip=[
                alt.Tooltip("hour:Q", title="Hour"),
                alt.Tooltip("tci:Q", title="TCI", format=",.2f"),
            ],
        )
        .add_params(nearest)
    )

    chart = (
        alt.layer(lines_dim, lines_focus, rule_one)
        .properties(title=f"TCI by hour and year-month for {group_name}")
        .add_params(picked)
    )
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def hourly_tci_by_geog(
    df,
    agg_spatial,
    agg_column,
    agg_spatial_name,
    group_name,
    start_date=None,
    end_date=None,
    dow=None,
):
    """
    Plot the average hourly Traffic Congestion Index (TCI) by geography.

    Parameters:
    - df (DataFrame): Dask/Pandas DataFrame containing traffic data.
    - agg_spatial (str): Name of column used for spatial aggregation.
    - agg_column (str): Name of column used for aggregation.
    - agg_spatial_name (str): Name of the spatial aggregation level for labeling the plot.
    - group_name (str): Label used in the plot title (e.g. region name).
    - start_date (str, optional): Period start date (YYYY-MM-DD).
    - end_date (str, optional): Period end date (YYYY-MM-DD).
    - dow (list, optional): Days of week to include. If None, all days are included.

    Returns:
    - alt.Chart: Altair line chart, one line per geography.
    """
    hourly = utils.hourly_tci_by_geography(
        df, agg_spatial, agg_column, start_date=start_date, end_date=end_date, dow=dow
    )

    regions = sorted(hourly[agg_spatial].dropna().unique().tolist())

    ALL = "All"
    input_dropdown = alt.binding_select(
        options=[ALL] + regions, name=f"{agg_spatial.capitalize()}: "
    )
    picked = alt.param(name=f"{agg_spatial}_hour_pick", value=ALL, bind=input_dropdown)
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["hour"], empty=False
    )

    is_highlighted = (
        f"{picked.name} == '{ALL}' || datum['{agg_spatial}'] == {picked.name}"
    )

    base = alt.Chart(hourly).encode(
        x=alt.X(
            "hour:Q",
            title="Hour of day",
            axis=alt.Axis(values=list(range(0, 24, 4))),
        ),
    )

    lines_dim = (
        base.mark_line(color=attaviz.REFERENCE)
        .transform_filter(f"!({is_highlighted})")
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            opacity=alt.value(0.15),
            detail=f"{agg_spatial}:N",
        )
    )

    lines_focus = (
        base.mark_line()
        .transform_filter(is_highlighted)
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            color=alt.Color(
                f"{agg_spatial}:N",
                title=agg_spatial.capitalize(),
                scale=alt.Scale(range=attaviz.CATEGORICAL),
                legend=None,
            ),
            detail=f"{agg_spatial}:N",
        )
    )

    rule_opacity = alt.when(nearest).then(alt.value(0.3)).otherwise(alt.value(0))
    rule_style = dict(color=attaviz.GREY_300, strokeWidth=0.5)

    rule_one = (
        alt.Chart(hourly)
        .mark_rule(**rule_style)
        .transform_filter(
            f"{picked.name} == '{ALL}' || datum['{agg_spatial}'] == {picked.name}"
        )
        .encode(
            x=alt.X("hour:Q"),
            opacity=rule_opacity,
            tooltip=[
                alt.Tooltip("hour:Q", title="Hour"),
                alt.Tooltip("tci:Q", title="TCI", format=",.2f"),
            ],
        )
        .add_params(nearest)
    )

    chart = (
        alt.layer(lines_dim, lines_focus, rule_one)
        .properties(
            title=f"TCI by hour and {agg_spatial.capitalize()} for {group_name}"
        )
        .add_params(picked)
    )
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def plot_year_to_year_tci(df, agg_column, start_date=None, end_date=None, dow=None):
    """
    Generate a line plot per year to compare TCI across the same months.

    Parameters:
    - df (DataFrame): Dask/Pandas DataFrame containing traffic data with year and month.
    - agg_column (str): Column to aggregate (e.g. 'length').
    - start_date (str, optional): Period start date (YYYY-MM-DD).
    - end_date (str, optional): Period end date (YYYY-MM-DD).
    - dow (list, optional): Days of week to include. If None, all days are included.

    Returns:
    - alt.Chart: Altair line chart, one line per year.
    """
    monthly = utils.tci_temporal_spatial(
        df,
        ["year", "month"],
        "region",
        agg_column,
        start_date=start_date,
        end_date=end_date,
        dow=dow,
    ).reset_index()
    monthly["year_str"] = monthly["year"].astype(str)

    years = sorted(monthly["year_str"].unique().tolist())

    ALL = "All"
    input_dropdown = alt.binding_select(options=[ALL] + years, name="Year: ")
    picked = alt.param(name="year_pick", value=ALL, bind=input_dropdown)
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["month"], empty=False
    )

    is_highlighted = f"{picked.name} == '{ALL}' || datum.year_str == {picked.name}"
    color_scale = alt.Scale(scheme="blues")

    base = alt.Chart(monthly).encode(
        x=alt.X("month:Q", title="Month", axis=alt.Axis(values=list(range(1, 13)))),
    )

    lines_dim = (
        base.mark_line(color=attaviz.REFERENCE)
        .transform_filter(f"!({is_highlighted})")
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            opacity=alt.value(0.15),
            detail="year_str:N",
        )
    )

    lines_focus = (
        base.mark_line()
        .transform_filter(is_highlighted)
        .encode(
            y=alt.Y(
                "tci:Q",
                title="TCI",
                axis=alt.Axis(labelExpr=attaviz.vega_scale_labelExpr()),
            ),
            color=alt.Color("year_str:N", title="Year", scale=color_scale, legend=None),
            detail="year_str:N",
        )
    )

    rule_opacity = alt.when(nearest).then(alt.value(0.3)).otherwise(alt.value(0))
    rule_style = dict(color=attaviz.GREY_300, strokeWidth=0.5)

    rule_one = (
        alt.Chart(monthly)
        .mark_rule(**rule_style)
        .transform_filter(
            f"{picked.name} != '{ALL}' && datum.year_str == {picked.name}"
        )
        .encode(
            x=alt.X("month:Q"),
            opacity=rule_opacity,
            tooltip=[
                alt.Tooltip("month:Q", title="Month"),
                alt.Tooltip("tci:Q", title="TCI", format=",.2f"),
            ],
        )
        .add_params(nearest)
    )

    chart = (
        alt.layer(lines_dim, lines_focus, rule_one)
        .properties(title="TCI by month and year")
        .add_params(picked)
    )
    chart = attaviz.add_caption(chart, "Source: Waze for Cities data").interactive()
    return chart


def map_tci(
    df, agg_spatial, agg_column, layer, start_date=None, end_date=None, dow=None
):
    """
    Render an interactive Folium choropleth of mean daily TCI by spatial unit.

    Useful for exploring results — pan, zoom, and inspect individual polygons.
    For a report-ready static-style figure with attaviz styling and a basemap
    overlay, see :func:`map_tci_static`.

    Parameters:
    - df (DataFrame): Dask/Pandas DataFrame containing traffic data.
    - agg_spatial (str): Spatial aggregation column.
    - agg_column (str): Column used in TCI computation.
    - layer (GeoDataFrame): Polygons with a 'Region' column.
    - start_date, end_date, dow: optional time filters.

    Returns:
    - folium.Map: Interactive Leaflet-based map produced by ``GeoDataFrame.explore``.
    """
    layer = layer.copy()
    layer.set_index("Region", inplace=True)
    layer["TCI"] = utils.mean_daily_tci_geog(
        df,
        agg_spatial,
        agg_column,
        layer,
        start_date=start_date,
        end_date=end_date,
        dow=dow,
    )
    layer = layer[layer["TCI"] > 0]
    layer.reset_index(inplace=True)
    return layer.explore(
        column="TCI",
        cmap="Spectral_r",
        tiles="CartoDB Positron",
        legend_kwds={"label": "TCI by Region", "orientation": "horizontal"},
    )


def _ensure_d3_winding(geom):
    """Ensure polygon ring orientation matches D3's spherical convention.

    D3's spherical projections (mercator, etc.) require CW exterior rings:
    a CCW ring is treated as the unbounded complement, which fills
    everything outside the polygon and leaves the interior transparent.
    GeoJSON / shapely default to CCW exteriors (right-hand rule on a plane),
    so we flip those before handing geometry to Vega. Geometries already in
    CW form (e.g. some GADM admin boundaries) are passed through unchanged.
    """
    if geom is None:
        return geom
    if geom.geom_type == "Polygon":
        if geom.exterior.is_ccw:
            return Polygon(
                list(geom.exterior.coords)[::-1],
                [list(r.coords)[::-1] for r in geom.interiors],
            )
        return geom
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([_ensure_d3_winding(p) for p in geom.geoms])
    return geom


def map_tci_static(
    df,
    agg_spatial,
    agg_column,
    layer,
    start_date=None,
    end_date=None,
    dow=None,
    provider="CartoDB.Positron",
    width=600,
    height=500,
    show_no_data_regions=False,
):
    """
    Render a static Altair choropleth of mean daily TCI over a tiled basemap.

    Basemap tiles are added by ``altair_tiles``, which handles tile fetching,
    auto-zoom, attribution, and projection alignment with the geoshape layer.
    The view auto-fits to the bounds of ``layer``.

    Parameters:
    - df (DataFrame): Dask/Pandas DataFrame containing traffic data.
    - agg_spatial (str): Spatial aggregation column.
    - agg_column (str): Column used in TCI computation.
    - layer (GeoDataFrame): Polygons with a 'Region' column, in any CRS
      (will be reprojected to EPSG:4326 if needed).
    - start_date, end_date, dow: optional time filters.
    - provider (str | xyzservices.TileProvider): XYZ tile provider, either a
      preconfigured name like ``"CartoDB.Positron"`` / ``"OpenStreetMap.Mapnik"``
      or a ``xyzservices`` provider object. See ``altair_tiles.providers``.
    - width, height (int): Chart dimensions in pixels.
    - show_no_data_regions (bool): If True, polygons without TCI data are
      rendered in the attaviz NO_DATA color. Default False — useful for
      tessellating geometries (e.g. H3 hexes) where rendering empty cells
      would obscure the basemap. Set True for admin/regional maps where
      no-data context is informative.

    Returns:
    - alt.LayerChart: Basemap tiles + attaviz SEQ_RED choropleth + attribution.
    """
    layer = layer.copy()
    if "Region" in layer.columns:
        layer = layer.set_index("Region")
    tci_series = utils.mean_daily_tci_geog(
        df,
        agg_spatial,
        agg_column,
        layer,
        start_date=start_date,
        end_date=end_date,
        dow=dow,
    )
    tci_series.index = tci_series.index.astype(layer.index.dtype)
    layer = layer.reset_index()

    if layer.crs is None or layer.crs.to_epsg() != 4326:
        layer = layer.to_crs(epsg=4326)

    tci_table = pd.DataFrame(
        {"Region": tci_series.index.astype(str), "TCI": tci_series.values}
    )
    tci_table["has_data"] = tci_table["TCI"].notna() & (tci_table["TCI"] > 0)

    if not show_no_data_regions:
        regions_with_data = set(tci_table.loc[tci_table["has_data"], "Region"])
        layer = layer[layer["Region"].astype(str).isin(regions_with_data)]

    layer = layer.assign(geometry=layer.geometry.apply(_ensure_d3_winding))
    geojson = json.loads(layer[["Region", "geometry"]].to_json())
    geo_data = alt.Data(values=geojson, format=alt.DataFormat(property="features"))

    select = alt.selection_point(
        fields=["properties.Region"], on="mouseover", empty=False
    )
    color = alt.condition(
        "datum.has_data",
        alt.Color("TCI:Q", title="TCI", scale=alt.Scale(range=attaviz.SEQ_RED)),
        alt.value(attaviz.NO_DATA),
    )
    stroke = (
        alt.when(select)
        .then(alt.value(attaviz.GREY_500))
        .otherwise(alt.value(attaviz.GREY_400))
    )
    stroke_width = alt.when(select).then(alt.value(2.5)).otherwise(alt.value(0.3))

    choropleth = (
        alt.Chart(geo_data, width=width, height=height)
        .mark_geoshape(strokeCap="round", strokeJoin="round", fillOpacity=0.7)
        .transform_lookup(
            lookup="properties.Region",
            from_=alt.LookupData(tci_table, "Region", ["TCI", "has_data"]),
        )
        .encode(
            color=color,
            stroke=stroke,
            strokeWidth=stroke_width,
            tooltip=[
                alt.Tooltip("properties.Region:N", title="Region"),
                alt.Tooltip("TCI:Q", title="TCI", format=",.2f"),
            ],
        )
        .add_params(select)
        .project(type="mercator")
    )

    chart = alt_tiles.add_tiles(
        choropleth, provider=provider, width=width, height=height
    ).properties(title=f"Mean daily TCI by {agg_spatial.capitalize()}")
    return attaviz.add_caption(chart, "Source: Waze for Cities data")
