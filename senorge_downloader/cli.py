import os
import re
import logging
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Tuple, Literal, Type

import geopandas as gpd
import pandas as pd
import plac
import xarray as xr
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from senorge_downloader.utils import fetch, open_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_CRS = "EPSG:25833"


def slugify(s: str) -> str:
    return re.sub(r"\W+", "-", s).lower()


def download(output_dir: Path, start_year: int, end_year: int, variable=None) -> None:
    def urls_for(year: int) -> List[str]:
        result: List[str] = []

        if variable in ["tg", "rr"] or variable is None:
            result.append(
                f"https://thredds.met.no/thredds/fileServer/senorge/seNorge_2018/Archive/seNorge2018_{year}.nc",
            )

        if (variable == "sd" or variable is None) and year >= 1958:
            result.append(
                f"https://thredds.met.no/thredds/fileServer/senorge/seNorge_snow/sd/sd_{year}.nc",
            )

        return result

    urls = [
        url
        for sublist in [urls_for(year) for year in range(start_year, end_year + 1)]
        for url in sublist
    ]

    process_map(
        partial(fetch, output_dir=output_dir),
        urls,
        max_workers=os.cpu_count(),
    )


def convert_file(path: Path, output_dir: Path):
    is_snow_depth = path.stem.startswith("sd_")

    variables = []

    logger.info(f"opening dataset @ {path}")
    ds = open_dataset(path)

    if is_snow_depth:
        variables.append("snow_depth")
    else:
        variables.extend(["tg", "rr"])

    for variable in variables:
        if not variable in ds.variables:
            raise ValueError(
                f"Expected {variable} for {path}, but not found in {list(ds.variables.keys())}"
            )

        output_path = output_dir / f"{variable}_{path.stem}.csv"
        logger.info(f"converting {path} => {output_path}")

        cols = ["year", "type", "x", "y", "value"]
        df = None

        if variable == "snow_depth":
            df_sd = (
                ds[variable]
                .resample(time="1Y")
                .mean()  # mean snow depth
                .to_dataframe()
                .reset_index()
                .dropna(subset=["snow_depth"])
                .rename(columns={variable: "value", "X": "x", "Y": "y"})
            )

            df_sd["type"] = "snow_depth"

            df_ski_days = (
                ds.where(ds.snow_depth > 25)
                .resample(time="1Y")
                .count()  # count ski days
                .to_dataframe()
                .reset_index()
                .dropna(subset=["snow_depth"])
                .rename(columns={variable: "value", "X": "x", "Y": "y"})
            )

            df_ski_days["type"] = "ski_days"

            df_snow_days = (
                ds.where(ds.snow_depth > 5)
                .resample(time="1Y")
                .count()  # count snow days
                .to_dataframe()
                .reset_index()
                .dropna(subset=["snow_depth"])
                .rename(columns={variable: "value", "X": "x", "Y": "y"})
            )

            df_snow_days["type"] = "snow_days"
            df = pd.concat([df_sd, df_ski_days, df_snow_days], ignore_index=True)
        elif variable == "tg":
            df = (
                ds[variable]
                .resample(time="1Y")
                .mean()  # mean temperature
                .to_dataframe()
                .reset_index()
                .dropna(subset=["tg"])
                .rename(columns={variable: "value", "X": "x", "Y": "y"})
            )

            df["type"] = "temperature"
        elif variable == "rr":
            df = (
                ds[variable]
                .resample(time="1Y")
                .sum()  # sum of daily precipitation
                .to_dataframe()
                .reset_index()
                .dropna(subset=["rr"])
                .rename(columns={variable: "value", "X": "x", "Y": "y"})
            )

            df["type"] = "precipitation"
        else:
            raise ValueError(f"Unknown variable {variable}")

        df["year"] = df["time"].dt.year.astype(int)
        df[cols].to_csv(output_path, index=False, header=True)


def convert_to_csv(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    non_fixed = [path for path in input_dir.glob("*.nc") if "_fixed" not in path.stem]

    # sequentially to keep down mem usage
    for file in tqdm(non_fixed):
        convert_file(file, output_dir)


def vg_format(
    input_dir: Path, output_dir: Path, type: Literal["maps", "timeseries", "normals"]
):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.csv"))

    if type == "timeseries":
        muni_cells = gpd.read_file("./data/geo/muni-cells-utm33.geojson")
        assert muni_cells.crs == EXPECTED_CRS

        build_muni_timeseries(files, output_dir, muni_cells)
    elif type == "maps":
        muni_areas = gpd.read_file("./data/geo/kommuner-2021-uten-hav-utm33.geojson")
        assert muni_areas.crs == EXPECTED_CRS

        build_map_summaries(
            files=files,
            munis=muni_areas,
            baseline_years=[1960, 1969],
            comparison_years=[2013, 2022],
        ).to_csv(output_dir / "map-diffs.csv", index=False, header=True)
    elif type == "normals":
        muni_areas = gpd.read_file("./data/geo/kommuner-2021-uten-hav-utm33.geojson")
        assert muni_areas.crs == EXPECTED_CRS

        build_map_summaries(
            files=files,
            munis=muni_areas,
            baseline_years=[1961, 1990],
            comparison_years=[1991, 2020],
        ).to_csv(output_dir / "map-normals.csv", index=False, header=True)


def build_muni_timeseries(
    input_files: List[Path], output_dir: Path, muni_cells: gpd.GeoDataFrame
):
    datasets = process_map(
        partial(mean_by_muni, muni_cells=muni_cells),
        input_files,
        max_workers=os.cpu_count(),
    )

    combined = pd.concat(datasets, ignore_index=True).set_index(["kommunekode"])

    for muni_code in combined.index.unique():
        out = output_dir / "municipalities" / f"{muni_code}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {out}")

        cols = ["type", "year", "value"]

        combined[combined.index == muni_code][cols].sort_values(
            ["type", "year"]
        ).to_csv(out, index=False)

    logger.info("Writing national averages")
    combined.groupby(["type", "year"])["value"].mean().reset_index().to_csv(
        output_dir / "norway-averages.csv", index=False
    )


def build_map_summaries(
    files: List[Path],
    munis: gpd.GeoDataFrame,
    baseline_years: List[int],
    comparison_years: List[int],
):
    logger.info(f"Building map summaries")

    def year_from(file: Path) -> int:
        md = re.search(r"(\d{4})$", file.stem)

        if md is None:
            raise ValueError(f"no year found in {file.stem}")

        return int(md.group(1))

    datasets = process_map(
        pd.read_csv,
        [
            file
            for file in files
            if year_from(file) in (baseline_years + comparison_years)
        ],
        max_workers=os.cpu_count(),
    )

    dat = pd.concat(datasets, ignore_index=True)

    logger.info("Calculating")

    baseline = (
        dat[(dat["year"] >= baseline_years[0]) & (dat["year"] <= baseline_years[-1])]  # type: ignore
        .groupby(["type", "x", "y"])["value"]  # type: ignore
        .mean()
    )

    latest = (
        dat[(dat["year"] >= comparison_years[0]) & (dat["year"] <= comparison_years[-1])]  # type: ignore
        .groupby(["type", "x", "y"])["value"]  # type: ignore
        .mean()
    )

    result = pd.concat([baseline, latest], axis=1, keys=["baseline", "latest"])

    # Calculate the absolute and relative difference
    result["diff_abs"] = (result["latest"] - result["baseline"]).round(4)
    result["diff_rel"] = (result["latest"] / result["baseline"]).round(4)

    cols = ["type", "x", "y", "diff_abs", "diff_rel"]
    result = result.reset_index()[cols]

    logger.info("Removing ocean cells")

    geo_result = gpd.GeoDataFrame(
        data=result.reset_index(),
        geometry=gpd.points_from_xy(result["x"], result["y"]),
        crs=munis.crs,
        copy=False,
    )  # type: ignore

    cleaned = gpd.sjoin(
        geo_result, munis, how="inner", predicate="intersects"
    ).reset_index()

    return cleaned[cols]


def mean_by_muni(file: Path, muni_cells: gpd.GeoDataFrame):
    logger.info(f"Finding intersects in {file}")

    dat = pd.read_csv(file, dtype={"year": str})

    geo = gpd.GeoDataFrame(
        data=dat,
        geometry=gpd.points_from_xy(dat["x"], dat["y"]),
        crs=muni_cells.crs,
        copy=False,
    )  # type: ignore

    # we find the first intersecting cell for each municipality that has a non-na value

    muni_dat = muni_cells.sjoin(geo, how="left", predicate="intersects").reset_index()
    muni_dat = muni_dat.sort_values(by=["kommunekode", "year", "rank"])

    def first_non_na(x):
        result = x[x["value"].notna()].head(1)

        if result.empty:
            return x.head(1)

        return result

    result = (
        muni_dat.groupby(["year", "type", "kommunekode", "kommunenavn"])
        .apply(first_non_na)
        .reset_index(drop=True)
    )

    result["value"] = result["value"].round(4)
    result["year"] = result["year"].astype(str)

    return result


# silly cloudfront
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def vg_comparisons(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # # https://www.epa.gov/climate-indicators/climate-change-indicators-us-and-global-temperature
    logger.info(f"Fetching EPA temperature")

    epa_temp = pd.read_csv(
        "https://www.epa.gov/system/files/other-files/2022-07/temperature_fig-2.csv",
        encoding="ISO-8859-1",
        skiprows=6,
        storage_options = {'User-Agent': USER_AGENT}
    ).rename(columns={"Earth's surface (land and ocean)": "value", "Year": "year"})

    epa_temp["type"] = "temperature"
    epa_temp["year"] = epa_temp["year"].astype(int)

    epa_temp = epa_temp[["year", "type", "value"]]
    epa_temp = epa_temp[epa_temp["year"] >= 1957]

    logger.info(f"Fetching EPA precipitation")

    epa_precip = pd.read_csv(
        "https://www.epa.gov/system/files/other-files/2022-07/precipitation_fig-2.csv",
        encoding="ISO-8859-1",
        skiprows=6,
        storage_options = {'User-Agent': USER_AGENT}
    )

    epa_precip.columns = epa_precip.columns.str.strip()
    epa_precip.rename(columns={"Anomaly": "value", "Year": "year"}, inplace=True)

    epa_precip["type"] = "precipitation"
    epa_precip = epa_precip[["year", "type", "value"]]
    epa_precip["year"] = epa_precip["year"].astype(int)
    epa_precip = epa_precip[epa_precip["year"] >= 1957]
    epa_precip["value"] = epa_precip["value"] * 25.4  # convert inches to mm

    out = output_dir / "global-anomalies-epa.csv"
    logger.info(f"Writing {out}")

    pd.concat([epa_temp, epa_precip], ignore_index=True).to_csv(
        out, index=False, header=True
    )

    proj_vars = ["precipitation_amount", "air_temperature"]
    scenarios = ["RCP45", "RCP85"]

    areas = [
        {"name": "Norge", "value": "NO"},
        {"name": "Region Østlandet", "value": "R1"},
        {"name": "Region Vestlandet", "value": "R2"},
        {"name": "Region Midt-Norge", "value": "R3"},
        {"name": "Region Nordland og Troms", "value": "R4"},
        {"name": "Region Finnmarksvidda", "value": "R5"},
        {"name": "Region Varanger", "value": "R6"},
        {"name": "Østfold", "value": "C1", "code": "01"},
        {"name": "Akershus", "value": "C2", "code": "02"},
        {"name": "Oslo", "value": "C3", "code": "03"},
        {"name": "Hedmark", "value": "C4", "code": "04"},
        {"name": "Oppland", "value": "C5", "code": "05"},
        {"name": "Buskerud", "value": "C6", "code": "06"},
        {"name": "Vestfold", "value": "C7", "code": "07"},
        {"name": "Telemark", "value": "C8", "code": "08"},
        {"name": "Aust-Agder", "value": "C9", "code": "09"},
        {"name": "Vest-Agder", "value": "C10", "code": "10"},
        {"name": "Rogaland", "value": "C11", "code": "11"},
        {"name": "Hordaland", "value": "C12", "code": "12"},
        {"name": "Sogn og Fjordane", "value": "C14", "code": "14"},
        {"name": "Møre og Romsdal", "value": "C15", "code": "15"},
        {"name": "Sør-Trøndelag", "value": "C16", "code": "16"},
        {"name": "Nord-Trøndelag", "value": "C17", "code": "17"},
        {"name": "Nordland", "value": "C18", "code": "18"},
        {"name": "Troms", "value": "C19", "code": "19"},
        {"name": "Finnmark", "value": "C20", "code": "20"},
    ]

    datasets = []

    for proj in proj_vars:
        for scenario in scenarios:
            for area in areas:
                print(f"Fetching predictions for {proj} {scenario} {area['name']}")
                url = f"https://prod.kss-backend.met.no/climateProjections?climateIndex={proj}&period=Annual&area={area['value']}&scenario={scenario}"

                print(url)

                dat = pd.read_json(url)
                min_max = pd.DataFrame(
                    dat["data"]["minMaxModelValues"], columns=["year", "min", "max"]
                )
                mod = pd.DataFrame(
                    dat["data"]["modelValues"], columns=["year", "model"]
                )
                obs = pd.DataFrame(dat["data"]["obsValues"], columns=["year", "obs"])

                result = (
                    min_max.merge(mod, on="year", how="outer")
                    .merge(obs, on="year", how="outer")
                    .sort_values(by="year")
                )

                result["scenario"] = scenario
                result["area"] = area["name"]
                result["area_code"] = area.get("code", None)

                if proj == "precipitation_amount":
                    result["type"] = "precipitation"
                elif proj == "air_temperature":
                    result["type"] = "temperature"
                else:
                    raise ValueError(f"Unknown projection variable {proj}")

                datasets.append(result)

    res = pd.concat(datasets, ignore_index=True).reset_index()
    cols = [
        "area",
        "area_code",
        "type",
        "year",
        "min",
        "max",
        "model",
        "obs",
        "scenario",
    ]

    for area in res["area"].unique():
        subset = res[res["area"] == area].reset_index()[cols]
        first_row = subset.iloc[0]

        name = (
            f"county-{first_row['area_code']}"
            if first_row["area_code"]
            else slugify(area)
        )

        out = output_dir / "projections" / f"{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {out}")
        subset.to_csv(out, index=False, header=True)


def vg_top_lists(dir: Path):
    baseline_years = [1960, 1969]
    comparison_years = [2013, 2022]
    top_lists = pd.DataFrame()

    for file in (dir / "municipalities").glob("*.csv"):
        code = file.stem
        dat = pd.read_csv(file)

        baseline = (
            dat[(dat["year"] >= baseline_years[0]) & (dat["year"] <= baseline_years[-1])]  # type: ignore
            .groupby(["type"])["value"]  # type: ignore
            .mean()
        )

        latest = (
            dat[(dat["year"] >= comparison_years[0]) & (dat["year"] <= comparison_years[-1])]  # type: ignore
            .groupby(["type"])["value"]  # type: ignore
            .mean()
        )

        result = pd.concat(
            [baseline, latest], axis=1, keys=["baseline", "latest", "type"]
        )

        # Calculate the absolute and relative difference
        result["diff_abs"] = (result["latest"] - result["baseline"]).round(4)
        result["diff_rel"] = (result["latest"] / result["baseline"]).round(4)
        result["code"] = code

        result.reset_index(inplace=True)

        top_lists = pd.concat([top_lists, result], ignore_index=True)

    top_lists.to_csv(dir / "top-lists.csv", index=False, header=True)


@plac.annotations(
    cmd=("Command", "positional", None, str),
    input=("Input directory", "option", None, Path),
    output=("Output directory", "option", None, Path),
    variable=plac.Annotation(
        "Variable",
        kind="option",
        type=str,
        choices=["tg", "rr", "sd"],
    ),
)
def senorge(cmd, input, output, variable=None):
    start_year = 1957
    end_year = datetime.now().year - 1  # current year is incomplete

    if cmd == "download":
        download(output, start_year=start_year, end_year=end_year, variable=variable)
    elif cmd == "csv":
        convert_to_csv(input, output)
    elif cmd == "vg-timeseries":
        vg_format(input, output, type="timeseries")
    elif cmd == "vg-maps":
        vg_format(input, output, type="maps")
    elif cmd == "vg-comparisons":
        vg_comparisons(output)
    elif cmd == "vg-top-lists":
        vg_top_lists(output)
    else:
        raise ValueError(f"Unknown command {cmd}")


def main():
    plac.call(senorge)


if __name__ == "__main__":
    main()
