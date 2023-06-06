# senorge-downloader

This code downloads NetCDF files from [MET](https://thredds.met.no/) and converts them to aggregated CSVs, then builds VG outputs.

## Setup the project

	pip install -U poetry
	poetry shell
	poetry install


## Usage

The process has four steps:

**1. Download data**

We currently fetch three both `seNorge` and `seNorge_snow` datasets. We are interested in the following variables:

* `tg`: Gridded average temperature by day
* `rr`: Gridded sum daily precipitation amount
* `sd`: Gridded snow depth by day (used to calculate snow and ski days)

```shell
poetry run senorge download -output ./data/raw
```

You can also download a single variable:

```shell
poetry run senorge download -output ./data/raw -variable tg
```

**2. Convert NetCDF to CSV**

We convert the downloaded files to CSVs and output one per year.

We resample the daily data to yearly, so that we end up with the following data for each year and grid cell:

* mean daily temperature
* sum daily precipitation
* mean snow depth
* count of ski days (depth > 25 cm)
* count snow days (depth > 5 cm)

```shell
poetry run senorge csv -input ./data/raw -output ./data/csv
```

**3. Find the largest grid cell in each municipality**

This step currently requires PostgreSQL with PostGIS support:

```shell
scripts/find-largest-pop-grid.sh
```

**4. Build VG-formatted outputs**

NB. You need PostgreSQL with n50 data with kommuner_uten_hav from our election maps to complete this step.

We create the following outputs:

1. Municipality timelines:
	- Yearly timelines for all five variables for the most densly populated grid cell
	- Only absolute values / no diffs
2. National timelines:
	- Created by averaging the municipalities in the previous step
3. Grid diffs
	- the absolute and relative diffs of 1960-1969 compared to the last ten years (currently 2013-2022) for all grid cells
	- Used for 3D maps
4. Downloaded world comparisons and Norway projections
5. Top lists of changes

```shell
poetry run senorge vg-timeseries -input ./data/csv -output ./data/vg
poetry run senorge vg-maps -input ./data/csv -output ./data/vg
poetry run senorge vg-comparisons -output ./data/vg
poetry run senorge vg-top-lists -output ./data/vg
```


