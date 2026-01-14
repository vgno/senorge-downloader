# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

senorge-downloader fetches Norwegian weather data (temperature, precipitation, snow depth) from MET's THREDDS server, processes daily gridded NetCDF data into yearly aggregates, and generates various outputs for VG news articles about climate change.

## Commands

```shell
uv sync                                              # Install dependencies
uv run senorge download -output ./data/raw           # Download NetCDF from MET
uv run senorge csv -input ./data/raw -output ./data/csv  # Convert to yearly CSV
uv run senorge vg-timeseries -input ./data/csv -output ./data/vg
uv run senorge vg-maps -input ./data/csv -output ./data/vg
uv run senorge vg-comparisons -output ./data/vg
uv run senorge vg-top-lists -output ./data/vg
```

## Architecture

**Data Pipeline:**
1. `download` - Fetches seNorge/seNorge_snow NetCDF files from thredds.met.no (1957-present)
2. `csv` - Resamples daily grids to yearly aggregates (mean temp, sum precip, mean snow depth, ski days >25cm, snow days >5cm)
3. `vg-*` commands - Generate VG-specific outputs using PostGIS for spatial joins

**Key Files:**
- `senorge_downloader/cli.py` - All CLI commands and data processing logic using plac
- `senorge_downloader/utils.py` - fetch() and open_dataset() helpers
- `scripts/find-largest-pop-grid.sh` - PostGIS script to find most populated grid cell per municipality

**Variables:**
- `tg`: Average daily temperature
- `rr`: Daily precipitation sum
- `sd`: Snow depth (available from 1958)

**Dependencies:**
- xarray/netcdf4 for NetCDF handling
- geopandas/shapely/pyproj for geospatial operations (EPSG:25833)
- pandas/dask for data processing

**External Data Requirements:**
- PostgreSQL with PostGIS and n50 municipality data for `vg-maps`
- `./data/geo/kommuner-2021-uten-hav-utm33.geojson` for municipality boundaries
- `./data/geo/muni-cells-utm33.geojson` for grid-to-municipality mapping
