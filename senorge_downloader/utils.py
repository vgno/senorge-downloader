from tqdm import tqdm
from urllib.request import urlretrieve
import os
import xarray as xr
from pathlib import Path


def fetch(url, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    out = os.path.join(output_dir, os.path.basename(url))

    if not Path(out).exists():
        print(f"{url} => {out}")
        urlretrieve(url, out)


def open_dataset(path) -> xr.Dataset:
    return xr.open_dataset(
        path,
        mask_and_scale=True,
        engine="netcdf4",
        decode_cf=True,
        decode_times=True,
    )
