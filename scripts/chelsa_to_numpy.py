"""Convert CHELSA v2.1 GeoTIFFs into the normalized land-pixel arrays train.py reads.

  python scripts/chelsa_to_numpy.py --raw-dir /path/to/chelsa_raw --out-dir /path/to/chelsa_numpy
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from ste.chelsa import MEAN, STD, VAR_NAMES

LAND_SEA_THRESHOLD = 30000


def raster_filename(var: str, month: str) -> str:
    if var == "pet":
        return f"CHELSA_pet_penman_{month}_1981-2010_V.2.1.tif"
    if var == "rsds":
        return f"CHELSA_rsds_1981-2010_{month}_V.2.1.tif"
    return f"CHELSA_{var}_{month}_1981-2010_V.2.1.tif"


def read_on_reference_grid(path, ref_profile):
    """Read band 1, resampling (nearest) onto the reference grid if shapes differ."""
    with rasterio.open(path) as src:
        if (src.height, src.width) == (ref_profile["height"], ref_profile["width"]):
            return src.read(1)
        dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
        return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir with one subdir of CHELSA tifs per variable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--months", nargs="*", default=[f"{m:02d}" for m in range(1, 13)])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ref_path = os.path.join(args.raw_dir, "cmi", raster_filename("cmi", "03"))
    with rasterio.open(ref_path) as ref:
        ref_profile = {"height": ref.height, "width": ref.width,
                       "transform": ref.transform, "crs": ref.crs}
        ref_vals = ref.read(1)
        ys = np.array([ref.xy(r, 0)[1] for r in range(ref.height)])  # row -> lat
        xs = np.array([ref.xy(0, c)[0] for c in range(ref.width)])   # col -> lon

    land_rows, land_cols = np.where(ref_vals <= LAND_SEA_THRESHOLD)
    n_land = land_rows.size
    print(f"Reference grid {ref_profile['height']}x{ref_profile['width']}, land pixels: {n_land/1e6:.1f}M")

    ptc_path = os.path.join(args.out_dir, "point_to_coord.npy")
    if not os.path.exists(ptc_path):
        np.save(ptc_path, np.stack([ys[land_rows], xs[land_cols]]).T)

    for month in args.months:
        print(f"Month {month}")
        land_only = np.empty((len(VAR_NAMES), n_land), dtype="float16")
        for i, var in enumerate(VAR_NAMES):
            path = os.path.join(args.raw_dir, var, raster_filename(var, month))
            raster = read_on_reference_grid(path, ref_profile).astype("float32")
            raster = (raster - MEAN[i]) / STD[i]
            land_only[i] = raster[land_rows, land_cols].astype("float16")
            print(f"  {var:9} land-mean={land_only[i].mean():+.4f}")
        np.save(os.path.join(args.out_dir, f"{month}_monthly_float16_land_only.npy"), land_only)

    print("Done.")


if __name__ == "__main__":
    main()
