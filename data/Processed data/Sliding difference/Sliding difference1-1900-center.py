import numpy as np
import xarray as xr
from pathlib import Path
ROOT = Path("D:/")
A_path = Path("D:/data/Processed data/TRW/AgeDepSpline (5+3).nc")
A_var = "AgeDepSpline"
B_path = Path("D:/data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc")
clim_path = Path("H:/Climatic data/month/climate14.nc")
clim_vars = ["SPEI12", "SPEI6", "scpdsi", "vap", "stl1", "cld", "RH", "VPD", "pet", "tmp"]
out_dir = Path("D:/data/Processed data/Sliding difference")
out_dir.mkdir(parents=True, exist_ok=True)
out_nc = Path("D:/data/Processed data/Sliding difference/Sliding difference.nc")
yr0, yr1 = 1900, 2025
WIN = 21
EPS = 1e-12
def open_tr_var(path, var):
    ds = xr.open_dataset(path)
    da = ds[var].sortby("year").sel(year=slice(yr0, yr1))
    years = np.asarray(da.year.values, dtype=float)
    if not np.isfinite(years).all() or not np.all(years == np.floor(years)):
        raise ValueError(f"Invalid year coordinates: {path}")
    if np.unique(years).size != years.size:
        raise ValueError(f"Duplicate year coordinates: {path}")
    return da.assign_coords(year=years.astype(np.int32))
def first_diff(da):
    return da.diff("year")
def sliding_windows(years, win):
    years = np.asarray(years, dtype=int)
    if years.size < win:
        raise ValueError("Insufficient years for a complete window")
    if not np.all(np.diff(years) == 1):
        raise ValueError("Years are not continuous")
    if win % 2 == 0:
        raise ValueError("Window length must be odd to define a unique center year")
    starts = np.arange(years.min(), years.max() - win + 2, dtype=int)
    ends = starts + win - 1
    centers = starts + win // 2
    return starts, ends, centers
def year_to_idx(starts, ends, years_min):
    return (starts - years_min).astype(np.int32), (ends - years_min).astype(np.int32)
def complete_pair_corr(x, y):
    valid = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
    x = np.where(valid[None, ...], x, 0.0)
    y = np.where(valid[None, ...], y, 0.0)
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    sx = np.max(np.abs(xc), axis=0)
    sy = np.max(np.abs(yc), axis=0)
    ok = valid & (sx > 0) & (sy > 0)
    xn = np.divide(xc, sx, out=np.zeros_like(xc), where=sx > 0)
    yn = np.divide(yc, sy, out=np.zeros_like(yc), where=sy > 0)
    numerator = (xn * yn).sum(axis=0)
    denominator = np.sqrt((xn * xn).sum(axis=0) * (yn * yn).sum(axis=0))
    result = np.divide(numerator, denominator, out=np.full(sx.shape, np.nan), where=ok & (denominator > 0))
    return np.clip(result, -1.0, 1.0)
def rolling_pair_corr(X, Y, s_idx, e_idx):
    result = np.full((len(s_idx),) + X.shape[1:], np.nan, dtype=np.float32)
    for w, (start, end) in enumerate(zip(s_idx, e_idx)):
        x = np.asarray(X[start:end + 1], dtype=np.float64)
        y = np.asarray(Y[start:end + 1], dtype=np.float64)
        result[w] = complete_pair_corr(x, y).astype(np.float32)
    return result
def rolling_partial_corr(X, Y, Z, s_idx, e_idx):
    result = np.full((len(s_idx),) + X.shape[1:], np.nan, dtype=np.float32)
    for w, (start, end) in enumerate(zip(s_idx, e_idx)):
        x = np.asarray(X[start:end + 1], dtype=np.float64)
        y = np.asarray(Y[start:end + 1], dtype=np.float64)
        z = np.asarray(Z[start:end + 1], dtype=np.float64)
        valid = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0) & np.isfinite(z).all(axis=0)
        x = np.where(valid[None, ...], x, np.nan)
        y = np.where(valid[None, ...], y, np.nan)
        z = np.where(valid[None, ...], z, np.nan)
        r_xy = complete_pair_corr(x, y)
        r_xz = complete_pair_corr(x, z)
        r_yz = complete_pair_corr(y, z)
        vx = 1.0 - r_xz ** 2
        vy = 1.0 - r_yz ** 2
        denominator = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        numerator = r_xy - r_xz * r_yz
        ok = valid & np.isfinite(numerator) & (vx > EPS) & (vy > EPS)
        partial = np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=ok)
        result[w] = np.clip(partial, -1.0, 1.0).astype(np.float32)
    return result
def build_clim_multi(path, var_names):
    ds = xr.open_dataset(path, decode_times=False)[var_names]
    t = np.asarray(ds.time.values, dtype=float)
    if not np.isfinite(t).all() or not np.all(t == np.floor(t)):
        raise ValueError("Climate time must use integer YYYYMM coordinates")
    t = t.astype(np.int64)
    year = (t // 100).astype(np.int32)
    month = (t % 100).astype(np.int32)
    if np.any((month < 1) | (month > 12)) or np.any((year < 1000) | (year > 3000)):
        raise ValueError("Climate time must use YYYYMM coordinates")
    if np.unique(t).size != t.size:
        raise ValueError("Duplicate climate months")
    ds = ds.assign_coords(year=("time", year), month=("time", month))
    out = {}
    for var in var_names:
        da = ds[var]
        annual = da.groupby("year").mean("time").sel(year=slice(yr0, yr1))
        summer_data = da.where((ds.month >= 6) & (ds.month <= 8), drop=True)
        summer = summer_data.groupby("year").mean("time").sel(year=slice(yr0, yr1))
        out[f"{var}_12"] = annual.astype(np.float32)
        out[f"{var}_68"] = summer.astype(np.float32)
    return out
A_raw = open_tr_var(A_path, A_var).astype(np.float32)
years = A_raw.year.values
if years.size == 0:
    raise ValueError("No tree-ring data in the requested period")
if not np.all(np.diff(years) == 1):
    raise ValueError("Tree-ring years are not continuous")
A_diff = first_diff(A_raw)
years_d = A_diff.year.values
starts, ends, centers = sliding_windows(years_d, WIN)
s_idx, e_idx = year_to_idx(starts, ends, years_d.min())
Y = A_diff.transpose("year", "latitude", "longitude").values
B = open_tr_var(B_path, "npp").astype(np.float32)
B = B.interp(latitude=A_raw.latitude, longitude=A_raw.longitude)
B = B.reindex(year=A_raw.year)
B_diff = first_diff(B)
Z = B_diff.transpose("year", "latitude", "longitude").values
coords = {"window_center_year": centers, "latitude": A_diff.latitude, "longitude": A_diff.longitude}
dims = ("window_center_year", "latitude", "longitude")
tree_mask = np.isfinite(A_raw).any("year")
print("Window center range:", int(centers.min()), int(centers.max()))
print("Required valid differences per window:", WIN)
r_raw = rolling_pair_corr(Y, Z, s_idx, e_idx)
r_raw_diff_win = xr.DataArray(r_raw, coords=coords, dims=dims).where(tree_mask)
ds_out = xr.Dataset({"r_raw_diff_win": r_raw_diff_win})
ds_out = ds_out.assign_coords(window_start_year=("window_center_year", starts), window_end_year=("window_center_year", ends))
ds_out["r_raw_diff_win"].attrs = {"long_name": "21-year moving correlation of first differences in RWI and NPP", "units": "1"}
clim_dict = build_clim_multi(clim_path, clim_vars)
for name, C0 in clim_dict.items():
    print("Processing:", name)
    C0 = C0.reindex(year=A_raw.year)
    C0 = C0.interp(latitude=A_raw.latitude, longitude=A_raw.longitude)
    C_diff = first_diff(C0)
    X = C_diff.transpose("year", "latitude", "longitude").values
    r_par = rolling_partial_corr(Y, Z, X, s_idx, e_idx)
    rp = xr.DataArray(r_par, coords=coords, dims=dims).where(tree_mask)
    out_name = f"delta_{name}"
    ds_out[out_name] = r_raw_diff_win - rp
    ds_out[out_name].attrs = {"long_name": f"Pairwise minus partial 21-year moving correlation controlling for {name}", "units": "1"}
ds_out.attrs = {"window_length": WIN, "missing_data_policy": "Complete windows only; all annual differences must be finite", "window_definition": "21 annual differences require 22 original annual observations", "center_definition": "Exact midpoint year of each 21-year difference window", "analysis_start_year": yr0, "analysis_end_year": yr1}
encoding = {name: {"zlib": True, "complevel": 6, "shuffle": True, "dtype": "float32", "_FillValue": np.float32(np.nan)} for name in ds_out.data_vars}
ds_out.to_netcdf(out_nc, encoding=encoding)
print("✅ Saved file:", out_nc)