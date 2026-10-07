from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
ROOT = Path("D:/")
A_PATH = ROOT / "data/Processed data/TRW/AgeDepSpline (5+3).nc"
A_VAR = "AgeDepSpline"
OUT_DIR = ROOT / "data/Processed data/median"
OUT_CSV = OUT_DIR / "Median_correlations.csv"
YR0, YR1 = 1981, 2010
MIN_PAIRS = 20
GROUPS = {
    "DGVM_S2": ROOT / "data/NPP and GPP/S2_npp9-8",
    "DGVM_S3": ROOT / "data/NPP and GPP/S3_npp9-8",
    "CMIP6": ROOT / "data/NPP and GPP/CMIP6 (NPP)",
    "NPP1-2": ROOT / "data/NPP and GPP/NPP (1-2)"
}
REGIONS = [
    ("NA", ROOT / "data/Map/Koppen_1991_2020_NA_big7.shp", (-167.27, 9.58, -55.0, 74.99)),
    ("EU", ROOT / "data/Map/Europe 7-class.shp", (-11.0, 30.0, 57.0, 75.0))
]
CLASSES = {"A": [1], "B": [2], "C": [3], "Dfa": [4], "Dfb": [5], "D other": [6], "E": [7], "Warm": [1, 2, 3, 4, 5], "Cold": [6, 7], "All": [1, 2, 3, 4, 5, 6, 7]}
CATEGORY_ORDER = ["DGVM_S2", "DGVM_S3", "CMIP6", "NPP1-2"]
def normalize(ds):
    aliases = {
        "latitude": ["lat", "Latitude", "LAT", "nav_lat"],
        "longitude": ["lon", "Longitude", "LON", "nav_lon"],
        "year": ["Year", "YEAR", "years"],
        "time": ["Time", "TIME"]
    }
    for target, names in aliases.items():
        if target not in ds.variables and target not in ds.dims:
            source = next((name for name in names if name in ds.variables or name in ds.dims), None)
            if source is not None:
                ds = ds.rename({source: target})
    for name in ["latitude", "longitude"]:
        if name not in ds.variables or ds[name].ndim != 1:
            raise ValueError(f"Missing one-dimensional {name} coordinate")
        dim = ds[name].dims[0]
        if dim != name:
            ds = ds.swap_dims({dim: name})
        values = np.asarray(ds[name].values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} coordinate")
        if name == "longitude":
            values = (values + 180) % 360 - 180
        values = np.round(values, 6)
        if np.unique(values).size != values.size:
            raise ValueError(f"Duplicate {name} coordinates")
        ds = ds.assign_coords({name: values}).sortby(name)
    return ds
def annual(da):
    special = next((name for name in ["year_gpp", "year_npp"] if name in da.coords or name in da.dims), None)
    if special is not None:
        if special not in da.coords:
            raise ValueError(f"Missing coordinate values for {special}")
        if "year" in da.coords and "year" != special:
            da = da.drop_vars("year")
        da = da.rename({special: "year"})
    if "year" in da.coords:
        if da.year.ndim != 1:
            raise ValueError("Year coordinate must be one-dimensional")
        dim = da.year.dims[0]
        if dim != "year":
            da = da.swap_dims({dim: "year"})
        years = np.asarray(da.year.values, dtype=float)
        if not np.isfinite(years).all() or not np.all(years == np.floor(years)):
            raise ValueError("Invalid year coordinate")
        da = da.assign_coords(year=years.astype(int))
    elif "time" in da.coords:
        if da.time.ndim != 1:
            raise ValueError("Time coordinate must be one-dimensional")
        dim = da.time.dims[0]
        if dim != "time":
            da = da.swap_dims({dim: "time"})
        try:
            years = np.asarray(da.time.dt.year.values)
        except Exception:
            years = np.asarray(da.time.values)
            if not np.issubdtype(years.dtype, np.number):
                raise ValueError("Time cannot be converted to years")
            if not np.isfinite(years).all() or not np.all(years == np.floor(years)) or not np.all((years >= 1000) & (years <= 3000)):
                raise ValueError("Numeric time is not an explicit year coordinate")
        da = da.assign_coords(time=years.astype(int)).rename({"time": "year"})
    else:
        raise ValueError("Missing annual coordinate")
    if np.unique(da.year.values).size != da.sizes["year"]:
        raise ValueError("Multiple records per year")
    for dim in list(da.dims):
        if dim not in {"year", "latitude", "longitude"}:
            if da.sizes[dim] != 1:
                raise ValueError(f"Unresolved dimension: {dim}={da.sizes[dim]}")
            da = da.isel({dim: 0}, drop=True)
    if set(da.dims) != {"year", "latitude", "longitude"}:
        raise ValueError(f"Unsupported dimensions: {da.dims}")
    da = da.sortby("year").sel(year=slice(YR0, YR1))
    if da.sizes["year"] < MIN_PAIRS + 1:
        raise ValueError("Insufficient annual coverage")
    return da.reindex(year=np.arange(YR0, YR1 + 1)).transpose("year", "latitude", "longitude")
def read_zones():
    frames = []
    mapping = {"A": 1, "B": 2, "C": 3, "Dfa": 4, "Dfb": 5, "D other": 6, "E": 7}
    for region, path, bounds in REGIONS:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"Missing shapefile CRS: {path}")
        gdf = gdf.to_crs(4326)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if "code" in gdf.columns:
            codes = pd.to_numeric(gdf["code"], errors="coerce")
        else:
            codes = None
            preferred = ["class", "Class", "CLASS", "label", "Label"]
            columns = preferred + [c for c in gdf.columns if c not in preferred and c != gdf.geometry.name]
            for column in columns:
                if column not in gdf.columns:
                    continue
                candidate = gdf[column].astype(str).str.strip().map(mapping)
                if candidate.notna().any():
                    codes = candidate
                    break
            if codes is None:
                raise ValueError(f"Climate class field not found: {path}")
        gdf["zone"] = codes
        gdf = gdf[gdf.zone.isin(range(1, 8))].copy()
        if gdf.empty:
            raise ValueError(f"No valid climate classes: {path}")
        gdf["zone"] = gdf.zone.astype(int)
        frames.append((region, gdf[["zone", gdf.geometry.name]], bounds))
    return frames
def classify_points(latitudes, longitudes, regions):
    labels = np.zeros(len(latitudes), dtype=np.int16)
    region_ids = np.full(len(latitudes), "", dtype="<U2")
    for region, polygons, bounds in regions:
        xmin, ymin, xmax, ymax = bounds
        selected = np.flatnonzero((longitudes >= xmin) & (longitudes <= xmax) & (latitudes >= ymin) & (latitudes <= ymax) & (labels == 0))
        if not selected.size:
            continue
        points = gpd.GeoDataFrame({"point_id": selected}, geometry=gpd.points_from_xy(longitudes[selected], latitudes[selected]), crs=4326)
        joined = gpd.sjoin(points, polygons, how="inner", predicate="intersects")
        if joined.empty:
            continue
        ambiguous = joined.groupby("point_id").zone.nunique()
        if (ambiguous > 1).any():
            print(f"{region}: boundary points assigned to lowest class code: {(ambiguous > 1).sum()}")
        joined = joined.sort_values("zone").drop_duplicates("point_id")
        indices = joined.point_id.to_numpy(dtype=int)
        labels[indices] = joined.zone.to_numpy(dtype=np.int16)
        region_ids[indices] = region
    return labels, region_ids
def correlation(x, y):
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)
    valid = np.isfinite(dx) & np.isfinite(dy)
    n = valid.sum(axis=0)
    sx = np.where(valid, dx, 0.0)
    sy = np.where(valid, dy, 0.0)
    mx = np.divide(sx.sum(axis=0), n, out=np.zeros(n.shape, dtype=float), where=n > 0)
    my = np.divide(sy.sum(axis=0), n, out=np.zeros(n.shape, dtype=float), where=n > 0)
    xc = np.where(valid, dx - mx, 0.0)
    yc = np.where(valid, dy - my, 0.0)
    den = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum(axis=0))
    r = np.divide((xc * yc).sum(axis=0), den, out=np.full(n.shape, np.nan), where=(n >= MIN_PAIRS) & (den > 0))
    return np.clip(r, -1, 1)
def compute_points(da, reference, latitudes, longitudes, region_ids, region=None):
    b = annual(da)
    yi = pd.Index(b.latitude.values).get_indexer(latitudes)
    xi = pd.Index(b.longitude.values).get_indexer(longitudes)
    matched = (yi >= 0) & (xi >= 0)
    if region is not None:
        matched &= region_ids == region
    indices = np.flatnonzero(matched)
    if not indices.size:
        raise ValueError("No matching spatial coordinates in the requested region")
    sampled = b.isel(latitude=xr.DataArray(yi[indices], dims="point"), longitude=xr.DataArray(xi[indices], dims="point")).transpose("year", "point")
    r = correlation(reference[:, indices], np.asarray(sampled.values, dtype=float))
    return indices, r
def summarize(category, product, variable, sources, indices, r, labels, region_ids, status="OK"):
    finite = np.isfinite(r)
    row = {
        "category": category,
        "product": product,
        "variable": variable,
        "status": status,
        "source": " | ".join(str(path) for path in sources),
        "period": f"{YR0}-{YR1}",
        "min_pairs": MIN_PAIRS,
        "matched_points": int(len(indices)),
        "valid_points": int(finite.sum()),
        "NA_n": int((finite & (region_ids[indices] == "NA")).sum()),
        "EU_n": int((finite & (region_ids[indices] == "EU")).sum())
    }
    zones = labels[indices]
    for name, codes in CLASSES.items():
        selected = r[np.isin(zones, codes) & finite]
        row[name] = float(np.median(selected)) if selected.size else np.nan
        row[f"{name}_n"] = int(selected.size)
    return row
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regions = read_zones()
    with xr.open_dataset(A_PATH) as raw:
        a = annual(normalize(raw)[A_VAR]).load()
    values = np.asarray(a.values, dtype=float)
    eligible = (np.isfinite(values[:-1]) & np.isfinite(values[1:])).sum(axis=0) >= MIN_PAIRS
    ilat, ilon = np.where(eligible)
    latitudes = a.latitude.values[ilat]
    longitudes = a.longitude.values[ilon]
    labels, region_ids = classify_points(latitudes, longitudes, regions)
    keep = labels > 0
    ilat, ilon = ilat[keep], ilon[keep]
    labels, region_ids = labels[keep], region_ids[keep]
    latitudes, longitudes = latitudes[keep], longitudes[keep]
    reference = values[:, ilat, ilon]
    if not labels.size:
        raise RuntimeError("No eligible tree-ring points within climate regions")
    print(f"Eligible tree-ring points: NA={(region_ids == 'NA').sum()}; EU={(region_ids == 'EU').sum()}")
    rows = []
    for category, folder in GROUPS.items():
        if not folder.is_dir():
            print(f"Missing directory: {folder}")
            continue
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".nc", ".nc4", ".cdf", ".netcdf"})
        print(f"{category}: {len(files)} files")
        for path in files:
            try:
                with xr.open_dataset(path) as raw:
                    ds = normalize(raw)
                    candidates = [name for name, da in ds.data_vars.items() if "latitude" in da.dims and "longitude" in da.dims and any(d in da.dims for d in ["year", "time", "year_gpp", "year_npp"]) and np.issubdtype(da.dtype, np.number)]
                    if not candidates:
                        print(f"Skipped {path.name}: no supported variables")
                        continue
                    for variable in candidates:
                        try:
                            indices, r = compute_points(ds[variable], reference, latitudes, longitudes, region_ids)
                            product = path.stem if len(candidates) == 1 else f"{path.stem}:{variable}"
                            row = summarize(category, product, variable, [path], indices, r, labels, region_ids)
                            rows.append(row)
                            print(f"{category} | {product} | NA={row['NA_n']} | EU={row['EU_n']} | total={row['valid_points']}")
                        except Exception as exc:
                            print(f"Skipped {path.name} [{variable}]: {exc}")
            except Exception as exc:
                print(f"Skipped {path}: {exc}")
    if not rows:
        raise RuntimeError("No products were processed")
    df = pd.DataFrame(rows)
    df["_order"] = df.category.map({name: i for i, name in enumerate(CATEGORY_ORDER)})
    df = df.sort_values(["_order", "product"]).drop(columns="_order")
    metadata = ["category", "product", "variable", "status", "period", "min_pairs", "matched_points", "valid_points", "NA_n", "EU_n"]
    columns = metadata + list(CLASSES) + [f"{name}_n" for name in CLASSES] + ["source"]
    df = df.reindex(columns=columns)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(df[["category", "product", "status", "NA_n", "EU_n", "All"]].to_string(index=False))
    print("✅ Saved table:", OUT_CSV)
if __name__ == "__main__":
    main()