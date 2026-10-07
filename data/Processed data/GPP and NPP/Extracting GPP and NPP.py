from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
ROOT = Path("D:/")
INPUT_DIRS = {"gpp": Path('D:/data/NPP and GPP/S3_gpp1-12'), "npp": Path('D:/data/NPP and GPP/S3_npp1-12')}
#The above output path needs to be adjusted according to the actual situation. Because the file is too large, Sample data is not provided.
OUTPUT_DIR = Path('D:/data/Processed data/GPP and NPP')
OUTPUT_FILES = {('cold', 'gpp'): Path('D:/data/Processed data/GPP and NPP/cold_gpp.csv'), ('cold', 'npp'): Path('D:/data/Processed data/GPP and NPP/cold_npp.csv'), ('warm', 'gpp'): Path('D:/data/Processed data/GPP and NPP/warm_gpp.csv'), ('warm', 'npp'): Path('D:/data/Processed data/GPP and NPP/warm_npp.csv')}
EU_SHP = ROOT / 'data/map/Europe 7-class.shp'
NA_SHP = ROOT / 'data/map/Koppen_1991_2020_NA_big7_simple.shp'
YEARS = np.arange(1900, 2025)
EARTH_RADIUS = 6371000.0
INVALID_VALUES = [-3000, 65535, -9999]
def climate_group(value):
    if pd.isna(value):
        return None
    name = str(value).strip().strip('"').strip("'")
    if name in {'1', '2', '3', '4', '5'} or name.split(' ')[0] in {'A', 'B', 'C', 'Dfa', 'Dfb'}:
        return 'warm'
    if name in {'6', '7'} or name == 'Cold' or name.startswith('D other') or name.split(' ')[0] == 'E':
        return 'cold'
    return None
def read_region_geometries(path):
    gdf = gpd.read_file(path)
    if 'class' not in gdf.columns:
        raise ValueError(f'Column "class" not found in {path}')
    if gdf.crs is None:
        raise ValueError(f'Coordinate reference system not found in {path}')
    gdf = gdf.to_crs('EPSG:4326')
    gdf['group'] = gdf['class'].map(climate_group)
    geometries = {}
    for group in ('warm', 'cold'):
        selected = gdf.loc[gdf['group'] == group, 'geometry']
        if selected.empty:
            raise ValueError(f'No {group} climate zones found in {path}')
        geometries[group] = selected.union_all()
    return geometries
def pixel_area(latitude, longitude):
    dlat = abs(np.diff(latitude).mean())
    dlon = abs(np.diff(longitude).mean())
    return EARTH_RADIUS ** 2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(np.deg2rad(latitude))[:, None]
def spatial_mask(geometry, latitude, longitude):
    longitude_180 = ((longitude + 180) % 360) - 180
    lon_grid, lat_grid = np.meshgrid(longitude_180, latitude)
    return shapely.contains_xy(geometry, lon_grid, lat_grid)
def annual_total(data, area, mask):
    valid = np.isfinite(data) & mask[None, :, :]
    weighted = np.where(valid, data * area[None, :, :], 0.0)
    totals = weighted.sum(axis=(1, 2))
    return np.where(valid.any(axis=(1, 2)), totals, np.nan)
eu_geometries = read_region_geometries(EU_SHP)
na_geometries = read_region_geometries(NA_SHP)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for variable, input_dir in INPUT_DIRS.items():
    if not input_dir.is_dir():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')
    files = sorted(input_dir.glob('*.nc'))
    if not files:
        raise FileNotFoundError(f'No NC files found in: {input_dir}')
    results = {'warm': pd.DataFrame({'Year': YEARS}), 'cold': pd.DataFrame({'Year': YEARS})}
    for file in files:
        try:
            with xr.open_dataset(file) as ds:
                required = {variable, 'year', 'latitude', 'longitude'}
                missing = required.difference(ds.variables).difference(ds.coords)
                if missing:
                    raise ValueError(f'Missing variables or coordinates: {sorted(missing)}')
                years = np.intersect1d(YEARS, ds['year'].values)
                if years.size == 0:
                    continue
                latitude = ds['latitude'].values.astype(float)
                longitude = ds['longitude'].values.astype(float)
                if latitude.size < 2 or longitude.size < 2:
                    raise ValueError('Latitude or longitude has fewer than two grid points')
                values = ds[variable].sel(year=years).transpose('year', 'latitude', 'longitude').values.astype(np.float64)
            values[np.isin(values, INVALID_VALUES)] = np.nan
            values *= 12.0
            area = pixel_area(latitude, longitude)
            for group in ('warm', 'cold'):
                eu_mask = spatial_mask(eu_geometries[group], latitude, longitude)
                na_mask = spatial_mask(na_geometries[group], latitude, longitude)
                totals = annual_total(values, area, eu_mask | na_mask)
                results[group][file.name] = pd.Series(totals, index=years).reindex(YEARS).to_numpy()
        except Exception as exc:
            print(f'Skipped {file.name}: {exc}')
    for group, df in results.items():
        output_csv = OUTPUT_FILES[(group, variable)]
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print('✅ Saved:', output_csv)