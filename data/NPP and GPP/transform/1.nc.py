from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from netCDF4 import Dataset
ROOT=Path("D:/")
map_dir=ROOT/"data"/"Map"
source_dir=map_dir/"natural_earth_land"
zip_path=map_dir/"ne_10m_land.zip"
land_path=source_dir/"ne_10m_land.shp"
out_nc=Path('D:/data/NPP and GPP/transform/1.nc')
download_url="https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
factor=10
na_lon_min,na_lon_max=-167.27,-55.0
na_lat_min,na_lat_max=9.58,74.99
eu_lon_min,eu_lon_max=-11.0,55.0
eu_lat_min,eu_lat_max=29.0,75.0
map_dir.mkdir(parents=True,exist_ok=True)
if not land_path.exists():
    if not zip_path.exists():
        print("Downloading Natural Earth land polygons")
        urlretrieve(download_url,zip_path)
    source_dir.mkdir(parents=True,exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(source_dir)
if not land_path.exists():
    raise FileNotFoundError(land_path)
longitude=np.arange(-167.25,55.0,0.5,dtype=np.float64)
latitude=np.arange(9.75,75.0,0.5,dtype=np.float64)
world=gpd.read_file(land_path)
if world.crs is None:
    raise ValueError("The land polygons have no coordinate reference system")
world=world.to_crs("EPSG:4326")
geometries=[geometry for geometry in world.geometry if geometry is not None and not geometry.is_empty]
if not geometries:
    raise ValueError("No land polygons were found")
fine_resolution=0.5/factor
fine_rows=len(latitude)*factor
fine_cols=len(longitude)*factor
fine_land=rasterize(((geometry,1) for geometry in geometries),out_shape=(fine_rows,fine_cols),transform=from_origin(-167.5,75.0,fine_resolution,fine_resolution),fill=0,all_touched=False,dtype="uint8")
fine_top_latitude=75.0-np.arange(fine_rows)*fine_resolution
fine_bottom_latitude=fine_top_latitude-fine_resolution
fine_area_weights=np.sin(np.deg2rad(fine_top_latitude))-np.sin(np.deg2rad(fine_bottom_latitude))
fine_area_weights=fine_area_weights.reshape(len(latitude),factor)
fine_land=fine_land.reshape(len(latitude),factor,len(longitude),factor)
longitude_fraction=fine_land.mean(axis=3)
fraction_descending=(longitude_fraction*fine_area_weights[:,:,None]).sum(axis=1)/fine_area_weights.sum(axis=1)[:,None]
land_fraction=np.flipud(fraction_descending)
na_region=(latitude[:,None]>=na_lat_min)&(latitude[:,None]<=na_lat_max)&(longitude[None,:]>=na_lon_min)&(longitude[None,:]<=na_lon_max)
eu_region=(latitude[:,None]>=eu_lat_min)&(latitude[:,None]<=eu_lat_max)&(longitude[None,:]>=eu_lon_min)&(longitude[None,:]<=eu_lon_max)
valid=(land_fraction>0.5)&(na_region|eu_region)
na_count=int(np.count_nonzero(valid&na_region))
eu_count=int(np.count_nonzero(valid&eu_region))
total_count=int(np.count_nonzero(valid))
values=np.ma.array(np.ones(valid.shape,dtype=np.float32),mask=~valid)
with Dataset(out_nc,"w",format="NETCDF4_CLASSIC") as dataset:
    dataset.createDimension("latitude",len(latitude))
    dataset.createDimension("longitude",len(longitude))
    lat_var=dataset.createVariable("latitude","f8",("latitude",))
    lon_var=dataset.createVariable("longitude","f8",("longitude",))
    land_var=dataset.createVariable("land_mask","f4",("latitude","longitude"),fill_value=np.float32(-9999.0))
    lat_var[:]=latitude
    lon_var[:]=longitude
    land_var[:]=values
    lat_var.units="degrees_north"
    lon_var.units="degrees_east"
    land_var.long_name="Grid cells with more than 50 percent land area"
    land_var.source="Natural Earth 1:10m land polygons"
print("North America land cells:",na_count)
print("Europe land cells:",eu_count)
print("Total land cells:",total_count)
print("✅ Saved file:",out_nc)