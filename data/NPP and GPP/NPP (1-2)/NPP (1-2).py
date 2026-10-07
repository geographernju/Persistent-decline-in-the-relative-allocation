from pathlib import Path
import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
ROOT=Path("D:/")
# 原始数据来源：https://www.nesdc.org.cn/sdo/detail?id=612f42ee7e28172cbed3d809 和 https://www.glass.hku.hk/download.html
INPUT=ROOT/"NPP (1-2)"
OUT=Path("D:/data/NPP and GPP/NPP (1-2)")
MASK_FILE=Path('D:/data/NPP and GPP/transform/1.nc')
OUT.mkdir(parents=True,exist_ok=True)
YEARS=np.arange(1982,2018,dtype=np.int16)
LATITUDE=np.arange(9.75,75.0,0.5,dtype=np.float32)
LONGITUDE=np.arange(-167.25,55.0,0.5,dtype=np.float32)
PAIRS=[("NPP1.nc","NPP1-2.nc","NPP1.nc"),("NPP2.nc","NPP2-2.nc","NPP2.nc")]
with Dataset(MASK_FILE) as mask_nc:
    mask_lat=np.asarray(mask_nc.variables["latitude"][:],dtype=float)
    mask_lon=np.asarray(mask_nc.variables["longitude"][:],dtype=float)
    mask_values=np.ma.asarray(mask_nc.variables["land_mask"][:],dtype=np.float64).filled(np.nan)
if mask_values.shape!=(len(LATITUDE),len(LONGITUDE)):
    raise ValueError("Land mask shape differs from output grid")
if not np.allclose(mask_lat,LATITUDE,rtol=0,atol=1e-6):
    raise ValueError("Land mask latitude differs from output grid")
if not np.allclose(mask_lon,LONGITUDE,rtol=0,atol=1e-6):
    raise ValueError("Land mask longitude differs from output grid")
LAND_MASK=np.isfinite(mask_values)&(mask_values==1)
LAND_COUNT=int(LAND_MASK.sum())
if LAND_COUNT==0:
    raise ValueError("Land mask contains no valid cells")
print(f"Land template cells: {LAND_COUNT}")
def get_exact_indices(source,target,name):
    source=np.asarray(source,dtype=float)
    target=np.asarray(target,dtype=float)
    indices=np.empty(len(target),dtype=int)
    for i,value in enumerate(target):
        matches=np.flatnonzero(np.isclose(source,value,rtol=0,atol=1e-6))
        if len(matches)!=1:
            raise ValueError(f"Cannot uniquely match {name}: {value}")
        indices[i]=matches[0]
    return indices
def check_pair(ds1,ds2,name1,name2):
    for var in ("npp","year_npp","latitude","longitude"):
        if var not in ds1:
            raise ValueError(f"{name1} missing variable: {var}")
        if var not in ds2:
            raise ValueError(f"{name2} missing variable: {var}")
    if ds1["npp"].dims!=("year_npp","latitude","longitude"):
        raise ValueError(f"Unexpected dimensions in {name1}: {ds1['npp'].dims}")
    if ds2["npp"].dims!=("year_npp","latitude","longitude"):
        raise ValueError(f"Unexpected dimensions in {name2}: {ds2['npp'].dims}")
    years1=np.asarray(ds1["year_npp"].values,dtype=int)
    years2=np.asarray(ds2["year_npp"].values,dtype=int)
    lat1=np.asarray(ds1["latitude"].values,dtype=float)
    lat2=np.asarray(ds2["latitude"].values,dtype=float)
    lon1=np.asarray(ds1["longitude"].values,dtype=float)
    lon2=np.asarray(ds2["longitude"].values,dtype=float)
    if not np.array_equal(years1,years2):
        raise ValueError(f"Year coordinates differ between {name1} and {name2}")
    if not np.allclose(lat1,lat2,rtol=0,atol=1e-10):
        raise ValueError(f"Latitude coordinates differ between {name1} and {name2}")
    if not np.allclose(lon1,lon2,rtol=0,atol=1e-10):
        raise ValueError(f"Longitude coordinates differ between {name1} and {name2}")
    if len(years1)!=len(np.unique(years1)):
        raise ValueError(f"Duplicate years in {name1}")
    return years1,lat1,lon1
def merge_values(primary,secondary):
    primary=np.asarray(primary,dtype=np.float64)
    secondary=np.asarray(secondary,dtype=np.float64)
    primary=np.where(np.isfinite(primary)&(primary>=0)&(primary<1e30),primary,np.nan)
    secondary=np.where(np.isfinite(secondary)&(secondary>=0)&(secondary<1e30),secondary,np.nan)
    pvalid=np.isfinite(primary)
    svalid=np.isfinite(secondary)
    overlap=pvalid&svalid
    added=(~pvalid)&svalid
    result=primary.copy()
    result[added]=secondary[added]
    return result,int(pvalid.sum()),int(svalid.sum()),int(overlap.sum()),int(added.sum())
def create_output(path):
    nc=Dataset(path,"w",format="NETCDF4")
    nc.createDimension("year",len(YEARS))
    nc.createDimension("latitude",len(LATITUDE))
    nc.createDimension("longitude",len(LONGITUDE))
    year_var=nc.createVariable("year","i2",("year",))
    lat_var=nc.createVariable("latitude","f4",("latitude",))
    lon_var=nc.createVariable("longitude","f4",("longitude",))
    data_var=nc.createVariable("npp","f4",("year","latitude","longitude"),fill_value=np.float32(np.nan),zlib=True,complevel=4)
    year_var[:]=YEARS
    lat_var[:]=LATITUDE
    lon_var[:]=LONGITUDE
    data_var[:]=np.nan
    year_var.long_name="year"
    lat_var.long_name="latitude"
    lon_var.long_name="longitude"
    lat_var.units="degrees_north"
    lon_var.units="degrees_east"
    data_var.long_name="Annual net primary productivity"
    return nc,data_var
def fill_land_cells(values):
    values=np.asarray(values,dtype=np.float64)
    result=np.full(LAND_MASK.shape,np.nan,dtype=np.float32)
    good=LAND_MASK&np.isfinite(values)&(values>=0)&(values<1e30)
    if not good.any():
        return result,0,0
    result[good]=values[good].astype(np.float32)
    missing=LAND_MASK&~good
    filled=0
    if missing.any():
        nearest=distance_transform_edt(~good,return_distances=False,return_indices=True)
        result[missing]=result[nearest[0,missing],nearest[1,missing]]
        filled=int(missing.sum())
    result[~LAND_MASK]=np.nan
    if int(np.isfinite(result).sum())!=LAND_COUNT:
        raise ValueError("Final land mask cell count mismatch")
    return result,int(good.sum()),filled
def process_pair(primary_name,secondary_name,output_name):
    primary_path=INPUT/primary_name
    secondary_path=INPUT/secondary_name
    output_path=OUT/output_name
    tmp_path=OUT/(Path(output_name).stem+".partial.nc")
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    if not secondary_path.exists():
        raise FileNotFoundError(secondary_path)
    tmp_path.unlink(missing_ok=True)
    with xr.open_dataset(primary_path,decode_times=False,mask_and_scale=True,engine="netcdf4") as ds1,xr.open_dataset(secondary_path,decode_times=False,mask_and_scale=True,engine="netcdf4") as ds2:
        source_years,source_lat,source_lon=check_pair(ds1,ds2,primary_name,secondary_name)
        lat_indices=get_exact_indices(source_lat,LATITUDE,"latitude")
        lon_indices=get_exact_indices(source_lon,LONGITUDE,"longitude")
        year_lookup={int(year):i for i,year in enumerate(source_years)}
        available_years=np.array([int(year) for year in YEARS if int(year) in year_lookup],dtype=int)
        if len(available_years)==0:
            raise ValueError("No years overlap output period")
        nc,var=create_output(tmp_path)
        total_primary=0
        total_secondary=0
        total_overlap=0
        total_added=0
        total_spatial_filled=0
        try:
            for year in available_years:
                source_index=year_lookup[int(year)]
                output_index=int(year-YEARS[0])
                a=ds1["npp"].isel(year_npp=source_index,latitude=lat_indices,longitude=lon_indices).transpose("latitude","longitude").values
                b=ds2["npp"].isel(year_npp=source_index,latitude=lat_indices,longitude=lon_indices).transpose("latitude","longitude").values
                merged,pcount,scount,overlap,added=merge_values(a,b)
                final,original_count,spatial_filled=fill_land_cells(merged)
                var[output_index,:,:]=final
                total_primary+=pcount
                total_secondary+=scount
                total_overlap+=overlap
                total_added+=added
                total_spatial_filled+=spatial_filled
                print(f"{output_name} {year}: primary={pcount}, secondary={scount}, overlap={overlap}, added={added}, merged_land={original_count}, spatial_filled={spatial_filled}",flush=True)
        finally:
            nc.close()
    os.replace(tmp_path,output_path)
    print(f"Source years: {int(source_years.min())}-{int(source_years.max())}",flush=True)
    print(f"Used years: {int(available_years.min())}-{int(available_years.max())} ({len(available_years)} years)",flush=True)
    print(f"Primary valid cells: {total_primary}",flush=True)
    print(f"Secondary valid cells: {total_secondary}",flush=True)
    print(f"Overlap cells: {total_overlap}",flush=True)
    print(f"Added from secondary: {total_added}",flush=True)
    print(f"Spatially filled land cells: {total_spatial_filled}",flush=True)
    print("✅ Saved file:",output_path,flush=True)
print(f"Input: {INPUT}")
print(f"Output: {OUT}")
print(f"Template: {YEARS[0]}-{YEARS[-1]}, {len(LATITUDE)} latitudes x {len(LONGITUDE)} longitudes")
for primary_name,secondary_name,output_name in PAIRS:
    print("="*100)
    print(f"Merging: {primary_name} + {secondary_name} -> {output_name}",flush=True)
    try:
        process_pair(primary_name,secondary_name,output_name)
    except Exception as exc:
        tmp=OUT/(Path(output_name).stem+".partial.nc")
        tmp.unlink(missing_ok=True)
        print(f"[Failed] {output_name}: {type(exc).__name__}: {exc}",flush=True)
print("Processing finished")