from pathlib import Path
import os
import numpy as np
import xarray as xr
import cftime
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
ROOT=Path("D:/")
INPUT=ROOT/"npp and tree ring/Model data/CMIP6/NPP_year"
OUT=Path('D:/data/NPP and GPP/CMIP6 (NPP)')
MASK_FILE=Path('D:/data/NPP and GPP/transform/1.nc')
OUT.mkdir(parents=True,exist_ok=True)
YEARS=np.arange(1980,2014,dtype=np.int16)
LATITUDE=np.arange(9.75,75.0,0.5,dtype=np.float32)
LONGITUDE=np.arange(-167.25,55.0,0.5,dtype=np.float32)
REGIONS=((-167.27,-55.0,9.58,74.99),(-11.0,55.0,29.0,75.0))
SECONDS_PER_YEAR=365.0*86400.0
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
def get_years(ds,da):
    if "year" in da.dims:
        values=np.asarray(ds["year"].values)
        years=np.rint(values.astype(float)).astype(int)
        return "year",years
    if "time" in da.dims:
        time=ds["time"]
        values=np.asarray(time.values)
        units=str(time.attrs.get("units",""))
        calendar=str(time.attrs.get("calendar","standard"))
        if np.issubdtype(values.dtype,np.datetime64):
            years=np.asarray(time.dt.year.values,dtype=int)
            return "time",years
        if units:
            dates=cftime.num2date(np.asarray(values,dtype=float),units=units,calendar=calendar,only_use_cftime_datetimes=True)
            years=np.array([d.year for d in dates],dtype=int)
            return "time",years
        values=np.asarray(values,dtype=float)
        if np.nanmin(values)>=1600 and np.nanmax(values)<2200:
            years=np.rint(values).astype(int)
            return "time",years
        raise ValueError("Cannot decode time coordinate")
    raise ValueError(f"No year or time dimension in npp: {da.dims}")
def prepare_region(ds,bounds):
    lon_min,lon_max,lat_min,lat_max=bounds
    source_lat_all=np.asarray(ds["lat"].values,dtype=float)
    source_lon_all=((np.asarray(ds["lon"].values,dtype=float)+180.0)%360.0)-180.0
    lat_indices=np.flatnonzero((source_lat_all>=lat_min-4.0)&(source_lat_all<=lat_max+4.0))
    lon_indices=np.flatnonzero((source_lon_all>=lon_min-4.0)&(source_lon_all<=lon_max+4.0))
    target_lat_indices=np.flatnonzero((LATITUDE>=lat_min)&(LATITUDE<=lat_max))
    target_lon_indices=np.flatnonzero((LONGITUDE>=lon_min)&(LONGITUDE<=lon_max))
    if len(lat_indices)<2 or len(lon_indices)<2:
        raise ValueError(f"Insufficient source grid coverage: {bounds}")
    lat_order=np.argsort(source_lat_all[lat_indices])
    lon_order=np.argsort(source_lon_all[lon_indices])
    source_lat=source_lat_all[lat_indices][lat_order]
    source_lon=source_lon_all[lon_indices][lon_order]
    if np.any(np.diff(source_lat)<=0):
        raise ValueError("Duplicate source latitude")
    if np.any(np.diff(source_lon)<=0):
        raise ValueError("Duplicate source longitude")
    target_lat=LATITUDE[target_lat_indices]
    target_lon=LONGITUDE[target_lon_indices]
    lat_right=np.clip(np.searchsorted(source_lat,target_lat,side="right"),1,len(source_lat)-1)
    lon_right=np.clip(np.searchsorted(source_lon,target_lon,side="right"),1,len(source_lon)-1)
    lat_left=lat_right-1
    lon_left=lon_right-1
    lat_weight=((target_lat-source_lat[lat_left])/(source_lat[lat_right]-source_lat[lat_left]))[:,None]
    lon_weight=((target_lon-source_lon[lon_left])/(source_lon[lon_right]-source_lon[lon_left]))[None,:]
    inside_lat=(target_lat>=source_lat[0])&(target_lat<=source_lat[-1])
    inside_lon=(target_lon>=source_lon[0])&(target_lon<=source_lon[-1])
    return {"lat_indices":lat_indices,"lon_indices":lon_indices,"lat_order":lat_order,"lon_order":lon_order,"target_lat_indices":target_lat_indices,"target_lon_indices":target_lon_indices,"lat_left":lat_left,"lat_right":lat_right,"lon_left":lon_left,"lon_right":lon_right,"lat_weight":lat_weight,"lon_weight":lon_weight,"inside_lat":inside_lat,"inside_lon":inside_lon}
def read_data(da,region,time_dim,time_index):
    selected=da.isel({time_dim:int(time_index),"lat":region["lat_indices"],"lon":region["lon_indices"]}).transpose("lat","lon")
    values=np.asarray(selected.values,dtype=np.float64)
    values=values[region["lat_order"],:][:,region["lon_order"]]
    values[~np.isfinite(values)]=np.nan
    return values
def convert_npp(values):
    values=np.where(np.isfinite(values)&(values<1e20),values,np.nan)
    values=np.where(values<0,0.0,values)
    return values*SECONDS_PER_YEAR
def regrid_bilinear(values,region):
    i0=region["lat_left"]
    i1=region["lat_right"]
    j0=region["lon_left"]
    j1=region["lon_right"]
    a=region["lat_weight"]
    b=region["lon_weight"]
    corners=np.stack((values[np.ix_(i0,j0)],values[np.ix_(i0,j1)],values[np.ix_(i1,j0)],values[np.ix_(i1,j1)]))
    weights=np.stack(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b))
    valid=np.isfinite(corners)
    numerator=np.sum(np.where(valid,corners,0.0)*weights,axis=0)
    denominator=np.sum(np.where(valid,weights,0.0),axis=0)
    result=np.divide(numerator,denominator,out=np.full(numerator.shape,np.nan,dtype=np.float64),where=denominator>0)
    result[~region["inside_lat"],:]=np.nan
    result[:,~region["inside_lon"]]=np.nan
    return result.astype(np.float32)
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
    data_var.units="kg C m-2 yr-1"
    return nc,data_var
def fill_land_cells(variable,available_year_indices):
    total_filled=0
    for year_index in available_year_indices:
        source=np.ma.asarray(variable[year_index,:,:],dtype=np.float64).filled(np.nan)
        good=LAND_MASK&np.isfinite(source)
        if not good.any():
            continue
        result=np.full(LAND_MASK.shape,np.nan,dtype=np.float32)
        result[good]=source[good].astype(np.float32)
        missing=LAND_MASK&~good
        if missing.any():
            nearest=distance_transform_edt(~good,return_distances=False,return_indices=True)
            result[missing]=result[nearest[0,missing],nearest[1,missing]]
            total_filled+=int(missing.sum())
        result[~LAND_MASK]=np.nan
        variable[year_index,:,:]=result
    return total_filled
def process_file(path):
    tmp=OUT/(path.stem+".partial.nc")
    final=OUT/path.name
    tmp.unlink(missing_ok=True)
    with xr.open_dataset(path,decode_times=False,mask_and_scale=True,engine="netcdf4") as ds:
        if "npp" not in ds:
            raise ValueError("Missing variable: npp")
        if "lat" not in ds:
            raise ValueError("Missing coordinate: lat")
        if "lon" not in ds:
            raise ValueError("Missing coordinate: lon")
        da=ds["npp"]
        if da.ndim!=3:
            raise ValueError(f"npp must be 3-D: {da.dims}")
        if "lat" not in da.dims or "lon" not in da.dims:
            raise ValueError(f"npp does not contain lat/lon dimensions: {da.dims}")
        if ds["lat"].ndim!=1 or ds["lon"].ndim!=1:
            raise ValueError("lat and lon must be one-dimensional")
        time_dim,source_years=get_years(ds,da)
        if len(source_years)!=da.sizes[time_dim]:
            raise ValueError("Year coordinate length differs from npp")
        unique_years,counts=np.unique(source_years,return_counts=True)
        if np.any(counts>1):
            raise ValueError(f"Duplicate years: {unique_years[counts>1].tolist()}")
        source_lookup={int(year):i for i,year in enumerate(source_years)}
        available_years=np.array([int(y) for y in YEARS if int(y) in source_lookup],dtype=int)
        if len(available_years)==0:
            raise ValueError("No source years overlap 1899-2024")
        regions=[prepare_region(ds,bounds) for bounds in REGIONS]
        nc,var=create_output(tmp)
        try:
            for region in regions:
                lat_target=region["target_lat_indices"]
                lon_target=region["target_lon_indices"]
                lat_slice=slice(lat_target[0],lat_target[-1]+1)
                lon_slice=slice(lon_target[0],lon_target[-1]+1)
                for year in available_years:
                    source_index=source_lookup[int(year)]
                    output_index=int(np.where(YEARS==year)[0][0])
                    values=read_data(da,region,time_dim,source_index)
                    values=convert_npp(values)
                    values=regrid_bilinear(values,region)
                    var[output_index,lat_slice,lon_slice]=values
            available_indices=np.array([int(np.where(YEARS==year)[0][0]) for year in available_years],dtype=int)
            filled_cells=fill_land_cells(var,available_indices)
        finally:
            nc.close()
    os.replace(tmp,final)
    return int(source_years.min()),int(source_years.max()),int(available_years.min()),int(available_years.max()),len(available_years),filled_cells
files=sorted(INPUT.glob("*.nc"))
print(f"Land template cells: {LAND_COUNT}")
print(f"Input files: {len(files)}")
print(f"Output template: {YEARS[0]}-{YEARS[-1]}")
for index,path in enumerate(files,1):
    print(f"[{index}/{len(files)}] Processing: {path.name}",flush=True)
    try:
        source_start,source_end,used_start,used_end,n_years,filled_cells=process_file(path)
        print(f"Source years: {source_start}-{source_end}",flush=True)
        print(f"Used years: {used_start}-{used_end} ({n_years} years)",flush=True)
        print(f"Filled land cells: {filled_cells}",flush=True)
        print("✅ Saved file:",OUT/path.name,flush=True)
    except Exception as exc:
        partial=OUT/(path.stem+".partial.nc")
        partial.unlink(missing_ok=True)
        print(f"[Failed] {path.name}: {type(exc).__name__}: {exc}",flush=True)
print("Processing finished")