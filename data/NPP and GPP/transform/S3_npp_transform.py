from pathlib import Path
from contextlib import ExitStack
from calendar import monthrange
import os
import re
import shutil
import numpy as np
import xarray as xr
import cftime
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
ROOT=Path("D:/")
# 原始数据下载脚本：D:/data/NPP and GPP/transform/wget.sh；来源：https://mdosullivan.github.io/GCB/
INPUT=ROOT/"S3_npp"
OUT_12=Path('D:/data/NPP and GPP/S3_npp1-12')
OUT_98=Path('D:/data/NPP and GPP/S3_npp9-8')
FAIL=Path("D:/data/NPP and GPP")
MASK_FILE=Path('D:/data/NPP and GPP/transform/1.nc')
for folder in (OUT_12,OUT_98,FAIL):
    folder.mkdir(parents=True,exist_ok=True)
YEARS=np.arange(1899,2025,dtype=np.int16)
LATITUDE=np.arange(9.75,75.0,0.5,dtype=np.float32)
LONGITUDE=np.arange(-167.25,55.0,0.5,dtype=np.float32)
REGIONS=((-167.27,-55.0,9.58,74.99),(-11.0,55.0,29.0,75.0))
with Dataset(MASK_FILE) as mask_nc:
    mask_lat=np.asarray(mask_nc.variables["latitude"][:],dtype=float)
    mask_lon=np.asarray(mask_nc.variables["longitude"][:],dtype=float)
    mask_values=np.ma.asarray(mask_nc.variables["land_mask"][:],dtype=np.float64).filled(np.nan)
if mask_values.shape!=(len(LATITUDE),len(LONGITUDE)):
    raise ValueError("Land mask shape differs from the output grid")
if not np.array_equal(mask_lat,LATITUDE) or not np.array_equal(mask_lon,LONGITUDE):
    raise ValueError("Land mask coordinates differ from the output grid")
LAND_MASK=np.isfinite(mask_values)&(mask_values==1)
LAND_COUNT=int(LAND_MASK.sum())
if LAND_COUNT==0:
    raise ValueError("The land mask contains no valid cells")
print(f"Land template cells: {LAND_COUNT}")
def month_days(year,month,calendar_name):
    name=str(calendar_name).lower()
    if name=="360_day":
        return 30
    if name in ("365_day","noleap"):
        return monthrange(2001,month)[1]
    if name in ("366_day","all_leap"):
        return monthrange(2000,month)[1]
    return monthrange(year,month)[1]
def annual_days(year,calendar_name):
    return sum(month_days(year,month,calendar_name) for month in range(1,13))
def parse_time(ds,path):
    values=np.asarray(ds["time"].values,dtype=float)
    units=str(ds["time"].attrs.get("units",""))
    calendar_name=str(ds["time"].attrs.get("calendar","standard"))
    if path.name.startswith("CARDAMOM_"):
        if not np.array_equal(values,np.arange(1,len(values)+1)):
            raise ValueError("Unexpected CARDAMOM time index")
        offsets=values.astype(int)-1
        years=2003+offsets//12
        months=offsets%12+1
    elif re.search(r"years\s+since\s+\d{1,4}-",units,re.I):
        match=re.search(r"years\s+since\s+(\d{1,4})-(\d{1,2})",units,re.I)
        years=int(match.group(1))+np.floor(values+1e-7).astype(int)
        months=np.full(len(values),int(match.group(2)),dtype=int)
    elif "months since" in units.lower():
        match=re.search(r"months\s+since\s+(\d{1,4})-(\d{1,2})",units,re.I)
        if match is None:
            raise ValueError(f"Cannot parse month origin: {units}")
        offsets=np.floor(values+1e-7).astype(int)
        positions=int(match.group(1))*12+int(match.group(2))-1+offsets
        years=positions//12
        months=positions%12+1
    elif not units and len(values) and np.nanmin(values)>=1600 and np.nanmax(values)<2200:
        years=np.floor(values+1e-7).astype(int)
        months=np.floor((values-years)*12+1e-6).astype(int)+1
    else:
        dates=cftime.num2date(values,units=units,calendar=calendar_name,only_use_cftime_datetimes=True)
        years=np.array([date.year for date in dates],dtype=int)
        months=np.array([date.month for date in dates],dtype=int)
    if np.any((months<1)|(months>12)):
        raise ValueError("Invalid months in time coordinate")
    return years,months,calendar_name
def prepare_region(ds,lat_dim,lon_dim,bounds):
    lon_min,lon_max,lat_min,lat_max=bounds
    source_lat_all=np.asarray(ds[lat_dim].values,dtype=float)
    source_lon_all=((np.asarray(ds[lon_dim].values,dtype=float)+180)%360)-180
    lat_indices=np.flatnonzero((source_lat_all>=lat_min-4)&(source_lat_all<=lat_max+4))
    lon_indices=np.flatnonzero((source_lon_all>=lon_min-4)&(source_lon_all<=lon_max+4))
    target_lat_indices=np.flatnonzero((LATITUDE>=lat_min)&(LATITUDE<=lat_max))
    target_lon_indices=np.flatnonzero((LONGITUDE>=lon_min)&(LONGITUDE<=lon_max))
    if len(lat_indices)<2 or len(lon_indices)<2:
        raise ValueError(f"Insufficient source grid coverage: {bounds}")
    lat_order=np.argsort(source_lat_all[lat_indices])
    lon_order=np.argsort(source_lon_all[lon_indices])
    source_lat=source_lat_all[lat_indices][lat_order]
    source_lon=source_lon_all[lon_indices][lon_order]
    if np.any(np.diff(source_lat)<=0) or np.any(np.diff(source_lon)<=0):
        raise ValueError("Source coordinates contain duplicates")
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
def read_data(da,region,time_indices,lat_dim,lon_dim):
    selected=da.isel({"time":np.asarray(time_indices,dtype=int),lat_dim:region["lat_indices"],lon_dim:region["lon_indices"]}).transpose("time",lat_dim,lon_dim)
    values=np.asarray(selected.values,dtype=np.float64)
    values=values[:,region["lat_order"],:][:,:,region["lon_order"]]
    values[~np.isfinite(values)]=np.nan
    return values
def regrid_bilinear(values,region):
    values=np.where(np.isfinite(values)&(values>=0)&(values<1e30),values,np.nan)
    i0=region["lat_left"]
    i1=region["lat_right"]
    j0=region["lon_left"]
    j1=region["lon_right"]
    a=region["lat_weight"]
    b=region["lon_weight"]
    corners=np.stack((values[np.ix_(i0,j0)],values[np.ix_(i0,j1)],values[np.ix_(i1,j0)],values[np.ix_(i1,j1)]))
    weights=np.stack(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b))
    valid=np.isfinite(corners)
    numerator=np.sum(np.where(valid,corners,0)*weights,axis=0)
    denominator=np.sum(valid*weights,axis=0)
    result=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    result[~region["inside_lat"],:]=np.nan
    result[:,~region["inside_lon"]]=np.nan
    return result.astype(np.float32)
def sum_complete_months(values):
    valid=np.isfinite(values).all(axis=0)
    total=np.nansum(values,axis=0)
    total[~valid]=np.nan
    return total
def create_output(path):
    nc=Dataset(path,"w",format="NETCDF4")
    nc.createDimension("year",len(YEARS))
    nc.createDimension("latitude",len(LATITUDE))
    nc.createDimension("longitude",len(LONGITUDE))
    year_var=nc.createVariable("year","i2",("year",))
    lat_var=nc.createVariable("latitude","f4",("latitude",))
    lon_var=nc.createVariable("longitude","f4",("longitude",))
    data_var=nc.createVariable("npp","f4",("year","latitude","longitude"),fill_value=np.float32(np.nan),zlib=False,contiguous=True)
    year_var[:]=YEARS
    lat_var[:]=LATITUDE
    lon_var[:]=LONGITUDE
    data_var[:]=np.nan
    year_var.long_name="year"
    lat_var.units="degrees_north"
    lon_var.units="degrees_east"
    data_var.long_name="Annual gross primary productivity"
    data_var.units="kg C m-2 yr-1"
    return nc,data_var
def enforce_land_mask(variable):
    complete_years=0
    empty_years=0
    filled_cells=0
    for year_index,year in enumerate(YEARS):
        source=np.ma.asarray(variable[year_index,:,:],dtype=np.float64).filled(np.nan)
        good=LAND_MASK&np.isfinite(source)&(source>=0)&(source<1e30)
        result=np.full(LAND_MASK.shape,np.nan,dtype=np.float32)
        if not good.any():
            variable[year_index,:,:]=result
            empty_years+=1
            continue
        result[good]=source[good].astype(np.float32)
        missing=LAND_MASK&~good
        if missing.any():
            nearest=distance_transform_edt(~good,return_distances=False,return_indices=True)
            result[missing]=result[nearest[0,missing],nearest[1,missing]]
            filled_cells+=int(missing.sum())
        if not np.array_equal(np.isfinite(result),LAND_MASK):
            raise ValueError(f"Land mask mismatch in year {int(year)}")
        if int(np.isfinite(result).sum())!=LAND_COUNT:
            raise ValueError(f"Land cell count mismatch in year {int(year)}")
        variable[year_index,:,:]=result
        complete_years+=1
    return complete_years,empty_years,filled_cells
def process_file(path):
    annual_input=path.name.lower().endswith("_nppannual.nc")
    tmp_12=OUT_12/(path.stem+".partial.nc")
    tmp_98=OUT_98/(path.stem+".partial.nc")
    tmp_12.unlink(missing_ok=True)
    tmp_98.unlink(missing_ok=True)
    with xr.open_dataset(path,decode_times=False,mask_and_scale=True,engine="netcdf4") as ds:
        variable="nppAnnual" if annual_input else "npp"
        if variable not in ds:
            raise ValueError(f"Missing variable: {variable}")
        da=ds[variable]
        lat_dim=next((dim for dim in da.dims if dim.lower() in ("lat","latitude")),None)
        lon_dim=next((dim for dim in da.dims if dim.lower() in ("lon","longitude")),None)
        if "time" not in da.dims or lat_dim is None or lon_dim is None or da.ndim!=3:
            raise ValueError(f"Unsupported npp dimensions: {da.dims}")
        if ds[lat_dim].ndim!=1 or ds[lon_dim].ndim!=1:
            raise ValueError("Latitude and longitude must be one-dimensional")
        raw_units=str(da.attrs.get("units",""))
        units=re.sub(r"[^a-z0-9]","",raw_units.lower())
        if not re.search(r"kg.*m2.*s1",units):
            raise ValueError(f"Unsupported npp units: {raw_units}")
        time_years,time_months,calendar_name=parse_time(ds,path)
        if len(time_years)!=da.sizes["time"]:
            raise ValueError("Time and npp lengths differ")
        keys=time_years.tolist() if annual_input else list(zip(time_years.tolist(),time_months.tolist()))
        if len(keys)!=len(set(keys)):
            raise ValueError("Duplicate year or year-month records")
        lookup={key:index for index,key in enumerate(keys)}
        regions=[prepare_region(ds,lat_dim,lon_dim,bounds) for bounds in REGIONS]
        with ExitStack() as stack:
            nc_12,var_12=create_output(tmp_12)
            stack.callback(nc_12.close)
            if not annual_input:
                nc_98,var_98=create_output(tmp_98)
                stack.callback(nc_98.close)
            for region in regions:
                lat_target=region["target_lat_indices"]
                lon_target=region["target_lon_indices"]
                lat_slice=slice(lat_target[0],lat_target[-1]+1)
                lon_slice=slice(lon_target[0],lon_target[-1]+1)
                previous_tail=None
                for year_index,year_value in enumerate(YEARS):
                    year=int(year_value)
                    if annual_input:
                        position=lookup.get(year)
                        if position is not None:
                            values=read_data(da,region,[position],lat_dim,lon_dim)[0]
                            values=values*annual_days(year,calendar_name)*86400
                            var_12[year_index,lat_slice,lon_slice]=regrid_bilinear(values,region)
                        continue
                    monthly=np.full((12,len(region["lat_indices"]),len(region["lon_indices"])),np.nan,dtype=np.float64)
                    month_list=[month for month in range(1,13) if (year,month) in lookup]
                    if month_list:
                        positions=[lookup[(year,month)] for month in month_list]
                        values=read_data(da,region,positions,lat_dim,lon_dim)
                        for offset,month in enumerate(month_list):
                            monthly[month-1]=values[offset]*month_days(year,month,calendar_name)*86400
                    if len(month_list)==12:
                        var_12[year_index,lat_slice,lon_slice]=regrid_bilinear(sum_complete_months(monthly),region)
                    if previous_tail is not None:
                        period=np.concatenate((previous_tail,monthly[:8]),axis=0)
                        var_98[year_index,lat_slice,lon_slice]=regrid_bilinear(sum_complete_months(period),region)
                    previous_tail=monthly[8:].copy()
            stats_12=enforce_land_mask(var_12)
            stats_98=enforce_land_mask(var_98) if not annual_input else None
    os.replace(tmp_12,OUT_12/path.name)
    if not annual_input:
        os.replace(tmp_98,OUT_98/path.name)
    return annual_input,stats_12,stats_98
files=sorted(INPUT.glob("*.nc"))
print(f"Input files: {len(files)}")
print(f"Template: {len(YEARS)} years x {len(LATITUDE)} latitudes x {len(LONGITUDE)} longitudes")
for index,path in enumerate(files,1):
    print(f"[{index}/{len(files)}] Processing: {path.name}",flush=True)
    try:
        annual_input,stats_12,stats_98=process_file(path)
        print(f"1-12: complete years={stats_12[0]}, empty years={stats_12[1]}, filled cells={stats_12[2]}",flush=True)
        print("✅ Saved file:",OUT_12/path.name,flush=True)
        if not annual_input:
            print(f"9-8: complete years={stats_98[0]}, empty years={stats_98[1]}, filled cells={stats_98[2]}",flush=True)
            print("✅ Saved file:",OUT_98/path.name,flush=True)
    except Exception as exc:
        (OUT_12/(path.stem+".partial.nc")).unlink(missing_ok=True)
        (OUT_98/(path.stem+".partial.nc")).unlink(missing_ok=True)
        destination=FAIL/path.name
        counter=1
        while destination.exists():
            destination=FAIL/f"{path.stem}_failed{counter}{path.suffix}"
            counter+=1
        try:
            shutil.move(str(path),str(destination))
            print(f"[Failed] {path.name}: {type(exc).__name__}: {exc}",flush=True)
            print("Moved to:",destination,flush=True)
        except Exception as move_error:
            print(f"[Failed] {path.name}: {type(exc).__name__}: {exc}",flush=True)
            print(f"[Move error] {type(move_error).__name__}: {move_error}",flush=True)
print("Processing finished")