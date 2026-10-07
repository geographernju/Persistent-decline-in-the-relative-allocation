# -*- coding: utf-8 -*-
from pathlib import Path
import re
import numpy as np
import xarray as xr
ROOT=Path("D:/")
a_path=Path('D:/data/Processed data/TRW/AgeDepSpline (5+3).nc')
b_dir=Path('D:/data/NPP and GPP/S2_npp9-8')
out_nc=Path('D:/data/Processed data/Difference_correlation/AgeDepSpline_TRW_vs_TRENDY_NPP_annual_difference_correlation_1981-2010.nc')
out_nc.parent.mkdir(parents=True,exist_ok=True)
yr0,yr1=1981,2010
def sel_year(ds):
    if "year" not in ds.coords: raise ValueError("Missing year coordinate")
    ds=ds.sel(year=slice(yr0,yr1))
    ds["year"]=ds["year"].astype(np.int32)
    return ds
def corr_time(x,y,min_n=3):
    valid=(~np.isnan(x))&(~np.isnan(y))
    n=np.sum(valid,axis=0)
    x=np.where(valid,x,np.nan)
    y=np.where(valid,y,np.nan)
    xm=x-np.nanmean(x,axis=0,keepdims=True)
    ym=y-np.nanmean(y,axis=0,keepdims=True)
    num=np.nansum(xm*ym,axis=0)
    den=np.sqrt(np.nansum(xm*xm,axis=0)*np.nansum(ym*ym,axis=0))
    r=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)
    return np.where(n>=10,r,np.nan)
def sanitize(name):
    s=re.sub(r'[^0-9A-Za-z_]+','_',name.strip())
    return s or "var"
out_nc.parent.mkdir(parents=True,exist_ok=True)
with xr.open_dataset(a_path) as da:
    da=da.load()
da=sel_year(da)
if "AgeDepSpline" not in da: raise ValueError("AgeDepSpline not found")
A=da["AgeDepSpline"].transpose("year","latitude","longitude")
vars_out={}
lat,lon=A.latitude.values.astype(np.float32),A.longitude.values.astype(np.float32)
for fpath in b_dir.glob("*.nc"):
    fname=fpath.name
    try:
        with xr.open_dataset(fpath) as db:
            db=db.load()
        db=sel_year(db)
        if "npp" not in db: continue
        B=db["npp"].transpose("year","latitude","longitude")
        A1,B1=xr.align(A,B,join="inner")
        if A1.sizes["year"]<4: continue
        A1,B1=A1.diff("year"),B1.diff("year")
        A1,B1=xr.align(A1,B1,join="inner")
        if A1.sizes["year"]<3: continue
        r=corr_time(A1.values.astype(float),B1.values.astype(float),min_n=3)
        vname=sanitize(fpath.stem)
        while vname in vars_out: vname+="_x"
        vars_out[vname]=(["latitude","longitude"],r)
        lat,lon=A1.latitude.values.astype(np.float32),A1.longitude.values.astype(np.float32)
    except Exception as e:
        print(f"[Skipped] {fname}: {e}")
if not vars_out: raise RuntimeError("No valid products available")
ds_out=xr.Dataset(data_vars=vars_out,coords=dict(latitude=("latitude",lat),longitude=("longitude",lon)),attrs=dict(description="Pixel-wise annual difference correlation r(dA,dB), A=AgeDepSpline, B=npp",period=f"{yr0}-{yr1}"))
encoding={k:{"zlib":True,"complevel":4,"dtype":"float32","_FillValue":np.float32(np.nan)} for k in vars_out}
ds_out.to_netcdf(out_nc,encoding=encoding)
print('✅ Saved file:',out_nc)