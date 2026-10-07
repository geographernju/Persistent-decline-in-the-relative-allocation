import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point,box
from shapely.prepared import prep
from scipy.spatial import cKDTree
ROOT=Path("D:/data")
points_csv=ROOT/"1. Basic information of global tree-ring sites.csv"
agespline_csv=Path('D:/data/Processed data/Detrending/fill/AgeDepSpline+fill.csv')
world_shp_path=ROOT/"Map"/"World_countries.shp"
out_nc=Path('D:/data/Processed data/TRW/recon_mask (3+3).nc')
grid_res,anchor_lon,anchor_lat,bbox_margin=0.5,-179.75,-89.75,5.0
radius_deg,min_points=3,3
full_years=np.arange(1900,2026,dtype=np.int32)
pts=pd.read_csv(points_csv)[["longitude","latitude","name"]].dropna().reset_index(drop=True)
if pts.empty:
    raise ValueError("Site table is empty")
points_xy=pts[["longitude","latitude"]].to_numpy()
names=pts["name"].astype(str).to_numpy()
ts=pd.read_csv(agespline_csv)
ts=ts.rename(columns={ts.columns[0]:"year"}).set_index("year").reindex(full_years)
used=[c for c in ts.columns if c in set(names)]
if not used:
    raise ValueError("No matching site names were found in the chronology table")
mat=ts[used].to_numpy(dtype=np.float64)
mat[mat==0]=np.nan
fin=np.isfinite(mat)
idx_by_name={nm:j for j,nm in enumerate(used)}
col_idx=np.array([idx_by_name.get(nm,-1) for nm in names],dtype=int)
has_col=col_idx>=0
def axis_aligned(vmin,vmax,anchor,step):
    a=int(np.ceil((vmin-anchor)/step))
    b=int(np.floor((vmax-anchor)/step))
    return (anchor+np.arange(a,b+1)*step).astype(np.float32)
lon=axis_aligned(points_xy[:,0].min()-bbox_margin,points_xy[:,0].max()+bbox_margin,anchor_lon,grid_res)
lat=axis_aligned(points_xy[:,1].min()-bbox_margin,points_xy[:,1].max()+bbox_margin,anchor_lat,grid_res)
LON,LAT=np.meshgrid(lon,lat)
grid=np.c_[LON.ravel(),LAT.ravel()]
G=grid.shape[0]
world=gpd.read_file(world_shp_path).to_crs(4326)
bbox=box(lon.min(),lat.min(),lon.max(),lat.max())
try:
    ids=list(world.sindex.query(bbox,predicate="intersects"))
    sub=world.iloc[ids]
except Exception:
    sub=world
try:
    land=sub.union_all()
except AttributeError:
    land=sub.unary_union
prep_land=prep(land)
land_mask=np.fromiter((prep_land.intersects(Point(xy)) for xy in grid),count=G,dtype=bool)
tree=cKDTree(points_xy)
k=128
dists,idxs=tree.query(grid,k=k,distance_upper_bound=radius_deg)
valid=np.isfinite(dists)&(idxs<len(pts))
safe_idxs=np.where(valid,idxs,0)
Y=full_years.size
mask3d=np.full((Y,lat.size,lon.size),np.nan,dtype=np.float32)
valid_by_point_year=np.zeros((Y,len(pts)),dtype=bool)
if np.any(has_col):
    valid_by_point_year[:,has_col]=fin[:,col_idx[has_col]]
for yi in range(Y):
    nn_year=valid_by_point_year[yi,:][safe_idxs]
    nn_year[~valid]=False
    ok=(np.sum(nn_year,axis=1)>=min_points)&land_mask
    mask3d[yi]=np.where(ok.reshape(lat.size,lon.size),1.0,np.nan).astype(np.float32)
ds=xr.Dataset(data_vars={"recon_mask":(["year","latitude","longitude"],mask3d)},coords={"year":full_years,"latitude":lat,"longitude":lon},attrs={"description":f"Per-year reconstructable mask over land with at least {min_points} usable tree-ring sites within {radius_deg} degrees","resolution":"0.5 degree","radius_deg":radius_deg,"min_points":min_points})
ds["recon_mask"].encoding.update({"zlib":True,"complevel":4,"_FillValue":np.nan})
out_nc.parent.mkdir(parents=True,exist_ok=True)
ds.to_netcdf(out_nc)
print('✅ Saved file:',out_nc)