import re,numpy as np,xarray as xr,matplotlib.pyplot as plt,geopandas as gpd,pandas as pd
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator,FuncFormatter,FixedLocator
from pathlib import Path
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['mathtext.rm']='Times New Roman'
plt.rcParams['mathtext.it']='Times New Roman:italic'
plt.rcParams['figure.dpi']=300
plt.rcParams['axes.labelsize']=26
plt.rcParams['xtick.labelsize']=26
plt.rcParams['ytick.labelsize']=26
ROOT=Path('D:/')
nc1=ROOT/'data/Processed data/Sliding difference/Sliding difference1-1900-center.nc'
tree_path=ROOT/'data/Processed data/TRW/AgeDepSpline (5+3).nc'
shp_world=ROOT/'data/Map/World.shp'
shp_boundary=ROOT/'data/Map/Boundary_of_warm_zone_and_cold_zone.shp'
eu_shp_path=ROOT/'data/Map/Europe 7-class.shp'
na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp'
r_diff_dir=ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (5+3)/All r_diff'
out_png=ROOT / 'results/Fig. 4.png'
Y0,Y1=1911,2014
PLOT_Y0,PLOT_Y1=1911,2014
YEAR_SHIFT=0
TIME_DIM='window_center_year'
MAP_YEARS=np.arange(Y0-YEAR_SHIFT,Y1-YEAR_SHIFT+1)
PLOT_YEARS=np.arange(PLOT_Y0-YEAR_SHIFT,PLOT_Y1-YEAR_SHIFT+1)
X_TICKS=[1920,1940,1960,1980,2000,2020]
VARIABLES=['delta_SPEI12_68','delta_scpdsi_68','delta_tmp_68']
COLORS={'delta_SPEI12_68':'#8c5e37','delta_scpdsi_68':'#d9a97c','delta_tmp_68':'#2b6f4e'}
r_diff_label=r'$\mathit{r}_{\mathrm{moving}}$ (NA & EU)'
eu_lon_min,eu_lon_max=-11.0,57.0
eu_lat_min,eu_lat_max=29.0,75.0
na_lon_min,na_lon_max=-167.27,-55.0
na_lat_min,na_lat_max=9.58,74.99
def get_zone_col(gdf):
    for name in ['class7','CLASS7','zone','ZONE','Koppen','koppen','CLIMATE','climate','Class','class']:
        if name in gdf.columns:return name
    return [c for c in gdf.columns if c!='geometry'][0]
def build_masks(ds):
    lat=ds['latitude'].values
    lon=ds['longitude'].values
    lon2,lat2=np.meshgrid(lon,lat)
    idx=np.arange(lon2.size)
    frame=pd.DataFrame({'lon':lon2.ravel(),'lat':lat2.ravel(),'idx':idx})
    points=gpd.GeoDataFrame(frame,geometry=gpd.points_from_xy(frame.lon,frame.lat),crs='EPSG:4326')
    eu=gpd.read_file(eu_shp_path)
    na=gpd.read_file(na_shp_path)
    eu=eu.set_crs('EPSG:4326') if eu.crs is None else eu.to_crs('EPSG:4326')
    na=na.set_crs('EPSG:4326') if na.crs is None else na.to_crs('EPSG:4326')
    eu_col,na_col=get_zone_col(eu),get_zone_col(na)
    eu=eu[['geometry',eu_col]].rename(columns={eu_col:'zone'})
    na=na[['geometry',na_col]].rename(columns={na_col:'zone'})
    climate=gpd.GeoDataFrame(pd.concat([eu,na],ignore_index=True),crs='EPSG:4326')
    joined=gpd.sjoin(points,climate,how='left',predicate='within')
    zones=np.array(joined.set_index('idx')['zone'].reindex(idx).astype(str).values,dtype=object)
    warm_codes={'A','B','C','Dfa','Dfb','Tropical','Arid','Temperate'}
    cold_codes={'D other','Dother','D','E','Cold','Subpolar','ET','EF'}
    warm=np.array([zone in warm_codes for zone in zones]).reshape(lat.size,lon.size)
    cold=np.array([zone in cold_codes for zone in zones]).reshape(lat.size,lon.size)
    coords={'latitude':lat,'longitude':lon}
    dims=('latitude','longitude')
    return xr.DataArray(warm,coords=coords,dims=dims),xr.DataArray(cold,coords=coords,dims=dims)
def spatial_mask(ds,lon_min,lon_max,lat_min,lat_max):
    return (ds['latitude']>=lat_min)&(ds['latitude']<=lat_max)&(ds['longitude']>=lon_min)&(ds['longitude']<=lon_max)
def coordinate_tolerance(coordinate):
    values=np.sort(np.unique(np.asarray(coordinate,dtype=float)))
    spacing=np.diff(values)
    spacing=spacing[spacing>0]
    return float(np.median(spacing)/2+1e-6) if spacing.size else 1e-6
def load_tree_mask(ds):
    with xr.open_dataset(tree_path) as tree:
        da=tree['AgeDepSpline'].rename({d:'latitude' if 'lat' in d else ('longitude' if 'lon' in d else d) for d in tree['AgeDepSpline'].dims})
        time_dim=[dim for dim in da.dims if dim not in ('latitude','longitude')][0]
        years=np.asarray(da[time_dim].values)
        years=np.rint(years.astype(float)).astype(int) if np.issubdtype(years.dtype,np.number) else pd.to_datetime(years).year.to_numpy()
        da=da.isel({time_dim:int(np.flatnonzero(years==2020)[0])}).squeeze(drop=True)
        lon=np.asarray(da['longitude'].values,dtype=float)
        if lon.min()>=0 and np.any(ds['longitude'].values<0):da=da.assign_coords(longitude=((lon+180)%360)-180)
        da=da.sortby('latitude').sortby('longitude')
        tree_valid=xr.apply_ufunc(np.isfinite,da).astype(float)
        tree_valid=tree_valid.reindex(latitude=ds['latitude'],method='nearest',tolerance=coordinate_tolerance(da['latitude'].values))
        tree_valid=tree_valid.reindex(longitude=ds['longitude'],method='nearest',tolerance=coordinate_tolerance(da['longitude'].values))
        return tree_valid.fillna(0).astype(bool).transpose('latitude','longitude')
def display_name(name):
    name=name.removeprefix('delta_').removesuffix('_68')
    return f"{re.sub('(?i)scpdsi','scPDSI',name)} (summer)"
def da_to_series(da):
    other=[dim for dim in da.dims if dim!=TIME_DIM]
    if other:da=da.mean(other,skipna=True)
    series=da.to_pandas()
    series.index=series.index+YEAR_SHIFT
    return series.loc[(series.index>=PLOT_Y0)&(series.index<=PLOT_Y1)]
def read_r_diff_series(zone):
    path=r_diff_dir/f'{zone}.csv'
    if not path.exists():raise FileNotFoundError(path)
    frame=pd.read_csv(path)
    if not {'year','mean_r'}.issubset(frame.columns):raise ValueError(f'Required columns are missing: {path}')
    frame=frame[['year','mean_r']].apply(pd.to_numeric,errors='coerce')
    frame=frame[np.isfinite(frame['year'])&np.isfinite(frame['mean_r'])]
    frame=frame.groupby('year',as_index=False)['mean_r'].mean().sort_values('year')
    frame=frame[(frame['year']>=PLOT_Y0)&(frame['year']<=PLOT_Y1)]
    return pd.Series(frame['mean_r'].to_numpy(dtype=float),index=frame['year'].to_numpy(dtype=int))
def assign_factors(values,region_mask):
    valid=np.isfinite(values).any(axis=0)&np.asarray(region_mask,dtype=bool)
    winners=np.argmax(np.where(np.isfinite(values),values,-np.inf),axis=0)
    assigned=np.ma.array(winners,mask=~valid)
    counts=np.bincount(winners[valid].ravel(),minlength=len(VARIABLES))
    return assigned,counts,int(valid.sum())
def spherical_points(lon,lat):
    lon,lat=np.deg2rad(lon),np.deg2rad(lat)
    return np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
def assign_map_factors(values,region_mask,tree_mask,lon2d,lat2d):
    region=np.asarray(region_mask,dtype=bool)
    target=region&np.asarray(tree_mask,dtype=bool)
    source_valid=np.isfinite(values).any(axis=0)
    donors=target&source_valid
    if not np.any(donors):donors=region&source_valid
    winners=np.argmax(np.where(np.isfinite(values),values,-np.inf),axis=0)
    result=np.zeros(target.shape,dtype=np.int16)
    direct=target&source_valid
    result[direct]=winners[direct]
    missing=target&~source_valid
    if np.any(missing):
        donor_pts=spherical_points(lon2d[donors],lat2d[donors])
        missing_pts=spherical_points(lon2d[missing],lat2d[missing])
        nearest=cKDTree(donor_pts).query(missing_pts,k=1)[1]
        result[missing]=winners[donors][nearest]
    return np.ma.array(result,mask=~target),np.bincount(result[target].ravel(),minlength=len(VARIABLES)),int(target.sum())
def draw_panel(ds,values,mask,label,ax,reference,ylabel=None,mark_years=None,label_x=0.86,legend_loc='upper left',legend_bbox=(0,0.9)):
    assigned,counts,total=assign_factors(values,mask.values)
    for index,name in enumerate(VARIABLES):
        series=da_to_series(ds[name].where(mask))
        ax.plot(series.index,series.values,lw=1.8,color=COLORS[name],label=f'{display_name(name)} ({counts[index]/total*100:.1f}%)')
    ax.plot(reference.index,reference.values,ls='-.',lw=2.2,color='k',label=r_diff_label)
    ax.margins(y=0.12)
    if ylabel is not None:ax.set_ylabel(ylabel,fontsize=30)
    ax.set_xlabel('Year',fontsize=30)
    ax.text(label_x,0.96,label,transform=ax.transAxes,ha='left',va='top',fontsize=32,fontweight='bold')
    years=[int(y) for y in mark_years] if mark_years else ([int(reference.idxmax())] if reference.notna().any() else [])
    for year in years:
        if PLOT_Y0<=year<=PLOT_Y1:
            ax.axvline(year,color='k',ls='--',lw=1.4)
            ax.text(year,0.035,str(year),transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=26)
    handles,labels=ax.get_legend_handles_labels()
    items=[(h,t) for h,t in zip(handles,labels) if t==r_diff_label]+[(h,t) for h,t in zip(handles,labels) if t!=r_diff_label]
    ax.legend([it[0] for it in items],[it[1] for it in items],loc=legend_loc,bbox_to_anchor=legend_bbox,frameon=True,fontsize=20)
def format_pct(counts):
    total=np.sum(counts)
    if total==0:return [0.0]*len(counts)
    pcts=np.round(counts/total*100,1)
    diff=round(100.0-float(np.sum(pcts)),1)
    if not np.isclose(diff,0):
        target_idx=int(np.argmax(counts))
        pcts[target_idx]=round(pcts[target_idx]+diff,1)
    return pcts
source=xr.open_dataset(nc1)
ds=source.reindex({TIME_DIM:PLOT_YEARS})
tree_mask=load_tree_mask(ds)
warm_mask,cold_mask=build_masks(ds)
warm_reference=read_r_diff_series('Warm')
cold_reference=read_r_diff_series('Cold')
na_mask=spatial_mask(ds,na_lon_min,na_lon_max,na_lat_min,na_lat_max)
eu_mask=spatial_mask(ds,eu_lon_min,eu_lon_max,eu_lat_min,eu_lat_max)
arrays=[ds[name].sel({TIME_DIM:MAP_YEARS}).mean(TIME_DIM,skipna=True) for name in VARIABLES]
values=xr.concat(arrays,dim='factor').transpose('factor','latitude','longitude').values
lon2d,lat2d=np.meshgrid(ds['longitude'].values,ds['latitude'].values)
na_assigned,na_counts,na_total=assign_map_factors(values,na_mask.values,tree_mask.values,lon2d,lat2d)
eu_assigned,eu_counts,eu_total=assign_map_factors(values,eu_mask.values,tree_mask.values,lon2d,lat2d)
map_pcts=format_pct(na_counts+eu_counts)
map_colors=[COLORS[name] for name in VARIABLES]
map_handles=[Patch(facecolor=COLORS[name],edgecolor='none',label=f'{display_name(name)} ({map_pcts[i]:.1f}%)') for i,name in enumerate(VARIABLES)]
map_handles.append(Line2D([0],[0],color='#111111',lw=2.4,ls=(0,(4,1.5)),label='Warm/Cold boundary'))
world=gpd.read_file(shp_world)
world=world.set_crs(epsg=4326) if world.crs is None else world.to_crs(epsg=4326)
boundary=gpd.read_file(shp_boundary)
boundary=boundary.set_crs(epsg=4326) if boundary.crs is None else boundary.to_crs(epsg=4326)
wr_na=(na_lon_max-na_lon_min)/(na_lat_max-na_lat_min)
wr_eu=(eu_lon_max-eu_lon_min)/(eu_lat_max-eu_lat_min)
fig=plt.figure(figsize=(18,14),dpi=300)
gs=gridspec.GridSpec(2,2,figure=fig,left=0.06,right=0.97,bottom=0.075,top=0.975,wspace=0.1,hspace=0.14,width_ratios=[wr_na,wr_eu],height_ratios=[1.35,1.0])
ax_na=fig.add_subplot(gs[0,0])
ax_eu=fig.add_subplot(gs[0,1])
ax_warm=fig.add_subplot(gs[1,0])
ax_cold=fig.add_subplot(gs[1,1])
ax_na.text(0.05,0.98,'a',transform=ax_na.transAxes,ha='right',va='top',fontsize=40,fontweight='bold')
ax_eu.text(0.06,0.98,'b',transform=ax_eu.transAxes,ha='right',va='top',fontsize=40,fontweight='bold')
ax_warm.text(0.05,0.98,'c',transform=ax_warm.transAxes,ha='right',va='top',fontsize=40,fontweight='bold')
ax_cold.text(0.06,0.98,'d',transform=ax_cold.transAxes,ha='right',va='top',fontsize=40,fontweight='bold')
world.cx[na_lon_min:na_lon_max,na_lat_min:na_lat_max].plot(ax=ax_na,facecolor='0.9',edgecolor='none',zorder=0)
ax_na.pcolormesh(lon2d,lat2d,na_assigned,cmap=ListedColormap(map_colors),vmin=-0.5,vmax=len(VARIABLES)-0.5,shading='nearest',zorder=1)
boundary.cx[na_lon_min:na_lon_max,na_lat_min:na_lat_max].plot(ax=ax_na,color='white',linewidth=4.0,zorder=2)
boundary.cx[na_lon_min:na_lon_max,na_lat_min:na_lat_max].plot(ax=ax_na,color='#111111',linewidth=2.4,linestyle=(0,(4,1.5)),zorder=3)
ax_na.set_aspect(1.3,adjustable='box')
ax_na.legend(handles=map_handles,loc='lower left',frameon=True,fontsize=18)
ax_na.set_xlim(na_lon_min,na_lon_max)
ax_na.set_ylim(na_lat_min,na_lat_max)
ax_na.margins(0,0)
ax_na.tick_params(axis='both',which='major',labelsize=26)
ax_na.xaxis.set_major_locator(MultipleLocator(20))
ax_na.yaxis.set_major_locator(MultipleLocator(10))
ax_na.xaxis.set_major_formatter(FuncFormatter(lambda v,p:'0°' if np.isclose(v,0) else f'{abs(v):g}°{"E" if v>0 else "W"}'))
ax_na.yaxis.set_major_formatter(FuncFormatter(lambda v,p:'0°' if np.isclose(v,0) else f'{abs(v):g}°{"N" if v>0 else "S"}'))
world.cx[eu_lon_min:eu_lon_max,eu_lat_min:eu_lat_max].plot(ax=ax_eu,facecolor='0.9',edgecolor='none',zorder=0)
ax_eu.pcolormesh(lon2d,lat2d,eu_assigned,cmap=ListedColormap(map_colors),vmin=-0.5,vmax=len(VARIABLES)-0.5,shading='nearest',zorder=1)
boundary.cx[eu_lon_min:eu_lon_max,eu_lat_min:eu_lat_max].plot(ax=ax_eu,color='white',linewidth=4.0,zorder=2)
boundary.cx[eu_lon_min:eu_lon_max,eu_lat_min:eu_lat_max].plot(ax=ax_eu,color='#111111',linewidth=2.4,linestyle=(0,(4,1.5)),zorder=3)
ax_eu.set_aspect(1.3,adjustable='box')
ax_eu.set_xlim(eu_lon_min,eu_lon_max)
ax_eu.set_ylim(eu_lat_min,eu_lat_max)
ax_eu.margins(0,0)
ax_eu.tick_params(axis='both',which='major',labelsize=26,labelleft=False,left=False,right=True,labelright=True)
ax_eu.xaxis.set_major_locator(MultipleLocator(10))
ax_eu.yaxis.set_major_locator(MultipleLocator(10))
ax_eu.xaxis.set_major_formatter(FuncFormatter(lambda v,p:'0°' if np.isclose(v,0) else f'{abs(v):g}°{"E" if v>0 else "W"}'))
ax_eu.yaxis.set_major_formatter(FuncFormatter(lambda v,p:'0°' if np.isclose(v,0) else f'{abs(v):g}°{"N" if v>0 else "S"}'))
draw_panel(ds,values,warm_mask,'Warm',ax_warm,warm_reference,ylabel=r'$\Delta r = r_{\mathrm{moving}} - r_{\mathrm{moving}\mid\mathrm{climate}}$',mark_years=[1933,1998],label_x=0.84,legend_loc='upper left',legend_bbox=(0,0.9))
draw_panel(ds,values,cold_mask,'Cold',ax_cold,cold_reference,mark_years=[1933,1998],label_x=0.86,legend_loc='upper right',legend_bbox=(1.0,0.9))
ymin_warm=ax_warm.get_ylim()[0]
ymin_cold=ax_cold.get_ylim()[0]
ax_warm.set_ylim(ymin_warm,0.85)
ax_cold.set_ylim(ymin_cold,0.5)
for ax in [ax_warm,ax_cold]:
    ax.set_xlim(PLOT_Y0,PLOT_Y1)
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.1f}'))
ax_cold.set_ylabel('')
ax_cold.tick_params(axis='y',which='major',left=True,labelleft=True)
out_png.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(out_png,bbox_inches='tight',pad_inches=0.02,facecolor='white',dpi=600)
plt.close()
print('✅ Saved figure:', out_png)