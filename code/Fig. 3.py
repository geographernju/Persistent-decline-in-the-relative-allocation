import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter,MaxNLocator,MultipleLocator,FormatStrFormatter
from pandas.api.types import is_numeric_dtype
ROOT=Path('D:/')
START_YEAR=1911
MIN_N=20
DPI=600
MARK_YEARS=[1933,1998]
plt.rcParams.update({'font.family':'Times New Roman','font.size':22,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','axes.unicode_minus':False,'figure.dpi':150,'savefig.dpi':DPI})
recon_path=ROOT/'data/Processed data/TRW/AgeDepSpline (5+3).nc'
model_path=ROOT/'data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc'
world_shp=ROOT/'data/Map/World.shp'
na_clim_shp=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp'
eu_clim_shp=ROOT/'data/Map/Europe 7-class.shp'
NA_r_diff_dir=ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (5+3)/North America r_diff'
EU_r_diff_dir=ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (5+3)/Europe r_diff'
ALL_r_diff_dir=ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (5+3)/All r_diff'
out_dir=ROOT / 'results'
out_dir.mkdir(parents=True,exist_ok=True)
out_png=ROOT / 'results/Fig. 3.png'
lonlat_list=[(-167.27,-55.0,9.58,74.99),(-11.0,57.0,29.0,75.0)]
base_classes=[1,2,3,4,5,6,7]
label_map={1:'Tropical',2:'Arid',3:'Temperate',4:'Dfa',5:'Dfb',6:'D other',7:'E'}
agg_defs=[('Warm',[1,2,3,4,5]),('Cold',[6,7]),('All',[1,2,3,4,5,6,7])]
series_styles=[('na_diff','NA','#1f77b4','--',2.0),('eu_diff','EU','#ff7f0e','--',2.0),('all_diff','NA & EU','#2ca02c','-',2.8)]
def format_lon(x,pos):
    return '0°' if np.isclose(x,0) else f'{abs(x):g}°{"E" if x>0 else "W"}'
def format_lat(y,pos):
    return '0°' if np.isclose(y,0) else f'{abs(y):g}°{"N" if y>0 else "S"}'
def format_r(value,pos):
    return '0' if abs(value)<1e-10 else f'{value:.3f}'.rstrip('0').rstrip('.')
def read_shape(path):
    gdf=gpd.read_file(path)
    return gdf.set_crs(epsg=4326) if gdf.crs is None else gdf.to_crs(epsg=4326)
def get_recon_da(ds):
    candidates=[v for v in ds.data_vars if 'year' in ds[v].dims]
    preferred=[v for v in candidates if 'recon' in v.lower() or 'agedep' in v.lower()]
    return ds[preferred[0] if preferred else candidates[0]]
def get_model_da(ds):
    candidates=[v for v in ds.data_vars if 'year' in ds[v].dims]
    preferred=[v for v in candidates if v.lower() in ('npp','gpp')]
    return ds[preferred[0] if preferred else candidates[0]]
def prepare_grid(da,bounds):
    lon_min,lon_max,lat_min,lat_max=bounds
    da=da.transpose('year','latitude','longitude')
    years=np.asarray(da.year.values,dtype=float)
    lat=np.round(np.asarray(da.latitude.values,dtype=float),6)
    lon=np.round((np.asarray(da.longitude.values,dtype=float)+180)%360-180,6)
    da=da.assign_coords(year=years.astype(int),latitude=lat,longitude=lon)
    da=da.sortby('year').sortby('latitude').sortby('longitude')
    return da.sel(year=slice(1980,2010),latitude=slice(lat_min,lat_max),longitude=slice(lon_min,lon_max)).reindex(year=np.arange(1980,2011))
def diff_corr_region(bounds):
    with xr.open_dataset(recon_path) as ds_a,xr.open_dataset(model_path) as ds_b:
        da_a=prepare_grid(get_recon_da(ds_a),bounds)
        da_b=prepare_grid(get_model_da(ds_b),bounds)
        da_a,da_b=xr.align(da_a,da_b,join='inner')
        lat=da_a.latitude.values.copy()
        lon=da_a.longitude.values.copy()
        a=np.asarray(da_a.values,dtype=np.float64)
        b=np.asarray(da_b.values,dtype=np.float64)
    ad=np.diff(a,axis=0)
    bd=np.diff(b,axis=0)
    valid=np.isfinite(ad)&np.isfinite(bd)
    n=valid.sum(axis=0)
    x=np.where(valid,ad,0.0)
    y=np.where(valid,bd,0.0)
    xm=np.divide(x.sum(axis=0),n,out=np.zeros(n.shape,dtype=float),where=n>0)
    ym=np.divide(y.sum(axis=0),n,out=np.zeros(n.shape,dtype=float),where=n>0)
    xc=np.where(valid,ad-xm,0.0)
    yc=np.where(valid,bd-ym,0.0)
    numerator=(xc*yc).sum(axis=0)
    denominator=np.sqrt((xc*xc).sum(axis=0)*(yc*yc).sum(axis=0))
    r=np.divide(numerator,denominator,out=np.full(n.shape,np.nan),where=(n>=MIN_N)&(denominator>0))
    return np.clip(r,-1.0,1.0).astype(np.float32),lat,lon
def detect_class_col(gdf):
    numeric_cols=[c for c in gdf.columns if is_numeric_dtype(gdf[c])]
    for col in numeric_cols:
        values=gdf[col].dropna().to_numpy(dtype=float)
        if values.size and np.all(np.isin(values,base_classes)):
            return col
    raise ValueError('No valid climate-class column found')
def zone_box_data(r,lat,lon,shp_path,include_tropical=True):
    class_ids=[1,2,3,4,5] if include_tropical else [2,3,4,5]
    definitions=[(label_map[c],[c]) for c in class_ids]+agg_defs
    names=[name for name,members in definitions]
    lat2d,lon2d=np.meshgrid(lat,lon,indexing='ij')
    valid=np.isfinite(r)
    if not valid.any():
        return names,[np.array([np.nan]) for name in names]
    shp=read_shape(shp_path)
    col=detect_class_col(shp)
    frame=pd.DataFrame({'lon':lon2d[valid],'lat':lat2d[valid],'r':r[valid]})
    points=gpd.GeoDataFrame(frame,geometry=gpd.points_from_xy(frame.lon,frame.lat),crs='EPSG:4326')
    joined=gpd.sjoin(points,shp[[col,'geometry']],how='left',predicate='intersects')
    data=[]
    for name,members in definitions:
        values=joined.loc[joined[col].isin(members),'r'].to_numpy(dtype=float)
        values=values[np.isfinite(values)]
        data.append(values if values.size else np.array([np.nan]))
    return names,data
def combine_box_data(names_na,data_na,names_eu,data_eu):
    order=['Tropical','Arid','Temperate','Dfa','Dfb','Warm','Cold','All']
    na_dict={name:values for name,values in zip(names_na,data_na)}
    eu_dict={name:values for name,values in zip(names_eu,data_eu)}
    combined=[]
    for name in order:
        arrays=[]
        if name in na_dict:
            values=np.asarray(na_dict[name],dtype=float)
            values=values[np.isfinite(values)]
            if values.size:
                arrays.append(values)
        if name in eu_dict:
            values=np.asarray(eu_dict[name],dtype=float)
            values=values[np.isfinite(values)]
            if values.size:
                arrays.append(values)
        combined.append(np.concatenate(arrays) if arrays else np.array([np.nan]))
    return order,combined
def read_curve(directory,tag):
    path=directory/f'{tag}.csv'
    if not path.exists():
        return None
    frame=pd.read_csv(path)
    if not {'year','mean_r'}.issubset(frame.columns):
        return None
    frame=frame[['year','mean_r']].apply(pd.to_numeric,errors='coerce')
    frame=frame[np.isfinite(frame.year)]
    frame['mean_r']=frame['mean_r'].where(np.isfinite(frame['mean_r']))
    frame=frame.groupby('year',as_index=False)['mean_r'].mean().sort_values('year')
    return frame.year.to_numpy(dtype=float),frame.mean_r.to_numpy(dtype=float)
def visible_series(series):
    if series is None:
        return None
    x,y=series
    mask=np.isfinite(x)&(x>=START_YEAR)
    return x[mask],y[mask]
def value_at_year(series,year):
    if series is None:
        return None
    x,y=series
    mask=np.isfinite(x)&np.isfinite(y)&np.isclose(x,float(year))
    if not mask.any():
        return None
    index=np.where(mask)[0][0]
    return int(year),float(y[index])
def set_adaptive_ylim(ax,panel_curves):
    arrays=[]
    for series in panel_curves.values():
        if series is not None:
            values=series[1]
            values=values[np.isfinite(values)]
            if values.size:
                arrays.append(values)
    if not arrays:
        ax.set_ylim(-0.1,0.1)
    else:
        values=np.concatenate(arrays)
        low,high=float(values.min()),float(values.max())
        span=high-low
        if span<=0:
            span=max(abs(high)*0.2,0.05)
        ax.set_ylim(low-span*0.12,high+span*0.30)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(format_r))
def annotate_year(ax,point,text_height=0.92):
    if point is None:
        return
    year,value=point
    ax.axvline(year,color='#2ca02c',linestyle='--',linewidth=1.4,zorder=2)
    ax.scatter([year],[value],s=32,color='#2ca02c',edgecolors='white',linewidths=0.7,zorder=5)
    ax.text(year,text_height,str(year),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=18,color='#2ca02c',zorder=6)
results=[diff_corr_region(bounds) for bounds in lonlat_list]
r_list=[item[0] for item in results]
finite_r=[np.abs(r[np.isfinite(r)]) for r in r_list if np.isfinite(r).any()]
vabs=float(np.percentile(np.concatenate(finite_r),95)) if finite_r else 0.1
vabs=vabs if np.isfinite(vabs) and vabs>0 else 0.1
norm=TwoSlopeNorm(vmin=-vabs,vcenter=0.0,vmax=vabs)
world=read_shape(world_shp)
na_all=read_curve(NA_r_diff_dir,'All')
eu_all=read_curve(EU_r_diff_dir,'All')
all_all=read_curve(ALL_r_diff_dir,'All')
all_panel={'na_diff':visible_series(na_all),'eu_diff':visible_series(eu_all),'all_diff':visible_series(all_all)}
end_years=[]
for series in all_panel.values():
    if series is not None:
        x,y=series
        mask=np.isfinite(x)&np.isfinite(y)
        if mask.any():
            end_years.append(float(x[mask].max()))
end_year=max(end_years) if end_years else START_YEAR+20
names_na,data_na=zone_box_data(*results[0],na_clim_shp,include_tropical=True)
names_eu,data_eu=zone_box_data(*results[1],eu_clim_shp,include_tropical=False)
box_names,box_data=combine_box_data(names_na,data_na,names_eu,data_eu)
finite_box=[d[np.isfinite(d)] for d in box_data if np.isfinite(d).any()]
r_sig=0.36
box_min=float(np.min(np.concatenate(finite_box))) if finite_box else -1.0
box_min=min(box_min,r_sig)
box_min-=max((1.0-box_min)*0.04,0.02)
fig=plt.figure(figsize=(14,12))
gs=gridspec.GridSpec(2,2,figure=fig,left=0.075,right=0.95,top=0.97,bottom=0.075,hspace=0.22,wspace=0.16,width_ratios=[1.15,1.0],height_ratios=[1.05,1.0])
ax_map_na=fig.add_subplot(gs[0,0])
ax_map_eu=fig.add_subplot(gs[0,1])
ax_box=fig.add_subplot(gs[1,0])
ax_line=fig.add_subplot(gs[1,1])
for index,ax in enumerate((ax_map_na,ax_map_eu)):
    r,lat,lon=results[index]
    lon2d,lat2d=np.meshgrid(lon,lat)
    world.plot(ax=ax,color='lightgrey',edgecolor='grey',linewidth=0.4,zorder=0)
    im=ax.pcolormesh(lon2d,lat2d,r,cmap='BrBG',norm=norm,shading='auto',zorder=1)
    marked=np.isfinite(r)&(np.abs(r)>=r_sig)
    if marked.any():
        ax.scatter(lon2d[marked],lat2d[marked],s=0.08,color='black',linewidths=0,zorder=2)
    lon_min,lon_max,lat_min,lat_max=lonlat_list[index]
    ax.set(xlim=(lon_min,lon_max),ylim=(lat_min,lat_max),xlabel='',ylabel='')
    ax.xaxis.set_major_locator(MultipleLocator(20 if index==0 else 10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.tick_params(axis='both',labelsize=18)
    ax.set_aspect('auto')
    if index==1:
        ax.yaxis.tick_right()
        ax.tick_params(axis='y',right=True,left=False)
    median=float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan
    if index==0:
        ax.text(0.03,0.03,rf'Median $\mathit{{r}}$ = {median:.2f}',transform=ax.transAxes,ha='left',va='bottom',fontsize=19)
    else:
        ax.text(0.10,0.97,rf'Median $\mathit{{r}}$ = {median:.2f}',transform=ax.transAxes,ha='left',va='top',fontsize=19)
cax=ax_map_na.inset_axes([0.02,0.25,0.35,0.04])
cbar=fig.colorbar(im,cax=cax,orientation='horizontal',extend='both')
cbar.ax.tick_params(labelsize=18)
rng=np.random.default_rng(42)
bp=ax_box.boxplot(box_data,tick_labels=box_names,showfliers=False)
for line in bp['medians']:
    line.set_color('#4b5f83')
for position,values in enumerate(box_data,start=1):
    valid=values[np.isfinite(values)]
    if valid.size:
        jitter=rng.uniform(-0.175,0.175,valid.size)
        ax_box.scatter(position+jitter,valid,s=1,color='#1f77b4',alpha=0.1)
        median=float(np.median(valid))
        ax_box.text(position,median,f'{median:.2f}',ha='center',va='center',fontsize=17,color='#d62728')
ax_box.set_ylim(box_min,1.1)
ax_box.yaxis.set_major_locator(MultipleLocator(0.2))
ax_box.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax_box.tick_params(axis='x',labelrotation=45,labelsize=17)
ax_box.tick_params(axis='y',labelsize=18)
ax_box.axhline(r_sig,color='black',linestyle='--',linewidth=1.2)
ax_box.text(4.0,r_sig-0.08,r'$\mathit{r}$ = 0.36, $\mathit{p}$ < 0.05',ha='center',va='top',fontsize=18,color='black')
ax_box.set_ylabel(r'$\mathit{r}$',fontsize=22)
for key,label,color,linestyle,linewidth in series_styles:
    series=all_panel[key]
    if series is not None and np.isfinite(series[1]).any():
        ax_line.plot(series[0],series[1],label=label,color=color,linestyle=linestyle,linewidth=linewidth)
ax_line.set_xlim(START_YEAR,end_year)
set_adaptive_ylim(ax_line,all_panel)
ymax=ax_line.get_ylim()[1]
ymax=max(0.3,np.ceil(ymax*10)/10)
ax_line.set_ylim(0.15,ymax)
ax_line.yaxis.set_major_locator(MultipleLocator(0.1))
ax_line.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax_line.xaxis.set_major_locator(MultipleLocator(20))
ax_line.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax_line.set_xlabel('Year',fontsize=22)
ax_line.set_ylabel(r'$\mathit{r}_{\mathrm{moving}}$',fontsize=22)
ax_line.tick_params(axis='both',labelsize=18)
for year in MARK_YEARS:
    point=value_at_year(all_panel['all_diff'],year)
    annotate_year(ax_line,point,text_height=0.92)
handles,labels=ax_line.get_legend_handles_labels()
if handles:
    ax_line.legend(handles,labels,fontsize=18,frameon=True,fancybox=False,edgecolor='black',loc='lower left',borderpad=0.3,handlelength=1.7,handletextpad=0.5)
for label,ax in zip('abcd',[ax_map_na,ax_map_eu,ax_box,ax_line]):
    ax.text(0.02,0.98,label,transform=ax.transAxes,fontsize=26,fontweight='bold',ha='left',va='top')
fig.savefig(out_png,bbox_inches='tight',pad_inches=0.12,dpi=DPI,facecolor='white')
plt.close(fig)
print('✅ Saved figure:', out_png)