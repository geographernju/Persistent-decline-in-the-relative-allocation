import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from matplotlib.ticker import MultipleLocator,FuncFormatter,FixedLocator
from pathlib import Path
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['mathtext.rm']='Times New Roman'
plt.rcParams['mathtext.it']='Times New Roman:italic'
plt.rcParams['figure.dpi']=600
plt.rcParams['axes.labelsize']=26
plt.rcParams['xtick.labelsize']=22
plt.rcParams['ytick.labelsize']=22
ROOT=Path('D:/')
nc1=ROOT/'data/Processed data/Sliding difference/Sliding difference1-1900-center.nc'
eu_shp_path=ROOT/'data/Map/Europe 7-class.shp'
na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp'
r_diff_dir=ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (5+3)/All r_diff'
out_png=ROOT / 'results/Extended Data Fig. 7.png'
PLOT_Y0,PLOT_Y1=1911,2014
YEAR_SHIFT=0
TIME_DIM='window_center_year'
PLOT_YEARS=np.arange(PLOT_Y0-YEAR_SHIFT,PLOT_Y1-YEAR_SHIFT+1)
X_TICKS=[1920,1940,1960,1980,2000,2020]
VARIABLES=['delta_SPEI12_68','delta_scpdsi_68','delta_tmp_68']
COLORS={'delta_SPEI12_68':'#8c5e37','delta_scpdsi_68':'#d9a97c','delta_tmp_68':'#2b6f4e'}
r_diff_label=r'$\mathit{r}_{\mathrm{moving}}$ (NA & EU)'
ZONES=['Arid','Temperate','Dfa','Dfb']
def get_zone_col(gdf):
    for name in ['class7','CLASS7','zone','ZONE','Koppen','koppen','CLIMATE','climate','Class','class']:
        if name in gdf.columns:return name
    return [c for c in gdf.columns if c!='geometry'][0]
def build_zone_masks(ds):
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
    zone_values=np.array(joined.set_index('idx')['zone'].reindex(idx).astype(str).values,dtype=object)
    zone_values=np.char.strip(zone_values.astype(str))
    aliases={'Arid':{'Arid','B'},'Temperate':{'Temperate','C'},'Dfa':{'Dfa'},'Dfb':{'Dfb'}}
    masks={}
    for zone_name,names in aliases.items():
        mask=np.array([z in names for z in zone_values]).reshape(lat.size,lon.size)
        masks[zone_name]=xr.DataArray(mask,coords={'latitude':lat,'longitude':lon},dims=('latitude','longitude'))
    return masks
def display_name(name):
    name=name.removeprefix('delta_').removesuffix('_68')
    return f"{re.sub('(?i)scpdsi','scPDSI',name)} (summer)"
def da_to_series(da):
    other_dims=[dim for dim in da.dims if dim!=TIME_DIM]
    if other_dims:da=da.mean(other_dims,skipna=True)
    series=da.to_pandas()
    series.index=series.index+YEAR_SHIFT
    return series.loc[(series.index>=PLOT_Y0)&(series.index<=PLOT_Y1)]
def read_r_diff_series(zone):
    file_map={'Arid':'B','Temperate':'C','Dfa':'Dfa','Dfb':'Dfb'}
    if zone not in file_map:raise ValueError(f'Unknown zone: {zone}')
    path=r_diff_dir/f"{file_map[zone]}.csv"
    if not path.exists():raise FileNotFoundError(path)
    frame=pd.read_csv(path)
    if not {'year','mean_r'}.issubset(frame.columns):raise ValueError(f'Required columns are missing: {path}')
    frame=frame[['year','mean_r']].apply(pd.to_numeric,errors='coerce')
    frame=frame[np.isfinite(frame['year'])&np.isfinite(frame['mean_r'])]
    frame=frame.groupby('year',as_index=False)['mean_r'].mean().sort_values('year')
    frame=frame[(frame['year']>=PLOT_Y0)&(frame['year']<=PLOT_Y1)]
    return pd.Series(frame['mean_r'].to_numpy(dtype=float),index=frame['year'].to_numpy(dtype=int))
def draw_zone_panel(ds,zone_mask,zone_name,reference,ax,letter,ylabel=None):
    for name in VARIABLES:
        series=da_to_series(ds[name].where(zone_mask))
        ax.plot(series.index,series.values,lw=1.8,color=COLORS[name],label=display_name(name))
    ax.plot(reference.index,reference.values,ls='-.',lw=2.2,color='k',label=r_diff_label)
    ax.set_xlim(PLOT_Y0,PLOT_Y1)
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.1f}'))
    ax.set_xlabel('Year',fontsize=26)
    if ylabel is not None:ax.set_ylabel(ylabel,fontsize=26)
    ax.text(0.02,0.98,letter,transform=ax.transAxes,ha='left',va='top',fontsize=30,fontweight='bold')
    ax.text(0.98,0.98,zone_name,transform=ax.transAxes,ha='right',va='top',fontsize=24,fontweight='bold')
    ax.legend(loc='upper left',bbox_to_anchor=(0.0,0.88),frameon=True,fontsize=17)
    ax.margins(y=0.12)
source=xr.open_dataset(nc1)
ds=source.reindex({TIME_DIM:PLOT_YEARS})
zone_masks=build_zone_masks(ds)
references={zone:read_r_diff_series(zone) for zone in ZONES}
fig,axes=plt.subplots(2,2,figsize=(18,13),dpi=600,sharex=True,sharey=False)
axes=axes.ravel()
letters=['a','b','c','d']
for ax,zone_name,letter in zip(axes,ZONES,letters):
    ylabel=r'$\Delta r = r_{\mathrm{moving}} - r_{\mathrm{moving}\mid\mathrm{climate}}$' if letter in ['a','c'] else None
    draw_zone_panel(ds,zone_masks[zone_name],zone_name,references[zone_name],ax,letter,ylabel)
ymin_ab=min(axes[0].get_ylim()[0],axes[1].get_ylim()[0])
ymax_ab=max(axes[0].get_ylim()[1],axes[1].get_ylim()[1])
axes[0].set_ylim(ymin_ab,ymax_ab)
axes[1].set_ylim(ymin_ab,ymax_ab)
ymin_cd=min(axes[2].get_ylim()[0],axes[3].get_ylim()[0])
ymax_cd=max(axes[2].get_ylim()[1],axes[3].get_ylim()[1])
axes[2].set_ylim(ymin_cd,ymax_cd)
axes[3].set_ylim(ymin_cd,ymax_cd)
for ax in [axes[0],axes[1]]:
    ax.set_xlabel('')
    ax.tick_params(axis='x',which='major',labelbottom=False,bottom=True)
axes[1].set_ylabel('')
axes[3].set_ylabel('')
axes[1].tick_params(axis='y',which='both',labelleft=False,left=True)
axes[3].tick_params(axis='y',which='both',labelleft=False,left=True)
fig.tight_layout()
out_png.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(out_png,bbox_inches='tight',pad_inches=0.02,facecolor='white',dpi=600)
plt.close()
source.close()
print('✅ Saved figure:', out_png)