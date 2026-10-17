
import xarray as xr,numpy as np,geopandas as gpd,matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from pathlib import Path
plt.rcParams['font.family']='Times New Roman';plt.rcParams['mathtext.fontset']='custom';plt.rcParams['mathtext.rm']='Times New Roman';plt.rcParams['mathtext.it']='Times New Roman:italic';plt.rcParams['font.size']=24;plt.rcParams['axes.unicode_minus']=False
ROOT=Path('D:/')
npp_paths=[ROOT/'data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc',ROOT/'data/NPP and GPP/S2_npp9-8/ CLASSIC_S2_npp.nc',ROOT/'data/NPP and GPP/S2_npp9-8/iMAPLE_S2_npp.nc']
npp_names=['ISBA-CTRIP','CLASSIC','iMAPLE']
eu_shp_path=ROOT/'data/Map/Europe 7-class.shp';na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp';out_png=ROOT/'results/Supplementary Fig. 3.png'
models=[('ModNegExp',ROOT/'data/Processed data/Detrending5/ModNegExp (5+3).nc'),('AgeDepSpline',ROOT/'data/Processed data/Detrending5/AgeDepSpline (5+3).nc'),('SFRCS',ROOT/'data/Processed data/Detrending5/SFRCS (5+3).nc'),('Spline',ROOT/'data/Processed data/Detrending5/Spline (5+3).nc')]
PLOT_YEARS=np.arange(1900,1996)
def temporal_std(da):
    count=da.count('year');mean=da.mean('year',skipna=True)
    return np.sqrt(((da-mean)**2).sum('year',skipna=True)/count.where(count>0))
def prepare_data(ds,var):
    da=ds[var];time_dim=[d for d in da.dims if d not in ('latitude','longitude')][0]
    if time_dim=='year':da=da.assign_coords(year=ds['year'].values)
    else:da=da.assign_coords({time_dim:ds['year'].values}).rename({time_dim:'year'})
    if float(np.nanmin(da.longitude.values))>=0:da=da.assign_coords(longitude=(da.longitude+180)%360-180)
    return da.sortby('year').sortby('latitude').sortby('longitude')
eu_shp=gpd.read_file(eu_shp_path);eu_shp=eu_shp.set_crs('EPSG:4326') if eu_shp.crs is None else eu_shp.to_crs('EPSG:4326')
na_shp=gpd.read_file(na_shp_path);na_shp=na_shp.set_crs('EPSG:4326') if na_shp.crs is None else na_shp.to_crs('EPSG:4326')
region_cache={}
def region_indices(da):
    key=(tuple(da.latitude.values),tuple(da.longitude.values))
    if key not in region_cache:
        stacked=da.isel(year=0).stack(points=('latitude','longitude'))
        points=gpd.GeoDataFrame({'index':np.arange(stacked.sizes['points'])},geometry=gpd.points_from_xy(stacked.longitude.values,stacked.latitude.values),crs='EPSG:4326')
        eu_ids=np.unique(gpd.sjoin(points,eu_shp,how='inner',predicate='within')['index'].values)
        na_ids=np.unique(gpd.sjoin(points,na_shp,how='inner',predicate='within')['index'].values)
        if not len(eu_ids) or not len(na_ids):raise ValueError('No region grid cells found')
        region_cache[key]=(eu_ids,na_ids)
    return region_cache[key]
def extract_series(model,trw_path,npp_path):
    if not npp_path.exists() and npp_path.name.startswith(' '):npp_path=npp_path.with_name(npp_path.name.lstrip())
    with xr.open_dataset(trw_path) as ds_trw,xr.open_dataset(npp_path) as ds_npp:
        trw=prepare_data(ds_trw,model);npp=prepare_data(ds_npp,'npp')
        years=np.intersect1d(trw.year.values,npp.year.values)
        years=years[(years>=1900)&(years<=1995)]
        if years.size<3:raise ValueError(f'Insufficient common years: {model}, {npp_path.name}')
        trw=trw.sel(year=years);npp=npp.sel(year=years)
        ratio=trw/npp;ratio=ratio.where(np.isfinite(ratio))
        mean=ratio.mean('year',skipna=True);std=temporal_std(ratio)
        ratio_z=((ratio-mean)/std).where(std>0)
        eu_ids,na_ids=region_indices(ratio_z)
        stacked=ratio_z.stack(points=('latitude','longitude'))
        def series(ids):
            q=stacked.isel(points=ids).quantile([0.25,0.5,0.75],dim='points',skipna=True)
            return np.stack([q.sel(quantile=0.5).values,q.sel(quantile=0.25).values,q.sel(quantile=0.75).values])
        return years,series(eu_ids),series(na_ids)
def combine_series(series):
    arr=np.stack(series)
    counts=np.isfinite(arr).sum(axis=0)
    return np.divide(np.nansum(arr,axis=0),counts,out=np.full(arr.shape[1:],np.nan),where=counts>0)
def correlation(eu,na):
    mask=np.isfinite(eu)&np.isfinite(na)
    if mask.sum()<3 or np.std(eu[mask])==0 or np.std(na[mask])==0:return np.nan,np.nan
    return pearsonr(eu[mask],na[mask])
fig,axes=plt.subplots(2,2,figsize=(18,11),dpi=300,sharex=True,sharey=True)
axes=axes.ravel();letters=['a','b','c','d']
for ax,(model,path),letter in zip(axes,models,letters):
    eu_models=[];na_models=[]
    for npp_name,npp_path in zip(npp_names,npp_paths):
        years,eu,na=extract_series(model,path,npp_path)
        aligned_eu=np.full((3,len(PLOT_YEARS)),np.nan);aligned_na=np.full((3,len(PLOT_YEARS)),np.nan)
        aligned_eu[:,years-PLOT_YEARS[0]]=eu;aligned_na[:,years-PLOT_YEARS[0]]=na
        eu_models.append(aligned_eu);na_models.append(aligned_na)
        r_model,p_model=correlation(eu[0],na[0])
        print(f'{model} | {npp_name}: r={r_model:.3f}, p={p_model:.4g}')
    eu=combine_series(eu_models);na=combine_series(na_models)
    r,p=correlation(eu[0],na[0])
    line_eu,=ax.plot(PLOT_YEARS,eu[0],linewidth=2.2,label='EU')
    ax.fill_between(PLOT_YEARS,eu[1],eu[2],color=line_eu.get_color(),alpha=0.25)
    line_na,=ax.plot(PLOT_YEARS,na[0],linewidth=2.2,label='NA')
    ax.fill_between(PLOT_YEARS,na[1],na[2],color=line_na.get_color(),alpha=0.25)
    ax.axvline(x=1945,color='black',linestyle='--',linewidth=1.8,zorder=5)
    ax.text(1945,0.03,'1945',color='black',fontsize=24,ha='center',va='bottom',transform=ax.get_xaxis_transform(),zorder=6)
    ax.set_xlim(1900,1995);ax.tick_params(labelsize=22)
    ax.xaxis.set_major_locator(MultipleLocator(20));ax.yaxis.set_major_locator(MultipleLocator(1));ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.text(0.02,0.98,letter,transform=ax.transAxes,ha='left',va='top',fontweight='bold',fontsize=30)
    ax.text(0.5,0.98,model,transform=ax.transAxes,ha='center',va='top',fontsize=24)
    if np.isfinite(p):p_text=r'$\mathit{p}$ < 0.001' if p<0.001 else rf'$\mathit{{p}}$ = {p:.3f}'
    else:p_text=r'$\mathit{p}$ = NA'
    r_text=rf'$\mathit{{r}}$ = {r:.2f}' if np.isfinite(r) else r'$\mathit{r}$ = NA'
    ax.text(0.97,0.08,r_text+', '+p_text,transform=ax.transAxes,ha='right',va='bottom',fontsize=21)
    print(f'{model} | Combined: r={r:.3f}, p={p:.4g}')
axes[0].legend(frameon=True,fancybox=False,edgecolor='black',fontsize=18,loc='upper right')
axes[2].set_xlabel('Year',fontsize=24);axes[3].set_xlabel('Year',fontsize=24)
fig.text(0.015,0.5,'Standardized TRW/NPP',rotation=90,ha='center',va='center',fontsize=24)
fig.tight_layout(rect=[0.035,0.02,1,1]);out_png.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out_png,bbox_inches='tight',pad_inches=0.12,facecolor='white',dpi=600)
plt.close(fig)
print('✅ Saved figure:',out_png)
