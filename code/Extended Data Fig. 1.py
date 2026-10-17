
import warnings,xarray as xr,numpy as np,geopandas as gpd,matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from pathlib import Path
warnings.filterwarnings('ignore',message='Mean of empty slice')
plt.rcParams['font.family']='Times New Roman';plt.rcParams['mathtext.fontset']='custom';plt.rcParams['mathtext.rm']='Times New Roman';plt.rcParams['mathtext.it']='Times New Roman:italic';plt.rcParams['font.size']=24;plt.rcParams['axes.unicode_minus']=False
ROOT=Path('D:/')
npp_paths=[ROOT/'data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc',ROOT/'data/NPP and GPP/S2_npp9-8/CLASSIC_S2_npp.nc',ROOT/'data/NPP and GPP/S2_npp9-8/iMAPLE_S2_npp.nc']
npptree_path=ROOT/'data/NPP and GPP/S2_npptree9-8/ISBA-CTRIP_S2_nppTree.nc'
eu_shp_path=ROOT/'data/Map/Europe 7-class.shp'
na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp'
out_png=ROOT/'results/Extended Data Fig. 1.png'
models=[('ModNegExp (5+3)',ROOT/'data/Processed data/TRW/ModNegExp (5+3).nc','ModNegExp'),('SFRCS (5+3)',ROOT/'data/Processed data/TRW/SFRCS (5+3).nc','SFRCS'),('Spline (5+3)',ROOT/'data/Processed data/TRW/Spline (5+3).nc','Spline'),('AgeDepSpline (4+3)',ROOT/'data/Processed data/TRW/AgeDepSpline (4+3).nc','AgeDepSpline'),('AgeDepSpline (3+3)',ROOT/'data/Processed data/TRW/AgeDepSpline (3+3).nc','AgeDepSpline'),('AgeDepSpline (5+3) + nppTree',ROOT/'data/Processed data/TRW/AgeDepSpline (5+3).nc','AgeDepSpline')]
plot_years=np.arange(1900,2026)
def temporal_std(da):
    count=da.count('year');mean=da.mean('year',skipna=True)
    return np.sqrt(((da-mean)**2).sum('year',skipna=True)/count.where(count>0))
def find_data_var(ds,preferred):
    for name in preferred:
        if name in ds.data_vars:return name
    lower_map={name.lower():name for name in ds.data_vars}
    for name in preferred:
        if name.lower() in lower_map:return lower_map[name.lower()]
    candidates=[name for name in ds.data_vars if 'year' in name.lower() or any(d.lower()=='year' for d in ds[name].dims)]
    if candidates:return candidates[0]
    candidates=list(ds.data_vars)
    if candidates:return candidates[0]
    raise ValueError('No valid data variable found')
def normalize_year(da,ds):
    time_dims=[d for d in da.dims if d not in ('latitude','longitude')]
    if len(time_dims)!=1:raise ValueError(f'Unexpected dimensions: {da.dims}')
    time_dim=time_dims[0]
    if 'year' in ds.variables:years=np.asarray(ds['year'].values)
    elif time_dim in ds.coords:years=np.asarray(ds[time_dim].values)
    else:raise ValueError('No year coordinate found')
    years=np.asarray(years,dtype=float).astype(int)
    if time_dim=='year':da=da.assign_coords(year=years)
    else:da=da.assign_coords({time_dim:years}).rename({time_dim:'year'})
    if float(np.nanmin(da.longitude.values))>=0:da=da.assign_coords(longitude=(da.longitude+180)%360-180)
    return da.sortby('year').sortby('latitude').sortby('longitude')
def safe_quantiles(reg):
    values=np.asarray(reg.transpose('year','points').values,dtype=np.float64)
    result=np.full((3,values.shape[0]),np.nan,dtype=np.float64)
    for i in np.flatnonzero(np.any(np.isfinite(values),axis=1)):
        row=values[i];row=row[np.isfinite(row)]
        result[:,i]=np.percentile(row,[50,25,75])
    return result
eu_shp=gpd.read_file(eu_shp_path);eu_shp=eu_shp.set_crs('EPSG:4326') if eu_shp.crs is None else eu_shp.to_crs('EPSG:4326')
na_shp=gpd.read_file(na_shp_path);na_shp=na_shp.set_crs('EPSG:4326') if na_shp.crs is None else na_shp.to_crs('EPSG:4326')
region_cache={}
def region_indices(da):
    key=(da.latitude.values.tobytes(),da.longitude.values.tobytes())
    if key not in region_cache:
        stacked=da.isel(year=0).stack(points=('latitude','longitude'))
        gdf=gpd.GeoDataFrame({'point_index':np.arange(stacked.sizes['points'])},geometry=gpd.points_from_xy(stacked.longitude.values,stacked.latitude.values),crs='EPSG:4326')
        eu_join=gpd.sjoin(gdf,eu_shp,how='inner',predicate='within')
        na_join=gpd.sjoin(gdf,na_shp,how='inner',predicate='within')
        region_cache[key]=(np.unique(eu_join.point_index.values),np.unique(na_join.point_index.values))
    return region_cache[key]
def extract_series(ds_trw,trw_var,npp_file,npp_preferred):
    with xr.open_dataset(npp_file) as ds_npp:
        trw=normalize_year(ds_trw[find_data_var(ds_trw,[trw_var])],ds_trw)
        npp=normalize_year(ds_npp[find_data_var(ds_npp,npp_preferred)],ds_npp)
        years=np.intersect1d(npp.year.values,trw.year.values)
        years=years[(years>=1900)&(years<=2025)]
        if years.size<2:raise ValueError(f'No sufficient overlapping years: {npp_file.name}')
        npp=npp.sel(year=years);trw=trw.sel(year=years)
        same_grid=np.array_equal(npp.latitude.values,trw.latitude.values) and np.array_equal(npp.longitude.values,trw.longitude.values)
        if not same_grid:npp=npp.interp(latitude=trw.latitude,longitude=trw.longitude)
        valid=np.isfinite(trw)&np.isfinite(npp)&(npp!=0)
        ratio=xr.where(valid,trw/npp,np.nan)
        mean_ratio=ratio.mean('year',skipna=True);std_ratio=temporal_std(ratio)
        ratio_z=xr.where(np.isfinite(std_ratio)&(std_ratio>0),(ratio-mean_ratio)/std_ratio,np.nan)
        eu_ids,na_ids=region_indices(ratio_z)
        if len(eu_ids)==0 or len(na_ids)==0:raise ValueError(f'No regional grid cells: {npp_file.name}')
        stacked=ratio_z.stack(points=('latitude','longitude'))
        eu=safe_quantiles(stacked.isel(points=eu_ids))
        na=safe_quantiles(stacked.isel(points=na_ids))
        return years,eu,na
def combine_model_series(results,region):
    arrays=[]
    for years,eu,na in results:
        data=eu if region=='EU' else na
        aligned=np.full((3,plot_years.size),np.nan)
        aligned[:,years-plot_years[0]]=data
        arrays.append(aligned)
    return np.nanmean(np.stack(arrays),axis=0)
fig,axes=plt.subplots(3,2,figsize=(20,18),dpi=600,sharex=True,sharey=True)
axes=axes.ravel();letters=['a','b','c','d','e','f']
for index,(ax,config,letter) in enumerate(zip(axes,models,letters)):
    title,trw_path,trw_var=config
    input_files=npp_paths if index<5 else [npptree_path]
    results=[]
    with xr.open_dataset(trw_path) as ds_trw:
        for path in input_files:
            if not path.exists() and (path.parent/(' '+path.name)).exists():path=path.parent/(' '+path.name)
            results.append(extract_series(ds_trw,trw_var,path,['npp'] if index<5 else ['nppTree','npptree','npp_tree','npp']))
    eu=combine_model_series(results,'EU');na=combine_model_series(results,'NA')
    valid=np.isfinite(eu[0])&np.isfinite(na[0])
    r,p=pearsonr(eu[0,valid],na[0,valid]) if np.count_nonzero(valid)>=3 else (np.nan,np.nan)
    l1,=ax.plot(plot_years,eu[0],linewidth=2.2,label='EU')
    ax.fill_between(plot_years,eu[1],eu[2],color=l1.get_color(),alpha=0.25)
    l2,=ax.plot(plot_years,na[0],linewidth=2.2,label='NA')
    ax.fill_between(plot_years,na[1],na[2],color=l2.get_color(),alpha=0.25)
    ax.axvline(x=1945,color='black',linestyle='--',linewidth=1.8,zorder=5)
    ax.text(1945,0.03,'1945',color='black',fontsize=22,ha='center',va='bottom',transform=ax.get_xaxis_transform(),zorder=6)
    ax.set_xlim(1900,2025);ax.tick_params(labelsize=20)
    ax.xaxis.set_major_locator(MultipleLocator(20));ax.yaxis.set_major_locator(MultipleLocator(1));ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.text(0.02,0.98,letter,transform=ax.transAxes,ha='left',va='top',fontweight='bold',fontsize=30)
    ax.text(0.5,0.98,title,transform=ax.transAxes,ha='center',va='top',fontsize=22)
    if np.isfinite(p):
        p_text=r'$\mathit{p}$ < 0.001' if p<0.001 else rf'$\mathit{{p}}$ = {p:.3f}'
        stat_text=rf'$\mathit{{r}}$ = {r:.2f}, '+p_text
    else:stat_text=r'$\mathit{r}$ = NA, $\mathit{p}$ = NA'
    ax.text(0.97,0.08,stat_text,transform=ax.transAxes,ha='right',va='bottom',fontsize=19)
    if index%2==0:ax.set_ylabel('Standardized TRW/NPP',fontsize=23)
    if index>=4:ax.set_xlabel('Year',fontsize=23)
axes[0].legend(frameon=True,fancybox=False,edgecolor='black',fontsize=18,loc='upper right')
fig.tight_layout(h_pad=0.8,w_pad=0.8)
out_png.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out_png,bbox_inches='tight',pad_inches=0.12,facecolor='white',dpi=600)
plt.close(fig)
print('✅ Saved figure:', out_png)
