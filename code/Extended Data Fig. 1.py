import xarray as xr,numpy as np,geopandas as gpd,matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from pathlib import Path
plt.rcParams['font.family']='Times New Roman';plt.rcParams['mathtext.fontset']='custom';plt.rcParams['mathtext.rm']='Times New Roman';plt.rcParams['mathtext.it']='Times New Roman:italic';plt.rcParams['font.size']=24;plt.rcParams['axes.unicode_minus']=False
ROOT=Path('D:/')
npp_path=ROOT/'data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc'
npptree_path=ROOT/'data/NPP and GPP/S2_npptree9-8/ISBA-CTRIP_S2_nppTree.nc'
eu_shp_path=ROOT/'data/Map/Europe 7-class.shp'
na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp'
out_png=ROOT/'results/Extended Data Fig. 1.png'
models=[('ModNegExp',ROOT/'data/Processed data/TRW/ModNegExp (5+3).nc','ModNegExp',npp_path,['npp']),('SFRCS',ROOT/'data/Processed data/TRW/SFRCS (5+3).nc','SFRCS',npp_path,['npp']),('Spline',ROOT/'data/Processed data/TRW/Spline (5+3).nc','Spline',npp_path,['npp']),('AgeDepSpline (4+3)',ROOT/'data/Processed data/TRW/AgeDepSpline (4+3).nc','AgeDepSpline',npp_path,['npp']),('AgeDepSpline (3+3)',ROOT/'data/Processed data/TRW/AgeDepSpline (3+3).nc','AgeDepSpline',npp_path,['npp']),('AgeDepSpline (5+3) + nppTree',ROOT/'data/Processed data/TRW/AgeDepSpline (5+3).nc','AgeDepSpline',npptree_path,['nppTree','npptree','npp_tree','npp'])]
def temporal_std(da):
    count=da.count('year')
    mean=da.mean('year',skipna=True)
    variance=((da-mean)**2).sum('year',skipna=True)/count.where(count>0)
    return np.sqrt(variance)
def find_data_var(ds,preferred):
    for name in preferred:
        if name in ds.data_vars:return name
    lower_map={name.lower():name for name in ds.data_vars}
    for name in preferred:
        if name.lower() in lower_map:return lower_map[name.lower()]
    candidates=[name for name in ds.data_vars if any(d.lower()=='year' for d in ds[name].dims) or 'year' in ds[name].dims]
    if candidates:return candidates[0]
    candidates=list(ds.data_vars)
    if candidates:return candidates[0]
    raise ValueError('No valid data variable found')
def normalize_year(da,ds):
    time_dims=[d for d in da.dims if d not in ('latitude','longitude')]
    if len(time_dims)!=1:raise ValueError(f'Unexpected dimensions: {da.dims}')
    time_dim=time_dims[0]
    if 'year' in ds.coords:
        years=np.asarray(ds['year'].values)
    elif 'year' in ds.variables:
        years=np.asarray(ds['year'].values)
    elif time_dim in ds.coords:
        years=np.asarray(ds[time_dim].values)
    else:
        raise ValueError('No year coordinate found')
    years=np.asarray(years,dtype=float).astype(int)
    if time_dim=='year':return da.assign_coords(year=years)
    return da.assign_coords({time_dim:years}).rename({time_dim:'year'})
def safe_quantiles(reg):
    values=np.asarray(reg.transpose('year','points').values,dtype=np.float64)
    n=values.shape[0]
    q25=np.full(n,np.nan,dtype=np.float64)
    q50=np.full(n,np.nan,dtype=np.float64)
    q75=np.full(n,np.nan,dtype=np.float64)
    valid_rows=np.any(np.isfinite(values),axis=1)
    for i in np.flatnonzero(valid_rows):
        row=values[i]
        row=row[np.isfinite(row)]
        q25[i],q50[i],q75[i]=np.percentile(row,[25,50,75])
    return q50,q25,q75
eu_shp=gpd.read_file(eu_shp_path)
eu_shp=eu_shp.set_crs('EPSG:4326') if eu_shp.crs is None else eu_shp.to_crs('EPSG:4326')
na_shp=gpd.read_file(na_shp_path)
na_shp=na_shp.set_crs('EPSG:4326') if na_shp.crs is None else na_shp.to_crs('EPSG:4326')
def extract_series(title,trw_path,trw_var,npp_file,npp_preferred):
    ds_npp=xr.open_dataset(npp_file)
    ds_trw=xr.open_dataset(trw_path)
    try:
        npp_var=find_data_var(ds_npp,npp_preferred)
        if trw_var in ds_trw.data_vars:
            trw_name=trw_var
        else:
            trw_name=find_data_var(ds_trw,[trw_var])
        npp=normalize_year(ds_npp[npp_var],ds_npp)
        trw=normalize_year(ds_trw[trw_name],ds_trw)
        if float(np.nanmin(npp['longitude'].values))>=0:npp=npp.assign_coords(longitude=(npp['longitude']+180)%360-180).sortby('longitude')
        if float(np.nanmin(trw['longitude'].values))>=0:trw=trw.assign_coords(longitude=(trw['longitude']+180)%360-180).sortby('longitude')
        npp=npp.sortby('year').sortby('latitude').sortby('longitude')
        trw=trw.sortby('year').sortby('latitude').sortby('longitude')
        years=np.intersect1d(npp['year'].values,trw['year'].values)
        years=years[(years>=1900)&(years<=2025)]
        if years.size<2:raise ValueError(f'No sufficient common years for {title}')
        npp=npp.sel(year=years)
        trw=trw.sel(year=years)
        same_grid=npp.sizes.get('latitude')==trw.sizes.get('latitude') and npp.sizes.get('longitude')==trw.sizes.get('longitude')
        if same_grid:same_grid=np.allclose(npp['latitude'],trw['latitude']) and np.allclose(npp['longitude'],trw['longitude'])
        if not same_grid:npp=npp.interp(latitude=trw['latitude'],longitude=trw['longitude'])
        valid=np.isfinite(trw)&np.isfinite(npp)&(npp!=0)
        ratio=xr.where(valid,trw/npp,np.nan)
        mu_ratio=ratio.mean('year',skipna=True)
        std_ratio=temporal_std(ratio)
        ratio_z=xr.where(np.isfinite(std_ratio)&(std_ratio>0),(ratio-mu_ratio)/std_ratio,np.nan)
        stacked=ratio_z.stack(points=('latitude','longitude'))
        lon=stacked['longitude'].values
        lat=stacked['latitude'].values
        pts=stacked['points'].values
        gdf=gpd.GeoDataFrame({'points':pts},geometry=gpd.points_from_xy(lon,lat),crs='EPSG:4326')
        def region(shp):
            join=gpd.sjoin(gdf,shp,how='inner',predicate='within')
            if join.empty:return None
            ids=np.unique(join['points'].values)
            reg=stacked.sel(points=ids)
            return safe_quantiles(reg)
        eu=region(eu_shp)
        na=region(na_shp)
        if eu is None or na is None:raise ValueError(f'No valid regional TRW/NPP data for {title}')
        mask=np.isfinite(eu[0])&np.isfinite(na[0])
        if mask.sum()<2:
            r,p=np.nan,np.nan
        else:
            r,p=pearsonr(eu[0][mask],na[0][mask])
        return years,eu,na,r,p
    finally:
        ds_npp.close()
        ds_trw.close()
fig,axes=plt.subplots(3,2,figsize=(20,18),dpi=600,sharex=True,sharey=True)
axes=axes.ravel()
letters=['a','b','c','d','e','f']
for index,(ax,config,letter) in enumerate(zip(axes,models,letters)):
    title,trw_path,trw_var,npp_file,npp_preferred=config
    years,eu,na,r,p=extract_series(title,trw_path,trw_var,npp_file,npp_preferred)
    l1,=ax.plot(years,eu[0],linewidth=2.2,label='EU')
    ax.fill_between(years,eu[1],eu[2],color=l1.get_color(),alpha=0.25)
    l2,=ax.plot(years,na[0],linewidth=2.2,label='NA')
    ax.fill_between(years,na[1],na[2],color=l2.get_color(),alpha=0.25)
    ax.axvline(x=1945,color='black',linestyle='--',linewidth=1.8,zorder=5)
    ax.text(1945,0.03,'1945',color='black',fontsize=22,fontweight='normal',ha='center',va='bottom',transform=ax.get_xaxis_transform(),zorder=6)
    ax.set_xlim(1900,2025)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.text(0.02,0.98,letter,transform=ax.transAxes,ha='left',va='top',fontweight='bold',fontsize=30)
    ax.text(0.5,0.98,title,transform=ax.transAxes,ha='center',va='top',fontsize=22)
    if np.isfinite(p):
        p_text=r'$\mathit{p}$ < 0.001' if p<0.001 else rf'$\mathit{{p}}$ = {p:.3f}'
        stat_text=rf'$\mathit{{r}}$ = {r:.2f}, '+p_text
    else:
        stat_text=r'$\mathit{r}$ = NA, $\mathit{p}$ = NA'
    ax.text(0.97,0.08,stat_text,transform=ax.transAxes,ha='right',va='bottom',fontsize=19)
    if index%2==0:ax.set_ylabel('Standardized TRW/NPP',fontsize=23)
    if index>=4:ax.set_xlabel('Year',fontsize=23)
axes[0].legend(frameon=True,fancybox=False,edgecolor='black',fontsize=18,loc='upper right')
fig.tight_layout(h_pad=0.8,w_pad=0.8)
out_png.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out_png,bbox_inches='tight',pad_inches=0.12,facecolor='white',dpi=600)
plt.close(fig)
print('✅ Saved figure:',out_png)