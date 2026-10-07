import xarray as xr,numpy as np,geopandas as gpd,matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from pathlib import Path
plt.rcParams['font.family']='Times New Roman';plt.rcParams['mathtext.fontset']='custom';plt.rcParams['mathtext.rm']='Times New Roman';plt.rcParams['mathtext.it']='Times New Roman:italic';plt.rcParams['font.size']=24;plt.rcParams['axes.unicode_minus']=False
ROOT=Path('D:/');npp_path=ROOT/'data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc';eu_shp_path=ROOT/'data/Map/Europe 7-class.shp';na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp';out_png=ROOT/'results/Supplementary Fig. 3.png'
models=[('ModNegExp',ROOT/'data/Processed data/Detrending5/ModNegExp (5+3).nc'),('AgeDepSpline',ROOT/'data/Processed data/Detrending5/AgeDepSpline (5+3).nc'),('SFRCS',ROOT/'data/Processed data/Detrending5/SFRCS (5+3).nc'),('Spline',ROOT/'data/Processed data/Detrending5/Spline (5+3).nc')]
def temporal_std(da):
    count=da.count('year');mean=da.mean('year',skipna=True);variance=((da-mean)**2).sum('year',skipna=True)/count.where(count>0);return np.sqrt(variance)
eu_shp=gpd.read_file(eu_shp_path);eu_shp=eu_shp.set_crs('EPSG:4326') if eu_shp.crs is None else eu_shp.to_crs('EPSG:4326')
na_shp=gpd.read_file(na_shp_path);na_shp=na_shp.set_crs('EPSG:4326') if na_shp.crs is None else na_shp.to_crs('EPSG:4326')
def extract_series(model,path):
    ds_npp=xr.open_dataset(npp_path);ds_trw=xr.open_dataset(path);npp=ds_npp['npp'];trw=ds_trw[model];year_npp=ds_npp['year'].values;year_trw=ds_trw['year'].values;time_dim_npp=[d for d in npp.dims if d not in ('latitude','longitude')][0];time_dim_trw=[d for d in trw.dims if d not in ('latitude','longitude')][0];npp=npp.assign_coords({time_dim_npp:year_npp}).rename({time_dim_npp:'year'});trw=trw.assign_coords({time_dim_trw:year_trw}).rename({time_dim_trw:'year'});years=np.intersect1d(npp['year'].values,trw['year'].values);years=years[(years>=1900)&(years<=1995)];npp=npp.sel(year=years);trw=trw.sel(year=years)
    if float(np.nanmin(npp['longitude'].values))>=0:npp=npp.assign_coords(longitude=(npp['longitude']+180)%360-180).sortby('longitude')
    if float(np.nanmin(trw['longitude'].values))>=0:trw=trw.assign_coords(longitude=(trw['longitude']+180)%360-180).sortby('longitude')
    ratio=(trw/npp).where(np.isfinite(trw/npp));mu_ratio=ratio.mean('year',skipna=True);std_ratio=temporal_std(ratio);ratio_z=((ratio-mu_ratio)/std_ratio).where(std_ratio>0);stacked=ratio_z.stack(points=('latitude','longitude'));lon=stacked['longitude'].values;lat=stacked['latitude'].values;pts=stacked['points'].values;gdf=gpd.GeoDataFrame({'points':pts},geometry=gpd.points_from_xy(lon,lat),crs='EPSG:4326')
    def region(shp):
        join=gpd.sjoin(gdf,shp,how='inner',predicate='within')
        if join.empty:return None
        ids=np.unique(join['points'].values);reg=stacked.sel(points=ids);q=reg.quantile(q=[0.25,0.5,0.75],dim='points',skipna=True);return q.sel(quantile=0.5).values,q.sel(quantile=0.25).values,q.sel(quantile=0.75).values
    eu=region(eu_shp);na=region(na_shp);ds_npp.close();ds_trw.close()
    if eu is None or na is None:raise ValueError(f'No valid regional TRW/NPP data for {model}')
    mask=np.isfinite(eu[0])&np.isfinite(na[0]);r,p=pearsonr(eu[0][mask],na[0][mask]);return years,eu,na,r,p
fig,axes=plt.subplots(2,2,figsize=(18,11),dpi=300,sharex=True,sharey=True);axes=axes.ravel();letters=['a','b','c','d']
for ax,(model,path),letter in zip(axes,models,letters):
    years,eu,na,r,p=extract_series(model,path);ax.plot(years,eu[0],linewidth=2.2,label='EU');ax.fill_between(years,eu[1],eu[2],alpha=0.25);ax.plot(years,na[0],linewidth=2.2,label='NA');ax.fill_between(years,na[1],na[2],alpha=0.25)
    ax.axvline(x=1945,color='black',linestyle='--',linewidth=1.8,zorder=5)
    ax.text(1945,0.03,'1945',color='black',fontsize=24,ha='center',va='bottom',transform=ax.get_xaxis_transform(),zorder=6)
    ax.set_xlim(1900,1995);ax.tick_params(labelsize=22);ax.xaxis.set_major_locator(MultipleLocator(20));ax.yaxis.set_major_locator(MultipleLocator(1));ax.yaxis.set_major_formatter(FormatStrFormatter('%d'));ax.text(0.02,0.98,letter,transform=ax.transAxes,ha='left',va='top',fontweight='bold',fontsize=30);ax.text(0.5,0.98,model,transform=ax.transAxes,ha='center',va='top',fontsize=24);p_text=r'$\mathit{p}$ < 0.001' if p<0.001 else rf'$\mathit{{p}}$ = {p:.3f}';ax.text(0.97,0.08,rf'$\mathit{{r}}$ = {r:.2f}, '+p_text,transform=ax.transAxes,ha='right',va='bottom',fontsize=21)
axes[0].legend(frameon=True,fancybox=False,edgecolor='black',fontsize=18,loc='upper right');axes[2].set_xlabel('Year',fontsize=24);axes[3].set_xlabel('Year',fontsize=24);fig.text(0.015,0.5,'Standardized TRW/NPP',rotation=90,ha='center',va='center',fontsize=24);fig.tight_layout(rect=[0.035,0.02,1,1]);out_png.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out_png,bbox_inches='tight',pad_inches=0.12,facecolor='white',dpi=600);plt.close(fig);print('✅ Saved figure:',out_png)