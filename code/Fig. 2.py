import warnings,xarray as xr,numpy as np,pandas as pd,geopandas as gpd,matplotlib.pyplot as plt,matplotlib as mpl,os
from pathlib import Path
from scipy.stats import pearsonr,t as student_t
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore',message='All-NaN slice encountered')
ROOT=Path('D:/');FS=24
plt.rcParams.update({'font.family':'Times New Roman','font.size':FS,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic'})
mpl.rcParams['savefig.dpi']=600
npp_path=ROOT/'data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc'
age_path=ROOT/'data/Processed data/TRW/AgeDepSpline (5+3).nc'
eu_shp_path=ROOT/'data/Map/Europe 7-class.shp';na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp';shp_world=ROOT/'data/Map/World.shp'
out_dir=ROOT / 'results';os.makedirs(out_dir,exist_ok=True);out_png=ROOT / 'results/Fig. 2.png'
MIN_N=30
eu_lon_min,eu_lon_max=-11.,57.;eu_lat_min,eu_lat_max=29,75.
na_lon_min,na_lon_max=-167.27,-55.;na_lat_min,na_lat_max=9.58,74.99
BLUE='#2166ac';ORANGE='#d95f02';GRAY='0.75'
def format_lon(x,pos):return '0°' if np.isclose(x,0) else f'{abs(x):g}°{"E" if x>0 else "W"}'
def format_lat(y,pos):return '0°' if np.isclose(y,0) else f'{abs(y):g}°{"N" if y>0 else "S"}'
def temporal_std(da):
    n=da.count('year');m=da.mean('year',skipna=True)
    return np.sqrt(((da-m)**2).sum('year',skipna=True)/n.where(n>0))
def region_quantile_series(da,gdf,path):
    shp=gpd.read_file(path);shp=shp.set_crs('EPSG:4326') if shp.crs is None else shp.to_crs('EPSG:4326')
    j=gpd.sjoin(gdf,shp,how='inner',predicate='within');reg=da.sel(points=np.unique(j.points.values))
    q=reg.quantile([.25,.5,.75],dim='points',skipna=True)
    return q.sel(quantile=.5).values,q.sel(quantile=.25).values,q.sel(quantile=.75).values
def regional_flux(mask):
    nr=npp.where(mask);tr=trw.where(mask);pair=np.isfinite(nr)&np.isfinite(tr);nr=nr.where(pair);tr=tr.where(pair)
    nz=((nr-nr.mean('year',skipna=True))/temporal_std(nr));tz=((tr-tr.mean('year',skipna=True))/temporal_std(tr))
    return nz.mean(('latitude','longitude'),skipna=True).values,tz.mean(('latitude','longitude'),skipna=True).values,nz.quantile(.25,dim=('latitude','longitude'),skipna=True).values,nz.quantile(.75,dim=('latitude','longitude'),skipna=True).values,tz.quantile(.25,dim=('latitude','longitude'),skipna=True).values,tz.quantile(.75,dim=('latitude','longitude'),skipna=True).values
def plot_class_hist(ax,data,pvals,bins,region_text,show_ylabel=True):
    sig=np.isfinite(pvals)&(pvals<.05)
    neg=data[sig&(data<0)];ns=data[~sig];pos=data[sig&(data>0)]
    total=len(data);pneg=len(neg)/total*100;pns=len(ns)/total*100;ppos=len(pos)/total*100
    labels=[rf'$\mathit{{P}}$ < 0.05 ({pneg:.1f}%)',
            rf'$\mathit{{P}}$ ≥ 0.05 ({pns:.1f}%)',
            rf'$\mathit{{P}}$ < 0.05 ({ppos:.1f}%)']
    ax.hist([neg,ns,pos],bins=bins,stacked=True,color=[BLUE,GRAY,ORANGE],
            edgecolor='white',linewidth=.4,label=labels)
    ax.axvline(0,color='black',lw=1,ls='--')
    ax.set_xlabel('Trend in TRW/NPP (yr$^{-1}$)',fontsize=FS)
    ax.set_ylabel('Number of 0.5° × 0.5° pixels' if show_ylabel else '',fontsize=FS)
    ax.text(.98,.965,region_text,transform=ax.transAxes,ha='right',va='top',fontsize=30,fontweight='bold',fontname='Times New Roman')
    ax.legend(frameon=True,fancybox=False,edgecolor='black',fontsize=FS-3,
              loc='upper right',bbox_to_anchor=(.98,.75))
ds_npp=xr.open_dataset(npp_path);ds_trw=xr.open_dataset(age_path)
npp=ds_npp['npp'];trw=ds_trw['AgeDepSpline']
dn=[d for d in npp.dims if d not in ('latitude','longitude')][0];dt=[d for d in trw.dims if d not in ('latitude','longitude')][0]
npp=npp.assign_coords({dn:ds_npp.year.values}).rename({dn:'year'})
trw=trw.assign_coords({dt:ds_trw.year.values}).rename({dt:'year'})
years=np.intersect1d(npp.year.values,trw.year.values);years=years[(years>=1900)&(years<=2025)]
npp=npp.sel(year=years);trw=trw.sel(year=years)
if float(np.nanmin(npp.longitude.values))>=0:npp=npp.assign_coords(longitude=(npp.longitude+180)%360-180).sortby('longitude')
if float(np.nanmin(trw.longitude.values))>=0:trw=trw.assign_coords(longitude=(trw.longitude+180)%360-180).sortby('longitude')
world=gpd.read_file(shp_world);world=world.set_crs(epsg=4326) if world.crs is None else world.to_crs(epsg=4326)
ratio=(trw/npp).where(np.isfinite(trw/npp));mu=ratio.mean('year',skipna=True);sd=temporal_std(ratio);ratio_z=((ratio-mu)/sd).where(sd>0)
ratio_z_trend=ratio_z.sel(year=ratio_z.year>=1945);years_trend=ratio_z_trend.year.values
mask_ratio=ratio_z_trend.notnull();n_valid=mask_ratio.sum('year')
x=xr.DataArray(years_trend.astype(float),coords={'year':years_trend},dims='year')
x_mean=x.where(mask_ratio).mean('year',skipna=True);z_mean=ratio_z_trend.where(mask_ratio).mean('year',skipna=True)
xm=x-x_mean;zm=ratio_z_trend-z_mean;num=(xm*zm).where(mask_ratio).sum('year',skipna=True);den=(xm*xm).where(mask_ratio).sum('year',skipna=True)
slope=(num/den).where((n_valid>=MIN_N)&(den>0))
fit=(z_mean+slope*(x-x_mean)).where(mask_ratio);resid=(ratio_z_trend-fit).where(mask_ratio)
df=(n_valid-2).where(n_valid>2);rss=(resid**2).sum('year',skipna=True);se=np.sqrt((rss/df)/den);tval=(slope/se).where(np.isfinite(se)&(se>0))
pval=xr.full_like(slope,np.nan,dtype=float);m=np.isfinite(tval.values)&np.isfinite(df.values)&(df.values>0)
pval.values[m]=2*student_t.sf(np.abs(tval.values[m]),df.values[m])
na_s=slope.sel(longitude=slice(na_lon_min,na_lon_max),latitude=slice(na_lat_min,na_lat_max))
eu_s=slope.sel(longitude=slice(eu_lon_min,eu_lon_max),latitude=slice(eu_lat_min,eu_lat_max))
na_p=pval.sel(longitude=slice(na_lon_min,na_lon_max),latitude=slice(na_lat_min,na_lat_max))
eu_p=pval.sel(longitude=slice(eu_lon_min,eu_lon_max),latitude=slice(eu_lat_min,eu_lat_max))
allv=np.concatenate([na_s.values.ravel(),eu_s.values.ravel()]);allv=allv[np.isfinite(allv)]
vmax=max(float(np.nanpercentile(np.abs(allv),95)),1e-6);vmin=-vmax
rz=ratio_z.stack(points=('latitude','longitude'))
gdf=gpd.GeoDataFrame({'points':rz.points.values},geometry=gpd.points_from_xy(rz.longitude.values,rz.latitude.values),crs='EPSG:4326')
eu_med,eu25,eu75=region_quantile_series(rz,gdf,eu_shp_path)
na_med,na25,na75=region_quantile_series(rz,gdf,na_shp_path)
lon=npp.longitude;lat=npp.latitude;lat2d,lon2d=xr.broadcast(lat,lon)
eu_mask=(lon2d>=eu_lon_min)&(lon2d<=eu_lon_max)&(lat2d>=eu_lat_min)&(lat2d<=eu_lat_max)
na_mask=(lon2d>=na_lon_min)&(lon2d<=na_lon_max)&(lat2d>=na_lat_min)&(lat2d<=na_lat_max)
roi=eu_mask|na_mask
roi_npp_mean,roi_trw_mean,roi_npp25,roi_npp75,roi_trw25,roi_trw75=regional_flux(roi)
na_valid=np.isfinite(na_s.values);eu_valid=np.isfinite(eu_s.values)
na_hist=na_s.values[na_valid];na_hist_p=na_p.values[na_valid]
eu_hist=eu_s.values[eu_valid];eu_hist_p=eu_p.values[eu_valid]
bins=np.linspace(np.nanmin(np.r_[na_hist,eu_hist]),np.nanmax(np.r_[na_hist,eu_hist]),31)
wr1=(na_lon_max-na_lon_min)/(na_lat_max-na_lat_min);wr2=(eu_lon_max-eu_lon_min)/(eu_lat_max-eu_lat_min)
fig=plt.figure(figsize=(18,19),dpi=300)
gs=gridspec.GridSpec(3,2,figure=fig,left=.06,right=.97,bottom=.045,top=.975,wspace=.1,hspace=.21,
                     width_ratios=[wr1,wr2],height_ratios=[1.35,1,1])
ax_a=fig.add_subplot(gs[0,0]);ax_b=fig.add_subplot(gs[0,1])
ax_c=fig.add_subplot(gs[1,0]);ax_d=fig.add_subplot(gs[1,1])
ax_e=fig.add_subplot(gs[2,0]);ax_f=fig.add_subplot(gs[2,1])
im=ax_a.pcolormesh(na_s.longitude,na_s.latitude,na_s,cmap='RdBu_r',vmin=vmin,vmax=vmax,shading='nearest',alpha=.95)
world.cx[na_lon_min:na_lon_max,na_lat_min:na_lat_max].boundary.plot(ax=ax_a,color='black',linewidth=.6,zorder=3)
yy,xx=np.where(np.isfinite(na_s.values)&(na_p.values<.05))
ax_a.scatter(na_s.longitude.values[xx],na_s.latitude.values[yy],s=.5,c='black',linewidths=0,alpha=.7,zorder=4)
ax_a.set(xlim=(na_lon_min,na_lon_max),ylim=(na_lat_min,na_lat_max));ax_a.set_aspect(1.3,adjustable='box')
ax_a.xaxis.set_major_formatter(FuncFormatter(format_lon));ax_a.yaxis.set_major_formatter(FuncFormatter(format_lat))
cax=inset_axes(ax_a,width='36%',height='4.2%',loc='lower left',borderpad=2.4)
cb=fig.colorbar(im,cax=cax,orientation='horizontal',extend='both')
cb.set_label('Trend in TRW/NPP (yr$^{-1}$)',fontsize=FS,labelpad=-3);cb.ax.tick_params(labelsize=FS)
ax_b.pcolormesh(eu_s.longitude,eu_s.latitude,eu_s,cmap='RdBu_r',vmin=vmin,vmax=vmax,shading='nearest',alpha=.95)
world.cx[eu_lon_min:eu_lon_max,eu_lat_min:eu_lat_max].boundary.plot(ax=ax_b,color='black',linewidth=.6,zorder=3)
yy,xx=np.where(np.isfinite(eu_s.values)&(eu_p.values<.05))
ax_b.scatter(eu_s.longitude.values[xx],eu_s.latitude.values[yy],s=.5,c='black',linewidths=0,alpha=.7,zorder=4)
ax_b.set(xlim=(eu_lon_min,eu_lon_max),ylim=(eu_lat_min,eu_lat_max));ax_b.set_aspect(1.3,adjustable='box');ax_b.yaxis.tick_right()
ax_b.xaxis.set_major_formatter(FuncFormatter(format_lon));ax_b.yaxis.set_major_locator(MultipleLocator(10));ax_b.yaxis.set_major_formatter(FuncFormatter(format_lat))
plot_class_hist(ax_c,na_hist,na_hist_p,bins,'NA',True)
plot_class_hist(ax_d,eu_hist,eu_hist_p,bins,'EU',False)
ax_e.plot(years,eu_med,lw=2.2,color=ORANGE,label='EU');ax_e.fill_between(years,eu25,eu75,color=ORANGE,alpha=.18)
ax_e.plot(years,na_med,lw=2.2,color=BLUE,label='NA');ax_e.fill_between(years,na25,na75,color=BLUE,alpha=.18)
ax_e.axvline(x=1945,color='black',linestyle='--',linewidth=1.8,zorder=5)
ax_e.text(1945,0.03,'1945',color='black',fontsize=24,fontweight='normal',ha='center',va='bottom',transform=ax_e.get_xaxis_transform(),zorder=6)
ax_e.set(xlim=(1900,2025),xlabel='Year',ylabel='Standardized TRW/NPP')
ax_e.xaxis.set_major_locator(MultipleLocator(20));ax_e.legend(frameon=True,fancybox=False,edgecolor='black',fontsize=FS-3,loc='upper right')
m=np.isfinite(eu_med)&np.isfinite(na_med);r_e,p_e=pearsonr(eu_med[m],na_med[m])
pe=r'$\mathit{p}$ < 0.001' if p_e<.001 else rf'$\mathit{{p}}$ = {p_e:.3f}'
ax_e.text(.25,.28,rf'$\mathit{{r}}$ = {r_e:.2f}, '+pe,transform=ax_e.transAxes,fontsize=FS)
ax_f.plot(years,roi_npp_mean,lw=2.2,color=BLUE,label='NPP (NA & EU)')
ax_f.plot(years,roi_trw_mean,lw=2.2,color=ORANGE,label='TRW (NA & EU)')
ax_f.fill_between(years,roi_npp25,roi_npp75,color=BLUE,alpha=.12)
ax_f.fill_between(years,roi_trw25,roi_trw75,color=ORANGE,alpha=.12)
ax_f.set(xlim=(1900,2025),xlabel='Year',ylabel='Standardized value')
ax_f.xaxis.set_major_locator(MultipleLocator(20))
ax_f.legend(frameon=True,fancybox=False,edgecolor='black',fontsize=FS-5,loc='lower right')
mf=np.isfinite(roi_npp_mean)&np.isfinite(roi_trw_mean);r_f,p_f=pearsonr(roi_npp_mean[mf],roi_trw_mean[mf])
pf=r'$\mathit{p}$ < 0.001' if p_f<.001 else rf'$\mathit{{p}}$ = {p_f:.3f}'
ax_f.text(.52,.28,rf'$\mathit{{r}}$ = {r_f:.2f}, '+pf,transform=ax_f.transAxes,fontsize=FS)
ax_e.set_ylim(-2,2)
ax_f.set_ylim(-2,2)
yticks=[-1,0,1]
ax_e.set_yticks(yticks)
ax_f.set_yticks(yticks)
ax_e.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax_f.yaxis.set_major_formatter(FormatStrFormatter('%d'))
for lab,ax in zip('abcdef',[ax_a,ax_b,ax_c,ax_d,ax_e,ax_f]):
    ax.text(.02,.98,lab,transform=ax.transAxes,ha='left',va='top',fontweight='bold',fontsize=32)
    ax.tick_params(axis='both',labelsize=FS)
fig.savefig(out_png,bbox_inches='tight',pad_inches=.12,facecolor='white',dpi=600)
plt.close(fig)
na_sig=np.isfinite(na_hist_p)&(na_hist_p<.05);eu_sig=np.isfinite(eu_hist_p)&(eu_hist_p<.05)
print('NA:',round(np.sum(na_sig&(na_hist<0))/len(na_hist)*100,2),'%,',round(np.sum(~na_sig)/len(na_hist)*100,2),'%,',round(np.sum(na_sig&(na_hist>0))/len(na_hist)*100,2),'%')
print('EU:',round(np.sum(eu_sig&(eu_hist<0))/len(eu_hist)*100,2),'%,',round(np.sum(~eu_sig)/len(eu_hist)*100,2),'%,',round(np.sum(eu_sig&(eu_hist>0))/len(eu_hist)*100,2),'%')
print('✅ Saved figure:',out_png)