import re,numpy as np,pandas as pd,xarray as xr,matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
plt.rcParams['font.family']='Times New Roman';plt.rcParams['mathtext.fontset']='custom';plt.rcParams['mathtext.rm']='Times New Roman';plt.rcParams['mathtext.it']='Times New Roman:italic';plt.rcParams['font.size']=36;plt.rcParams['axes.unicode_minus']=False
ROOT=Path('D:/');gpp_dt_csv=ROOT/'data/8. Annual GPP data of GPP_DT_VUT method.csv';gpp_nt_csv=ROOT/'data/9. Annual GPP data of GPP_NT_VUT method.csv';site_csv=ROOT/'data/4. Coordinates of North American and European flux tower sites.csv';out_png=ROOT / 'results/Supplementary Fig. 6.png'
nc_dirs=[ROOT/'data/NPP and GPP/S3_gpp1-12']
MIN_YEARS=10
sites=pd.read_csv(site_csv);gpp_dt=pd.read_csv(gpp_dt_csv);gpp_nt=pd.read_csv(gpp_nt_csv);sites['Site ID']=sites['Site ID'].astype(str).str.strip();site_ids=[sid for sid in sites['Site ID'] if sid in gpp_dt.columns or sid in gpp_nt.columns];sites=sites[sites['Site ID'].isin(site_ids)].drop_duplicates('Site ID').copy();site_ids=sites['Site ID'].tolist();gpp_dt['year']=pd.to_numeric(gpp_dt['year'],errors='coerce');gpp_nt['year']=pd.to_numeric(gpp_nt['year'],errors='coerce')
def model_name(path):
    s=path.stem;s=re.sub(r'^\d+[_\s-]*','',s);s=re.sub(r'_S[23]_(gpp|npp)$','',s,flags=re.I);s=re.sub(r'_(gpp|npp)$','',s,flags=re.I);return s
def detect_coord(ds,names):
    for n in names:
        if n in ds.coords or n in ds.dims:return n
    return None
def select_var(ds,latn,lonn):
    pref=['gpp','npp','ndvi','evi','lai','fpar','cwood','savi','ndmi','tcw','nbr'];cand=[]
    for v in ds.data_vars:
        da=ds[v]
        if latn in da.dims and lonn in da.dims and np.issubdtype(da.dtype,np.number):
            vn=v.lower();score=min([i for i,k in enumerate(pref) if k in vn],default=999);cand.append((score,-da.ndim,v))
    return sorted(cand)[0][2] if cand else None
def series_at_site(da,latn,lonn,lat,lon):
    lon=float(lon);nc_lon=np.asarray(da[lonn].values,dtype=float)
    if np.nanmin(nc_lon)>=0 and lon<0:lon%=360
    if np.nanmax(nc_lon)<=180 and lon>180:lon=((lon+180)%360)-180
    s=da.sel({latn:float(lat),lonn:lon},method='nearest');tdim=next((d for d in s.dims if 'time' in d.lower() or 'year' in d.lower()),None)
    if tdim is None:return None
    for d in list(s.dims):
        if d!=tdim:s=s.mean(d,skipna=True)
    vals=np.asarray(s.values).squeeze();tv=np.asarray(s[tdim].values)
    if vals.ndim!=1 or len(vals)!=len(tv):return None
    if np.issubdtype(tv.dtype,np.datetime64):yrs=pd.DatetimeIndex(tv).year.to_numpy()
    else:
        yrs=pd.to_numeric(pd.Series(tv),errors='coerce').to_numpy()
        if np.isnan(yrs).all():
            try:yrs=pd.to_datetime(tv).year.to_numpy()
            except:return None
    d=pd.DataFrame({'year':yrs,'value':pd.to_numeric(pd.Series(vals),errors='coerce')}).dropna();d['year']=d['year'].astype(int);return d.groupby('year')['value'].mean()
def obs_series(table,sid):
    if sid not in table.columns:return pd.Series(dtype=float)
    d=pd.DataFrame({'year':table['year'],'value':pd.to_numeric(table[sid],errors='coerce')}).dropna();d['year']=d['year'].astype(int);return d.groupby('year')['value'].mean()
def corr_one(obs,prod):
    if prod is None or len(obs)==0:return np.nan,1.0,0
    d=pd.DataFrame({'OBS':obs,'PROD':prod.reindex(obs.index)}).dropna()
    if len(d)<MIN_YEARS or d['OBS'].nunique()<2 or d['PROD'].nunique()<2:return np.nan,1.0,len(d)
    r,p=pearsonr(d['OBS'],d['PROD']);return float(r),float(p),len(d)
nc_files=[]
for d in nc_dirs:
    if d.exists():nc_files.extend(sorted(d.rglob('*.nc')))
if not nc_files:raise ValueError('No NetCDF files were found')
results=[];site_results={}
for k,path in enumerate(nc_files,1):
    try:
        ds=xr.open_dataset(path);latn=detect_coord(ds,['latitude','lat','Latitude','LAT']);lonn=detect_coord(ds,['longitude','lon','Longitude','LON'])
        if latn is None or lonn is None:ds.close();print(f'Skip {path}: coordinates not found');continue
        var=select_var(ds,latn,lonn)
        if var is None:ds.close();print(f'Skip {path}: suitable variable not found');continue
        da=ds[var];per_dt={};per_nt={}
        for _,row in sites.iterrows():
            sid=row['Site ID'];prod=series_at_site(da,latn,lonn,row['Lat'],row['Long'])
            if prod is None:continue
            odt=obs_series(gpp_dt,sid);ont=obs_series(gpp_nt,sid)
            r,p,n=corr_one(odt,prod)
            if np.isfinite(r):per_dt[sid]=(r,p,n)
            r,p,n=corr_one(ont,prod)
            if np.isfinite(r):per_nt[sid]=(r,p,n)
        ds.close();rd=[v[0] for v in per_dt.values()];rn=[v[0] for v in per_nt.values()];med_dt=float(np.median(rd)) if rd else np.nan;med_nt=float(np.median(rn)) if rn else np.nan;key=str(path)
        results.append({'path':key,'model':model_name(path),'variable':var,'median_dt':med_dt,'median_nt':med_nt,'n_dt':len(rd),'n_nt':len(rn)});site_results[key]={'dt':per_dt,'nt':per_nt}
        print(f'[{k}/{len(nc_files)}] {path.name} | variable={var} | DT median r={med_dt:.4f} n={len(rd)} | NT median r={med_nt:.4f} n={len(rn)}')
    except Exception as e:print(f'Skip {path}: {e}')
rank=pd.DataFrame(results)
if rank.empty:raise ValueError('No valid NetCDF products were evaluated')
rank_dt=rank[np.isfinite(rank['median_dt'])].sort_values(['median_dt','n_dt'],ascending=[False,False]).reset_index(drop=True)
rank_nt=rank[np.isfinite(rank['median_nt'])].sort_values(['median_nt','n_nt'],ascending=[False,False]).reset_index(drop=True)
if rank_dt.empty or rank_nt.empty:raise ValueError('No valid DT or NT model ranking could be calculated')
best_dt=rank_dt.iloc[0];best_nt=rank_nt.iloc[0];dt_data=site_results[best_dt['path']]['dt'];nt_data=site_results[best_nt['path']]['nt'];plot_sites=[sid for sid in site_ids if sid in dt_data or sid in nt_data];x=np.arange(len(plot_sites));dt_r=np.array([dt_data[sid][0] if sid in dt_data else np.nan for sid in plot_sites]);dt_p=np.array([dt_data[sid][1] if sid in dt_data else np.nan for sid in plot_sites]);nt_r=np.array([nt_data[sid][0] if sid in nt_data else np.nan for sid in plot_sites]);nt_p=np.array([nt_data[sid][1] if sid in nt_data else np.nan for sid in plot_sites])
print('\nGPP_DT ranking:');print(rank_dt[['model','variable','median_dt','n_dt','path']].to_string(index=False));print('\nGPP_NT ranking:');print(rank_nt[['model','variable','median_nt','n_nt','path']].to_string(index=False));print('\nBest GPP_DT model:',best_dt['model'],'median r=',round(best_dt['median_dt'],4));print('Best GPP_NT model:',best_nt['model'],'median r=',round(best_nt['median_nt'],4))
fig,axes=plt.subplots(2,1,figsize=(25,12),dpi=300,sharex=True)
axes[0].bar(x,dt_r,width=0.72,color='royalblue',zorder=2,label=rf'$\mathit{{r}}$ (GPP_DT & {best_dt["model"]})')
axes[0].axhline(0,color='k',linewidth=1,zorder=1)
axes[0].set_ylim(-1,1)
axes[0].tick_params(axis='y',labelsize=25)
axes[0].legend(frameon=True,edgecolor='black',fancybox=False,fontsize=24,loc='lower right')
axes[0].text(0.99,0.25,'Highest median '+r'$\mathit{r}$'+f' across all products = {best_dt["median_dt"]:.3f}',transform=axes[0].transAxes,ha='right',va='bottom',fontsize=24)
for i,(r,p) in enumerate(zip(dt_r,dt_p)):
    if np.isfinite(r) and np.isfinite(p) and p<0.05:axes[0].text(i,r+0.01 if r>=0 else r-0.06,'*',ha='center',va='bottom' if r>=0 else 'top',fontsize=30)
axes[1].bar(x,nt_r,width=0.72,color='darkorange',zorder=2,label=rf'$\mathit{{r}}$ (GPP_NT & {best_nt["model"]})')
axes[1].axhline(0,color='k',linewidth=1,zorder=1)
axes[1].set_ylim(-1,1)
axes[1].tick_params(axis='y',labelsize=25)
axes[1].legend(frameon=True,edgecolor='black',fancybox=False,fontsize=24,loc='lower right')
axes[1].text(0.99,0.25,'Highest median '+r'$\mathit{r}$'+f' across all products = {best_nt["median_nt"]:.3f}',transform=axes[1].transAxes,ha='right',va='bottom',fontsize=24)
axes[1].set_xlabel('Site ID',fontsize=26)
axes[1].set_xticks(x)
axes[1].set_xticklabels(plot_sites,rotation=90,fontsize=15)
for i,(r,p) in enumerate(zip(nt_r,nt_p)):
    if np.isfinite(r) and np.isfinite(p) and p<0.05:axes[1].text(i,r+0.01 if r>=0 else r-0.06,'*',ha='center',va='bottom' if r>=0 else 'top',fontsize=30)
fig.text(0.04,0.53,r'$\mathit{r}$ (Flux tower & DGVM_S3 (GPP))',rotation=90,ha='center',va='center',fontsize=30)
fig.tight_layout(rect=[0.035,0,1,1])
out_png.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out_png,dpi=600,bbox_inches='tight')
plt.close(fig)
print('Saved figure:',out_png)