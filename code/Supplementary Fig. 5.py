import os,glob,numpy as np,pandas as pd,xarray as xr,matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['mathtext.rm']='Times New Roman'
plt.rcParams['mathtext.it']='Times New Roman:italic'
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=14
plt.rcParams['axes.labelsize']=17
ROOT=Path('D:/')
PAD=0.2
L_N_OFF=0.015
L_R_OFF=0.012
L_STAR_OFF=0.08
R_STAR_OFF=0.15
R_R_OFF=0.08
R_N_OFF=0.08
site_csv1=ROOT/'data/4. Coordinates of North American and European flux tower sites.csv'
gpp_obs_candidates=[ROOT/'data/8. Annual GPP data of GPP_DT_VUT method.csv',ROOT/'data/8. Annual GPP data of GPP_DT_VUT method.csv']
obs_csv1=next((p for p in gpp_obs_candidates if p.exists()),gpp_obs_candidates[-1])
csv_data2=ROOT/'data/6. Annual NPP values at observation sites.csv'
coord_csv2=ROOT/'data/5. NPP site coordinate table.csv'
gpp_dir=ROOT/'data/NPP and GPP/S3_gpp1-12'
npp_dir=ROOT/'data/NPP and GPP/S3_npp1-12'
gpp_exclude={'VISIT_S3_gpp.nc','VISIT-UT_S3_gpp.nc','LPJmL_S3_gpp.nc'}
npp_exclude={'SDGVM_S3_nppAnnual.nc'}
out_dir=ROOT / 'results'
os.makedirs(out_dir,exist_ok=True)
out_png=ROOT / 'results/Supplementary Fig. 5.png'
def stars(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
def coord_names(ds):
    lat='latitude' if 'latitude' in ds.coords else 'lat' if 'lat' in ds.coords else None
    lon='longitude' if 'longitude' in ds.coords else 'lon' if 'lon' in ds.coords else None
    if lat is None or lon is None:
        raise ValueError('Latitude or longitude coordinate not found')
    return lat,lon
def set_xlim_pad(ax,rvals,pad=PAD,rev=False):
    rmin,rmax=float(np.nanmin(rvals)),float(np.nanmax(rvals))
    if rmin==rmax:
        rmin-=0.5
        rmax+=0.5
    rng=rmax-rmin
    x0,x1=rmin-pad*rng,rmax+pad*rng
    ax.set_xlim((x1,x0) if rev else (x0,x1))
    return abs(x1-x0)
def compute_gpp_results():
    sites=pd.read_csv(site_csv1)
    obs=pd.read_csv(obs_csv1).replace([-9999,-9999.0],np.nan)
    years=pd.to_numeric(obs['year'],errors='coerce')
    valid_year=years.notna()
    obs=obs.loc[valid_year].copy()
    years=years.loc[valid_year].astype(int)
    obs['year']=years.values
    rows=[]
    files=sorted(glob.glob(os.path.join(gpp_dir,'*.nc')))
    if not files:
        raise FileNotFoundError(f'No NetCDF files found in {gpp_dir}')
    for fp in files:
        fn=os.path.basename(fp)
        if fn in gpp_exclude:
            continue
        try:
            ds=xr.open_dataset(fp)
        except Exception:
            continue
        if 'gpp' not in ds.data_vars or 'year' not in ds.coords:
            ds.close()
            continue
        try:
            lat_name,lon_name=coord_names(ds)
        except Exception:
            ds.close()
            continue
        gpp=ds['gpp']
        try:
            gpp=gpp.sel(year=slice(int(years.min()),int(years.max())))
        except Exception:
            ds.close()
            continue
        xs,ys=[],[]
        for _,r in sites.iterrows():
            sid=r.get('Site ID')
            if sid not in obs.columns:
                continue
            try:
                ts=gpp.sel({lat_name:float(r['Lat']),lon_name:float(r['Long'])},method='nearest').to_pandas()
            except Exception:
                continue
            if isinstance(ts,pd.DataFrame):
                if ts.shape[1]!=1:
                    continue
                ts=ts.iloc[:,0]
            ts.index=pd.to_numeric(ts.index,errors='coerce')
            ts=ts[~pd.isna(ts.index)]
            ts.index=ts.index.astype(int)
            df=pd.DataFrame({'year':years.values,'obs':pd.to_numeric(obs[sid],errors='coerce').values})
            mod=ts.rename('mod').reset_index()
            mod.columns=['year','mod']
            df=df.merge(mod,on='year',how='left').replace([-9999,-9999.0],np.nan).dropna(subset=['obs','mod'])
            if df.empty:
                continue
            xs.append(df['obs'].to_numpy(float))
            ys.append(df['mod'].to_numpy(float))
        ds.close()
        if sum(map(len,xs))<10:
            continue
        x=np.concatenate(xs)
        y=np.concatenate(ys)
        valid=np.isfinite(x)&np.isfinite(y)
        x=x[valid]
        y=y[valid]
        if len(x)<10 or np.nanstd(x)==0 or np.nanstd(y)==0:
            continue
        r,p=pearsonr(x,y)
        rows.append({'Model':Path(fn).stem,'r':r,'p':p,'n':len(x)})
    if not rows:
        raise ValueError('No valid GPP results were calculated')
    df=pd.DataFrame(rows).sort_values('r',ascending=True).reset_index(drop=True)
    df['sig']=df['p'].apply(stars)
    return df
def compute_npp_results():
    lon_min,lon_max=-167.27,-55.0
    lat_min,lat_max=9.58,74.99
    d=pd.read_csv(csv_data2,usecols=['site_ID','begin_year','end_year','NPP_tot1'])
    xy=pd.read_csv(coord_csv2,usecols=['site_ID','latitude','longitude'])
    d=pd.merge(d,xy,on='site_ID').dropna(subset=['latitude','longitude','NPP_tot1'])
    d=d[(d['longitude']>=lon_min)&(d['longitude']<=lon_max)&(d['latitude']>=lat_min)&(d['latitude']<=lat_max)]
    d['begin_year']=pd.to_numeric(d['begin_year'],errors='coerce')
    d['end_year']=pd.to_numeric(d['end_year'],errors='coerce')
    d['NPP_tot1']=pd.to_numeric(d['NPP_tot1'],errors='coerce')
    d=d.dropna(subset=['begin_year','end_year','NPP_tot1'])
    rows=[]
    files=sorted(glob.glob(os.path.join(npp_dir,'*.nc')))
    if not files:
        raise FileNotFoundError(f'No NetCDF files found in {npp_dir}')
    for fp in files:
        fn=os.path.basename(fp)
        if fn in npp_exclude:
            continue
        try:
            ds=xr.open_dataset(fp)
        except Exception:
            continue
        if 'npp' not in ds.data_vars or 'year' not in ds.coords:
            ds.close()
            continue
        try:
            lat_name,lon_name=coord_names(ds)
        except Exception:
            ds.close()
            continue
        npp=ds['npp']
        yrs=pd.to_numeric(pd.Series(ds['year'].values),errors='coerce').to_numpy()
        xs,ys=[],[]
        for _,r in d.iterrows():
            y0,y1=int(r['begin_year']),int(r['end_year'])
            m=np.isfinite(yrs)&(yrs>=y0)&(yrs<=y1)
            if not np.any(m):
                continue
            selected_years=ds['year'].values[m]
            try:
                ts=npp.sel({lat_name:float(r['latitude']),lon_name:float(r['longitude'])},method='nearest').sel(year=selected_years)
                val=float(ts.mean(skipna=True).item())
            except Exception:
                continue
            if np.isfinite(val) and np.isfinite(r['NPP_tot1']):
                xs.append(float(r['NPP_tot1']))
                ys.append(val)
        ds.close()
        if len(xs)<3:
            continue
        x=np.asarray(xs,dtype=float)
        y=np.asarray(ys,dtype=float)
        valid=np.isfinite(x)&np.isfinite(y)
        x=x[valid]
        y=y[valid]
        if len(x)<3 or np.nanstd(x)==0 or np.nanstd(y)==0:
            continue
        r,p=pearsonr(x,y)
        rows.append({'Model':Path(fn).stem,'r':r,'p':p,'n':len(x)})
    if not rows:
        raise ValueError('No valid NPP results were calculated')
    df=pd.DataFrame(rows).sort_values('r',ascending=True).reset_index(drop=True)
    df['sig']=df['p'].apply(stars)
    return df
def clean_model_name(name):
    name=str(name)
    for suffix in ['_S3_gpp','_S3_npp','_gpp','_npp']:
        if name.endswith(suffix):
            name=name[:-len(suffix)]
            break
    return name
def draw_left(ax,df,xlabel):
    y=np.arange(len(df))
    bars=ax.barh(y,df['r'],edgecolor='black')
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([clean_model_name(name) for name in df['Model']],fontsize=14)
    ax.set_xlabel(xlabel,fontsize=17)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True,linestyle='--',alpha=0.5)
    span=set_xlim_pad(ax,df['r'].values,pad=PAD,rev=False)
    for yi,b,(n,r,sig) in zip(y,bars,df[['n','r','sig']].to_numpy()):
        w=b.get_width()
        ax.text(w-L_N_OFF*span,yi,f'n={int(n)}',va='center',ha='right',fontsize=11)
        ax.text(w+L_R_OFF*span,yi,f'{r:.2f}',va='center',ha='left',fontsize=12)
        if sig:
            ax.text(w+(L_R_OFF+L_STAR_OFF)*span,yi,sig,va='center',ha='left',fontsize=12,color='red')
def draw_right(ax,df,xlabel):
    y=np.arange(len(df))
    bars=ax.barh(y,df['r'],edgecolor='black')
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([clean_model_name(name) for name in df['Model']],fontsize=14)
    ax.yaxis.tick_right()
    ax.tick_params(labelleft=False,labelright=True)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(xlabel,fontsize=17)
    ax.xaxis.grid(True,linestyle='--',alpha=0.5)
    span=set_xlim_pad(ax,df['r'].values,pad=PAD,rev=True)
    for yi,b,(n,r,sig) in zip(y,bars,df[['n','r','sig']].to_numpy()):
        w=b.get_width()
        if sig:
            ax.text(w+R_STAR_OFF*span,yi,sig,va='center',ha='left',fontsize=12,color='red')
        ax.text(w+R_R_OFF*span,yi,f'{r:.2f}',va='center',ha='left',fontsize=12)
        ax.text(w-R_N_OFF*span,yi,f'n={int(n)}',va='center',ha='right',fontsize=11)
df1=compute_gpp_results()
df2=compute_npp_results()
print(df1[['Model','r','p','n']].to_string(index=False))
print(df2[['Model','r','p','n']].to_string(index=False))
h=max(6,len(df1)*0.48,len(df2)*0.48)
fig,axes=plt.subplots(1,2,figsize=(13,h),constrained_layout=True,gridspec_kw={'wspace':0.08})
draw_left(axes[0],df1,r'$\mathit{r}$ (Flux tower& DGVM_S3 (GPP))')
draw_right(axes[1],df2,r'$\mathit{r}$ (NPP site & DGVM_S3 (NPP))')
axes[0].text(0.02,0.98,'a',transform=axes[0].transAxes,ha='left',va='top',fontsize=22,fontweight='bold')
axes[1].text(0.02,0.98,'b',transform=axes[1].transAxes,ha='left',va='top',fontsize=22,fontweight='bold')
fig.savefig(out_png,dpi=600,bbox_inches='tight')
plt.close(fig)
print('✅ Saved figure:',out_png)