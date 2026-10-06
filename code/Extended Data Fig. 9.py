import warnings,numpy as np,pandas as pd,xarray as xr,matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import solve,pinvh,LinAlgWarning
from scipy.stats import pearsonr
from pykrige.ok import OrdinaryKriging
from matplotlib.ticker import MultipleLocator

ROOT=Path('D:/');POINTS=ROOT/'data/1. Basic information of global tree-ring sites.csv';TRW=ROOT/'data/Processed data/Detrending/fill/AgeDepSpline+fill.csv';MASK=ROOT/'data/Processed data/TRW/recon_mask (5+3).nc';OUT=ROOT / 'results/Extended Data Fig. 9.png'
MODELS=['spherical','exponential','gaussian','linear','power'];REGIONS=((-11,57,30,75),(-167.27,-55,9.58,74.99))
N_NEIGHBORS=100;N_YEARS=12;TEST_N=300;TARGETS=250;MAX_D=10.;STEP=.5;MAX_PAIRS=120000;SEED=1234;rng=np.random.default_rng(SEED)
plt.rcParams.update({'font.family':'Times New Roman','font.size':18,'axes.unicode_minus':False,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic'})

def kriging_predict(train_xy,train_z,target_xy,model):
    ok=OrdinaryKriging(train_xy[:,0],train_xy[:,1],train_z,variogram_model=model,coordinates_type='euclidean',verbose=False,enable_plotting=False,enable_statistics=False);par=np.asarray(ok.variogram_model_parameters,float);vg=ok.variogram_function;k=min(N_NEIGHBORS,len(train_xy));d,ids=cKDTree(train_xy).query(target_xy,k=k)
    if k==1:d=d[:,None];ids=ids[:,None]
    pred=np.full(len(target_xy),np.nan)
    for n in range(len(target_xy)):
        ii=np.atleast_1d(ids[n]);dd=np.atleast_1d(d[n]);kk=len(ii);A=np.ones((kk+1,kk+1));A[-1,-1]=0;A[:kk,:kk]=-vg(par,cdist(train_xy[ii],train_xy[ii]));np.fill_diagonal(A[:kk,:kk],0);b=np.r_[-vg(par,dd),1.];b[:kk][dd<=1e-10]=0
        try:
            with warnings.catch_warnings():warnings.simplefilter('error',LinAlgWarning);w=solve(A,b,assume_a='sym',check_finite=False)
            if not np.isfinite(w).all():raise np.linalg.LinAlgError
        except (np.linalg.LinAlgError,LinAlgWarning):w=pinvh(A,check_finite=False)@b
        pred[n]=w[:kk]@train_z[ii]
    return pred,ok

p=pd.read_csv(POINTS,usecols=['name','latitude','longitude']).dropna();p['name']=p['name'].astype(str);p[['latitude','longitude']]=p[['latitude','longitude']].apply(pd.to_numeric,errors='coerce');p=p.dropna(subset=['latitude','longitude']).reset_index(drop=True)
t=pd.read_csv(TRW);t=t.rename(columns={t.columns[0]:'year'});t['year']=pd.to_numeric(t['year'],errors='coerce');t=t.dropna(subset=['year']);t['year']=t['year'].astype(int);t=t[(t.year>=1950)&(t.year<=2026)].set_index('year').sort_index()
p=p[p.name.isin(t.columns)].reset_index(drop=True);names=p.name.tolist();years=t.index.to_numpy(int);v=t[names].to_numpy(float);v[(v==0)|~np.isfinite(v)]=np.nan;coords=p[['longitude','latitude']].to_numpy(float);xy,inv=np.unique(coords,axis=0,return_inverse=True);z=np.full((len(years),len(xy)),np.nan)
for i in range(len(xy)):
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);z[:,i]=np.nanmean(v[:,inv==i],axis=1)

with xr.open_dataset(MASK) as ds:mask_years=ds.year.values.astype(int);lat=ds.latitude.values;lon=ds.longitude.values;recon_mask=ds.recon_mask.transpose('year','latitude','longitude').values
common=np.intersect1d(years,mask_years);sample_years=common[np.unique(np.linspace(0,len(common)-1,N_YEARS).round().astype(int))]

tree=cKDTree(xy);pairs=tree.query_pairs(MAX_D,output_type='ndarray');total_pairs=len(pairs)
if len(pairs)>MAX_PAIRS:pairs=pairs[rng.choice(len(pairs),MAX_PAIRS,False)]
dist=np.linalg.norm(xy[pairs[:,0]]-xy[pairs[:,1]],axis=1);corr=np.full(len(pairs),np.nan)
for i,(a,b) in enumerate(pairs):
    m=np.isfinite(z[:,a])&np.isfinite(z[:,b])
    if m.sum()>=20 and np.std(z[m,a])>0 and np.std(z[m,b])>0:corr[i]=np.corrcoef(z[m,a],z[m,b])[0,1]
edges=np.arange(0,MAX_D+STEP,STEP);cx=(edges[:-1]+edges[1:])/2;cmed=[];cq25=[];cq75=[]
for a,b in zip(edges[:-1],edges[1:]):
    q=corr[(dist>a)&(dist<=b)&np.isfinite(corr)];cmed.append(np.median(q) if len(q) else np.nan);cq25.append(np.percentile(q,25) if len(q) else np.nan);cq75.append(np.percentile(q,75) if len(q) else np.nan)
cmed,cq25,cq75=map(np.asarray,(cmed,cq25,cq75))

records=[]
for year in sample_years:
    yi=np.where(years==year)[0][0];valid=np.flatnonzero(np.isfinite(z[yi]));yrng=np.random.default_rng(SEED+int(year));test=yrng.choice(valid,min(TEST_N,len(valid)-3),False);train=np.setdiff1d(valid,test);obs=z[yi,test]
    for model in MODELS:
        try:
            pred,_=kriging_predict(xy[train],z[yi,train],xy[test],model);m=np.isfinite(obs)&np.isfinite(pred);o=obs[m];pr=pred[m];r=pearsonr(o,pr).statistic;rmse=np.sqrt(np.mean((pr-o)**2));mae=np.mean(abs(pr-o));bias=np.mean(pr-o);nrmse=rmse/np.std(o) if np.std(o)>0 else np.nan;records.append([year,model,r,rmse,nrmse,mae,bias])
        except Exception as e:print(year,model,'FAILED:',e)
cv=pd.DataFrame(records,columns=['year','model','r','rmse','nrmse','mae','bias'])
summary=cv.groupby('model').agg(r=('r','median'),r25=('r',lambda x:np.percentile(x,25)),r75=('r',lambda x:np.percentile(x,75)),rmse=('rmse','median'),nrmse=('nrmse','median'),mae=('mae','median'),bias=('bias','median')).reindex(MODELS)

glon,glat=np.meshgrid(lon,lat);region=np.zeros(glon.shape,bool)
for a,b,c,d in REGIONS:region|=(glon>=a)&(glon<=b)&(glat>=c)&(glat<=d)
weight_samples=[];all_dist=[]
for year in sample_years:
    yi=np.where(years==year)[0][0];valid=np.isfinite(z[yi]);xyy=xy[valid];zz=z[yi,valid];ok=OrdinaryKriging(xyy[:,0],xyy[:,1],zz,variogram_model='spherical',coordinates_type='euclidean',verbose=False,enable_plotting=False,enable_statistics=False);par=np.asarray(ok.variogram_model_parameters,float);vg=ok.variogram_function;mi=np.where(mask_years==year)[0][0];flat=np.flatnonzero((region&np.isfinite(recon_mask[mi])&(recon_mask[mi]>.5)).ravel())
    if len(flat)>TARGETS:flat=np.random.default_rng(SEED+year+999).choice(flat,TARGETS,False)
    targets=np.c_[glon.ravel()[flat],glat.ravel()[flat]];k=min(N_NEIGHBORS,len(xyy));dd_all,id_all=cKDTree(xyy).query(targets,k=k)
    for dd,ii in zip(dd_all,id_all):
        dd=np.atleast_1d(dd);ii=np.atleast_1d(ii);kk=len(ii);A=np.ones((kk+1,kk+1));A[-1,-1]=0;A[:kk,:kk]=-vg(par,cdist(xyy[ii],xyy[ii]));np.fill_diagonal(A[:kk,:kk],0);b=np.r_[-vg(par,dd),1.]
        try:w=solve(A,b,assume_a='sym',check_finite=False)[:kk]
        except:w=(pinvh(A,check_finite=False)@b)[:kk]
        aw=np.abs(w);order=np.argsort(dd)
        if np.isfinite(aw).all() and aw.sum()>0:weight_samples.append((dd[order],np.cumsum(aw[order])/aw.sum()));all_dist.extend(dd.tolist())

dmax=max(5.,np.percentile(all_dist,99));dg=np.linspace(0,dmax,150);curves=np.array([np.interp(dg,d,c,left=0,right=1) for d,c in weight_samples]);wmed=np.median(curves,axis=0)*100;w25=np.percentile(curves,25,axis=0)*100;w75=np.percentile(curves,75,axis=0)*100
D=lambda q:dg[np.where(wmed>=q)[0][0]];d50,d80,d90=D(50),D(80),D(90)

fig,ax=plt.subplots(1,3,figsize=(18,5.8),dpi=600)

ax[0].fill_between(cx,cq25,cq75,alpha=.22)
ax[0].plot(cx,cmed,lw=2.4,label=r'Median $\mathit{r}$')
ax[0].axhline(0,color='black',ls='--',lw=1)
ax[0].set_xlim(0,10)
ax[0].set_xlabel('Distance between tree-ring sites (°)',fontsize=18)
ax[0].set_ylabel(r'Pairwise TRW correlation ($\mathit{r}$)',fontsize=18)
ax[0].xaxis.set_major_locator(MultipleLocator(2))
ax[0].legend(frameon=True,fancybox=False,edgecolor='black',fontsize=16,loc='upper right')

x=np.arange(5);r=summary.r.values;low=r-summary.r25.values;high=summary.r75.values-r
ax[1].bar(x,r,width=.68)
ax[1].errorbar(x,r,yerr=[low,high],fmt='none',ecolor='black',capsize=4,lw=1.2)
ax[1].set_xticks(x)
ax[1].set_xticklabels(['Spherical','Exponential','Gaussian','Linear','Power'],rotation=25,ha='center',rotation_mode='anchor')
ax[1].tick_params(axis='x',pad=14)
ax[1].set_ylabel(r'Cross-validation correlation ($\mathit{r}$)',fontsize=18)
ax[1].set_ylim(0,max(.6,np.nanmax(summary.r75)+.08))
for xx,yy in zip(x,r):ax[1].text(xx,yy+.012,f'{yy:.2f}',ha='center',va='bottom',fontsize=18)

ax[2].fill_between(dg,w25,w75,alpha=.22,label='IQR')
ax[2].plot(dg,wmed,lw=2.4,label='Median cumulative absolute weight')
for y,d in [(50,d50),(80,d80),(90,d90)]:
    ax[2].axhline(y,color='black',lw=.8,ls=':')
    ax[2].axvline(d,color='black',lw=.8,ls=':')
    ax[2].scatter(d,y,s=34,zorder=5)
ax[2].set_xlim(0,dmax)
ax[2].set_ylim(0,100)
ax[2].set_xlabel('Distance from target grid cell (°)',fontsize=18)
ax[2].set_ylabel('Cumulative absolute kriging weight (%)',fontsize=18)
ax[2].yaxis.set_major_locator(MultipleLocator(20))
ax[2].text(.97,.47,f'50% = {d50:.2f}°\n80% = {d80:.2f}°\n90% = {d90:.2f}°',transform=ax[2].transAxes,ha='right',va='center',fontsize=18)
ax[2].legend(frameon=True,fancybox=False,edgecolor='black',fontsize=16,loc='lower right')

for i,a in enumerate(ax):
    a.text(.02,.98,'abc'[i],transform=a.transAxes,ha='left',va='top',fontweight='bold',fontsize=24)
    a.tick_params(axis='both',labelsize=18)
    a.spines[['top','right']].set_visible(False)
    a.grid(axis='y',ls=':',lw=.6,alpha=.35)

fig.tight_layout()
OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=600,bbox_inches='tight',pad_inches=.12,facecolor='white')
plt.close(fig)

print('\n========== Panel B ==========\n',summary.to_string(float_format=lambda x:f'{x:.4f}'))
print('\n========== Panel C ==========')
print('50% cumulative weight:',round(d50,3),'degrees')
print('80% cumulative weight:',round(d80,3),'degrees')
print('90% cumulative weight:',round(d90,3),'degrees')
print('✅ Saved figure:',OUT)