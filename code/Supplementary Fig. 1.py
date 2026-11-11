
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import solve,pinvh,LinAlgWarning
from pykrige.ok import OrdinaryKriging
from joblib import Parallel,delayed,parallel_config
ROOT=Path('D:/')
POINTS_CSV=ROOT/'data/1. Basic information of global tree-ring sites.csv'
INPUT_DIR=ROOT/'data/Processed data/Detrending/fill'
NC_DIR=ROOT/'data/Processed data/TRW'
MASK_FILE=NC_DIR/'recon_mask (5+3).nc'
OUTPUT_DIR=NC_DIR/'leave-one-out'
OUTPUT_PNG=ROOT/'results/Supplementary Fig. 1.png'
METHODS=('AgeDepSpline','ModNegExp','SFRCS','Spline')
REGIONS=((-11.,57.,30.,75.),(-167.27,-55.,9.58,74.99))
YEAR_MIN,YEAR_MAX=1900,2023
REMOVE_FRACTION=0.10
N_REPEATS=5
N_NEIGHBORS=100
N_JOBS=5
STRATA_DEG=5.0
SEED=20261110
MAX_GRID_CELLS=0
DISPLAY_POINTS_PER_REPEAT=50000
PREDICTION_BATCH=5000
CHECK_BASELINE=True
BASELINE_TOL=0.02
DISPLAY_QUANTILES=(0.005,0.995)
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','font.size':19,'axes.labelsize':19,'xtick.labelsize':15,'ytick.labelsize':15,'axes.unicode_minus':False})
def region_ids(xy):
    result=np.full(len(xy),2,dtype=np.int8)
    for i,(xmin,xmax,ymin,ymax) in enumerate(REGIONS):
        valid=(xy[:,0]>=xmin)&(xy[:,0]<=xmax)&(xy[:,1]>=ymin)&(xy[:,1]<=ymax)
        result[valid]=i
    return result
def stratified_sample(ids,xy,region,count,seed):
    ids=np.asarray(ids,dtype=np.int32)
    count=min(len(ids),max(0,int(count)))
    if count==0:
        return np.empty(0,dtype=np.int32)
    rng=np.random.default_rng(seed)
    labels=np.column_stack((region[ids],np.floor(xy[ids]/STRATA_DEG).astype(np.int32)))
    _,inverse=np.unique(labels,axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    expected=count*counts/len(ids)
    quota=np.floor(expected).astype(int)
    extra=count-int(quota.sum())
    if extra:
        fractions=expected-quota
        chosen=rng.choice(len(quota),size=extra,replace=False,p=fractions/fractions.sum())
        quota[chosen]+=1
    selected=[]
    for i,n in enumerate(quota):
        if n:
            selected.extend(rng.choice(ids[inverse==i],size=int(n),replace=False).tolist())
    return np.asarray(sorted(selected),dtype=np.int32)
def load_mask():
    with xr.open_dataset(MASK_FILE) as ds:
        latitude=np.asarray(ds['latitude'].values,dtype=np.float64)
        longitude=np.asarray(ds['longitude'].values,dtype=np.float64)
        original_years=np.asarray(ds['year'].values,dtype=int)
        years=np.unique(original_years[(original_years>=YEAR_MIN)&(original_years<=YEAR_MAX)])
        if len(years)==0:
            raise ValueError('No valid reconstruction years')
        lookup={int(y):i for i,y in enumerate(original_years)}
        selected=[lookup[int(y)] for y in years]
        mask=np.asarray(ds['recon_mask'].transpose('year','latitude','longitude').isel(year=selected).values)
    grid_lon,grid_lat=np.meshgrid(longitude,latitude)
    all_xy=np.column_stack((grid_lon.ravel(),grid_lat.ravel()))
    region=region_ids(all_xy)
    mask_flat=mask.reshape(len(years),-1)
    valid=np.isfinite(mask_flat)&(mask_flat>0.5)&(region[None,:]<2)
    codes=np.flatnonzero(np.any(valid,axis=0))
    if len(codes)==0:
        raise ValueError('No valid 0.5-degree grid cells')
    if MAX_GRID_CELLS>0 and len(codes)>MAX_GRID_CELLS:
        chosen=stratified_sample(np.arange(len(codes)),all_xy[codes],region[codes],MAX_GRID_CELLS,SEED+888)
        codes=codes[chosen]
    return years,latitude,longitude,codes,all_xy[codes],valid[:,codes]
def load_site_matrices(years):
    points=pd.read_csv(POINTS_CSV,usecols=['name','longitude','latitude']).dropna().copy()
    points['name']=points['name'].astype(str)
    points['longitude']=pd.to_numeric(points['longitude'],errors='coerce')
    points['latitude']=pd.to_numeric(points['latitude'],errors='coerce')
    points=points.loc[np.isfinite(points['longitude'])&np.isfinite(points['latitude'])].reset_index(drop=True)
    xy,inverse=np.unique(points[['longitude','latitude']].to_numpy(dtype=np.float64),axis=0,return_inverse=True)
    names=points['name'].to_numpy()
    year_lookup={int(y):i for i,y in enumerate(years)}
    matrices={}
    for method in METHODS:
        path=INPUT_DIR/f'{method}+fill.csv'
        if not path.is_file():
            raise FileNotFoundError(path)
        table=pd.read_csv(path)
        table=table.rename(columns={table.columns[0]:'year'})
        raw_years=pd.to_numeric(table['year'],errors='raise').to_numpy(dtype=float)
        if not np.isfinite(raw_years).all() or not np.equal(raw_years,np.floor(raw_years)).all():
            raise ValueError(f'Invalid years: {method}')
        table['year']=raw_years.astype(int)
        if table['year'].duplicated().any():
            raise ValueError(f'Duplicate years: {method}')
        table=table.set_index('year').sort_index()
        matched=np.flatnonzero(np.isin(names,table.columns))
        common_years=np.intersect1d(years,table.index.to_numpy(dtype=int))
        if not len(matched) or not len(common_years):
            raise ValueError(f'No matched sites or years: {method}')
        values=table.loc[common_years,names[matched].tolist()].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
        ids=inverse[matched]
        matrix=np.full((len(years),len(xy)),np.nan,dtype=np.float64)
        for row,year in enumerate(common_years):
            z=values[row]
            good=np.isfinite(z)&(z!=0)
            if not good.any():
                continue
            counts=np.bincount(ids[good],minlength=len(xy))
            total=np.bincount(ids[good],weights=z[good],minlength=len(xy))
            use=counts>0
            matrix[year_lookup[int(year)],use]=total[use]/counts[use]
        matrices[method]=matrix
        print(f'{method}: {len(matched)} matched records, {np.any(np.isfinite(matrix),axis=0).sum()} valid coordinates',flush=True)
    candidate=np.logical_or.reduce([np.any(np.isfinite(matrices[m]),axis=0) for m in METHODS])
    return xy,matrices,np.flatnonzero(candidate)
def read_reference(method,years,latitude,longitude,codes,valid_mask):
    path=NC_DIR/f'{method} (5+3).nc'
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        if method not in ds:
            raise ValueError(f'Missing NC variable: {method}')
        if not np.allclose(ds['latitude'].values,latitude,rtol=0,atol=1e-5):
            raise ValueError(f'Latitude mismatch: {method}')
        if not np.allclose(ds['longitude'].values,longitude,rtol=0,atol=1e-5):
            raise ValueError(f'Longitude mismatch: {method}')
        source_years=np.asarray(ds['year'].values,dtype=int)
        lookup={int(y):i for i,y in enumerate(source_years)}
        if not all(int(y) in lookup for y in years):
            raise ValueError(f'Missing reconstruction years: {method}')
        data=ds[method].transpose('year','latitude','longitude').isel(year=[lookup[int(y)] for y in years])
        full=np.asarray(data.values,dtype=np.float32).reshape(len(years),-1)[:,codes].copy()
    full[~valid_mask]=np.nan
    return full
def manual_kriging(model,xy,z,target,k):
    distances,neighbors=cKDTree(xy).query(target,k=k)
    if k==1:
        distances=distances[:,None]
        neighbors=neighbors[:,None]
    pars=np.asarray(model.variogram_model_parameters,dtype=float)
    gamma=model.variogram_function
    matrix=np.empty((k+1,k+1),dtype=float)
    matrix[-1,:]=1.
    matrix[:,-1]=1.
    matrix[-1,-1]=0.
    rhs=np.empty(k+1,dtype=float)
    rhs[-1]=1.
    pred=np.empty(len(target),dtype=np.float32)
    for i in range(len(target)):
        ids=neighbors[i]
        d=distances[i]
        matrix[:k,:k]=-gamma(pars,cdist(xy[ids],xy[ids]))
        np.fill_diagonal(matrix[:k,:k],0.)
        rhs[:k]=-gamma(pars,d)
        rhs[:k][d<=1e-10]=0.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',LinAlgWarning)
                w=solve(matrix,rhs,assume_a='sym',check_finite=False)
            if not np.isfinite(w).all():
                raise np.linalg.LinAlgError('Invalid weights')
        except (np.linalg.LinAlgError,LinAlgWarning):
            w=pinvh(matrix,check_finite=False)@rhs
        pred[i]=w[:k]@z[ids]
    return pred
def kriging_predict(xy,z,target):
    if len(xy)<3:
        raise ValueError('Fewer than three source coordinates')
    if len(target)==0:
        return np.empty(0,dtype=np.float32)
    if np.all(z==z[0]):
        return np.full(len(target),z[0],dtype=np.float32)
    model=OrdinaryKriging(x=xy[:,0],y=xy[:,1],z=z,variogram_model='spherical',coordinates_type='euclidean',enable_plotting=False,verbose=False,enable_statistics=False)
    pars=np.asarray(model.variogram_model_parameters,dtype=float)
    if not np.isfinite(pars).all():
        raise ValueError('Nonfinite variogram parameters')
    k=min(N_NEIGHBORS,len(xy))
    predictions=[]
    use_manual=False
    for start in range(0,len(target),PREDICTION_BATCH):
        locations=target[start:start+PREDICTION_BATCH]
        if use_manual:
            result=manual_kriging(model,xy,z,locations,k)
        else:
            try:
                estimated,_=model.execute('points',locations[:,0],locations[:,1],backend='C',n_closest_points=k)
                result=np.asarray(np.ma.filled(estimated,np.nan),dtype=np.float32).reshape(-1)
                if not np.isfinite(result).all():
                    raise ValueError('Nonfinite C-backend predictions')
            except (ImportError,ModuleNotFoundError,NotImplementedError,RuntimeError,ValueError,np.linalg.LinAlgError):
                use_manual=True
                result=manual_kriging(model,xy,z,locations,k)
        predictions.append(result)
    output=np.concatenate(predictions)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite kriging predictions')
    return output
def check_baseline(method,years,xy,matrix,grid_xy,full):
    rng=np.random.default_rng(SEED+700)
    errors=[]
    for yi in sorted(set((0,len(years)//2,len(years)-1))):
        ids=np.flatnonzero(np.isfinite(full[yi]))
        if not len(ids):
            continue
        selected=rng.choice(ids,size=min(25,len(ids)),replace=False)
        good=np.isfinite(matrix[yi])
        predicted=kriging_predict(xy[good],matrix[yi,good],grid_xy[selected])
        errors.extend(np.abs(predicted.astype(float)-full[yi,selected].astype(float)).tolist())
    if not errors:
        raise ValueError(f'Unable to verify reference NC: {method}')
    maximum=max(errors)
    print(f'{method}: baseline max absolute difference = {maximum:.6f}',flush=True)
    if maximum>BASELINE_TOL:
        raise ValueError(f'{method}: original NC and current inputs or kriging settings differ; maximum difference = {maximum:.6f}')
def predict_year(yi,xy,z,grid_xy,valid,removed):
    targets=np.flatnonzero(valid)
    if len(targets)==0:
        return yi,targets,np.empty(0,dtype=np.float32)
    training=np.isfinite(z)&~removed
    if training.sum()<3:
        raise ValueError(f'Insufficient training sites at year index {yi}')
    estimated=kriging_predict(xy[training],z[training],grid_xy[targets])
    return yi,targets,estimated
def calculate_metrics(full,reduced):
    valid=np.isfinite(full)&np.isfinite(reduced)
    x=full[valid].astype(np.float64)
    y=reduced[valid].astype(np.float64)
    if len(x)<3:
        raise ValueError('Insufficient paired grid values')
    sx=np.std(x,ddof=1)
    sy=np.std(y,ddof=1)
    r=float(np.corrcoef(x,y)[0,1]) if sx>0 and sy>0 else np.nan
    rmse=float(np.sqrt(np.mean((y-x)**2)))
    nrmse=float(rmse/sx) if sx>0 else np.nan
    return {'r':r,'NRMSE':nrmse,'RMSE':rmse,'n':len(x)}
def analyze_method(method,years,xy,matrix,grid_xy,valid_mask,latitude,longitude,codes,removed_sets):
    full=read_reference(method,years,latitude,longitude,codes,valid_mask)
    if CHECK_BASELINE:
        check_baseline(method,years,xy,matrix,grid_xy,full)
    valid=np.isfinite(full)&valid_mask
    results=[]
    display_x=[]
    display_y=[]
    for repeat,removed_ids in enumerate(removed_sets):
        removed=np.zeros(len(xy),dtype=bool)
        removed[removed_ids]=True
        tasks=[(yi,matrix[yi],valid[yi]) for yi in range(len(years)) if np.any(valid[yi])]
        with parallel_config(backend='loky',inner_max_num_threads=1):
            rows=Parallel(n_jobs=N_JOBS,batch_size=1,pre_dispatch=N_JOBS)(delayed(predict_year)(yi,xy,z,grid_xy,year_valid,removed) for yi,z,year_valid in tasks)
        reduced=np.full_like(full,np.nan)
        for yi,indices,predicted in rows:
            reduced[yi,indices]=predicted
        scores=calculate_metrics(full,reduced)
        results.append(scores)
        paired=np.flatnonzero(np.isfinite(full.ravel())&np.isfinite(reduced.ravel()))
        rng=np.random.default_rng(SEED+repeat*1237+METHODS.index(method)*17)
        chosen=rng.choice(paired,size=min(DISPLAY_POINTS_PER_REPEAT,len(paired)),replace=False)
        display_x.append(full.ravel()[chosen])
        display_y.append(reduced.ravel()[chosen])
        print(f'{method} | repeat {repeat+1}/{N_REPEATS} | r={scores["r"]:.4f} | NRMSE={scores["NRMSE"]:.4f} | RMSE={scores["RMSE"]:.4f} | paired values={scores["n"]}',flush=True)
    return {'scores':results,'x':np.concatenate(display_x),'y':np.concatenate(display_y)}
def summary(record,key):
    values=np.asarray([item[key] for item in record['scores']],dtype=float)
    return float(np.nanmedian(values)),float(np.nanmin(values)),float(np.nanmax(values))
def draw_panel(ax,record,method,letter):
    x=record['x'].astype(float)
    y=record['y'].astype(float)
    low,high=np.quantile(np.concatenate((x,y)),DISPLAY_QUANTILES)
    padding=max((high-low)*0.035,1e-4)
    low-=padding
    high+=padding
    visible=(x>=low)&(x<=high)&(y>=low)&(y<=high)
    ax.hexbin(x[visible],y[visible],gridsize=50,mincnt=1,bins='log',cmap='viridis',linewidths=0,rasterized=True)
    ax.plot([low,high],[low,high],color='black',ls='--',lw=1.5)
    ax.set_xlim(low,high)
    ax.set_ylim(low,high)
    ax.set_aspect('equal',adjustable='box')
    ax.text(0.035,0.965,f'{letter}  {method}',transform=ax.transAxes,ha='left',va='top',fontsize=18,fontweight='bold',bbox={'facecolor':'white','edgecolor':'none','alpha':0.88,'pad':2})
    r=summary(record,'r')[0]
    nrmse=summary(record,'NRMSE')[0]
    ax.text(0.965,0.035,r'$r$'+f' = {r:.3f}\nNRMSE = {nrmse:.3f}',transform=ax.transAxes,ha='right',va='bottom',fontsize=17,bbox={'facecolor':'white','edgecolor':'none','alpha':0.90,'pad':3})
    ax.set_xlabel('TRW from 100% sites')
    ax.set_ylabel('TRW from 90% sites')
    ax.tick_params(direction='out',length=5,width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
def make_figure(records):
    fig,axes=plt.subplots(2,2,figsize=(13.5,12))
    fig.subplots_adjust(left=0.11,right=0.98,bottom=0.09,top=0.98,wspace=0.28,hspace=0.25)
    for i,method in enumerate(METHODS):
        draw_panel(axes.flat[i],records[method],method,'abcd'[i])
    fig.savefig(OUTPUT_PNG,dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    started=time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    years,latitude,longitude,codes,grid_xy,valid_mask=load_mask()
    xy,matrices,candidates=load_site_matrices(years)
    region=region_ids(xy)
    remove_count=int(round(REMOVE_FRACTION*len(candidates)))
    removed_sets=[stratified_sample(candidates,xy,region,remove_count,SEED+repeat*100003) for repeat in range(N_REPEATS)]
    print(f'Years: {years[0]}-{years[-1]}',flush=True)
    print(f'Valid 0.5-degree grid cells: {len(codes)}',flush=True)
    print(f'Eligible unique site coordinates: {len(candidates)}',flush=True)
    print(f'Removed coordinates per repeat: {remove_count}',flush=True)
    print(f'Repeats: {N_REPEATS}',flush=True)
    records={}
    for method in METHODS:
        records[method]=analyze_method(method,years,xy,matrices[method],grid_xy,valid_mask,latitude,longitude,codes,removed_sets)
        print(f'FINAL SUMMARY | {method}',flush=True)
        for key in ('r','NRMSE'):
            median,minimum,maximum=summary(records[method],key)
            print(f'{key}: median={median:.4f}, range={minimum:.4f}-{maximum:.4f}',flush=True)
    make_figure(records)
    print(f'Saved image: {OUTPUT_PNG}',flush=True)
    print(f'Elapsed: {(time.perf_counter()-started)/60:.1f} minutes',flush=True)
if __name__=='__main__':
    main()
