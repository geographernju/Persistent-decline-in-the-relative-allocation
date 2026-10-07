import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
import gc
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.stats import t as student_t
from joblib import Parallel,delayed,parallel_config
ROOT=Path("D:/")
DATA_DIR=Path('D:/data/Processed data/Detrending')
OUTPUT_DIR=Path('D:/data/Processed data/Detrending/fill')
CACHE_DIR=Path('D:/data/Processed data/Detrending/fill/climate_cache')
METHODS=("Spline","ModNegExp","AgeDepSpline","SFRCS")
SITE_PATH=ROOT/"data"/"1. Basic information of global tree-ring sites.csv"
CLIM_PATH=Path("H:/")/"Climatic data"/"month"/"climate14.nc"
START_YEAR=1950
END_YEAR=2025
MIN_PAIRS=30
P_THRESHOLD=0.01
N_JOBS=10
CELLS_PER_BATCH=30
SPATIAL_BLOCK=64
TIME_BLOCK=24
ALL_YEARS=np.arange(1700,END_YEAR+1)
TARGET_ROWS=(ALL_YEARS>=START_YEAR)&(ALL_YEARS<=END_YEAR)
TARGET_YEARS=ALL_YEARS[TARGET_ROWS]
WINDOWS=[(first,first+length-1) for length in (1,3,6,12) for first in range(1,14-length)]
def find_coord(ds,options):
    for collection in (ds.coords,ds.dims):
        for candidate in options:
            for key in collection:
                if key.lower()==candidate:
                    return key
    raise ValueError(f"Coordinate not found: {options}")
def load_growth():
    sites=pd.read_csv(SITE_PATH,encoding="utf-8-sig")
    sites.columns=sites.columns.astype(str).str.strip()
    if not {"name","latitude","longitude"}.issubset(sites.columns):
        raise ValueError("Required site columns are missing")
    sites["name"]=sites["name"].astype(str).str.strip()
    sites["latitude"]=pd.to_numeric(sites["latitude"],errors="coerce")
    sites["longitude"]=pd.to_numeric(sites["longitude"],errors="coerce")
    sites["lon180"]=((sites["longitude"]+180)%360)-180
    na=sites["lon180"].between(-167.27,-55.0)&sites["latitude"].between(9.58,74.99)
    eu=sites["lon180"].between(-11.0,55.0)&sites["latitude"].between(29.0,75.0)
    sites=sites.loc[na|eu].drop_duplicates("name").set_index("name")
    growth={}
    selected={}
    records={}
    for method in METHODS:
        path=DATA_DIR/f"{method}.csv"
        table=pd.read_csv(path,encoding="utf-8-sig")
        table.columns=table.columns.astype(str).str.strip()
        if "year" not in table.columns:
            raise ValueError(f"Missing year column: {path}")
        years=pd.to_numeric(table["year"],errors="raise").to_numpy(dtype=float)
        if not np.isfinite(years).all() or not np.equal(years,np.floor(years)).all():
            raise ValueError(f"Invalid years: {path}")
        table["year"]=years.astype(int)
        if table["year"].duplicated().any():
            raise ValueError(f"Duplicate years: {path}")
        table=table.set_index("year").reindex(ALL_YEARS)
        growth[method]={}
        selected[method]=[]
        for name in table.columns:
            records[(method,name)]={"method":method,"site":name,"climate_variable":"","months":"","R":np.nan,"P":np.nan,"paired_years":np.nan}
            if name not in sites.index:
                continue
            values=pd.to_numeric(table[name],errors="coerce").to_numpy(dtype=float)
            values[~np.isfinite(values)]=np.nan
            if np.isfinite(values[TARGET_ROWS]).sum()>=MIN_PAIRS:
                selected[method].append(name)
                growth[method][name]=values
        print(f"{method}: {len(selected[method])} eligible sites",flush=True)
    all_names=sorted(set().union(*(set(selected[method]) for method in METHODS)))
    if not all_names:
        raise ValueError("No eligible sites")
    return sites,growth,selected,records,all_names
def inspect_climate(sites,all_names):
    with xr.open_dataset(CLIM_PATH,decode_times=False) as ds:
        time_key=find_coord(ds,["time","date"])
        lat_key=find_coord(ds,["latitude","lat"])
        lon_key=find_coord(ds,["longitude","lon"])
        if ds[lat_key].ndim!=1 or ds[lon_key].ndim!=1:
            raise ValueError("One-dimensional spatial coordinates are required")
        if ds[time_key].ndim!=1:
            raise ValueError("One-dimensional time coordinate is required")
        time_dim=ds[time_key].dims[0]
        lat_dim=ds[lat_key].dims[0]
        lon_dim=ds[lon_key].dims[0]
        raw_time=np.asarray(ds[time_key].values)
        if np.issubdtype(raw_time.dtype,np.datetime64):
            dates=pd.DatetimeIndex(raw_time)
        else:
            time_text=pd.Series(raw_time).astype(str).str.replace(r"\.0$","",regex=True)
            dates=pd.DatetimeIndex(pd.to_datetime(time_text,format="%Y%m",errors="raise"))
        years=np.asarray(dates.year)
        months=np.asarray(dates.month)
        time_keep=np.flatnonzero((years>=START_YEAR)&(years<=END_YEAR))
        if time_keep.size==0:
            raise ValueError("No climate months in the target period")
        month_positions=(years[time_keep]-START_YEAR)*12+months[time_keep]-1
        var_names=[]
        required={time_dim,lat_dim,lon_dim}
        for name,variable in ds.data_vars.items():
            if not np.issubdtype(variable.dtype,np.number):
                continue
            if not required.issubset(variable.dims):
                continue
            if any(variable.sizes[dim]!=1 for dim in variable.dims if dim not in required):
                continue
            var_names.append(name)
        if not var_names:
            raise ValueError("No usable climate variables")
        latitudes=np.asarray(ds[lat_key].values,dtype=float)
        longitudes=np.asarray(ds[lon_key].values,dtype=float)
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("Invalid climate coordinates")
    groups={}
    outside=0
    for name in all_names:
        lat=float(sites.at[name,"latitude"])
        lon=float(sites.at[name,"lon180"])
        target_lon=lon%360 if longitudes.min()>=0 else lon
        if not (latitudes.min()<=lat<=latitudes.max() and longitudes.min()<=target_lon<=longitudes.max()):
            outside+=1
            continue
        i=int(np.abs(latitudes-lat).argmin())
        j=int(np.abs(longitudes-target_lon).argmin())
        groups.setdefault((i,j),[]).append(name)
    if not groups:
        raise ValueError("No eligible sites within the climate grid")
    groups=dict(sorted(groups.items()))
    print(f"Climate variables: {len(var_names)}; windows per variable: {len(WINDOWS)}",flush=True)
    print(f"Sites outside climate grid: {outside}; unique climate cells: {len(groups)}",flush=True)
    return groups,var_names,time_keep,month_positions,time_dim,lat_dim,lon_dim
def build_cache(groups,var_names,time_keep,time_dim,lat_dim,lon_dim):
    started=time.perf_counter()
    cells=np.asarray(list(groups),dtype=np.int64)
    stat=CLIM_PATH.stat()
    specification={"version":1,"source":str(CLIM_PATH.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"variables":var_names,"cells":cells.tolist(),"time_indices":time_keep.tolist(),"dimensions":[time_dim,lat_dim,lon_dim],"dtype":"float64"}
    signature=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()[:24]
    path=CACHE_DIR/f"climate_{signature}.npy"
    temporary=CACHE_DIR/f"climate_{signature}.partial.npy"
    shape=(len(cells),len(var_names),len(time_keep))
    if path.exists():
        try:
            cached=np.load(path,mmap_mode="r",allow_pickle=False)
            valid=cached.shape==shape and cached.dtype==np.dtype("float64")
            del cached
            if valid:
                print("Using climate cache:",path,flush=True)
                return path
        except Exception:
            pass
    print(f"Building climate cache: {np.prod(shape)*8/1024**2:.1f} MiB",flush=True)
    cache=None
    try:
        cache=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.float64,shape=shape)
        cache[:]=np.nan
        with Dataset(CLIM_PATH) as ds:
            for var_index,name in enumerate(var_names):
                variable_started=time.perf_counter()
                variable=ds.variables[name]
                dims=variable.dimensions
                chunks=variable.chunking()
                if isinstance(chunks,(list,tuple)):
                    block_lat=min(int(chunks[dims.index(lat_dim)]),SPATIAL_BLOCK)
                    block_lon=min(int(chunks[dims.index(lon_dim)]),SPATIAL_BLOCK)
                    block_time=min(int(chunks[dims.index(time_dim)]),TIME_BLOCK)
                else:
                    block_lat=SPATIAL_BLOCK
                    block_lon=SPATIAL_BLOCK
                    block_time=TIME_BLOCK
                block_lat=max(1,block_lat)
                block_lon=max(1,block_lon)
                block_time=max(1,block_time)
                spatial_groups={}
                for cell_index,(i,j) in enumerate(cells):
                    key=(int(i)//block_lat,int(j)//block_lon)
                    spatial_groups.setdefault(key,[]).append(cell_index)
                time_groups={}
                for output_index,source_index in enumerate(time_keep):
                    key=int(source_index)//block_time
                    time_groups.setdefault(key,[]).append(output_index)
                retained_dims=[dim for dim in dims if dim in (time_dim,lat_dim,lon_dim)]
                permutation=[retained_dims.index(dim) for dim in (time_dim,lat_dim,lon_dim)]
                for (tile_i,tile_j),cell_indices in spatial_groups.items():
                    cell_indices=np.asarray(cell_indices,dtype=int)
                    lat_start=tile_i*block_lat
                    lon_start=tile_j*block_lon
                    lat_stop=min(lat_start+block_lat,len(ds.dimensions[lat_dim]))
                    lon_stop=min(lon_start+block_lon,len(ds.dimensions[lon_dim]))
                    local_i=cells[cell_indices,0]-lat_start
                    local_j=cells[cell_indices,1]-lon_start
                    for tile_t,output_indices in time_groups.items():
                        output_indices=np.asarray(output_indices,dtype=int)
                        time_start=tile_t*block_time
                        time_stop=min(time_start+block_time,len(ds.dimensions[time_dim]))
                        selectors=[]
                        for dim in dims:
                            if dim==time_dim:
                                selectors.append(slice(time_start,time_stop))
                            elif dim==lat_dim:
                                selectors.append(slice(lat_start,lat_stop))
                            elif dim==lon_dim:
                                selectors.append(slice(lon_start,lon_stop))
                            else:
                                selectors.append(0)
                        block=np.ma.asarray(variable[tuple(selectors)],dtype=np.float64).filled(np.nan)
                        block=np.transpose(block,permutation)
                        local_t=time_keep[output_indices]-time_start
                        extracted=block[local_t[:,None],local_i[None,:],local_j[None,:]].T
                        extracted[~np.isfinite(extracted)]=np.nan
                        cache[cell_indices[:,None],var_index,output_indices[None,:]]=extracted
                cache.flush()
                print(f"Climate cache [{var_index+1}/{len(var_names)}] {name}: {time.perf_counter()-variable_started:.1f} s",flush=True)
        cache.flush()
        del cache
        cache=None
        gc.collect()
        temporary.replace(path)
    finally:
        if cache is not None:
            del cache
            gc.collect()
        temporary.unlink(missing_ok=True)
    print(f"Climate extraction time: {time.perf_counter()-started:.1f} s",flush=True)
    return path
def process_cell(raw_monthly,month_positions,var_names,payload):
    n_years=len(TARGET_YEARS)
    n_vars=len(var_names)
    n_features=n_vars*len(WINDOWS)
    monthly=np.full((n_vars,n_years*12),np.nan,dtype=np.float64)
    for index,values in enumerate(raw_monthly):
        finite=np.isfinite(values)
        if not finite.any():
            continue
        sums=np.bincount(month_positions[finite],weights=values[finite],minlength=n_years*12)
        counts=np.bincount(month_positions[finite],minlength=n_years*12)
        np.divide(sums,counts,out=monthly[index],where=counts>0)
    monthly=monthly.reshape(n_vars,n_years,12)
    finite_months=np.isfinite(monthly)
    cumulative_values=np.concatenate((np.zeros((n_vars,n_years,1)),np.cumsum(np.where(finite_months,monthly,0.0),axis=2)),axis=2)
    cumulative_counts=np.concatenate((np.zeros((n_vars,n_years,1),dtype=int),np.cumsum(finite_months,axis=2)),axis=2)
    blocks=[]
    for first,last in WINDOWS:
        length=last-first+1
        values=(cumulative_values[:,:,last]-cumulative_values[:,:,first-1])/length
        counts=cumulative_counts[:,:,last]-cumulative_counts[:,:,first-1]
        blocks.append(np.where(counts==length,values,np.nan))
    features=np.stack(blocks,axis=1).reshape(n_features,n_years)
    finite_features=np.isfinite(features)
    results=[]
    for method,name,original in payload:
        y=original[TARGET_ROWS]
        valid=finite_features&np.isfinite(y)[None,:]
        counts=valid.sum(axis=1)
        eligible=counts>=MIN_PAIRS
        if not eligible.any():
            results.append((method,name,original.copy(),None,0,0,int((~np.isfinite(y)).sum()),"no_model"))
            continue
        x_mean=np.divide(np.where(valid,features,0.0).sum(axis=1),counts,out=np.zeros(n_features),where=counts>0)
        y_mean=np.divide(np.where(valid,y[None,:],0.0).sum(axis=1),counts,out=np.zeros(n_features),where=counts>0)
        x_centered=np.where(valid,features-x_mean[:,None],0.0)
        y_centered=np.where(valid,y[None,:]-y_mean[:,None],0.0)
        covariance=np.sum(x_centered*y_centered,axis=1)
        x_ss=np.sum(x_centered*x_centered,axis=1)
        y_ss=np.sum(y_centered*y_centered,axis=1)
        denominator=np.sqrt(x_ss*y_ss)
        correlations=np.divide(covariance,denominator,out=np.full(n_features,np.nan),where=eligible&(denominator>0))
        candidates=np.flatnonzero(np.isfinite(correlations))
        if candidates.size==0:
            results.append((method,name,original.copy(),None,0,0,int((~np.isfinite(y)).sum()),"no_model"))
            continue
        correlations=np.clip(correlations,-1.0,1.0)
        ranked=candidates[np.argsort(-np.abs(correlations[candidates]),kind="stable")]
        candidate_r=correlations[candidates]
        candidate_n=counts[candidates]
        with np.errstate(divide="ignore",invalid="ignore"):
            statistics=np.abs(candidate_r)*np.sqrt((candidate_n-2)/(1.0-candidate_r**2))
        p_values=np.full(n_features,np.nan)
        p_values[candidates]=2.0*student_t.sf(statistics,df=candidate_n-2)
        best=int(ranked[0])
        first,last=WINDOWS[best%len(WINDOWS)]
        month_label=f"M{first:02d}" if first==last else f"M{first:02d}_to_M{last:02d}"
        record={"climate_variable":var_names[best//len(WINDOWS)],"months":month_label,"R":float(correlations[best]),"P":float(p_values[best]),"paired_years":int(counts[best])}
        allowed=ranked[np.isfinite(p_values[ranked])&(p_values[ranked]<=P_THRESHOLD)]
        filled_y=y.copy()
        primary_count=0
        fallback_count=0
        for feature_index in allowed:
            missing=~np.isfinite(filled_y)
            if not missing.any():
                break
            positions=np.flatnonzero(missing&finite_features[feature_index])
            if positions.size==0:
                continue
            slope=covariance[feature_index]/x_ss[feature_index]
            intercept=y_mean[feature_index]-slope*x_mean[feature_index]
            if not np.isfinite(slope) or not np.isfinite(intercept):
                continue
            with np.errstate(over="ignore",invalid="ignore"):
                predictions=intercept+slope*features[feature_index,positions]
            usable=np.isfinite(predictions)
            filled_y[positions[usable]]=predictions[usable]
            if feature_index==best:
                primary_count+=int(usable.sum())
            else:
                fallback_count+=int(usable.sum())
        filled=original.copy()
        filled[TARGET_ROWS]=filled_y
        remaining=int((~np.isfinite(filled_y)).sum())
        status="eligible" if allowed.size else "no_significant_model"
        results.append((method,name,filled,record,primary_count,fallback_count,remaining,status))
    return results
def process_batch(cache_path,batch,month_positions,var_names):
    cache=np.load(cache_path,mmap_mode="r",allow_pickle=False)
    results=[]
    try:
        for cell_index,payload in batch:
            results.extend(process_cell(cache[cell_index],month_positions,var_names,payload))
    finally:
        del cache
    return results
def generate_batches(groups,growth):
    batch=[]
    for cell_index,names in enumerate(groups.values()):
        payload=[(method,name,growth[method][name]) for method in METHODS for name in names if name in growth[method]]
        batch.append((cell_index,payload))
        if len(batch)>=CELLS_PER_BATCH:
            yield batch
            batch=[]
    if batch:
        yield batch
def main():
    started=time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    sites,growth,selected,records,all_names=load_growth()
    groups,var_names,time_keep,month_positions,time_dim,lat_dim,lon_dim=inspect_climate(sites,all_names)
    cache_path=build_cache(groups,var_names,time_keep,time_dim,lat_dim,lon_dim)
    print(f"Computing: workers={N_JOBS}; cells per batch={CELLS_PER_BATCH}; maximum P={P_THRESHOLD}",flush=True)
    compute_started=time.perf_counter()
    with parallel_config(backend="loky",inner_max_num_threads=1):
        batch_results=Parallel(n_jobs=N_JOBS,batch_size=1,pre_dispatch=N_JOBS)(delayed(process_batch)(str(cache_path),batch,month_positions,var_names) for batch in generate_batches(groups,growth))
    print(f"Calculation time: {time.perf_counter()-compute_started:.1f} s",flush=True)
    filled_columns={method:{} for method in METHODS}
    primary_total={method:0 for method in METHODS}
    fallback_total={method:0 for method in METHODS}
    remaining_total={method:0 for method in METHODS}
    no_model={method:0 for method in METHODS}
    no_significant={method:0 for method in METHODS}
    for results in batch_results:
        for method,name,filled,record,primary_count,fallback_count,remaining,status in results:
            filled_columns[method][name]=filled
            if record is not None:
                records[(method,name)].update(record)
            primary_total[method]+=primary_count
            fallback_total[method]+=fallback_count
            remaining_total[method]+=remaining
            no_model[method]+=int(status=="no_model")
            no_significant[method]+=int(status=="no_significant_model")
    for method in METHODS:
        if not filled_columns[method]:
            print(f"[No output] {method}: no processed sites",flush=True)
            continue
        result=pd.DataFrame({"year":ALL_YEARS,**{name:filled_columns[method][name] for name in selected[method] if name in filled_columns[method]}})
        output_file=OUTPUT_DIR/f"{method}+fill.csv"
        result.to_csv(output_file,index=False,encoding="utf-8-sig")
        print(f"{method}: retained sites={len(filled_columns[method])}; no model={no_model[method]}; no significant model={no_significant[method]}; primary fills={primary_total[method]}; fallback fills={fallback_total[method]}; remaining missing={remaining_total[method]}",flush=True)
        for year in (2024,2025):
            row=result.loc[result["year"]==year].iloc[:,1:]
            values=row.to_numpy(dtype=float)
            missing=~np.isfinite(values)
            print(f"{method} {year}: valid sites={int(np.isfinite(values).sum())}; missing sites={int(missing.sum())}",flush=True)
        print("✅ Saved file:",output_file,flush=True)
    summary=pd.DataFrame(records.values(),columns=["method","site","climate_variable","months","R","P","paired_years"])
    summary_file=Path('D:/data/Processed data/Detrending/fill/best_climate_correlations.csv')
    summary.to_csv(summary_file,index=False,encoding="utf-8-sig")
    print("✅ Saved file:",summary_file,flush=True)
    print(f"Total time: {time.perf_counter()-started:.2f} s",flush=True)
if __name__=="__main__":
    main()