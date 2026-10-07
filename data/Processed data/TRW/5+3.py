import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import solve,pinvh,LinAlgWarning
from joblib import Parallel,delayed,parallel_config
from pykrige.ok import OrdinaryKriging
ROOT=Path("D:/")
POINTS_CSV=ROOT/"data"/"1. Basic information of global tree-ring sites.csv"
INPUT_DIR=Path('D:/data/Processed data/Detrending/fill')
MASK_FILE=Path('D:/data/Processed data/TRW/recon_mask (5+3).nc')
OUTPUT_DIR=Path('D:/data/Processed data/TRW')
METHODS=("AgeDepSpline","ModNegExp","RCS","SFRCS","Spline")
VARIOGRAM_MODEL="spherical"
YEAR_MIN,YEAR_MAX=1700,2026
N_NEIGHBORS=100
N_JOBS=5
REGIONS=((-11.0,57.0,30.0,75.0),(-167.27,-55.00,9.58,74.99))
def interpolate_year(method,year,coordinates,values,target_coordinates):
    started=time.perf_counter()
    valid=np.isfinite(values)&np.isfinite(coordinates).all(axis=1)
    raw_count=int(valid.sum())
    result={"year":int(year),"values":None,"raw_sites":raw_count,"unique_sites":0,"fallbacks":0,"seconds":0.0,"error":""}
    try:
        if len(target_coordinates)==0:
            result["values"]=np.empty(0,dtype=np.float32)
            return result
        if raw_count<3:
            raise ValueError("Fewer than three valid sites")
        xy,inverse=np.unique(coordinates[valid],axis=0,return_inverse=True)
        sums=np.bincount(inverse,weights=values[valid])
        counts=np.bincount(inverse)
        z=sums/counts
        result["unique_sites"]=len(xy)
        if len(xy)<3:
            raise ValueError("Fewer than three unique valid coordinates")
        if np.all(z==z[0]):
            result["values"]=np.full(len(target_coordinates),z[0],dtype=np.float32)
            return result
        model=OrdinaryKriging(x=xy[:,0],y=xy[:,1],z=z,variogram_model=VARIOGRAM_MODEL,coordinates_type="euclidean",enable_plotting=False,verbose=False,enable_statistics=False)
        parameters=np.asarray(model.variogram_model_parameters,dtype=np.float64)
        if not np.isfinite(parameters).all():
            raise ValueError("Nonfinite variogram parameters")
        variogram=model.variogram_function
        k=min(N_NEIGHBORS,len(xy))
        tree=cKDTree(xy)
        distances,neighbors=tree.query(target_coordinates,k=k)
        predicted=np.empty(len(target_coordinates),dtype=np.float64)
        matrix=np.empty((k+1,k+1),dtype=np.float64)
        matrix[-1,:]=1.0
        matrix[:,-1]=1.0
        matrix[-1,-1]=0.0
        rhs=np.empty(k+1,dtype=np.float64)
        rhs[-1]=1.0
        for index in range(len(target_coordinates)):
            ids=neighbors[index]
            local_distances=distances[index]
            pair_distances=cdist(xy[ids],xy[ids],metric="euclidean")
            matrix[:k,:k]=-variogram(parameters,pair_distances)
            np.fill_diagonal(matrix[:k,:k],0.0)
            rhs[:k]=-variogram(parameters,local_distances)
            rhs[:k][local_distances<=1e-10]=0.0
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error",LinAlgWarning)
                    weights=solve(matrix,rhs,assume_a="sym",check_finite=False)
                if not np.isfinite(weights).all():
                    raise np.linalg.LinAlgError("Nonfinite kriging weights")
            except (np.linalg.LinAlgError,LinAlgWarning):
                weights=pinvh(matrix,check_finite=False)@rhs
                result["fallbacks"]+=1
            predicted[index]=np.dot(weights[:k],z[ids])
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite interpolated values")
        result["values"]=predicted.astype(np.float32)
    except Exception as error:
        result["error"]=f"{type(error).__name__}: {error}"
    finally:
        result["seconds"]=time.perf_counter()-started
        status="FAILED" if result["error"] else "DONE"
        print(f"[{status}] {method} {year} | sites={raw_count} | unique={result['unique_sites']} | targets={len(target_coordinates)} | pinv={result['fallbacks']} | seconds={result['seconds']:.1f}",flush=True)
        if result["error"]:
            print(f"{method} {year}: {result['error']}",flush=True)
    return result
def load_mask():
    with xr.open_dataset(MASK_FILE) as ds:
        if "recon_mask" not in ds:
            raise ValueError("Variable recon_mask was not found")
        years=np.asarray(ds["year"].values,dtype=np.int32)
        latitude=np.asarray(ds["latitude"].values,dtype=np.float32)
        longitude=np.asarray(ds["longitude"].values,dtype=np.float32)
        selected=np.flatnonzero((years>=YEAR_MIN)&(years<=YEAR_MAX))
        years=years[selected]
        mask=ds["recon_mask"].transpose("year","latitude","longitude").isel(year=selected).values
    if len(years)==0 or len(np.unique(years))!=len(years):
        raise ValueError("Mask years are empty or duplicated")
    if latitude.ndim!=1 or longitude.ndim!=1:
        raise ValueError("Grid coordinates must be one-dimensional")
    grid_lon,grid_lat=np.meshgrid(longitude,latitude)
    region_mask=np.zeros(grid_lon.shape,dtype=bool)
    for lon_min,lon_max,lat_min,lat_max in REGIONS:
        region_mask|=(grid_lon>=lon_min)&(grid_lon<=lon_max)&(grid_lat>=lat_min)&(grid_lat<=lat_max)
    targets={}
    for index,year in enumerate(years):
        valid=region_mask&np.isfinite(mask[index])&(mask[index]>0.5)
        flat_indices=np.flatnonzero(valid.ravel())
        xy=np.column_stack((grid_lon.ravel()[flat_indices],grid_lat.ravel()[flat_indices])).astype(np.float64)
        targets[int(year)]=(flat_indices,xy)
    return years,latitude,longitude,targets
def process_method(method,points,mask_years,latitude,longitude,targets):
    started=time.perf_counter()
    input_path=INPUT_DIR/f"{method}+fill.csv"
    output_path=OUTPUT_DIR/f"{method} (5+3).nc"
    temporary_path=OUTPUT_DIR/f"{method} (5+3).partial.nc"
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    table=pd.read_csv(input_path)
    table=table.rename(columns={table.columns[0]:"year"})
    year_values=pd.to_numeric(table["year"],errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(year_values).all() or not np.equal(year_values,np.floor(year_values)).all():
        raise ValueError("Years must be finite integers")
    table["year"]=year_values.astype(np.int32)
    if table["year"].duplicated().any():
        raise ValueError("Duplicate years in input table")
    table=table.set_index("year").sort_index()
    site_names=points["name"].to_numpy()
    matched=np.array([name in table.columns for name in site_names],dtype=bool)
    if not matched.any():
        raise ValueError("No matching site names")
    names=site_names[matched].tolist()
    coordinates=points.loc[matched,["longitude","latitude"]].to_numpy(dtype=np.float64)
    common_years=np.intersect1d(table.index.to_numpy(dtype=np.int32),mask_years)
    if len(common_years)==0:
        raise ValueError("No overlapping years between input and mask")
    values=table.loc[common_years,names].to_numpy(dtype=np.float64)
    values[(values==0)|~np.isfinite(values)]=np.nan
    print("Input:",input_path,flush=True)
    print(f"{method}: matched sites={len(names)}, years={len(common_years)}, workers={N_JOBS}, neighbors={N_NEIGHBORS}",flush=True)
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results=Parallel(n_jobs=N_JOBS,batch_size=1,pre_dispatch=N_JOBS)(delayed(interpolate_year)(method,int(year),coordinates,values[index],targets[int(year)][1]) for index,year in enumerate(common_years))
    output_values=np.full((len(common_years),len(latitude),len(longitude)),np.nan,dtype=np.float32)
    failed_years=[]
    fallback_count=0
    for index,result in enumerate(results):
        if result["error"]:
            failed_years.append(result["year"])
            continue
        flat_indices=targets[result["year"]][0]
        output_values[index].reshape(-1)[flat_indices]=result["values"]
        fallback_count+=result["fallbacks"]
    if failed_years:
        raise RuntimeError(f"Failed years: {failed_years}. Output was not replaced")
    if not np.isfinite(output_values).any():
        raise ValueError("No valid output cells")
    output=xr.Dataset(data_vars={method:(("year","latitude","longitude"),output_values)},coords={"year":common_years.astype(np.int32),"latitude":latitude,"longitude":longitude},attrs={"description":f"Local ordinary kriging of gap-filled {method} data within the target regions and reconstruction mask","source_data":str(input_path),"source_mask":str(MASK_FILE),"variogram_model":VARIOGRAM_MODEL,"variogram_fit":"All unique valid site coordinates in each year","duplicate_coordinates":"Annual valid values averaged at identical coordinates","prediction":"Nearest valid site coordinates for each target cell","maximum_neighbors":N_NEIGHBORS,"coordinates_type":"euclidean","distance_units":"degrees","parallel_workers":N_JOBS,"pseudoinverse_fallback_count":fallback_count})
    output["latitude"].attrs["units"]="degrees_north"
    output["longitude"].attrs["units"]="degrees_east"
    output[method].attrs["units"]="same as input"
    output[method].encoding.update({"zlib":True,"complevel":4,"_FillValue":np.float32(np.nan)})
    try:
        output.to_netcdf(temporary_path,engine="netcdf4")
        output.close()
        temporary_path.replace(output_path)
    finally:
        output.close()
        temporary_path.unlink(missing_ok=True)
    print(f"{method}: completed in {(time.perf_counter()-started)/60:.1f} minutes",flush=True)
    print("✅ Saved file:",output_path,flush=True)
def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    points=pd.read_csv(POINTS_CSV,usecols=["name","latitude","longitude"])
    points=points.dropna(subset=["name","latitude","longitude"]).reset_index(drop=True)
    points["name"]=points["name"].astype(str)
    points["latitude"]=pd.to_numeric(points["latitude"],errors="raise")
    points["longitude"]=pd.to_numeric(points["longitude"],errors="raise")
    points=points.loc[np.isfinite(points["latitude"])&np.isfinite(points["longitude"])].reset_index(drop=True)
    if points.empty:
        raise ValueError("No valid site coordinates")
    mask_years,latitude,longitude,targets=load_mask()
    print(f"Grid: {len(latitude)} latitudes x {len(longitude)} longitudes",flush=True)
    print(f"Workers: {N_JOBS}, maximum neighbors: {N_NEIGHBORS}",flush=True)
    failed_methods=[]
    for index,method in enumerate(METHODS,1):
        print(f"[{index}/{len(METHODS)}] Processing {method}",flush=True)
        try:
            process_method(method,points,mask_years,latitude,longitude,targets)
        except Exception as error:
            failed_methods.append(method)
            print(f"[Failed] {method}: {type(error).__name__}: {error}",flush=True)
    print(f"Processing finished: {len(METHODS)-len(failed_methods)}/{len(METHODS)} files saved",flush=True)
    if failed_methods:
        print("Failed methods:",", ".join(failed_methods),flush=True)
if __name__=="__main__":
    main()