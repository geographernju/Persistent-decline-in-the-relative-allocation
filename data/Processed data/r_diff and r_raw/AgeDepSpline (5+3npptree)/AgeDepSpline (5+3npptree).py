import numpy as np,xarray as xr,geopandas as gpd,matplotlib.pyplot as plt,matplotlib as mpl,pandas as pd
from pathlib import Path
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box
ROOT=Path("D:/")
recon_path=Path('D:/data/Processed data/TRW/AgeDepSpline (5+3).nc')
model_paths=[Path('D:/data/NPP and GPP/S2_npptree9-8/ISBA-CTRIP_S2_nppTree.nc'),Path('D:/data/NPP and GPP/S2_npp9-8/SDGVM_S2_npp.nc'),Path('D:/data/NPP and GPP/S2_npp9-8/JULES_S2_npp.nc')]
base_out=Path('D:/data/Processed data/r_diff and r_raw/AgeDepSpline (5+3npptree)')
YEAR_MIN,YEAR_MAX=1900,2025
WIN=21
MIN_N=10
regions={"NA":{"shp":ROOT/"data"/"map"/"Koppen_1991_2020_NA_big7.shp","lon_min":-167.27,"lon_max":-55.0,"lat_min":9.58,"lat_max":74.99,"base":[1,2,3,4,5,6,7],"plot":[1,2,3,4,5,102,103,104],"agg":{102:[1,2,3,4,5],103:[6,7],104:[1,2,3,4,5,6,7]}},"EU":{"shp":ROOT/"data"/"map"/"Europe 7.shp","lon_min":-11.0,"lon_max":57.0,"lat_min":30.0,"lat_max":75.0,"base":[2,3,4,5,6,7],"plot":[2,3,4,5,102,103,104],"agg":{102:[2,3,4,5],103:[6,7],104:[2,3,4,5,6,7]}}}
label_map={1:"A",2:"B",3:"C",4:"Dfa",5:"Dfb",6:"D other",7:"E",102:"Warm",103:"Cold",104:"All"}
output_defs={"NA":{"name":"North America r_diff","png":"North_America_21-year_diff_correlation_1900-2025.png","shape":(2,4),"figsize":(16,6.5)},"EU":{"name":"Europe r_diff","png":"Europe_21-year_diff_correlation_1900-2025.png","shape":(2,4),"figsize":(16,7)},"ALL":{"name":"All r_diff","png":"All_21-year_diff_correlation_1900-2025.png","shape":(2,5),"figsize":(22,8.5)}}
mpl.rcParams["font.family"]="Times New Roman"
mpl.rcParams["axes.unicode_minus"]=False
mpl.rcParams["font.size"]=16
def subset_box(da,lon_min,lon_max,lat_min,lat_max):
    lon=da["longitude"].values
    lat=da["latitude"].values
    lon_slice=slice(lon_min,lon_max) if lon[0]<=lon[-1] else slice(lon_max,lon_min)
    lat_slice=slice(lat_min,lat_max) if lat[0]<=lat[-1] else slice(lat_max,lat_min)
    return da.sel(longitude=lon_slice,latitude=lat_slice)
def rasterize_classes(path,lon,lat,lon_min,lon_max,lat_min,lat_max):
    gdf=gpd.read_file(path)
    if gdf.crs is None:gdf=gdf.set_crs(epsg=4326)
    else:gdf=gdf.to_crs(epsg=4326)
    gdf=gpd.clip(gdf,box(lon_min,lat_min,lon_max,lat_max))
    prefer=["code","CODE","Code","class","Class","CLASS","label","Label"]
    col=next((c for c in prefer if c in gdf.columns),None)
    if col is None:col=next(c for c in gdf.columns if c.lower() not in ("geometry","geom","id","fid","ogc_fid"))
    if pd.api.types.is_numeric_dtype(gdf[col]):gdf["__code__"]=pd.to_numeric(gdf[col],errors="coerce").astype("Int16")
    else:
        values=gdf[col].astype(str).str.strip().str.extract(r"^(Dfa|Dfb|[ABCDE])(?:\s|$)",expand=False)
        gdf["__code__"]=values.map({"A":1,"B":2,"C":3,"Dfa":4,"Dfb":5,"D":6,"E":7}).astype("Int16")
    gdf=gdf.dropna(subset=["__code__"])
    transform=from_bounds(float(lon.min()),float(lat.min()),float(lon.max()),float(lat.max()),len(lon),len(lat))
    grid=rasterize([(geom,int(value)) for geom,value in zip(gdf.geometry,gdf["__code__"])],out_shape=(len(lat),len(lon)),transform=transform,fill=0,dtype="int16")
    if lat[0]<lat[-1]:grid=np.flipud(grid)
    return grid
def zone_mask(grid,code,region):
    if code in [1,2,3,4,5,6,7]:return grid==code
    return np.isin(grid,region["agg"][code])
def get_recon_da(ds):
    candidates=[name for name in ds.data_vars if "year" in ds[name].dims]
    preferred=[name for name in candidates if "recon" in name.lower() or "agedep" in name.lower()]
    if not candidates:raise ValueError("No tree-ring variable with year dimension found")
    return ds[preferred[0] if preferred else candidates[0]]
def get_model_var(ds):
    preferred=["npp","NPP"]
    for name in preferred:
        if name in ds.data_vars and "year" in ds[name].dims:return name
    candidates=[name for name in ds.data_vars if "year" in ds[name].dims and "npp" in name.lower()]
    if candidates:return candidates[0]
    candidates=[name for name in ds.data_vars if "year" in ds[name].dims]
    if candidates:return candidates[0]
    raise ValueError("No model variable with year dimension found")
def rolling_corr_from_diff(a,b,window,min_n):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    length=a.shape[0]-window+1
    result=np.full((length,)+a.shape[1:],np.nan,dtype=np.float32)
    for t in range(length):
        x=a[t:t+window]
        y=b[t:t+window]
        valid=np.isfinite(x)&np.isfinite(y)
        n=valid.sum(axis=0)
        sx=np.where(valid,x,0.0).sum(axis=0)
        sy=np.where(valid,y,0.0).sum(axis=0)
        mx=np.divide(sx,n,out=np.zeros_like(sx),where=n>0)
        my=np.divide(sy,n,out=np.zeros_like(sy),where=n>0)
        xc=np.where(valid,x-mx,0.0)
        yc=np.where(valid,y-my,0.0)
        vx=np.sum(xc*xc,axis=0)
        vy=np.sum(yc*yc,axis=0)
        cov=np.sum(xc*yc,axis=0)
        denominator=np.sqrt(vx*vy)
        usable=(n>=min_n)&(denominator>0)&np.isfinite(denominator)
        r=np.full(denominator.shape,np.nan,dtype=np.float32)
        r[usable]=np.clip(cov[usable]/denominator[usable],-1.0,1.0)
        result[t]=r
    return result
def frame_sum_count(correlations,mask):
    sums=np.full(correlations.shape[0],np.nan,dtype=np.float64)
    counts=np.zeros(correlations.shape[0],dtype=np.int64)
    for i,frame in enumerate(correlations):
        valid=mask&np.isfinite(frame)
        counts[i]=int(valid.sum())
        if counts[i]>0:sums[i]=float(frame[valid].sum(dtype=np.float64))
    return sums,counts
def sum_count_to_mean(years,sums,counts):
    values=np.divide(sums,counts,out=np.full(len(years),np.nan,dtype=np.float64),where=counts>0)
    return np.asarray(years,dtype=int),values
def combine_region_stats(stats):
    all_years=np.array(sorted(set().union(*(set(years.tolist()) for years,_,_ in stats))),dtype=int)
    total_sum=np.zeros(all_years.size,dtype=np.float64)
    total_count=np.zeros(all_years.size,dtype=np.int64)
    index={year:i for i,year in enumerate(all_years)}
    for years,sums,counts in stats:
        for year,value,count in zip(years,sums,counts):
            if count>0 and np.isfinite(value):
                j=index[int(year)]
                total_sum[j]+=value
                total_count[j]+=count
    values=np.divide(total_sum,total_count,out=np.full(all_years.size,np.nan,dtype=np.float64),where=total_count>0)
    return all_years,values
def mean_model_series(series_list):
    all_years=np.array(sorted(set().union(*(set(years.tolist()) for years,values in series_list))),dtype=int)
    matrix=np.full((len(series_list),len(all_years)),np.nan,dtype=np.float64)
    index={year:i for i,year in enumerate(all_years)}
    for row,(years,values) in enumerate(series_list):
        for year,value in zip(years,values):
            if np.isfinite(value):matrix[row,index[int(year)]]=value
    count=np.isfinite(matrix).sum(axis=0)
    values=np.divide(np.nansum(matrix,axis=0),count,out=np.full(len(all_years),np.nan,dtype=np.float64),where=count>0)
    return all_years,values
def write_csv(out_dir,label,years,values):
    out_dir.mkdir(parents=True,exist_ok=True)
    with open(out_dir/f"{label.replace(' ','_')}.csv","w",encoding="utf-8") as output:
        output.write("year,mean_r\n")
        for year,value in zip(years,values):
            if np.isfinite(value):output.write(f"{int(year)},{value:.6f}\n")
def plot_result(series,plot_order,out_key):
    cfg=output_defs[out_key]
    out_dir=base_out/cfg["name"]
    out_dir.mkdir(parents=True,exist_ok=True)
    finite_all=[values[np.isfinite(values)] for years,values in series.values() if np.isfinite(values).any()]
    if finite_all:
        global_min=min(float(v.min()) for v in finite_all)
        global_max=max(float(v.max()) for v in finite_all)
        global_range=global_max-global_min
        if global_range<=0:global_range=0.2
    else:global_range=0.2
    fig,axes=plt.subplots(cfg["shape"][0],cfg["shape"][1],figsize=cfg["figsize"],constrained_layout=True)
    axes=np.asarray(axes).ravel()
    for index,code in enumerate(plot_order):
        ax=axes[index]
        label=label_map[code]
        if code not in series:
            ax.text(0.5,0.5,"No data",ha="center",va="center",fontsize=16)
            ax.set_axis_off()
            continue
        years,values=series[code]
        valid=values[np.isfinite(values)]
        if not valid.size:
            ax.text(0.5,0.5,"No data",ha="center",va="center",fontsize=16)
            ax.set_axis_off()
            continue
        ax.plot(years,values,linewidth=2.2,label="Top 3 mean")
        midpoint=0.5*(float(valid.min())+float(valid.max()))
        ax.set_ylim(midpoint-global_range/2,midpoint+global_range/2)
        ax.margins(x=0.01)
        ax.set_xlabel("Year",fontsize=16)
        ax.set_ylabel("Mean r",fontsize=16)
        ax.tick_params(axis="both",labelsize=14)
        ax.text(0.02,0.92,label,transform=ax.transAxes,ha="left",va="top",fontsize=16,fontweight="bold")
        ax.legend(fontsize=12,frameon=True,loc="lower right",borderpad=0.2,handlelength=1.5,handletextpad=0.5)
        write_csv(out_dir,label,years,values)
    for index in range(len(plot_order),len(axes)):fig.delaxes(axes[index])
    out_png=out_dir/cfg["png"]
    fig.savefig(out_png,dpi=600,bbox_inches="tight")
    plt.close(fig)
    print('✅ Saved figure:', out_png)
def main():
    if not recon_path.exists():raise FileNotFoundError(recon_path)
    for model_path in model_paths:
        if not model_path.exists():raise FileNotFoundError(model_path)
    region_data={}
    with xr.open_dataset(recon_path) as ds:
        recon=get_recon_da(ds).sel(year=slice(YEAR_MIN,YEAR_MAX))
        for region_name,region in regions.items():
            tree=subset_box(recon,region["lon_min"],region["lon_max"],region["lat_min"],region["lat_max"]).load()
            class_grid=rasterize_classes(region["shp"],tree["longitude"].values,tree["latitude"].values,region["lon_min"],region["lon_max"],region["lat_min"],region["lat_max"])
            region_data[region_name]={"tree":tree,"class_grid":class_grid}
    model_region_series=[]
    model_all_series=[]
    for model_path in model_paths:
        print("Processing:",model_path.name)
        region_stats={"NA":{},"EU":{}}
        with xr.open_dataset(model_path) as ds:
            variable=get_model_var(ds)
            print("Model variable:",variable)
            model_all=ds[variable].sel(year=slice(YEAR_MIN,YEAR_MAX))
            for region_name,region in regions.items():
                tree=region_data[region_name]["tree"]
                class_grid=region_data[region_name]["class_grid"]
                model=subset_box(model_all,region["lon_min"],region["lon_max"],region["lat_min"],region["lat_max"])
                tree_years=tree["year"].values.astype(int)
                model_years=model["year"].values.astype(int)
                common=np.intersect1d(tree_years,model_years)
                if common.size<WIN+1:raise ValueError(f"Insufficient common years for {region_name}: {common.size}")
                if not np.all(np.diff(common)==1):raise ValueError(f"Common years are not continuous for {region_name}")
                a=tree.sel(year=common)
                b=model.sel(year=common)
                same_grid=b.sizes.get("latitude")==a.sizes["latitude"] and b.sizes.get("longitude")==a.sizes["longitude"]
                if same_grid:same_grid=np.allclose(b["latitude"],a["latitude"]) and np.allclose(b["longitude"],a["longitude"])
                if not same_grid:b=b.interp(latitude=a["latitude"],longitude=a["longitude"])
                a_values=a.values.astype(np.float64)
                b_values=b.load().values.astype(np.float64)
                a_diff=np.diff(a_values,axis=0)
                b_diff=np.diff(b_values,axis=0)
                correlations=rolling_corr_from_diff(a_diff,b_diff,WIN,MIN_N)
                diff_years=common[1:]
                years=diff_years[WIN//2:WIN//2+correlations.shape[0]]
                for code in region["plot"]:
                    mask=zone_mask(class_grid,code,region)
                    sums,counts=frame_sum_count(correlations,mask)
                    region_stats[region_name][code]=(years,sums,counts)
        na_series={}
        for code in regions["NA"]["plot"]:
            if code in region_stats["NA"]:
                years,sums,counts=region_stats["NA"][code]
                na_series[code]=sum_count_to_mean(years,sums,counts)
        eu_series={}
        for code in regions["EU"]["plot"]:
            if code in region_stats["EU"]:
                years,sums,counts=region_stats["EU"][code]
                eu_series[code]=sum_count_to_mean(years,sums,counts)
        all_plot=[1,2,3,4,5,6,7,102,103,104]
        all_series={}
        for code in all_plot:
            stats=[]
            if code in region_stats["NA"]:stats.append(region_stats["NA"][code])
            if code in region_stats["EU"]:stats.append(region_stats["EU"][code])
            if stats:all_series[code]=combine_region_stats(stats)
        model_region_series.append({"NA":na_series,"EU":eu_series})
        model_all_series.append(all_series)
    na_final={}
    for code in regions["NA"]["plot"]:
        available=[item["NA"][code] for item in model_region_series if code in item["NA"]]
        if available:na_final[code]=mean_model_series(available)
    eu_final={}
    for code in regions["EU"]["plot"]:
        available=[item["EU"][code] for item in model_region_series if code in item["EU"]]
        if available:eu_final[code]=mean_model_series(available)
    all_plot=[1,2,3,4,5,6,7,102,103,104]
    all_final={}
    for code in all_plot:
        available=[item[code] for item in model_all_series if code in item]
        if available:all_final[code]=mean_model_series(available)
    plot_result(na_final,regions["NA"]["plot"],"NA")
    plot_result(eu_final,regions["EU"]["plot"],"EU")
    plot_result(all_final,all_plot,"ALL")
    print("Done")
if __name__=="__main__":main()