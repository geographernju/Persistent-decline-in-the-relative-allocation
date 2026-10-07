import numpy as np,xarray as xr,geopandas as gpd,matplotlib.pyplot as plt,matplotlib as mpl,pandas as pd
from pathlib import Path
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box
ROOT=Path("D:/")
recon_path=Path('D:/data/Processed data/TRW/SFRCS (5+3).nc')
model_dir=Path('D:/data/NPP and GPP/S2_npp9-8')
YEAR_MIN,YEAR_MAX=1900,2025
WIN=21
MIN_N=10
mpl.rcParams["font.family"]="Times New Roman"
mpl.rcParams["axes.unicode_minus"]=False
mpl.rcParams["font.size"]=18
REGIONS={"EU":{"shp":ROOT/"data"/"map"/"Europe 7.shp","lon_min":-11.0,"lon_max":57.0,"lat_min":30.0,"lat_max":75.0,"base_classes":[2,3,4,5,6,7],"agg":{102:[2,3,4,5],103:[6,7],104:[2,3,4,5,6,7]},"plot_order":[2,3,4,5,102,103,104],"label_map":{2:"B",3:"C",4:"Dfa",5:"Dfb",102:"Warm",103:"Cold",104:"All"}},"NA":{"shp":ROOT/"data"/"map"/"Koppen_1991_2020_NA_big7.shp","lon_min":-167.27,"lon_max":-55.0,"lat_min":9.58,"lat_max":74.99,"base_classes":[1,2,3,4,5],"agg":{102:[1,2,3,4,5],103:[6,7],104:[1,2,3,4,5,6,7]},"plot_order":[1,2,3,4,5,102,103,104],"label_map":{1:"A",2:"B",3:"C",4:"Dfa",5:"Dfb",102:"Warm",103:"Cold",104:"All"}}}
ALL_PLOT_ORDER=[1,2,3,4,5,6,7,102,103,104]
ALL_LABEL_MAP={1:"A",2:"B",3:"C",4:"Dfa",5:"Dfb",6:"D other",7:"E",102:"Warm",103:"Cold",104:"All"}
OUTPUTS={"EU":{"dir":Path('D:/data/Processed data/r_diff and r_raw/SFRCS (5+3)/Europe r_diff'),"png":"Europe_21-year_diff_correlation_1900-2025.png","figsize":(16,7),"shape":(2,4)},"NA":{"dir":Path('D:/data/Processed data/r_diff and r_raw/SFRCS (5+3)/North America r_diff'),"png":"North_America_21-year_diff_correlation_1900-2025.png","figsize":(16,6.5),"shape":(2,4)},"ALL":{"dir":Path('D:/data/Processed data/r_diff and r_raw/SFRCS (5+3)/All r_diff'),"png":"All_21-year_diff_correlation_1900-2025.png","figsize":(22,8.5),"shape":(2,5)}}
LEGEND_POS={"EU":{2:"lower left",3:"lower left",4:"lower left",5:"lower right",102:"lower left",103:"lower right",104:"lower left"},"NA":{1:"lower right",2:"lower right",3:"lower center",4:"lower center",5:"lower right",102:"lower right",103:"lower right",104:"lower right"},"ALL":{1:"lower right",2:"lower left",3:"lower center",4:"lower center",5:"lower right",6:"lower right",7:"lower right",102:"lower left",103:"lower right",104:"lower right"}}
def subset_box(da,cfg):
    lon=da["longitude"].values
    lat=da["latitude"].values
    lon_slice=slice(cfg["lon_min"],cfg["lon_max"]) if lon[0]<=lon[-1] else slice(cfg["lon_max"],cfg["lon_min"])
    lat_slice=slice(cfg["lat_min"],cfg["lat_max"]) if lat[0]<=lat[-1] else slice(cfg["lat_max"],cfg["lat_min"])
    return da.sel(longitude=lon_slice,latitude=lat_slice)
def rasterize_classes(path,lon,lat,cfg):
    gdf=gpd.read_file(path)
    if gdf.crs is None:gdf=gdf.set_crs(epsg=4326)
    else:gdf=gdf.to_crs(epsg=4326)
    gdf=gpd.clip(gdf,box(cfg["lon_min"],cfg["lat_min"],cfg["lon_max"],cfg["lat_max"]))
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
def zone_mask(grid,code,cfg):
    if code in [1,2,3,4,5,6,7]:return grid==code
    return np.isin(grid,cfg["agg"][code])
def get_recon_da(ds):
    candidates=[name for name in ds.data_vars if "year" in ds[name].dims]
    preferred=[name for name in candidates if "recon" in name.lower() or "agedep" in name.lower()]
    return ds[preferred[0] if preferred else candidates[0]]
def get_model_var(ds):
    candidates=[name for name in ds.data_vars if "year" in ds[name].dims and name.lower()=="npp"]
    return candidates[0] if candidates else None
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
def sums_counts_to_values(years,sums,counts):
    values=np.divide(sums,counts,out=np.full(len(years),np.nan,dtype=np.float64),where=counts>0)
    return years,values
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
def calculate_region(region_name,tree,model_all,class_grid):
    cfg=REGIONS[region_name]
    model=subset_box(model_all,cfg)
    tree_years=tree["year"].values.astype(int)
    model_years=model["year"].values.astype(int)
    common=np.intersect1d(tree_years,model_years)
    if common.size<WIN+1:return None
    if not np.all(np.diff(common)==1):return None
    a=tree.sel(year=common)
    b=model.sel(year=common)
    same_grid=b.sizes.get("latitude")==a.sizes["latitude"] and b.sizes.get("longitude")==a.sizes["longitude"]
    if same_grid:same_grid=np.allclose(b["latitude"],a["latitude"]) and np.allclose(b["longitude"],a["longitude"])
    if not same_grid:b=b.interp(latitude=a["latitude"],longitude=a["longitude"])
    a_values=a.values.astype(np.float64)
    b_values=b.load().values.astype(np.float64)
    correlations=rolling_corr_from_diff(np.diff(a_values,axis=0),np.diff(b_values,axis=0),WIN,MIN_N)
    diff_years=common[1:]
    years=diff_years[WIN//2:WIN//2+correlations.shape[0]]
    result={}
    codes=sorted(set(ALL_PLOT_ORDER)|set(cfg["plot_order"]))
    for code in codes:
        if code in [102,103,104] and code not in cfg["agg"]:continue
        mask=zone_mask(class_grid,code,cfg)
        sums,counts=frame_sum_count(correlations,mask)
        result[code]=(years,sums,counts)
    return result
def select_top3(series,plot_order):
    selected={}
    minima={}
    maxima={}
    for code in plot_order:
        ranked=[(name,float(np.nanmean(values))) for name,(years,values) in series[code].items() if np.isfinite(values).any()]
        ranked.sort(key=lambda item:item[1],reverse=True)
        selected[code]=[name for name,score in ranked[:3]]
        plotted=[series[code][name][1] for name in selected[code]]
        finite=[values[np.isfinite(values)] for values in plotted if np.isfinite(values).any()]
        minima[code]=min(float(values.min()) for values in finite) if finite else np.nan
        maxima[code]=max(float(values.max()) for values in finite) if finite else np.nan
    return selected,minima,maxima
def plot_output(output_key,series,plot_order,label_map):
    cfg=OUTPUTS[output_key]
    out_dir=cfg["dir"]
    out_dir.mkdir(parents=True,exist_ok=True)
    out_png=out_dir/cfg["png"]
    selected,minima,maxima=select_top3(series,plot_order)
    ranges=[maxima[code]-minima[code] for code in plot_order if np.isfinite(minima[code]) and np.isfinite(maxima[code]) and maxima[code]>minima[code]]
    common_range=max(ranges) if ranges else 0.2
    fig,axes=plt.subplots(cfg["shape"][0],cfg["shape"][1],figsize=cfg["figsize"],constrained_layout=True)
    axes=np.asarray(axes).ravel()
    for index,code in enumerate(plot_order):
        ax=axes[index]
        names=selected[code]
        label=label_map[code]
        if not names:
            ax.text(0.5,0.5,"No data",ha="center",va="center",fontsize=18)
            ax.set_axis_off()
            continue
        for name in names:
            years,values=series[code][name]
            ax.plot(years,values,label=name,linewidth=1.6)
        common_years=sorted(set.intersection(*(set(series[code][name][0].tolist()) for name in names)))
        if len(common_years)>=2:
            curves=[]
            for name in names:
                years,values=series[code][name]
                curves.append(pd.Series(values,index=years).reindex(common_years).to_numpy(dtype=float))
            mean_curve=np.ma.masked_invalid(np.vstack(curves)).mean(axis=0).filled(np.nan)
            ax.plot(common_years,mean_curve,linewidth=2.8,label=f"Mean of top {len(names)}")
            csv_name=label.replace(" ","_") if output_key=="ALL" else label
            with open(out_dir/f"{csv_name}.csv","w",encoding="utf-8") as output:
                output.write("year,mean_r\n")
                for year,value in zip(common_years,mean_curve):
                    if np.isfinite(value):output.write(f"{int(year)},{value:.6f}\n")
        midpoint=0.5*(minima[code]+maxima[code]) if np.isfinite(minima[code]) and np.isfinite(maxima[code]) else 0.0
        ax.set_ylim(midpoint-common_range/2,midpoint+common_range/2)
        ax.margins(x=0.01)
        ax.set_xlabel("Year",fontsize=18)
        ax.set_ylabel("Mean r",fontsize=18)
        ax.tick_params(axis="both",labelsize=16)
        ax.text(0.02,0.92,label,transform=ax.transAxes,ha="left",va="top",fontsize=18,fontweight="bold")
        legend=ax.legend(fontsize=11,frameon=True,loc=LEGEND_POS[output_key].get(code,"lower right"),borderpad=0.05,handlelength=1.2,handletextpad=0.4)
        if legend is not None:
            try:legend.get_frame().set_boxstyle("sawtooth",pad=0.05)
            except Exception:pass
    for index in range(len(plot_order),len(axes)):fig.delaxes(axes[index])
    fig.savefig(out_png,dpi=600,bbox_inches="tight")
    plt.close(fig)
    print('✅ Saved figure:', out_png)
def main():
    files=sorted((path for path in model_dir.rglob("*") if path.is_file() and path.suffix.lower()==".nc"),key=lambda path:str(path).lower())
    if not files:raise FileNotFoundError(model_dir)
    region_data={}
    with xr.open_dataset(recon_path) as ds:
        recon=get_recon_da(ds).sel(year=slice(YEAR_MIN,YEAR_MAX))
        for region_name,cfg in REGIONS.items():
            tree=subset_box(recon,cfg).load()
            class_grid=rasterize_classes(cfg["shp"],tree["longitude"].values,tree["latitude"].values,cfg)
            region_data[region_name]={"tree":tree,"class_grid":class_grid}
    region_series={region_name:{code:{} for code in REGIONS[region_name]["plot_order"]} for region_name in REGIONS}
    all_series={code:{} for code in ALL_PLOT_ORDER}
    for path in files:
        model_name=path.relative_to(model_dir).with_suffix("").as_posix().replace("/","_")
        all_stats={code:[] for code in ALL_PLOT_ORDER}
        with xr.open_dataset(path) as ds:
            variable=get_model_var(ds)
            if variable is None:continue
            model_all=ds[variable].sel(year=slice(YEAR_MIN,YEAR_MAX))
            for region_name,cfg in REGIONS.items():
                result=calculate_region(region_name,region_data[region_name]["tree"],model_all,region_data[region_name]["class_grid"])
                if result is None:continue
                for code,(years,sums,counts) in result.items():
                    if code in region_series[region_name]:
                        region_series[region_name][code][model_name]=sums_counts_to_values(years,sums,counts)
                    if code in all_stats:all_stats[code].append((years,sums,counts))
        for code in ALL_PLOT_ORDER:
            if all_stats[code]:all_series[code][model_name]=combine_region_stats(all_stats[code])
    plot_output("EU",region_series["EU"],REGIONS["EU"]["plot_order"],REGIONS["EU"]["label_map"])
    plot_output("NA",region_series["NA"],REGIONS["NA"]["plot_order"],REGIONS["NA"]["label_map"])
    plot_output("ALL",all_series,ALL_PLOT_ORDER,ALL_LABEL_MAP)
if __name__=="__main__":main()