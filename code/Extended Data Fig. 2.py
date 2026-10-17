
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress,kruskal
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,MaxNLocator
ROOT=Path("D:/")
points_csv=ROOT/"data/1. Basic information of global tree-ring sites.csv"
trw_csv=ROOT/"data/Processed data/Detrending/fill/AgeDepSpline+fill.csv"
npp_paths=[("ISBA-CTRIP",ROOT/"data/NPP and GPP/S2_npp9-8/ISBA-CTRIP_S2_npp.nc"),("CLASSIC",ROOT/"data/NPP and GPP/S2_npp9-8/ CLASSIC_S2_npp.nc"),("iMAPLE",ROOT/"data/NPP and GPP/S2_npp9-8/iMAPLE_S2_npp.nc")]
map_path=ROOT/"data/Map/World_countries.shp"
out_dir=ROOT/"results"
out_png=ROOT/"results/Extended Data Fig. 2.png"
NPP_VAR=None
DISPLAY_START,DISPLAY_END=1900,2023
TREND_START,TREND_END=1945,2023
MIN_RATIO_YEARS=20
MIN_TREND_YEARS=30
MIN_TREND_SPAN=40
TOP_SPECIES=12
MAJOR_SHARE=0.80
ZERO_TRW_AS_MISSING=False
TRW_MISSING_VALUES=[]
na_extent=(-167.27,-55.0,9.58,74.99)
eu_extent=(-11.0,55.0,29.0,75.0)
colors={"Major":"#C43C39","Minor":"#E8AE55","Opposing":"#3578A8","Neutral":"#888888","Unavailable":"#BDBDBD"}
plant_markers={"Gymnosperm":"^","Angiosperm":"o","Unknown":"s"}
plt.ioff()
plt.rcParams.update({"font.family":"Times New Roman","font.size":21,"axes.labelsize":24,"xtick.labelsize":19,"ytick.labelsize":19,"legend.fontsize":18,"axes.linewidth":1.2,"savefig.dpi":600,"mathtext.fontset":"custom","mathtext.rm":"Times New Roman","mathtext.it":"Times New Roman:italic"})
def read_csv(path):
    for encoding in ("utf-8-sig","gb18030","cp1252"):
        try:
            frame=pd.read_csv(path,encoding=encoding)
            frame.columns=frame.columns.astype(str).str.strip()
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode CSV: {path}")
def inside(frame,extent):
    xmin,xmax,ymin,ymax=extent
    return frame["longitude"].between(xmin,xmax)&frame["latitude"].between(ymin,ymax)
def normalize_group(series,lookup):
    values=series.fillna("").astype(str).str.strip().str.lower()
    values=values.str.replace(r"[\s_\-/]+","",regex=True)
    return values.map(lookup).fillna("Unknown")
def panel_label(ax,letter):
    ax.text(0.015,0.975,letter,transform=ax.transAxes,ha="left",va="top",fontsize=28,fontweight="bold",zorder=20)
def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out",length=6,width=1.2)
def p_text(value,threshold=0.01):
    return rf"$P<{threshold:g}$" if value<threshold else rf"$P={value:.3f}$"
def group_test(arrays):
    if len(arrays)<2 or any(len(values)<2 for values in arrays):return None
    if np.ptp(np.concatenate(arrays))==0:return None
    return kruskal(*arrays)
pts=read_csv(points_csv)
required=["name","latitude","longitude","species","Gymnosperm/Angiosperm","Needle/broad"]
missing=[column for column in required if column not in pts.columns]
if missing:raise ValueError(f"Missing site columns: {missing}")
pts=pts.dropna(subset=["name"]).copy()
pts["name"]=pts["name"].astype(str).str.strip()
for column in ["latitude","longitude"]:
    pts[column]=pd.to_numeric(pts[column],errors="coerce")
pts=pts.dropna(subset=["latitude","longitude"])
pts=pts.loc[pts["latitude"].between(-90,90)&pts["longitude"].between(-180,180)]
pts=pts.loc[inside(pts,na_extent)|inside(pts,eu_extent)].copy()
pts=pts.drop_duplicates(subset=required)
if pts["name"].duplicated().any():
    duplicated=pts.loc[pts["name"].duplicated(keep=False),"name"].unique()
    raise ValueError(f"Conflicting duplicate site names: {duplicated[:20].tolist()}")
pts["species"]=pts["species"].fillna("Unknown").astype(str).str.strip().replace("","Unknown")
pts["plant_group"]=normalize_group(pts["Gymnosperm/Angiosperm"],{"gymnosperm":"Gymnosperm","gymnosperms":"Gymnosperm","angiosperm":"Angiosperm","angiosperms":"Angiosperm"})
pts["leaf_group"]=normalize_group(pts["Needle/broad"],{"needle":"Needle","needles":"Needle","needleleaf":"Needle","needleleaved":"Needle","broad":"Broad","broadleaf":"Broad","broadleaved":"Broad"})
pts["region"]=np.where(inside(pts,na_extent),"NA","EU")
trw=read_csv(trw_csv)
if "year" not in trw.columns:raise ValueError("The tree-ring CSV must contain a year column.")
if trw.columns.duplicated().any():raise ValueError("Duplicate tree-ring column names were found.")
trw["year"]=pd.to_numeric(trw["year"],errors="coerce")
trw=trw.dropna(subset=["year"])
if not np.all(np.isfinite(trw["year"])&(trw["year"]==np.floor(trw["year"]))):raise ValueError("Tree-ring years must be finite integers.")
trw["year"]=trw["year"].astype(int)
if trw["year"].duplicated().any():raise ValueError("Duplicate tree-ring years were found.")
trw=trw.set_index("year").sort_index()
trw=trw.loc[(trw.index>=DISPLAY_START)&(trw.index<=DISPLAY_END)]
pts=pts.loc[pts["name"].isin(trw.columns)].reset_index(drop=True)
if pts.empty:raise ValueError("No matching sites remain within NA and EU.")
all_sites=pts.copy()
names=pts["name"].tolist()
trw=trw[names].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
if TRW_MISSING_VALUES:trw=trw.replace(TRW_MISSING_VALUES,np.nan)
if ZERO_TRW_AS_MISSING:trw=trw.mask(trw==0)
full_years=np.arange(DISPLAY_START,DISPLAY_END+1)
def load_npp_sites(path):
    if not path.exists() and path.name.startswith(" ") and path.with_name(path.name.lstrip()).exists():path=path.with_name(path.name.lstrip())
    if not path.exists():raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        if NPP_VAR is None:
            candidates=[name for name in ds.data_vars if "npp" in name.lower() and not any(token in name.lower() for token in ["bnd","bound"])]
            if len(candidates)!=1:raise ValueError(f"Set NPP_VAR explicitly for {path.name}: {candidates}")
            selected_var=candidates[0]
        else:selected_var=NPP_VAR
        da=ds[selected_var]
        renames={}
        for standard,aliases in {"latitude":["latitude","lat"],"longitude":["longitude","lon"],"time":["time","year"]}.items():
            found=[name for name in aliases if name in da.dims]
            if len(found)!=1:raise ValueError(f"Cannot identify {standard} dimension in {da.dims}.")
            if found[0]!=standard:renames[found[0]]=standard
        da=da.rename(renames)
        for dim in list(da.dims):
            if dim not in ["time","latitude","longitude"]:
                if da.sizes[dim]!=1:raise ValueError(f"Select a level for extra NPP dimension: {dim}")
                da=da.isel({dim:0},drop=True)
        if da["latitude"].ndim!=1 or da["longitude"].ndim!=1:raise ValueError("One-dimensional spatial coordinates required")
        da=da.sortby("latitude").sortby("longitude")
        if pd.Index(da["latitude"].values).has_duplicates or pd.Index(da["longitude"].values).has_duplicates:raise ValueError("Duplicate NPP spatial coordinates")
        numeric_time=np.issubdtype(da["time"].dtype,np.number)
        if numeric_time:
            time_values=da["time"].values.astype(float)
            if not np.all(np.isfinite(time_values)&(time_values==np.floor(time_values))&(time_values>=1000)&(time_values<=3000)):raise ValueError("Numeric NPP time is not a year coordinate")
            npp_years=time_values.astype(int)
        else:npp_years=da["time"].dt.year.values.astype(int)
        selected_time=(npp_years>=DISPLAY_START)&(npp_years<=DISPLAY_END)
        if not selected_time.any():raise ValueError(f"No NPP data in {DISPLAY_START}–{DISPLAY_END}: {path.name}")
        da=da.isel(time=np.flatnonzero(selected_time));npp_years=npp_years[selected_time]
        lon_values=pts["longitude"].to_numpy(dtype=float)
        grid_lons=da["longitude"].values
        if grid_lons.min()>=0 and grid_lons.max()>180:lon_values=lon_values%360
        inside_grid=pts["latitude"].between(float(da["latitude"].min()),float(da["latitude"].max())).to_numpy()&(lon_values>=grid_lons.min())&(lon_values<=grid_lons.max())
        analysis_pts=pts.loc[inside_grid].reset_index(drop=True);lon_values=lon_values[inside_grid]
        if analysis_pts.empty:raise ValueError(f"No sites overlap NPP grid: {path.name}")
        analysis_names=analysis_pts["name"].tolist()
        lat_indexer=xr.DataArray(analysis_pts["latitude"].to_numpy(),dims="site")
        lon_indexer=xr.DataArray(lon_values,dims="site")
        sampled=da.sel(latitude=lat_indexer,longitude=lon_indexer,method="nearest").transpose("time","site")
        sampled_values=sampled.load().values.astype(float)
        sampled_values[~np.isfinite(sampled_values)]=np.nan
        npp_frame=pd.DataFrame(sampled_values,index=npp_years,columns=analysis_names)
        if pd.Series(npp_years).duplicated().any():
            if numeric_time:raise ValueError("Repeated numeric NPP years require explicit temporal aggregation")
            months=da["time"].dt.month.values.astype(int)
            if pd.MultiIndex.from_arrays([npp_years,months]).has_duplicates:raise ValueError("Aggregate NPP to monthly or annual data first")
            days=da["time"].dt.days_in_month.values.astype(float)
            weighted_sum=npp_frame.mul(days,axis=0).groupby(level=0).sum(min_count=1)
            valid_days=npp_frame.notna().mul(days,axis=0).groupby(level=0).sum()
            npp=weighted_sum/valid_days
            valid_months=npp_frame.notna().groupby(level=0).sum()
            npp=npp.where(valid_months==12)
        else:npp=npp_frame.sort_index()
        return analysis_names,npp.where(npp>0).reindex(full_years)
def calculate_model(model_name,npp_file):
    analysis_names,npp_full=load_npp_sites(npp_file)
    trw_full=trw[analysis_names].reindex(full_years)
    paired_valid=trw_full.notna()&npp_full.notna()
    ratio_raw=(trw_full.where(paired_valid)/npp_full.where(paired_valid)).replace([np.inf,-np.inf],np.nan)
    ratio_mean=ratio_raw.mean(axis=0)
    ratio_sd=ratio_raw.std(axis=0,ddof=0)
    ratio_counts=ratio_raw.notna().sum(axis=0)
    standardizable=(ratio_counts>=MIN_RATIO_YEARS)&np.isfinite(ratio_mean)&np.isfinite(ratio_sd)&(ratio_sd>0)
    standardized_names=standardizable.index[standardizable].tolist()
    if not standardized_names:raise ValueError(f"No standardizable sites: {model_name}")
    ratio_all=ratio_raw[standardized_names].sub(ratio_mean[standardized_names],axis=1).div(ratio_sd[standardized_names],axis=1)
    ratio_all=ratio_all.replace([np.inf,-np.inf],np.nan)
    if not np.allclose(ratio_all.mean(axis=0).to_numpy(),0.0,atol=1e-8):raise RuntimeError(f"Ratio centering failed: {model_name}")
    if not np.allclose(ratio_all.std(axis=0,ddof=0).to_numpy(),1.0,atol=1e-8):raise RuntimeError(f"Ratio standardization failed: {model_name}")
    ratio_trend_all=ratio_all.loc[TREND_START:TREND_END]
    trend_counts=ratio_trend_all.notna().sum(axis=0)
    trend_spans=pd.Series(np.nan,index=ratio_all.columns,dtype=float)
    retained_names=[]
    for name in ratio_all.columns:
        valid=ratio_trend_all[name].notna()
        if valid.any():
            vy=ratio_trend_all.index[valid].to_numpy(dtype=int)
            trend_spans[name]=int(vy.max()-vy.min())
            if len(vy)>=MIN_TREND_YEARS and trend_spans[name]>=MIN_TREND_SPAN:retained_names.append(name)
    if not retained_names:raise ValueError(f"No sites satisfy trend thresholds: {model_name}")
    ratio=ratio_all[retained_names]
    annual_count=ratio.notna().sum(axis=1)
    annual_mean=ratio.mean(axis=1)
    annual_q25=ratio.quantile(0.25,axis=1)
    annual_q75=ratio.quantile(0.75,axis=1)
    trend_valid=annual_mean.notna()&(annual_mean.index>=TREND_START)&(annual_mean.index<=TREND_END)
    trend_years=full_years[trend_valid.to_numpy()].astype(float)
    if len(trend_years)<2:raise ValueError(f"Too few years to compute trend: {model_name}")
    overall=linregress(trend_years,annual_mean.loc[trend_valid].to_numpy())
    centered_years=trend_years-trend_years.mean()
    denominator=np.sum(centered_years**2)
    components=ratio.loc[trend_valid].div(annual_count.loc[trend_valid],axis=0).fillna(0.0)
    contributions=pd.Series(centered_years@components.to_numpy()/denominator*10,index=retained_names)
    site_slopes=pd.Series(index=retained_names,dtype=float)
    site_trend_years=ratio.loc[TREND_START:TREND_END].index.to_numpy(dtype=float)
    for name in retained_names:
        values=ratio.loc[TREND_START:TREND_END,name].to_numpy(dtype=float)
        valid=np.isfinite(values)
        if valid.sum()>=MIN_TREND_YEARS and np.ptp(site_trend_years[valid])>=MIN_TREND_SPAN:
            site_slopes[name]=linregress(site_trend_years[valid],values[valid]).slope*10
    if not np.isclose(contributions.sum(),overall.slope*10,rtol=1e-8,atol=1e-10):raise RuntimeError(f"Trend contribution decomposition failed: {model_name}")
    print(f"{model_name}: grid-overlap={len(analysis_names)}, standardized={len(standardized_names)}, retained={len(retained_names)}, slope/10 years={overall.slope*10:.6g}")
    return {"name":model_name,"analysis_names":analysis_names,"standardized_names":standardized_names,"retained_names":retained_names,"paired_counts":ratio_counts,"trend_counts":trend_counts,"trend_spans":trend_spans,"annual_mean":annual_mean,"annual_q25":annual_q25,"annual_q75":annual_q75,"contributions":contributions,"site_slopes":site_slopes,"overall":overall,"fit_start":int(trend_years.min()),"fit_end":int(trend_years.max())}
model_results=[calculate_model(model_name,path) for model_name,path in npp_paths]
model_count=len(model_results)
annual_mean=pd.concat([result["annual_mean"].rename(result["name"]) for result in model_results],axis=1).mean(axis=1)
annual_q25=pd.concat([result["annual_q25"].rename(result["name"]) for result in model_results],axis=1).mean(axis=1)
annual_q75=pd.concat([result["annual_q75"].rename(result["name"]) for result in model_results],axis=1).mean(axis=1)
overall_slope=float(np.mean([result["overall"].slope for result in model_results]))
overall_intercept=float(np.mean([result["overall"].intercept for result in model_results]))
fit_start=max(result["fit_start"] for result in model_results)
fit_end=min(result["fit_end"] for result in model_results)
if fit_start>=fit_end:raise ValueError("Model trend periods do not overlap")
metadata=all_sites.set_index("name").copy()
metadata["paired_years"]=pd.concat([result["paired_counts"].rename(result["name"]) for result in model_results],axis=1).reindex(metadata.index).max(axis=1).fillna(0).astype(int)
metadata["trend_years"]=pd.concat([result["trend_counts"].rename(result["name"]) for result in model_results],axis=1).reindex(metadata.index).max(axis=1).fillna(0).astype(int)
metadata["trend_span"]=pd.concat([result["trend_spans"].rename(result["name"]) for result in model_results],axis=1).reindex(metadata.index).max(axis=1)
model_contributions=pd.concat([result["contributions"].rename(result["name"]) for result in model_results],axis=1).reindex(metadata.index)
model_slopes=pd.concat([result["site_slopes"].rename(result["name"]) for result in model_results],axis=1).reindex(metadata.index)
metadata["models_available"]=model_contributions.notna().sum(axis=1)
metadata["contribution_per_10 years"]=model_contributions.fillna(0).sum(axis=1).div(model_count).where(metadata["models_available"]>0)
metadata["site_slope_per_10 years"]=model_slopes.mean(axis=1,skipna=True)
metadata["contribution_class"]="Unavailable"
if not np.isclose(metadata["contribution_per_10 years"].sum(skipna=True),overall_slope*10,rtol=1e-8,atol=1e-10):raise RuntimeError("Ensemble trend contribution decomposition failed")
total_sites=len(metadata)
analysis_site_count=int((metadata["models_available"]>0).sum())
na_count=int((metadata["region"]=="NA").sum())
eu_count=int((metadata["region"]=="EU").sum())
if na_count+eu_count!=total_sites:raise RuntimeError("Regional counts do not match the total count")
print(f"Candidate tree-ring sites in NA and EU: {total_sites}")
print(f"Sites overlapping at least one NPP grid: {len(set().union(*(set(result['analysis_names']) for result in model_results)))}")
print(f"Sites with >= {MIN_RATIO_YEARS} paired TRW-NPP years in at least one model: {len(set().union(*(set(result['standardized_names']) for result in model_results)))}")
print(f"Unique sites retained for long-term trend analysis: {analysis_site_count}")
print(f"Ensemble slope per 10 years: {overall_slope*10:.6g}")
available=metadata["contribution_per_10 years"].notna()
direction=np.sign(overall_slope)
if direction==0:
    metadata.loc[available,"contribution_class"]="Neutral"
else:
    aligned=metadata.loc[available,"contribution_per_10 years"]*direction
    metadata.loc[aligned.index,"contribution_class"]=np.where(aligned>0,"Minor",np.where(aligned<0,"Opposing","Neutral"))
    supporting=aligned.loc[aligned>0].sort_values(ascending=False)
    if len(supporting):
        cumulative=supporting.cumsum()/supporting.sum()
        major_number=min(int(np.searchsorted(cumulative.to_numpy(),MAJOR_SHARE,side="left"))+1,len(supporting))
        metadata.loc[supporting.index[:major_number],"contribution_class"]="Major"
print(f"NA unavailable: {(metadata.loc[metadata['region']=='NA','contribution_class']=='Unavailable').sum()}")
print(f"EU unavailable: {(metadata.loc[metadata['region']=='EU','contribution_class']=='Unavailable').sum()}")
world=gpd.read_file(map_path)
if world.crs is None:raise ValueError("The world shapefile has no CRS information.")
world=world.to_crs("EPSG:4326")
fig=plt.figure(figsize=(24,21))
gs=fig.add_gridspec(3,2,height_ratios=[0.85,1.08,1.10],left=0.06,right=0.985,bottom=0.065,top=0.985,wspace=0.31,hspace=0.25)
ax_a=fig.add_subplot(gs[0,:])
ax_b=fig.add_subplot(gs[1,0])
ax_c=fig.add_subplot(gs[1,1])
ax_d=fig.add_subplot(gs[2,0])
ax_e=fig.add_subplot(gs[2,1])
mean_values=annual_mean.to_numpy(dtype=float)
lower_values=annual_q25.to_numpy(dtype=float)
upper_values=annual_q75.to_numpy(dtype=float)
ax_a.fill_between(full_years,lower_values,upper_values,color="#808080",alpha=0.35,linewidth=0,zorder=1)
ax_a.plot(full_years,mean_values,color="#222222",marker="o",markersize=4,linewidth=1.8,label=f"All sites (n = {analysis_site_count:,})",zorder=3)
line_years=np.array([fit_start,fit_end],dtype=float)
line_values=overall_intercept+overall_slope*line_years
ax_a.plot(line_years,line_values,color="#C43C39",linewidth=2.8,label=f"Mean model trend ({fit_start}–{fit_end})",zorder=5)
slope_x=fit_start+0.60*(fit_end-fit_start)
slope_y=overall_intercept+overall_slope*slope_x
ax_a.annotate(f"Slope = {overall_slope*10:.1f} / 10 years",xy=(slope_x,slope_y),xytext=(0,22),textcoords="offset points",ha="center",va="bottom",fontsize=23,color="#C43C39",bbox={"facecolor":"white","edgecolor":"none","alpha":0.8,"pad":2},zorder=6)
ax_a.axhline(0,color="#BBBBBB",linewidth=1,linestyle=":",zorder=0)
ax_a.set_xlabel("Year")
ax_a.set_ylabel("Standardized TRW/NPP")
ax_a.set_xlim(DISPLAY_START,DISPLAY_END)
ax_a.margins(y=0.12)
ax_a.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax_a.set_xticks([1900,1920,1940,1960,1980,2000,2020])
ax_a.axvline(1945,color="#666666",linestyle="--",linewidth=1.5,zorder=2)
ax_a.text(1945,0.025,"1945",transform=ax_a.get_xaxis_transform(),ha="center",va="bottom",fontsize=21,bbox={"facecolor":"white","edgecolor":"none","alpha":0.8,"pad":1},zorder=7)
ax_a.legend(loc="best",frameon=False,fontsize=21,ncol=2)
tidy(ax_a)
panel_label(ax_a,"a")
def draw_map(ax,extent,region,letter):
    world.plot(ax=ax,facecolor="#D3D3D3",edgecolor="none",linewidth=0,zorder=1)
    subset=metadata.loc[metadata["region"]==region]
    if region=="EU":subset=subset.loc[subset["plant_group"].isin(["Gymnosperm","Angiosperm"])]
    for category in ["Unavailable","Neutral","Opposing","Minor","Major"]:
        for plant,marker in plant_markers.items():
            rows=subset.loc[(subset["contribution_class"]==category)&(subset["plant_group"]==plant)]
            if rows.empty:continue
            ax.scatter(rows["longitude"],rows["latitude"],s=46 if category=="Major" else 30,marker=marker,color=colors[category],edgecolors="white",linewidths=0.25,alpha=0.85,zorder=5 if category=="Major" else 3)
    xmin,xmax,ymin,ymax=extent
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_aspect("auto")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{abs(x):g}°{'E' if x>0 else 'W' if x<0 else ''}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f"{abs(y):g}°{'N' if y>0 else 'S' if y<0 else ''}"))
    ax.text(0.97,0.97,f"{region} (n = {len(subset):,})",transform=ax.transAxes,ha="right",va="top",fontsize=24,fontweight="bold")
    categories=[category for category in ["Major","Minor","Opposing","Neutral","Unavailable"] if (subset["contribution_class"]==category).any()]
    handles=[Line2D([],[],marker="o",linestyle="none",color=colors[category],markersize=9,label=f"{category} (n = {(subset['contribution_class']==category).sum():,})") for category in categories]
    legend_kwargs={"loc":"upper left","bbox_to_anchor":(0.005,0.95)} if region=="EU" else {"loc":"lower left"}
    legend=ax.legend(handles=handles,frameon=True,facecolor="white",edgecolor="none",framealpha=0.9,fontsize=18,handletextpad=0.4,**legend_kwargs)
    ax.add_artist(legend)
    plant_handles=[Line2D([],[],marker=marker,linestyle="none",color="#555555",markersize=8,label=plant) for plant,marker in plant_markers.items() if (subset["plant_group"]==plant).any()]
    if plant_handles:ax.legend(handles=plant_handles,loc="lower right",frameon=True,facecolor="white",edgecolor="none",framealpha=0.9,fontsize=17,handletextpad=0.4)
    ax.tick_params(direction="out",length=6,width=1.2)
    panel_label(ax,letter)
draw_map(ax_b,na_extent,"NA","b")
draw_map(ax_c,eu_extent,"EU","c")
group_specs=[("plant_group","Gymnosperm",1.0,"#57946C"),("plant_group","Angiosperm",2.0,"#B885B7"),("leaf_group","Needle",3.5,"#57946C"),("leaf_group","Broad",4.5,"#D79956")]
group_arrays={}
for column,label,position,color in group_specs:
    values=metadata.loc[metadata[column]==label,"site_slope_per_10 years"].dropna().to_numpy()
    group_arrays[label]=values
    if len(values):
        boxes=ax_d.boxplot([values],positions=[position],widths=0.58,patch_artist=True,showfliers=False,medianprops={"color":"black","linewidth":2},whiskerprops={"linewidth":1.5},capprops={"linewidth":1.5},boxprops={"linewidth":1.5},manage_ticks=False)
        boxes["boxes"][0].set_facecolor(color)
        boxes["boxes"][0].set_alpha(0.8)
ax_d.set_xticks([item[2] for item in group_specs])
ax_d.set_xticklabels([f"{item[1]}\n(n = {len(group_arrays[item[1]]):,})" for item in group_specs],fontsize=19)
ax_d.set_xlim(0.35,5.15)
ax_d.axhline(0,color="#777777",linewidth=1,linestyle="--",zorder=0)
ax_d.axvline(2.75,ymin=0,ymax=1,color="#777777",linewidth=1.5,linestyle="--",zorder=1)
ax_d.set_ylabel("Standardized TRW/NPP slope per 10 years")
ax_d.ticklabel_format(axis="y",style="sci",scilimits=(-3,3),useMathText=True)
for pair,position in [(("Gymnosperm","Angiosperm"),1.5),(("Needle","Broad"),4.0)]:
    result=group_test([group_arrays[pair[0]],group_arrays[pair[1]]])
    if result is not None:ax_d.text(position,0.965,p_text(result.pvalue,threshold=0.01),transform=ax_d.get_xaxis_transform(),ha="center",va="top",fontsize=22,zorder=5)
tidy(ax_d)
panel_label(ax_d,"d")
species_data=metadata.loc[metadata["site_slope_per_10 years"].notna()&(metadata["species"]!="Unknown")]
species_counts=species_data["species"].value_counts()
selected_species=species_counts.head(TOP_SPECIES).index.tolist()
if selected_species:
    species_arrays=[species_data.loc[species_data["species"]==species,"site_slope_per_10 years"].to_numpy() for species in selected_species]
    order=np.argsort([np.median(values) for values in species_arrays])
    selected_species=[selected_species[i] for i in order]
    species_arrays=[species_arrays[i] for i in order]
    boxes=ax_e.boxplot(species_arrays,vert=False,widths=0.60,patch_artist=True,showfliers=False,medianprops={"color":"black","linewidth":1.8},boxprops={"linewidth":1.2},whiskerprops={"linewidth":1.2},capprops={"linewidth":1.2})
    palette=plt.get_cmap("tab20")
    for i,patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(palette(i%20))
        patch.set_alpha(0.8)
    ax_e.set_yticks(np.arange(1,len(selected_species)+1))
    ax_e.set_yticklabels([f"{species} (n = {len(values):,})" for species,values in zip(selected_species,species_arrays)],fontsize=18)
    ax_e.set_ylim(0,len(selected_species)+1)
    ax_e.axvline(0,color="#777777",linewidth=1,linestyle="--")
    species_test=group_test(species_arrays)
    if species_test is not None:ax_e.text(0.98,0.025,p_text(species_test.pvalue),transform=ax_e.transAxes,ha="right",va="bottom",fontsize=23,bbox={"facecolor":"white","edgecolor":"none","alpha":0.9,"pad":3})
else:ax_e.text(0.5,0.5,"No species with estimable slopes",transform=ax_e.transAxes,ha="center",va="center")
ax_e.set_xlabel("Standardized TRW/NPP slope per 10 years")
ax_e.ticklabel_format(axis="x",style="sci",scilimits=(-3,3),useMathText=True)
tidy(ax_e)
panel_label(ax_e,"e")
out_dir.mkdir(parents=True,exist_ok=True)
fig.savefig(out_png,dpi=600,bbox_inches="tight",pad_inches=0.08,facecolor="white")
plt.close(fig)
print('✅ Saved figure:', out_png)
