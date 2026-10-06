import os,glob,numpy as np,xarray as xr,geopandas as gpd,matplotlib.pyplot as plt,matplotlib as mpl,pandas as pd
from pathlib import Path
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
ROOT=Path('D:/')
recon_path=ROOT/'data/Processed data/TRW/AgeDepSpline (5+3).nc'
model_dir=ROOT/'data/NPP and GPP/S2_npp9-8'
out_png=ROOT / 'results/Extended Data Fig. 6.png'
YEAR_MIN,YEAR_MAX=1899,2025
WIN=20
MIN_N=20
DPI=600
regions={'NA':{'shp':ROOT/'data/Map/Koppen_1991_2020_NA_big7.shp','lon_min':-167.27,'lon_max':-55.0,'lat_min':9.58,'lat_max':74.99,'agg':{102:[1,2,3,4,5],103:[6,7],104:[1,2,3,4,5,6,7]}},'EU':{'shp':ROOT/'data/Map/Europe 7.shp','lon_min':-11.0,'lon_max':57.0,'lat_min':30.0,'lat_max':75.0,'agg':{102:[2,3,4,5],103:[6,7],104:[2,3,4,5,6,7]}}}
label_map={1:'Tropical',2:'Arid',3:'Temperate',4:'Dfa',5:'Dfb',102:'Warm',103:'Cold',104:'All'}
base_classes=[1,2,3,4,5,6,7]
plot_order=[1,2,3,4,5,102,103,104]
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['mathtext.rm']='Times New Roman'
plt.rcParams['mathtext.it']='Times New Roman:italic'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size']=18
def subset_box(da,lon_min,lon_max,lat_min,lat_max):
    lon=da['longitude'].values
    lat=da['latitude'].values
    lon_slice=slice(lon_min,lon_max) if lon[0]<=lon[-1] else slice(lon_max,lon_min)
    lat_slice=slice(lat_min,lat_max) if lat[0]<=lat[-1] else slice(lat_max,lat_min)
    return da.sel(longitude=lon_slice,latitude=lat_slice)
def rasterize_classes(path,lon,lat,lon_min,lon_max,lat_min,lat_max):
    gdf=gpd.read_file(path)
    if gdf.crs is None:
        gdf=gdf.set_crs(epsg=4326)
    else:
        gdf=gdf.to_crs(epsg=4326)
    gdf=gpd.clip(gdf,box(lon_min,lat_min,lon_max,lat_max))
    prefer=['code','CODE','Code','class','Class','CLASS','label','Label']
    col=next((c for c in prefer if c in gdf.columns),None)
    if col is None:
        col=next(c for c in gdf.columns if c.lower() not in ('geometry','geom','id','fid','ogc_fid'))
    if pd.api.types.is_numeric_dtype(gdf[col]):
        gdf['__code__']=pd.to_numeric(gdf[col],errors='coerce').astype('Int16')
    else:
        values=gdf[col].astype(str).str.strip().str.extract(r'^(Dfa|Dfb|[ABCDE])',expand=False)
        gdf['__code__']=values.map({'A':1,'B':2,'C':3,'Dfa':4,'Dfb':5,'D':6,'E':7}).astype('Int16')
    gdf=gdf.dropna(subset=['__code__'])
    transform=from_bounds(float(lon.min()),float(lat.min()),float(lon.max()),float(lat.max()),len(lon),len(lat))
    grid=rasterize([(geom,int(value)) for geom,value in zip(gdf.geometry,gdf['__code__'])],out_shape=(len(lat),len(lon)),transform=transform,fill=0,dtype='int16')
    if lat[0]<lat[-1]:
        grid=np.flipud(grid)
    return grid
def zone_mask(grid,code,region):
    if code in base_classes:
        return grid==code
    return np.isin(grid,region['agg'][code])
def get_recon_da(ds):
    candidates=[name for name in ds.data_vars if 'year' in ds[name].dims]
    preferred=[name for name in candidates if 'recon' in name.lower() or 'agedep' in name.lower()]
    if not candidates:
        raise ValueError('No yearly reconstruction variable found')
    return ds[preferred[0] if preferred else candidates[0]]
def get_model_var(ds):
    candidates=[name for name in ds.data_vars if 'year' in ds[name].dims]
    preferred=[name for name in candidates if name.lower() in ('npp','gpp')]
    return preferred[0] if preferred else candidates[0] if candidates else None
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
        if counts[i]>0:
            sums[i]=float(frame[valid].sum(dtype=np.float64))
    return sums,counts
def combine_region_stats(stats):
    all_years=np.array(sorted(set().union(*(set(years.tolist()) for years,_,_ in stats))),dtype=float)
    total_sum=np.zeros(all_years.size,dtype=np.float64)
    total_count=np.zeros(all_years.size,dtype=np.int64)
    index={year:i for i,year in enumerate(all_years)}
    for years,sums,counts in stats:
        for year,value,count in zip(years,sums,counts):
            if count>0 and np.isfinite(value):
                j=index[year]
                total_sum[j]+=value
                total_count[j]+=count
    values=np.divide(total_sum,total_count,out=np.full(all_years.size,np.nan,dtype=np.float64),where=total_count>0)
    return all_years,values
def top3(data):
    ranked=[]
    for name,(years,values) in data.items():
        if np.isfinite(values).any():
            ranked.append((name,float(np.nanmean(values))))
    ranked.sort(key=lambda x:x[1],reverse=True)
    return [name for name,value in ranked[:3]]
def clean_model_name(name):
    return name.replace('_S2_npp','')
def choose_step(low,high):
    span=max(high-low,1e-6)
    if span<=0.3:
        return 0.05
    if span<=0.8:
        return 0.1
    if span<=1.6:
        return 0.2
    if span<=3.2:
        return 0.4
    return 0.5
def compute_row_limits(series,row_codes):
    limits=[]
    for code in row_codes:
        names=top3(series[code])
        for name in names:
            _,values=series[code][name]
            finite=values[np.isfinite(values)]
            if finite.size:
                limits.append((float(finite.min()),float(finite.max())))
    if not limits:
        return -0.1,0.1,0.1
    low=min(v[0] for v in limits)
    high=max(v[1] for v in limits)
    span=max(high-low,0.2)
    mid=0.5*(low+high)
    low=mid-span/2
    high=mid+span/2
    step=choose_step(low,high)
    low=np.floor(low/step)*step
    high=np.ceil(high/step)*step
    return low,high,step
def draw_panel(ax,code,data,index,row_ylim):
    names=top3(data)
    if not names:
        ax.text(0.5,0.5,'No data',ha='center',va='center',fontsize=16)
        ax.set_axis_off()
        return
    used=set()
    for name in names:
        years,values=data[name]
        label=clean_model_name(name)
        if label in used:
            k=2
            while f'{label} ({k})' in used:
                k+=1
            label=f'{label} ({k})'
        used.add(label)
        ax.plot(years,values,linewidth=1.3,label=label)
    row=index//4
    col=index%4
    ylow,yhigh,ystep=row_ylim
    ax.set_xlim(YEAR_MIN+(WIN+1)/2,YEAR_MAX-(WIN-1)/2)
    ax.set_ylim(ylow,yhigh)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f' if ystep>=0.1 else '%.2f'))
    ax.tick_params(axis='both',labelsize=15)
    ax.tick_params(axis='x',which='major',bottom=True,top=False,labelbottom=True,length=4)
    ax.tick_params(axis='y',which='major',left=True,right=False,labelleft=col==0,length=4)
    ax.set_xlabel('Year' if row==1 else '',fontsize=18)
    ax.set_ylabel(r'$\mathit{r}_{\mathrm{moving}}$' if col==0 else '',fontsize=20)
    ax.text(0.02,0.95,chr(97+index),transform=ax.transAxes,ha='left',va='top',fontsize=21,fontweight='bold')
    ax.text(0.98,0.95,label_map[code],transform=ax.transAxes,ha='right',va='top',fontsize=17,fontweight='bold')
    legend_loc='upper center' if index==6 else 'lower right'
    ax.legend(fontsize=14,frameon=True,loc=legend_loc,borderpad=0.15,handlelength=1.4,handletextpad=0.4)
def main():
    out_png.parent.mkdir(parents=True,exist_ok=True)
    region_data={}
    with xr.open_dataset(recon_path) as ds:
        recon=get_recon_da(ds).sel(year=slice(YEAR_MIN,YEAR_MAX))
        for region_name,region in regions.items():
            tree=subset_box(recon,region['lon_min'],region['lon_max'],region['lat_min'],region['lat_max']).load()
            class_grid=rasterize_classes(region['shp'],tree['longitude'].values,tree['latitude'].values,region['lon_min'],region['lon_max'],region['lat_min'],region['lat_max'])
            region_data[region_name]={'tree':tree,'class_grid':class_grid}
    files=sorted([path for path in model_dir.rglob('*.nc') if not path.name.startswith('2')],key=lambda path:str(path).lower())
    if not files:
        raise FileNotFoundError(model_dir)
    series={code:{} for code in plot_order}
    for path in files:
        model_name=path.relative_to(model_dir).with_suffix('').as_posix().replace('/','_')
        region_stats={code:[] for code in plot_order}
        with xr.open_dataset(path) as ds:
            variable=get_model_var(ds)
            if variable is None:
                continue
            model_all=ds[variable].sel(year=slice(YEAR_MIN,YEAR_MAX))
            for region_name,region in regions.items():
                tree=region_data[region_name]['tree']
                class_grid=region_data[region_name]['class_grid']
                model=subset_box(model_all,region['lon_min'],region['lon_max'],region['lat_min'],region['lat_max'])
                tree_years=tree['year'].values.astype(int)
                model_years=model['year'].values.astype(int)
                common=np.intersect1d(tree_years,model_years)
                common=common[(common>=YEAR_MIN)&(common<=YEAR_MAX)]
                if common.size<WIN+1:
                    continue
                a=tree.sel(year=common)
                b=model.sel(year=common)
                same_grid=b.sizes.get('latitude')==a.sizes['latitude'] and b.sizes.get('longitude')==a.sizes['longitude']
                if same_grid:
                    same_grid=np.allclose(b['latitude'],a['latitude']) and np.allclose(b['longitude'],a['longitude'])
                if not same_grid:
                    b=b.interp(latitude=a['latitude'],longitude=a['longitude'])
                a_values=a.values.astype(np.float64)
                b_values=b.load().values.astype(np.float64)
                correlations=rolling_corr_from_diff(np.diff(a_values,axis=0),np.diff(b_values,axis=0),WIN,MIN_N)
                years=(common[1:common.size-WIN+1]+common[WIN:])/2
                for code in plot_order:
                    mask=zone_mask(class_grid,code,region)
                    sums,counts=frame_sum_count(correlations,mask)
                    region_stats[code].append((years,sums,counts))
        for code in plot_order:
            if region_stats[code]:
                series[code][model_name]=combine_region_stats(region_stats[code])
    row_ylims=[compute_row_limits(series,plot_order[:4]),compute_row_limits(series,plot_order[4:])]
    fig,axes=plt.subplots(2,4,figsize=(18,8.5),constrained_layout=True)
    axes=axes.ravel()
    for index,code in enumerate(plot_order):
        draw_panel(axes[index],code,series[code],index,row_ylims[index//4])
    fig.savefig(out_png,dpi=DPI,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('✅ Saved figure:',out_png)
if __name__=='__main__':
    main()