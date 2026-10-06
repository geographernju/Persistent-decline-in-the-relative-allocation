import re,numpy as np,pandas as pd,xarray as xr,matplotlib.pyplot as plt,geopandas as gpd
from shapely.geometry import shape as shp_shape
from rasterio.transform import from_bounds
from rasterio.features import shapes
from pathlib import Path
ROOT=Path('D:/');plt.rcParams['font.family']='Times New Roman';plt.rcParams['axes.unicode_minus']=False;plt.rcParams['font.size']=16;plt.rcParams['figure.dpi']=300
tree_site_csv=ROOT/'data/1. Basic information of global tree-ring sites.csv';flux_americas_csv=ROOT/'data/2. Coordinates of Americas flux tower sites.csv';flux_europe_csv=ROOT/'data/3. Coordinates of European ICOS flux tower sites.csv';flux_existing_csv=ROOT/'data/4. Coordinates of North American and European flux tower sites.csv';npp_coord_csv=ROOT/'data/5. NPP site coordinate table.csv';npp_annual_csv=ROOT/'data/6. Annual NPP values at observation sites.csv';gpp_annual_csv=ROOT/'data/8. Annual GPP data of GPP_DT_VUT method.csv';tree_annual_csv=ROOT/'data/Processed data/Detrending/AgeDepSpline.csv';recon_nc=ROOT/'data/Processed data/TRW/AgeDepSpline (5+3).nc';out_png=ROOT / 'results/Supplementary Fig. 4.png'
na_lon_min,na_lon_max=-167.27,-55.0;na_lat_min,na_lat_max=9.58,74.99;eu_lon_min,eu_lon_max=-11.0,55.0;eu_lat_min,eu_lat_max=29.0,75.0
def norm(s):return re.sub(r'[^a-z0-9]','',str(s).strip().lower())
def find_col(df,names):
    m={norm(c):c for c in df.columns}
    return next((m[norm(x)] for x in names if norm(x) in m),None)
def prep(df):
    la=find_col(df,['latitude','lat','site_latitude','sitelatitude']);lo=find_col(df,['longitude','lon','long','site_longitude','sitelongitude'])
    if la is None or lo is None:raise ValueError('Coordinate columns not found')
    df=df.copy();df['_lat']=pd.to_numeric(df[la],errors='coerce');df['_lon']=pd.to_numeric(df[lo],errors='coerce');df=df.dropna(subset=['_lat','_lon']);df['_lon']=((df['_lon']+180)%360)-180
    return df
def region(df):
    m1=df['_lon'].between(na_lon_min,na_lon_max)&df['_lat'].between(na_lat_min,na_lat_max);m2=df['_lon'].between(eu_lon_min,eu_lon_max)&df['_lat'].between(eu_lat_min,eu_lat_max)
    return df[m1|m2].copy()
def filter_by_recon_nc(df,nc_path):
    with xr.open_dataset(nc_path) as ds:
        cand=[v for v in ds.data_vars if 'year' in ds[v].dims]
        prefer=[v for v in cand if 'recon' in ds[v].name.lower() or 'agedep' in ds[v].name.lower()]
        da=ds[prefer[0] if prefer else cand[0]]
        lat_name=next(d for d in da.dims if 'lat' in d.lower())
        lon_name=next(d for d in da.dims if 'lon' in d.lower())
        valid_mask=np.isfinite(da).any(dim='year')
        pts_lat=xr.DataArray(df['_lat'].values,dims='points')
        pts_lon=xr.DataArray(df['_lon'].values,dims='points')
        is_valid=valid_mask.sel({lat_name:pts_lat,lon_name:pts_lon},method='nearest').values
    return df[is_valid].copy()
def valid_gdf_from_nc(nc_path):
    with xr.open_dataset(nc_path) as ds:
        da=ds['AgeDepSpline'].sel(year=slice(1981,2000))
        mask=np.isfinite(da).any(dim='year').values.astype('uint8')
        lat=ds['latitude'].values if 'latitude' in ds.coords else ds['lat'].values
        lon=ds['longitude'].values if 'longitude' in ds.coords else ds['lon'].values
    arr=mask[::-1,:] if bool(lat[-1]>lat[0]) else mask
    transform=from_bounds(float(lon.min()),float(lat.min()),float(lon.max()),float(lat.max()),len(lon),len(lat))
    polys=[shp_shape(geom) for geom,value in shapes(arr,mask=arr==1,transform=transform) if value==1]
    if not polys:
        raise ValueError('No valid NetCDF area was found for 1981-2000')
    geometry=gpd.GeoSeries(polys,crs='EPSG:4326').union_all().buffer(0)
    return gpd.GeoDataFrame(geometry=[geometry],crs='EPSG:4326')
def clip_like_fig1(df,valid_gdf):
    gdf=gpd.GeoDataFrame(df.copy(),geometry=gpd.points_from_xy(df['_lon'],df['_lat']),crs='EPSG:4326')
    clipped=gpd.clip(gdf,valid_gdf)
    return pd.DataFrame(clipped.drop(columns='geometry'))
def grids(df):return len(set(zip(np.floor((df['_lon']+180)/0.5).astype(int),np.floor((df['_lat']+90)/0.5).astype(int))))
tree=filter_by_recon_nc(region(prep(pd.read_csv(tree_site_csv))),recon_nc);tree['_name']=tree[find_col(tree,['name'])].astype(str).str.strip()
def read_flux(p):
    d=prep(pd.read_csv(p));c=find_col(d,['site_ID','siteid','site_id','site','sitecode','site_name','sitename','name']);d['_site_id']=d[c].astype(str).str.strip() if c else '';d['_lk']=d['_lon'].round(2);d['_ak']=d['_lat'].round(2);return d
valid_gdf=valid_gdf_from_nc(recon_nc)
flux=region(pd.concat([read_flux(flux_americas_csv),read_flux(flux_existing_csv),read_flux(flux_europe_csv)],ignore_index=True,sort=False).drop_duplicates(['_lk','_ak']))
flux=clip_like_fig1(flux,valid_gdf)
npp_coord=prep(pd.read_csv(npp_coord_csv,dtype=str));npp_annual=pd.read_csv(npp_annual_csv,dtype=str);cid=find_col(npp_coord,['site_ID','siteid','site_id']);aid=find_col(npp_annual,['site_ID','siteid','site_id']);vcol=find_col(npp_annual,['NPP_tot1','npptot1']);npp_coord['_site_id']=npp_coord[cid].astype(str).str.strip();npp_annual['_site_id']=npp_annual[aid].astype(str).str.strip();npp_annual['_npp']=pd.to_numeric(npp_annual[vcol],errors='coerce');valid_ids=set(npp_annual.loc[npp_annual['_npp'].notna(),'_site_id']);npp=region(npp_coord[npp_coord['_site_id'].isin(valid_ids)].drop_duplicates('_site_id'))
npp=clip_like_fig1(npp,valid_gdf)
ta=pd.read_csv(tree_annual_csv);names=list(dict.fromkeys(tree['_name'].dropna().astype(str).str.strip()));cm={str(c).strip():c for c in ta.columns};matched=[cm[x] for x in names if x in cm];yc=find_col(ta,['year','calendar_year','calendaryear','time'])
if yc is None:
    f=ta.columns[0];t=pd.to_numeric(ta[f],errors='coerce');yc=f if t.between(1800,2100).mean()>0.5 else None
if yc is None:raise ValueError('Year column not found in AgeDepSpline.csv')
years=pd.to_numeric(ta[yc],errors='coerce');tree_years=int(ta.loc[years.between(1950,2025),matched].apply(pd.to_numeric,errors='coerce').notna().sum().sum())
ga=pd.read_csv(gpp_annual_csv);flux_ids=set(flux['_site_id'].dropna().astype(str).str.strip());flux_ids.discard('');gsite=find_col(ga,['site_ID','siteid','site_id','site','sitecode','site_name','sitename','name'])
if gsite is not None:
    sub=ga[ga[gsite].astype(str).str.strip().isin(flux_ids)];gc=[c for c in sub.columns if 'gpp' in norm(c) and 'dtvut' in norm(c)] or [c for c in sub.columns if 'gpp' in norm(c)];flux_years=int(pd.to_numeric(sub[gc[0]],errors='coerce').notna().sum()) if gc else 0
else:
    gm={str(c).strip():c for c in ga.columns};gcols=[gm[x] for x in flux_ids if x in gm];flux_years=int(ga[gcols].apply(pd.to_numeric,errors='coerce').notna().sum().sum())
npp_ids=set(npp['_site_id'].dropna().astype(str).str.strip());npp_years=int(npp_annual[npp_annual['_site_id'].isin(npp_ids)]['_npp'].notna().sum())
tree_ring=[len(tree),grids(tree),tree_years];flux_tower=[len(flux),grids(flux),flux_years];npp_site=[len(npp),grids(npp),npp_years]
print('Tree-ring:',tree_ring);print('Flux tower:',flux_tower);print('NPP site:',npp_site)
labels=['Point Count','0.5° Grid Coverage','Accumulated Years'];names=['Tree-ring','Flux tower','NPP site'];colors=['gray','mediumseagreen','indianred'];letters=['a','b','c']
fig,axes=plt.subplots(1,3,figsize=(13.5,4.8),dpi=300,layout='constrained')
for i,ax in enumerate(axes):
    values=[tree_ring[i],flux_tower[i],npp_site[i]];bars=ax.bar(names,values,color=colors,width=0.68,zorder=2);ax.set_ylim(0,max(values)*1.28);ax.set_axisbelow(True);ax.grid(axis='y',linestyle='--',linewidth=0.8,alpha=0.35);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.tick_params(axis='x',labelsize=14,length=0);ax.tick_params(axis='y',labelsize=13);ax.text(0.5,0.97,labels[i],transform=ax.transAxes,ha='center',va='top',fontsize=16);ax.text(0.02,0.98,letters[i],transform=ax.transAxes,ha='left',va='top',fontsize=18,fontweight='bold')
    for bar,value in zip(bars,values):ax.annotate(f'{value:,}',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=14)
out_png.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out_png,dpi=600,bbox_inches='tight');plt.close(fig);print('✅ Saved figure:',out_png)