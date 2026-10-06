import os,numpy as np,pandas as pd,matplotlib.pyplot as plt,geopandas as gpd,xarray as xr
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from pathlib import Path
plt.rcParams['font.family']='Times New Roman'
ROOT=Path('D:/')
nc_path=ROOT/'data/Processed data/Sliding difference/Sliding difference1-1900-center.nc'
na_shp_path=ROOT/'data/Map/Koppen_1991_2020_NA_big7_simple.shp'
eu_shp_path=ROOT/'data/Map/Europe 7-class.shp'
out_dir=ROOT / 'results'
os.makedirs(out_dir,exist_ok=True)
png_bar=ROOT / 'results/Extended Data Fig. 8.png'
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=22
plt.rcParams['figure.dpi']=600
ds=xr.open_dataset(nc_path)
lon=ds['longitude'].values
lat=ds['latitude'].values
nx=lon.size
ny=lat.size
left=float(np.nanmin(lon))
right=float(np.nanmax(lon))
bottom=float(np.nanmin(lat))
top=float(np.nanmax(lat))
transform=from_bounds(left,bottom,right,top,nx,ny)
var_names=[v for v in ds.data_vars if v.startswith('delta_') and ('68' in v or 'summer' in v.lower())]
base_factors=[]
for v in var_names:
    base_factors.append(v.split('delta_')[1])
base_factors=list(dict.fromkeys(base_factors))
Moisture_keys=[s.lower() for s in ['pet_','VPD_','RH_']]
temp_keys=[s.lower() for s in ['vap_','tmp_','stl1_']]
rad_keys=[s.lower() for s in ['cld_']]
drought_keys=[s.lower() for s in ['spei6_','spei12_','scpdsi_']]
cat_map={}
for f in base_factors:
    s=f.lower()
    if any(k in s for k in Moisture_keys):
        cat_map[f]='Moisture'
    elif any(k in s for k in temp_keys):
        cat_map[f]='Temperature'
    elif any(k in s for k in rad_keys):
        cat_map[f]='Radiation'
    elif any(k in s for k in drought_keys):
        cat_map[f]='Drought'
cat_order=['Moisture','Temperature','Radiation','Drought']
final_factors=[f for c in cat_order for f in base_factors if cat_map.get(f)==c]
gdf_na=gpd.read_file(na_shp_path)
gdf_eu=gpd.read_file(eu_shp_path)
if gdf_na.crs is None:
    gdf_na=gdf_na.set_crs(epsg=4326)
else:
    gdf_na=gdf_na.to_crs(epsg=4326)
if gdf_eu.crs is None:
    gdf_eu=gdf_eu.set_crs(epsg=4326)
else:
    gdf_eu=gdf_eu.to_crs(epsg=4326)
cand_fields=['zone_name','big7','ZONE','Class','class','class_simple','Koppen','KOPPEN','NAME','name']
na_field=None
for f in cand_fields:
    if f in gdf_na.columns:
        na_field=f
        break
eu_field=None
for f in cand_fields:
    if f in gdf_eu.columns:
        eu_field=f
        break
if na_field is None:
    raise ValueError('No valid climate-zone field found in the North America shapefile')
if eu_field is None:
    raise ValueError('No valid climate-zone field found in the Europe shapefile')
base_zone_order=['A','B','C','Dfa','Dfb','D other','E']
gdf_na=gdf_na[gdf_na[na_field].isin(base_zone_order)].copy()
gdf_eu=gdf_eu[gdf_eu[eu_field].isin(base_zone_order)].copy()
gdf_na['zone_name']=gdf_na[na_field]
gdf_eu['zone_name']=gdf_eu[eu_field]
gdf=pd.concat([gdf_na,gdf_eu],ignore_index=True)
zones=['B','C','Dfa','Dfb','Warm','Cold','All']
name_map={'B':'Arid','C':'Temperate','Dfa':'Dfa','Dfb':'Dfb','Warm':'Warm','Cold':'Cold','All':'All'}
def median_ignore_nan(values,axis=None):
    masked=np.ma.masked_invalid(values)
    result=np.ma.median(masked,axis=axis)
    return np.asarray(np.ma.filled(result,np.nan))
def zone_mask_from_names(names):
    geom=list(gdf[gdf['zone_name'].isin(names)].geometry)
    if len(geom)==0:
        return np.zeros((ny,nx),dtype=bool)
    arr=rasterize([(g,1) for g in geom],out_shape=(ny,nx),transform=transform,fill=0,all_touched=True,dtype='uint8')
    arr=np.flipud(arr)
    return arr.astype(bool)
zone_masks={
    'B':zone_mask_from_names(['B']),
    'C':zone_mask_from_names(['C']),
    'Dfa':zone_mask_from_names(['Dfa']),
    'Dfb':zone_mask_from_names(['Dfb']),
    'Warm':zone_mask_from_names(['A','B','C','Dfa','Dfb']),
    'Cold':zone_mask_from_names(['D other','E']),
    'All':zone_mask_from_names(['A','B','C','Dfa','Dfb','D other','E'])
}
records=[]
for z in zones:
    m=zone_masks[z]
    for f in final_factors:
        var='delta_'+f
        if var not in ds.data_vars:
            med=np.nan
        else:
            da=ds[var].sel(window_center_year=slice(1980,2010))
            arr=da.values
            t_axes=tuple(da.dims.index(d) for d in da.dims if d not in ('latitude','longitude'))
            if t_axes:
                arr=median_ignore_nan(arr,axis=t_axes)
            vals=arr[m]
            med=float(median_ignore_nan(vals)) if vals.size>0 else np.nan
        records.append((z,f,med))
df=pd.DataFrame(records,columns=['zone_name','factor','median_delta_r_diff'])
def sample_colors(cmap_name,n,lo=0.35,hi=0.75):
    cmap=plt.get_cmap(cmap_name)
    xs=np.linspace(lo,hi,n)
    return [cmap(x) for x in xs]
cat_cmap={'Moisture':'Blues','Temperature':'Greens','Radiation':'Purples','Drought':'YlOrBr'}
cat_range={'Moisture':(0.4,0.75),'Temperature':(0.4,0.7),'Radiation':(0.45,0.75),'Drought':(0.45,0.7)}
items_by_cat={c:[v for v in final_factors if cat_map.get(v)==c] for c in cat_order}
colors_by_cat={c:sample_colors(cat_cmap[c],len(items_by_cat[c]),*cat_range[c]) for c in cat_order}
color_map={v:col for c in cat_order for v,col in zip(items_by_cat[c],colors_by_cat[c])}
cat_seq=[cat_map[f] for f in final_factors]
bounds=[i-0.5 for i in range(1,len(final_factors)) if cat_seq[i]!=cat_seq[i-1]]
data_by_zone={}
for z in zones:
    s=df[df['zone_name']==z].set_index('factor')['median_delta_r_diff']
    data_by_zone[z]=s.reindex(final_factors)
all_vals=[]
for z in zones:
    all_vals.append(data_by_zone[z].values.astype(float))
all_vals=np.concatenate(all_vals)
all_vals=all_vals[np.isfinite(all_vals)]
if all_vals.size==0:
    raise ValueError('No valid values are available for plotting')
vmin=min(0.0,float(all_vals.min()))
vmax=max(0.0,float(all_vals.max()))
tick_locator=MaxNLocator(nbins=6)
yticks=tick_locator.tick_values(vmin,vmax)
yticks=yticks[(yticks>=vmin-1e-12)&(yticks<=vmax+1e-12)]
def disp_name(s):
    name=s.replace('_68','').replace('_summer','')
    if name.lower()=='scpdsi':
        return 'scPDSI'
    return name
def legend_column(categories):
    entries=[]
    for cat in categories:
        items=items_by_cat[cat]
        if items:
            entries.append((Patch(facecolor='none',edgecolor='none'),f'$\\bf{{{cat}}}$'))
            for v in items:
                entries.append((Patch(facecolor=color_map[v],edgecolor='none'),disp_name(v)))
    return entries
fig,axes=plt.subplots(2,4,figsize=(24,14),sharey=True)
axes=axes.ravel()
for idx,z in enumerate(zones):
    ax=axes[idx]
    s=data_by_zone[z]
    x=np.arange(len(final_factors))
    ax.bar(x,np.nan_to_num(s.values.astype(float)),color=[color_map[f] for f in final_factors])
    for b in bounds:
        ax.axvline(b,ls='--',lw=1.2,color='k',alpha=0.5)
    ax.axhline(0,lw=1.5,color='k',alpha=0.6)
    ax.set_xlim(-0.5,len(final_factors)-0.5)
    ax.set_ylim(vmin,vmax)
    ax.set_yticks(yticks)
    ax.set_xticks([])
    if idx in [0,4]:
        ax.set_ylabel(r'$\Delta r = r_{\mathrm{moving}} - r_{\mathrm{moving}\mid\mathrm{climate}}$', fontsize=22)
        ax.tick_params(axis='y',labelleft=True)
    else:
        ax.tick_params(axis='y',labelleft=False)
    ax.text(0.02,0.93,chr(97+idx),transform=ax.transAxes,ha='left',va='top',fontsize=28,fontweight='bold')
    ax.text(0.98,0.93,name_map[z],transform=ax.transAxes,ha='right',va='top',fontsize=24,fontweight='bold')
axes[7].axis('off')
left_entries=legend_column(['Moisture','Temperature'])
right_entries=legend_column(['Radiation','Drought'])
nrows=max(len(left_entries),len(right_entries))
blank_handle=Patch(facecolor='none',edgecolor='none')
while len(left_entries)<nrows:
    left_entries.append((blank_handle,''))
while len(right_entries)<nrows:
    right_entries.append((blank_handle,''))
handles=[item[0] for item in left_entries]+[item[0] for item in right_entries]
labels=[item[1] for item in left_entries]+[item[1] for item in right_entries]
axes[7].legend(handles,labels,ncol=2,fontsize=20,loc='center',frameon=False,handlelength=1.2,handletextpad=0.6,columnspacing=2.0,borderpad=0.3)
fig.tight_layout()
fig.savefig(png_bar,bbox_inches='tight',dpi=600)
plt.close(fig)
ds.close()
print('✅ Saved figure:',png_bar)