import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,FuncFormatter
ROOT=Path('D:/')
METHOD='AgeDepSpline'
ZONES=['B','C','Dfa','Dfb','Warm','Cold']
ZONE_LABELS={'B':'Arid','C':'Temperate','Dfa':'Dfa','Dfb':'Dfb','Warm':'Warm','Cold':'Cold'}
START_YEAR=1911
DPI=600
PEAK_COLOR='#2ca02c'
MARK_YEARS=[1933,1998]
out_png=ROOT / 'results/Extended Data Fig. 5.png'
out_png.parent.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':22,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','axes.unicode_minus':False,'figure.dpi':150,'savefig.dpi':DPI})
styles=[('na','NA','#1f77b4','--',2.0),('eu','EU','#ff7f0e','--',2.0),('all','NA & EU',PEAK_COLOR,'-',2.8)]
def format_r(value,pos):
    return '0' if abs(value)<1e-10 else f'{value:.3f}'.rstrip('0').rstrip('.')
def read_curve(path):
    if not path.exists():
        return None
    frame=pd.read_csv(path)
    if not {'year','mean_r'}.issubset(frame.columns):
        return None
    frame=frame[['year','mean_r']].apply(pd.to_numeric,errors='coerce')
    frame=frame[np.isfinite(frame['year'])]
    frame['mean_r']=frame['mean_r'].where(np.isfinite(frame['mean_r']))
    series=frame.groupby('year')['mean_r'].mean().sort_index()
    series=series.loc[series.index>=START_YEAR]
    return series if len(series)>0 else None
def value_at_year(series,year):
    if series is None:
        return None
    index=np.asarray(series.index,dtype=float)
    values=series.to_numpy(dtype=float)
    mask=np.isfinite(index)&np.isfinite(values)&np.isclose(index,float(year))
    if not mask.any():
        return None
    i=np.where(mask)[0][0]
    return int(year),float(values[i])
def annotate_year(ax,point,text_height=0.90):
    if point is None:
        return
    year,value=point
    ax.axvline(year,color=PEAK_COLOR,linestyle='--',linewidth=1.2,zorder=2)
    ax.scatter(year,value,s=38,color=PEAK_COLOR,edgecolors='white',linewidths=0.8,zorder=5)
    ax.text(year,text_height,str(year),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=18,color=PEAK_COLOR,zorder=6)
def row_limits(zone_list,curves):
    arrays=[]
    for zone in zone_list:
        for series in curves[zone].values():
            if series is None:
                continue
            values=series.to_numpy(dtype=float)
            values=values[np.isfinite(values)]
            if values.size:
                arrays.append(values)
    if not arrays:
        return -0.1,0.1
    merged=np.concatenate(arrays)
    low=float(merged.min())
    high=float(merged.max())
    span=high-low
    if span<=0:
        span=max(abs(high)*0.2,0.05)
    return low-span*0.14,high+span*0.30
base_dir=ROOT/'data/Processed data/r_diff and r_raw'/f'{METHOD} (5+3)'
curves={}
end_years=[]
for zone in ZONES:
    na=read_curve(base_dir/'North America r_diff'/f'{zone}.csv')
    eu=read_curve(base_dir/'Europe r_diff'/f'{zone}.csv')
    all_series=read_curve(base_dir/'All r_diff'/f'{zone}.csv')
    curves[zone]={'na':na,'eu':eu,'all':all_series}
    for series in (na,eu,all_series):
        if series is None:
            continue
        values=series.to_numpy(dtype=float)
        mask=np.isfinite(values)
        if mask.any():
            end_years.append(float(series.index.to_numpy(dtype=float)[mask].max()))
if not end_years:
    raise ValueError('No valid moving-correlation data available from 1920 onward')
end_year=max(end_years)
if end_year<=START_YEAR:
    end_year=START_YEAR+20
row_groups=[['B','C'],['Dfa','Dfb'],['Warm','Cold']]
row_ylims=[row_limits(group,curves) for group in row_groups]
fig,axes=plt.subplots(3,2,figsize=(12,14),sharex=True)
fig.subplots_adjust(left=0.09,right=0.98,bottom=0.08,top=0.98,hspace=0.10,wspace=0.08)
axes=axes.ravel()
for index,zone in enumerate(ZONES):
    ax=axes[index]
    panel=curves[zone]
    for key,label,color,linestyle,linewidth in styles:
        series=panel[key]
        if series is None:
            continue
        values=series.to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        ax.plot(series.index.to_numpy(dtype=float),values,color=color,linestyle=linestyle,linewidth=linewidth,label=label)
    row=index//2
    col=index%2
    y_min,y_max=row_ylims[row]
    ax.set_xlim(START_YEAR,end_year)
    ax.set_ylim(y_min,y_max)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(format_r))
    ax.tick_params(axis='both',labelsize=18)
    ax.tick_params(axis='x',which='major',bottom=True,top=False,labelbottom=row==2,length=5)
    ax.tick_params(axis='y',which='both',left=True,right=False,labelleft=col==0,labelright=False)
    ax.set_xlabel('Year' if row==2 else '',fontsize=22)
    ax.set_ylabel(r'$\mathit{r}_{\mathrm{moving}}$' if col==0 else '',fontsize=22)
    ax.text(0.02,0.98,chr(97+index),transform=ax.transAxes,ha='left',va='top',fontsize=26,fontweight='bold')
    ax.text(0.98,0.98,ZONE_LABELS[zone],transform=ax.transAxes,ha='right',va='top',fontsize=20,fontweight='bold')
    for year in MARK_YEARS:
        point=value_at_year(panel['all'],year)
        annotate_year(ax,point,text_height=0.90)
handles,labels=axes[0].get_legend_handles_labels()
if handles:
    axes[0].legend(handles,labels,loc='lower right',fontsize=16,frameon=True,fancybox=False,edgecolor='black',borderpad=0.3,handlelength=1.7,handletextpad=0.5)
fig.savefig(out_png,dpi=DPI,bbox_inches='tight',pad_inches=0.12,facecolor='white')
plt.close(fig)
print('✅ Saved figure:', out_png)