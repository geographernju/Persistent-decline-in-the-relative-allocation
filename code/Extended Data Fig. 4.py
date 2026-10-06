import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,FuncFormatter
ROOT=Path('D:/')
START_YEAR=1911
DPI=600
PEAK_COLOR='#2ca02c'
out_png=ROOT / 'results/Extended Data Fig. 4.png'
out_png.parent.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':22,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','axes.unicode_minus':False,'figure.dpi':150,'savefig.dpi':DPI})
styles=[('na','NA','#1f77b4','--',2.0),('eu','EU','#ff7f0e','--',2.0),('all','NA & EU',PEAK_COLOR,'-',2.8)]
METHODS=[
    ('ModNegExp',ROOT/'data/Processed data/r_diff and r_raw/ModNegExp (5+3)'),
    ('SFRCS',ROOT/'data/Processed data/r_diff and r_raw/SFRCS (5+3)'),
    ('Spline',ROOT/'data/Processed data/r_diff and r_raw/Spline (5+3)'),
    ('AgeDepSpline (3+3)',ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (3+3)'),
    ('AgeDepSpline (4+3)',ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (4+3)'),
    ('AgeDepSpline (5+3) nppTree',ROOT/'data/Processed data/r_diff and r_raw/AgeDepSpline (5+3npptree)')
]
def format_r(value,pos):
    if abs(value)<1e-10:
        return '0'
    return f'{value:.1f}'
def read_curve(path,required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f'Missing moving-correlation file: {path}')
        return None
    frame=pd.read_csv(path)
    if not {'year','mean_r'}.issubset(frame.columns):
        if required:
            raise ValueError(f'Required columns are missing: {path}')
        return None
    frame=frame[['year','mean_r']].apply(pd.to_numeric,errors='coerce')
    frame=frame[np.isfinite(frame['year'])]
    frame['mean_r']=frame['mean_r'].where(np.isfinite(frame['mean_r']))
    series=frame.groupby('year')['mean_r'].mean().sort_index()
    return series.loc[series.index>=START_YEAR]
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
def annotate_year(ax,point):
    if point is None:
        return
    year,value=point
    xmin,xmax=ax.get_xlim()
    fraction=(year-xmin)/(xmax-xmin)
    alignment='left' if fraction<0.1 else 'right' if fraction>0.9 else 'center'
    ax.axvline(year,color=PEAK_COLOR,linestyle='--',linewidth=1.2,zorder=2)
    ax.scatter(year,value,s=38,color=PEAK_COLOR,edgecolors='white',linewidths=0.8,zorder=5)
    ax.annotate(str(year),xy=(year,value),xytext=(0,7),textcoords='offset points',ha=alignment,va='bottom',fontsize=20,color=PEAK_COLOR,zorder=6)
curves={}
end_years=[]
all_values=[]
for method,base_dir in METHODS:
    all_series=read_curve(base_dir/'All r_diff'/'All.csv',required=True)
    na=read_curve(base_dir/'North America r_diff'/'All.csv',required=False)
    eu=read_curve(base_dir/'Europe r_diff'/'All.csv',required=False)
    curves[method]={'na':na,'eu':eu,'all':all_series}
    for series in (na,eu,all_series):
        if series is None:
            continue
        years=np.asarray(series.index,dtype=float)
        values=series.to_numpy(dtype=float)
        valid=np.isfinite(years)&np.isfinite(values)
        if valid.any():
            end_years.append(float(years[valid].max()))
            all_values.extend(values[valid].tolist())
if not end_years:
    raise ValueError('No valid moving-correlation data available from 1920 onward')
if not all_values:
    raise ValueError('No valid moving-correlation values available')
end_year=max(end_years)
if end_year<=START_YEAR:
    end_year=START_YEAR+20
global_min=float(np.nanmin(all_values))
global_max=float(np.nanmax(all_values))
ymin=np.floor(global_min*10.0)/10.0
ymax=np.ceil(global_max*10.0)/10.0
if np.isclose(ymin,ymax):
    ymin-=0.1
    ymax+=0.1
fig,axes=plt.subplots(3,2,figsize=(18,16),sharex=True,sharey=True)
axes=axes.ravel()
fig.subplots_adjust(left=0.10,right=0.98,bottom=0.07,top=0.98,hspace=0.12,wspace=0.12)
for idx,(method,base_dir) in enumerate(METHODS):
    ax=axes[idx]
    panel=curves[method]
    row=idx//2
    col=idx%2
    for key,label,color,linestyle,linewidth in styles:
        series=panel[key]
        if series is None:
            continue
        ax.plot(series.index.to_numpy(dtype=float),series.to_numpy(dtype=float),color=color,linestyle=linestyle,linewidth=linewidth,label=label)
    ax.set_xlim(START_YEAR,end_year)
    ax.set_ylim(ymin,ymax)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(format_r))
    ax.tick_params(axis='both',direction='out',labelsize=20)
    ax.tick_params(axis='x',which='major',bottom=True,top=False,labelbottom=row==2,length=5)
    if col==0:
        ax.tick_params(axis='y',which='major',left=True,right=False,labelleft=True,labelright=False,length=5)
        ax.set_ylabel(r'$\mathit{r}_{\mathrm{moving}}$',fontsize=24)
    else:
        ax.tick_params(axis='y',which='major',left=True,right=False,labelleft=False,labelright=False,length=5)
        ax.set_ylabel('')
    ax.set_xlabel('Year' if row==2 else '',fontsize=24)
    ax.text(0.02,0.98,chr(97+idx),transform=ax.transAxes,ha='left',va='top',fontsize=28,fontweight='bold')
    ax.text(0.50,0.98,method,transform=ax.transAxes,ha='center',va='top',fontsize=23)
    for year in [1933,1998]:
        point=value_at_year(panel['all'],year)
        annotate_year(ax,point)
handles,labels=axes[0].get_legend_handles_labels()
if handles:
    axes[0].legend(handles,labels,loc='lower right',fontsize=19,frameon=True,fancybox=False,edgecolor='black',borderpad=0.3,handlelength=1.7,handletextpad=0.5)
fig.savefig(out_png,dpi=DPI,bbox_inches='tight',pad_inches=0.12,facecolor='white')
plt.close(fig)
print('✅ Saved figure:',out_png)