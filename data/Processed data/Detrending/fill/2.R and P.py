from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MaxNLocator
ROOT=Path("D:/")
FOLDER=Path('D:/data/Processed data/Detrending/fill')
INPUT=Path('D:/data/Processed data/Detrending/fill/best_climate_correlations.csv')
out_png=Path('D:/data/Processed data/Detrending/fill/R and P.png')
METHODS=("Spline","ModNegExp","RCS","AgeDepSpline","SFRCS")
plt.rcParams.update({"font.family":"Times New Roman","font.size":18,"axes.labelsize":21,"xtick.labelsize":17,"ytick.labelsize":17,"axes.linewidth":1.3,"mathtext.fontset":"stix","savefig.dpi":600})
df=pd.read_csv(INPUT,encoding="utf-8-sig")
df.columns=df.columns.astype(str).str.strip()
if not {"method","R","P"}.issubset(df.columns):
    raise ValueError("Required columns method, R and P were not found")
df["method"]=df["method"].astype(str).str.strip()
df["R"]=pd.to_numeric(df["R"],errors="coerce")
df["P"]=pd.to_numeric(df["P"],errors="coerce")
missing=[method for method in METHODS if method not in set(df["method"])]
if missing:
    raise ValueError(f"Methods not found: {', '.join(missing)}")
r_edges=np.linspace(-1,1,21)
p_colors=("#236B45","#71A879","#DAB967","#B6B6B6")
p_labels=(r"$<0.001$",r"$0.001$–$0.01$",r"$0.01$–$0.05$",r"$\geq0.05$")
fig=plt.figure(figsize=(17,18))
outer=fig.add_gridspec(5,1,left=0.075,right=0.99,bottom=0.055,top=0.985,hspace=0.55)
for index,method in enumerate(METHODS):
    sub=outer[index].subgridspec(1,2,width_ratios=[1,1.15],wspace=0.25)
    ax_r=fig.add_subplot(sub[0])
    ax_p=fig.add_subplot(sub[1])
    group=df.loc[df["method"]==method]
    r=group.loc[np.isfinite(group["R"])&group["R"].between(-1,1),"R"].to_numpy(dtype=float)
    p=group.loc[np.isfinite(group["P"])&group["P"].between(0,1),"P"].to_numpy(dtype=float)
    r_counts,_=np.histogram(r,bins=r_edges)
    r_percent=100*r_counts/len(r) if len(r) else np.zeros(len(r_counts))
    p_counts=np.array([np.count_nonzero(p<0.001),np.count_nonzero((p>=0.001)&(p<0.01)),np.count_nonzero((p>=0.01)&(p<0.05)),np.count_nonzero(p>=0.05)])
    p_percent=100*p_counts/len(p) if len(p) else np.zeros(4)
    ax_r.bar(r_edges[:-1],r_percent,width=np.diff(r_edges),align="edge",color="#477BA3",edgecolor="white",linewidth=0.6)
    ax_r.axvline(0,color="#444444",linewidth=1.0,linestyle="--")
    ax_r.set_xlim(-1,1)
    ax_r.set_xticks(np.arange(-1,1.01,0.5))
    ax_r.set_ylim(0,max(5,float(r_percent.max())*1.35))
    ax_r.set_xlabel(r"$R$",labelpad=3)
    ax_r.set_ylabel("Percentage")
    ax_r.text(0.01,1.07,f"{chr(97+index)}  {method}",transform=ax_r.transAxes,ha="left",va="bottom",fontsize=23,fontweight="bold")
    ax_r.text(0.98,0.95,f"n = {len(r):,}",transform=ax_r.transAxes,ha="right",va="top",fontsize=17)
    bars=ax_p.bar(np.arange(4),p_percent,width=0.68,color=p_colors,edgecolor="white",linewidth=0.7)
    ax_p.set_xticks(np.arange(4))
    ax_p.set_xticklabels(p_labels,fontsize=17)
    ax_p.set_ylim(0,120)
    ax_p.set_yticks((0,25,50,75,100))
    ax_p.set_xlabel(r"$P$",labelpad=3)
    ax_p.text(0.98,0.97,f"n = {len(p):,}",transform=ax_p.transAxes,ha="right",va="top",fontsize=17)
    for bar,value in zip(bars,p_percent):
        ax_p.text(bar.get_x()+bar.get_width()/2,value+2,f"{value:.1f}%",ha="center",va="bottom",fontsize=18)
    for ax in (ax_r,ax_p):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out",length=5,width=1.2)
    ax_r.yaxis.set_major_locator(MaxNLocator(nbins=4))
fig.savefig(out_png,dpi=600,bbox_inches="tight",facecolor="white")
print('✅ Saved figure:', out_png)
plt.show()