import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path("D:/")
data_dir = ROOT / "data/Processed data/GPP and NPP"
warm_gpp_csv = data_dir / "warm_gpp.csv"
warm_npp_csv = data_dir / "warm_npp.csv"
cold_gpp_csv = data_dir / "cold_gpp.csv"
cold_npp_csv = data_dir / "cold_npp.csv"
out_png = ROOT / 'results/Extended Data Fig. 3.png'
out_png.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "savefig.dpi": 600,
    "axes.unicode_minus": False
})

def load_df(path):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if year_col is None:
        raise ValueError(f"Missing year column: {path}")

    years = pd.to_numeric(df[year_col], errors="coerce")
    valid = np.isfinite(years)
    df = df.loc[valid].copy()
    years = years.loc[valid]

    if not np.all(years == np.floor(years)):
        raise ValueError(f"Invalid annual coordinates: {path}")

    df[year_col] = years.astype(int)
    df = df.set_index(year_col).sort_index()

    if df.empty:
        raise ValueError(f"No valid annual records: {path}")
    if df.index.duplicated().any():
        raise ValueError(f"Duplicate years: {path}")

    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")

    names = []
    for column in df.columns:
        name = re.sub(r"\.nc$", "", column, flags=re.IGNORECASE)
        name = re.sub(r"^\d+_", "", name)
        name = re.sub(r"_(S[123])_(gpp|npp)$", lambda m: "_" + m.group(1).upper(), name, flags=re.IGNORECASE)
        name = re.sub(r"_(gpp|npp)$", "", name, flags=re.IGNORECASE)
        names.append(name)
    df.columns = names

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate model names after normalization: {duplicates}")
    if df.shape[1] == 0:
        raise ValueError(f"No numeric model columns: {path}")

    print(f"{path.name}: years={len(df)}, models={df.shape[1]}")
    print("Models:", df.columns.tolist())
    return df

def fit_pca_series(g, label):
    g = g.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if g.shape[1] == 0:
        raise ValueError(f"{label}: no model columns available for PCA")

    variable = g.std(axis=0, ddof=0) > 0
    g = g.loc[:, variable]
    if g.shape[1] == 0:
        raise ValueError(f"{label}: no varying model series")

    g2 = g.dropna(axis=0, how="any")
    if len(g2) < 3:
        print(f"{label}: valid records per model")
        print(g.notna().sum().to_string())
        raise ValueError(f"{label}: only {len(g2)} complete years; check model coverage")

    g2 = g2.loc[:, g2.std(axis=0, ddof=0) > 0]
    if g2.shape[1] == 0:
        raise ValueError(f"{label}: all model series are constant in the common period")

    X = StandardScaler().fit_transform(g2.to_numpy(dtype=float))
    p = PCA(n_components=1)
    pc1 = p.fit_transform(X).ravel()
    evr = float(p.explained_variance_ratio_[0])

    ref = g2.mean(axis=1).to_numpy(dtype=float)
    a, b = np.polyfit(pc1, ref, 1)
    series = pd.Series(a * pc1 + b, index=g2.index)
    series = series.reindex(np.arange(g2.index.min(), g2.index.max() + 1))

    print(f"{label}: models={g2.shape[1]}, complete years={len(g2)}, period={g2.index.min()}-{g2.index.max()}, PC1={evr*100:.2f}%")
    return series, evr

def prep_block(gpp_csv, npp_csv, label):
    g_gpp = load_df(gpp_csv)
    g_npp = load_df(npp_csv)

    common_models = g_gpp.columns.intersection(g_npp.columns, sort=False)
    common_years = g_gpp.index.intersection(g_npp.index).sort_values()

    if common_models.empty:
        raise ValueError(f"{label}: no matching GPP and NPP model names")
    if common_years.empty:
        raise ValueError(f"{label}: no matching GPP and NPP years")

    g_gpp = g_gpp.loc[:, common_models]
    g_npp = g_npp.loc[:, common_models]

    ratio = g_npp.loc[common_years, common_models].div(
        g_gpp.loc[common_years, common_models].where(g_gpp.loc[common_years, common_models] != 0)
    )
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    pc1_gpp, evr_gpp = fit_pca_series(g_gpp, f"{label} GPP")
    pc1_npp, evr_npp = fit_pca_series(g_npp, f"{label} NPP")
    pc1_ratio, evr_ratio = fit_pca_series(ratio, f"{label} NPP/GPP")

    xmin = int(min(g_gpp.index.min(), g_npp.index.min(), ratio.index.min()))
    xmax = int(max(g_gpp.index.max(), g_npp.index.max(), ratio.index.max()))

    return {
        "gpp_plot": g_gpp,
        "npp_plot": g_npp,
        "ratio_plot": ratio,
        "pc1_gpp": pc1_gpp,
        "pc1_npp": pc1_npp,
        "pc1_ratio": pc1_ratio,
        "evr_gpp": evr_gpp,
        "evr_npp": evr_npp,
        "evr_ratio": evr_ratio,
        "xmin": xmin,
        "xmax": xmax
    }

def set_3_sci_int_ticks(ax, exp=None):
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    y0, y1 = ax.get_ylim()
    maximum = max(abs(y0), abs(y1))

    if exp is None:
        exp = int(np.floor(np.log10(maximum))) if maximum > 0 else 0

    scale = 10.0 ** exp
    ticks = np.linspace(y0, y1, 5)[1:-1]
    step = (ticks[1] - ticks[0]) / scale
    decimals = max(0, int(np.ceil(-np.log10(step)))) if step > 0 else 1

    ax.set_yticks(ticks)
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/scale:.{decimals}f}"))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(0.0, 1.015, rf"$\times10^{{{exp}}}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=19)

def set_3_float_ticks(ax, nd=3):
    y0, y1 = ax.get_ylim()
    ticks = np.linspace(y0, y1, 3)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.{nd}f}"))

def mark_peak_year(ax, series):
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None

    peak_year = int(valid.idxmax())
    ax.axvline(peak_year, ls="--", color="black", lw=1.1, alpha=0.9)
    ax.text(
        peak_year, 0.035, str(peak_year),
        transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=19
    )
    return peak_year

def draw_model_series(ax, df, cmap):
    columns = list(df.columns)
    n = max(1, len(columns))
    for i, column in enumerate(columns):
        s = df[column].replace([np.inf, -np.inf], np.nan)
        s = s.reindex(np.arange(int(s.index.min()), int(s.index.max()) + 1))
        smoothed = s.rolling(3, min_periods=1, center=True).mean().where(s.notna())
        color = cmap(0.4 + i * 0.45 / max(1, n - 1))
        ax.plot(s.index, smoothed, lw=1.15, alpha=0.55, color=color, zorder=2)

def draw_col(ax_gpp,ax_npp,ax_ratio,blk):
    draw_model_series(ax_gpp,blk["gpp_plot"],plt.get_cmap("Blues"))
    draw_model_series(ax_npp,blk["npp_plot"],plt.get_cmap("Oranges"))
    ax_gpp.plot(blk["pc1_gpp"].index,blk["pc1_gpp"],lw=3.2,color=plt.get_cmap("Blues")(0.95),zorder=4)
    ax_npp.plot(blk["pc1_npp"].index,blk["pc1_npp"],lw=3.2,color=plt.get_cmap("Oranges")(0.95),zorder=4)
    ax_ratio.plot(blk["pc1_ratio"].index,blk["pc1_ratio"],lw=3.2,color=plt.get_cmap("Greens")(0.95),zorder=4)

warm = prep_block(warm_gpp_csv, warm_npp_csv, "Warm")
cold = prep_block(cold_gpp_csv, cold_npp_csv, "Cold")

fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
draw_col(axes[0, 0], axes[1, 0], axes[2, 0], warm)
draw_col(axes[0, 1], axes[1, 1], axes[2, 1], cold)

xleft = 1900
xright = 2025
xticks = np.arange(1900, 2021, 20)

for row in range(3):
    for col in range(2):
        ax = axes[row, col]
        ax.set_xlim(xleft, xright)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.grid(True, ls="--", lw=0.6, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", direction="out", labelsize=20, pad=8)
        ax.tick_params(axis="x", direction="out", labelsize=20, pad=8)

for ax in axes.ravel():
    ax.set_autoscaley_on(True)
    ax.relim()
    ax.margins(y=0.08)
    ax.autoscale_view(scalex=False, scaley=True)

for row in [0, 1, 2]:
    for col in [0, 1]:
        if row < 2:
            set_3_sci_int_ticks(axes[row, col])
        else:
            set_3_float_ticks(axes[row, col], nd=3)

mark_peak_year(axes[0, 0], warm["pc1_gpp"])
mark_peak_year(axes[1, 0], warm["pc1_npp"])
mark_peak_year(axes[2, 0], warm["pc1_ratio"])
mark_peak_year(axes[0, 1], cold["pc1_gpp"])
mark_peak_year(axes[1, 1], cold["pc1_npp"])
mark_peak_year(axes[2, 1], cold["pc1_ratio"])

axes[0, 0].set_ylabel(r"GPP ($\mathrm{kg\ m^{-2}\ s^{-1}}$)", fontsize=24, labelpad=12)
axes[1, 0].set_ylabel(r"NPP ($\mathrm{kg\ m^{-2}\ s^{-1}}$)", fontsize=24, labelpad=12)
axes[2, 0].set_ylabel("NPP/GPP", fontsize=24, labelpad=12)

for col, block in enumerate([warm, cold]):
    axes[0, col].text(0.04, 0.90, f"PC1: {block['evr_gpp']*100:.1f}%", transform=axes[0, col].transAxes, ha="left", va="top", fontsize=19)
    axes[1, col].text(0.04, 0.90, f"PC1: {block['evr_npp']*100:.1f}%", transform=axes[1, col].transAxes, ha="left", va="top", fontsize=19)
    axes[2, col].text(0.04, 0.90, f"PC1: {block['evr_ratio']*100:.1f}%", transform=axes[2, col].transAxes, ha="left", va="top", fontsize=19)
    axes[2, col].set_xlabel("Year", fontsize=24, labelpad=10)

for col, region in enumerate(["Warm", "Cold"]):
    axes[0, col].text(0.5, 1.015, region, transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=23, fontweight="bold")

for ax, panel in zip(axes.ravel(), ["a", "b", "c", "d", "e", "f"]):
    ax.text(0.97, 0.97, panel, transform=ax.transAxes, fontweight="bold", fontsize=26, ha="right", va="top")

fig.subplots_adjust(left=0.15, right=0.98, bottom=0.105, top=0.96, wspace=0.24, hspace=0.15)
fig.align_ylabels(axes[:, 0])

fig.canvas.draw()
for ax in axes[2, :]:
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_ha("left")

fig.savefig(out_png, bbox_inches="tight", pad_inches=0.08, dpi=600, facecolor="white")
plt.close(fig)
print("✅ Saved figure:", out_png)