import re
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import shape as shp_shape
from rasterio.transform import from_bounds
from rasterio.features import shapes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
mpl.rcParams["mathtext.fontset"] = "stix"
ROOT = Path("D:/")
MAP = ROOT / "data/map"
TRW = ROOT / "data/Processed data/TRW"
na_lon_min, na_lon_max = -167.27, -55.0
na_lat_min, na_lat_max = 9.58, 74.99
eu_lon_min, eu_lon_max = -11.0, 55.0
eu_lat_min, eu_lat_max = 29.0, 75.0
csv_tree = ROOT / "data/1. Basic information of global tree-ring sites.csv"
trw_csv = ROOT / "data/Processed data/Detrending/fill/AgeDepSpline+fill.csv"
flux_csv_americas = ROOT / "data/2. Coordinates of Americas flux tower sites.csv"
flux_csv_existing = ROOT / "data/4. Coordinates of North American and European flux tower sites.csv"
flux_csv_europe = ROOT / "data/3. Coordinates of European ICOS flux tower sites.csv"
extra_site_csv = ROOT / "data/5. NPP site coordinate table.csv"
npp_site_csv = ROOT / "data/6. Annual NPP values at observation sites.csv"
correlation_csv = ROOT / "data/Processed data/median/Median_correlations.csv"
world_shp = MAP / "World.shp"
shp_boundary = ROOT / "data/Map/Boundary_of_warm_zone_and_cold_zone.shp"
koppen_na = MAP / "Koppen_1991_2020_NA_big7.shp"
koppen_eu = MAP / "Europe 7.shp"
nc_tree = TRW / "AgeDepSpline (5+3).nc"
out_png = ROOT / 'results/Fig. 1.png'
size_flux = 80
size_npp = 80
size_tree = 100
lw_tree = 1.2
aspect_y_over_x = 1.3
color_map = {"A": "#ff6e40", "B": "#ffd166", "C": "#9fbfad", "Dfa": "#cfe8ff", "Dfb": "#7ab6f2", "Cold": "#0f4c81"}
CATEGORIES = ["DGVM_S2", "DGVM_S3", "CMIP6", "NPP1-2"]
CATEGORY_LABELS = {"DGVM_S2": "DGVM_S2", "DGVM_S3": "DGVM_S3", "CMIP6": "CMIP6", "NPP1-2": "NPP (1-2)"}
CLIMATE_CLASSES = ["A", "B", "C", "Dfa", "Dfb", "Warm", "Cold", "All"]
CLIMATE_LABELS = ["Tropical", "Arid", "Temperate", "Dfa", "Dfb", "Warm", "Cold", "All"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "^", "P", "D", "X"]
OFFSETS = np.linspace(-0.28, 0.28, len(CATEGORIES))
MODEL_NAMES = {}
def cls_simple(s):
    s = str(s).strip()
    code = s.split()[0] if " " in s else s
    return code if code in ["A", "B", "C", "Dfa", "Dfb"] else "Cold"
def format_lon(x, pos):
    if np.isclose(x, 0):
        return "0°"
    return f'{abs(x):g}°{"E" if x > 0 else "W"}'
def format_lat(y, pos):
    if np.isclose(y, 0):
        return "0°"
    return f'{abs(y):g}°{"N" if y > 0 else "S"}'
def normalize_site(value):
    return str(value).replace("\ufeff", "").strip().casefold()
def read_tree_coordinates(metadata_path, series_path):
    headers = pd.read_csv(series_path, nrows=0, encoding="utf-8-sig").columns
    site_names = [str(col).strip() for col in headers if normalize_site(col) != "year" and not normalize_site(col).startswith("unnamed:")]
    if not site_names:
        raise ValueError(f"No site columns found in {series_path.name}")
    requested = pd.DataFrame({"trw_site": site_names})
    requested["site_key"] = requested["trw_site"].map(normalize_site)
    if requested["site_key"].duplicated().any():
        duplicates = requested.loc[requested["site_key"].duplicated(keep=False), "trw_site"].tolist()
        raise ValueError(f"Duplicate tree-ring site names: {duplicates}")
    metadata = pd.read_csv(metadata_path, dtype=str, encoding="utf-8-sig")
    metadata.columns = metadata.columns.str.replace("\ufeff", "", regex=False).str.strip()
    columns = {col.casefold(): col for col in metadata.columns}
    required = ["name", "longitude", "latitude"]
    missing_columns = [col for col in required if col not in columns]
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")
    metadata = metadata.rename(columns={columns[col]: col for col in required})
    metadata["site_key"] = metadata["name"].map(normalize_site)
    metadata = metadata[metadata["site_key"].isin(requested["site_key"])].copy()
    metadata["longitude"] = pd.to_numeric(metadata["longitude"], errors="coerce")
    metadata["latitude"] = pd.to_numeric(metadata["latitude"], errors="coerce")
    coordinate_valid = np.isfinite(metadata["longitude"]) & np.isfinite(metadata["latitude"]) & metadata["longitude"].between(-180, 360) & metadata["latitude"].between(-90, 90)
    metadata = metadata.loc[coordinate_valid].copy()
    metadata["longitude"] = ((metadata["longitude"] + 180) % 360) - 180
    coordinates = metadata[["site_key", "longitude", "latitude"]].drop_duplicates()
    conflicts = coordinates.loc[coordinates["site_key"].duplicated(keep=False), "site_key"].unique()
    if len(conflicts):
        names = requested.loc[requested["site_key"].isin(conflicts), "trw_site"].tolist()
        raise ValueError(f"Conflicting coordinates for tree-ring sites: {names}")
    result = requested.merge(coordinates, on="site_key", how="left", validate="one_to_one", indicator=True)
    missing = result.loc[result["_merge"].ne("both"), "trw_site"].tolist()
    if missing:
        raise ValueError(f"Missing or invalid coordinates for {len(missing)} tree-ring sites: {missing}")
    result = result.drop(columns=["_merge"])
    print("Tree-ring sites in source CSV:", len(requested))
    print("Tree-ring sites matched to coordinates:", len(result))
    return gpd.GeoDataFrame(result, geometry=gpd.points_from_xy(result["longitude"], result["latitude"]), crs="EPSG:4326")
def read_flux_coordinates(path):
    df = pd.read_csv(path)
    columns = {str(col).strip().lower().replace(" ", "").replace("_", ""): col for col in df.columns}
    lat_col = next((columns[name] for name in ["latitude", "lat", "sitelatitude"] if name in columns), None)
    lon_col = next((columns[name] for name in ["longitude", "long", "lon", "sitelongitude"] if name in columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Latitude or longitude column not found in {path.name}: {list(df.columns)}")
    result = pd.DataFrame({"latitude": pd.to_numeric(df[lat_col], errors="coerce"), "longitude": pd.to_numeric(df[lon_col], errors="coerce")})
    result = result.dropna(subset=["latitude", "longitude"]).copy()
    result["longitude"] = ((result["longitude"] + 180) % 360) - 180
    return result[result["latitude"].between(-90, 90)].copy()
def valid_gdf_from_nc(nc_path):
    with xr.open_dataset(nc_path) as ds:
        da = ds["AgeDepSpline"].sel(year=slice(1981, 2000))
        mask = np.isfinite(da).any(dim="year").values.astype("uint8")
        lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
    arr = mask[::-1, :] if bool(lat[-1] > lat[0]) else mask
    transform = from_bounds(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()), len(lon), len(lat))
    polys = [shp_shape(geom) for geom, value in shapes(arr, mask=arr == 1, transform=transform) if value == 1]
    if not polys:
        raise ValueError("No valid NetCDF area was found for 1981-2000")
    geometry = gpd.GeoSeries(polys, crs="EPSG:4326").union_all().buffer(0)
    return gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
def prep_region(lon_min, lon_max, lat_min, lat_max, tree_gdf, flux_gdf, world, valid_gdf, koppen_shp):
    world_bbox = world.cx[lon_min:lon_max, lat_min:lat_max]
    tree_visible = tree_gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()
    flux_clip = gpd.clip(flux_gdf, valid_gdf)
    kop = gpd.read_file(koppen_shp)
    kop = kop.set_crs("EPSG:4326") if kop.crs is None else kop.to_crs("EPSG:4326")
    kop = kop.cx[lon_min:lon_max, lat_min:lat_max].clip(valid_gdf)
    kop["class_simple"] = kop["class"].apply(cls_simple)
    kop7 = kop.dissolve(by="class_simple", as_index=False, aggfunc="first")
    kop7["facecolor"] = kop7["class_simple"].map(color_map).fillna(color_map["Cold"])
    bbox = np.array([lon_min, lat_min, lon_max, lat_max])
    return world_bbox, kop7, tree_visible, flux_clip, bbox
def visible_points(gdf, bbox):
    return gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
def model_name(row):
    name = str(row["product"])
    name = MODEL_NAMES.get(name, name)
    name = re.sub(r"\.nc$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^\d+[_\s]+", "", name)
    return re.sub(r"_S[23]_npp$", "", name, flags=re.IGNORECASE)
def plot_correlations(fig, ax, path):
    df = pd.read_csv(path)
    required = ["category", "product"] + CLIMATE_CLASSES
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[df["category"].isin(CATEGORIES)].copy()
    if "status" in df.columns:
        excluded = df[~df["status"].eq("OK")]
        if not excluded.empty:
            print("Excluded incomplete products:")
            print(excluded[["category", "product", "status"]].to_string(index=False))
        df = df[df["status"].eq("OK")].copy()
    for name in CLIMATE_CLASSES:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.reset_index(drop=True)
    values = df[CLIMATE_CLASSES].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise RuntimeError("No valid median correlations")
    if np.any(np.abs(finite) > 1.000001):
        raise ValueError("Correlation values outside [-1, 1]")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="in", length=6, labelsize=22, pad=8)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=1)
    positions = {}
    point_positions = []
    for j, category in enumerate(CATEGORIES):
        subset = df[df["category"].eq(category)]
        for i, climate in enumerate(CLIMATE_CLASSES):
            valid = subset[np.isfinite(subset[climate])]
            y = valid[climate].to_numpy(dtype=float)
            x = np.full(y.size, i + OFFSETS[j])
            ax.scatter(x, y, s=95, color=COLORS[j], marker=MARKERS[j], edgecolors="black", linewidths=0.55, alpha=0.85, zorder=3)
            for index, xp, yp in zip(valid.index, x, y):
                positions[(i, index)] = (float(xp), float(yp))
                point_positions.append((float(xp), float(yp)))
    lower = min(-0.2, float(np.floor((finite.min() - 0.04) * 5) / 5))
    upper = max(0.8, float(finite.max()) + 0.30)
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.65, len(CLIMATE_CLASSES) - 0.35)
    ax.set_xticks(np.arange(len(CLIMATE_CLASSES)))
    ax.set_xticklabels(CLIMATE_LABELS, fontsize=22)
    ax.set_xlabel("Climate zones", fontsize=25, labelpad=12)
    ax.set_ylabel(r"$r$ (Tree-ring & NPP)", fontsize=24, labelpad=12)
    handles = [Line2D([], [], linestyle="none", marker=MARKERS[j], color=COLORS[j], markeredgecolor="black", markeredgewidth=0.6, markersize=10, label=CATEGORY_LABELS[category]) for j, category in enumerate(CATEGORIES)]
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=4, frameon=False, fontsize=22, handletextpad=0.5, columnspacing=1.6, labelspacing=0.55, borderaxespad=0.35)
    panel_label = ax.text(0.005, 0.985, "c", transform=ax.transAxes, ha="left", va="top", fontsize=35, fontweight="bold", zorder=10)
    annotations = []
    for i, climate in enumerate(CLIMATE_CLASSES):
        column = df[climate].to_numpy(dtype=float)
        valid = np.isfinite(column)
        if not valid.any():
            continue
        maximum = float(np.max(column[valid]))
        indices = np.flatnonzero(valid & np.isclose(column, maximum, rtol=0, atol=1e-12))
        for index in indices:
            row = df.loc[index]
            xp, yp = positions[(i, index)]
            name = model_name(row)
            ax.scatter([xp], [yp], s=290, marker="o", facecolors="none", edgecolors="red", linewidths=2.1, zorder=5)
            annotation = ax.annotate(name, xy=(xp, yp), xytext=(0, 20), textcoords="offset points", ha="center", va="bottom", fontsize=17, color="black", arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.8, "shrinkA": 4, "shrinkB": 10}, annotation_clip=False, zorder=6)
            annotations.append((annotation, xp, yp))
            print(f"{climate}: {name} | {CATEGORY_LABELS[row['category']]} | median r={yp:.4f}")
    points = np.asarray(point_positions)
    for attempt in range(40):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        scale = fig.dpi / 72
        screen_points = ax.transData.transform(points)
        placed = []
        highest = ax.bbox.y0
        for annotation, xp, yp in annotations:
            anchor = ax.transData.transform((xp, yp))
            width, height, descent = renderer.get_text_width_height_descent(annotation.get_text(), annotation.get_fontproperties(), ismath=False)
            width += 8 * scale
            height += 8 * scale
            left, right = anchor[0] - width / 2, anchor[0] + width / 2
            nearby = (screen_points[:, 0] >= left - 12 * scale) & (screen_points[:, 0] <= right + 12 * scale)
            ceiling = screen_points[nearby, 1].max() if nearby.any() else anchor[1]
            bottom = max(anchor[1] + 20 * scale, ceiling + 16 * scale)
            while True:
                collisions = [box for box in placed if left < box[2] and right > box[0] and bottom < box[3] + 7 * scale and bottom + height > box[1] - 7 * scale]
                if not collisions:
                    break
                bottom = max(box[3] for box in collisions) + 9 * scale
            annotation.set_position((0, (bottom - anchor[1]) / scale))
            placed.append((left, bottom, right, bottom + height))
            highest = max(highest, bottom + height)
        reserved_bottom = min(legend.get_window_extent(renderer).y0, panel_label.get_window_extent(renderer).y0)
        if highest + 12 * scale < reserved_bottom:
            break
        current_lower, current_upper = ax.get_ylim()
        ax.set_ylim(current_lower, current_upper + max(0.08, (current_upper - current_lower) * 0.10))
    ax.set_yticks(np.round(np.arange(np.ceil(ax.get_ylim()[0] / 0.2) * 0.2, ax.get_ylim()[1] + 0.001, 0.2), 2))
tree_all = read_tree_coordinates(csv_tree, trw_csv)
flux_raw = pd.concat([read_flux_coordinates(path) for path in [flux_csv_americas, flux_csv_existing, flux_csv_europe]], ignore_index=True)
flux_raw["lon_key"] = flux_raw["longitude"].round(2)
flux_raw["lat_key"] = flux_raw["latitude"].round(2)
flux_raw = flux_raw.drop_duplicates(subset=["lon_key", "lat_key"]).drop(columns=["lon_key", "lat_key"]).reset_index(drop=True)
flux_all = gpd.GeoDataFrame(flux_raw, geometry=gpd.points_from_xy(flux_raw["longitude"], flux_raw["latitude"]), crs="EPSG:4326")
world = gpd.read_file(world_shp)
world = world.set_crs("EPSG:4326") if world.crs is None else world.to_crs("EPSG:4326")
boundary = gpd.read_file(shp_boundary)
boundary = boundary.set_crs("EPSG:4326") if boundary.crs is None else boundary.to_crs("EPSG:4326")
valid_gdf = valid_gdf_from_nc(nc_tree)
world_na_bbox, kop_na, base_na, flux_na, bbox_na = prep_region(na_lon_min, na_lon_max, na_lat_min, na_lat_max, tree_all, flux_all, world, valid_gdf, koppen_na)
world_eu_bbox, kop_eu, base_eu, flux_eu, bbox_eu = prep_region(eu_lon_min, eu_lon_max, eu_lat_min, eu_lat_max, tree_all, flux_all, world, valid_gdf, koppen_eu)
displayed_tree_keys = set(base_na["site_key"]) | set(base_eu["site_key"])
outside_tree = tree_all.loc[~tree_all["site_key"].isin(displayed_tree_keys), "trw_site"].tolist()
print("Tree-ring sites outside the map extents:", len(outside_tree))
if outside_tree:
    print("Sites outside the map extents:", ", ".join(outside_tree))
df_extra = pd.read_csv(extra_site_csv, usecols=["site_ID", "latitude", "longitude"], dtype={"site_ID": str})
df_npp = pd.read_csv(npp_site_csv, usecols=["site_ID", "NPP_tot1"], dtype={"site_ID": str})
df_extra["site_ID"] = df_extra["site_ID"].str.strip()
df_npp["site_ID"] = df_npp["site_ID"].str.strip()
df_extra["longitude"] = pd.to_numeric(df_extra["longitude"], errors="coerce")
df_extra["latitude"] = pd.to_numeric(df_extra["latitude"], errors="coerce")
df_npp["NPP_tot1"] = pd.to_numeric(df_npp["NPP_tot1"], errors="coerce")
df_extra = df_extra.dropna(subset=["site_ID", "latitude", "longitude"]).drop_duplicates(subset=["site_ID"])
valid_npp_ids = df_npp.loc[df_npp["NPP_tot1"].notna(), "site_ID"].dropna().drop_duplicates()
dfm = df_extra[df_extra["site_ID"].isin(valid_npp_ids)].copy()
gdf_extra = gpd.GeoDataFrame(dfm, geometry=gpd.points_from_xy(dfm["longitude"], dfm["latitude"]), crs="EPSG:4326")
gdf_extra = gpd.clip(gdf_extra, valid_gdf)
flux_na = visible_points(flux_na, bbox_na)
flux_eu = visible_points(flux_eu, bbox_eu)
extra_na = visible_points(gdf_extra, bbox_na)
extra_eu = visible_points(gdf_extra, bbox_eu)
n_flux = len(flux_na) + len(flux_eu)
n_npp = len(extra_na) + len(extra_eu)
n_tree = len(base_na) + len(base_eu)
fig = plt.figure(figsize=(18, 18), dpi=150)
ax_na = fig.add_axes([0.055, 0.58, 0.49, 0.37])
ax_eu = fig.add_axes([0.575, 0.58, 0.405, 0.37])
ax_c = fig.add_axes([0.075, 0.07, 0.93, 0.45])
for ax, world_region, koppen_region, tree_region, flux_region, npp_region, bbox in [
    (ax_na, world_na_bbox, kop_na, base_na, flux_na, extra_na, bbox_na),
    (ax_eu, world_eu_bbox, kop_eu, base_eu, flux_eu, extra_eu, bbox_eu)
]:
    world_region.plot(ax=ax, facecolor="#f0f0f0", edgecolor="gray", linewidth=0.5, zorder=1)
    koppen_region.plot(ax=ax, color=koppen_region["facecolor"], edgecolor="none", alpha=0.6, zorder=2.3)
    ax.scatter(flux_region.geometry.x.values, flux_region.geometry.y.values, s=size_flux, c="green", edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3.4)
    ax.scatter(npp_region.geometry.x.values, npp_region.geometry.y.values, s=size_npp, c="red", edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3.5)
    ax.scatter(tree_region.geometry.x.values, tree_region.geometry.y.values, s=size_tree, c="black", marker="+", linewidths=lw_tree, alpha=0.95, zorder=3.8)
    boundary.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].plot(ax=ax, color="white", linewidth=4.0, zorder=5)
    boundary.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].plot(ax=ax, color="#111111", linewidth=2.4, linestyle=(0, (4, 1.5)), zorder=6)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(aspect_y_over_x, adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.tick_params(axis="both", which="major", labelsize=19)
ax_eu.yaxis.tick_right()
ax_eu.yaxis.set_label_position("right")
ax_eu.tick_params(axis="y", which="major", right=True, left=False)
pt_flux = ax_na.scatter([], [], s=size_flux, color="green", edgecolor="white", linewidth=0.4, alpha=0.9, label=f"Flux tower (n={n_flux})")
pt_npp = ax_na.scatter([], [], s=size_npp, color="red", edgecolor="white", linewidth=0.4, alpha=0.9, label=f"NPP site (n={n_npp})")
pt_tree = ax_na.scatter([], [], s=size_tree, color="black", marker="+", linewidth=lw_tree, alpha=0.95, label=f"Tree-ring (n={n_tree})")
legend_name_map = {"A": "Tropical", "B": "Arid", "C": "Temperate", "Dfa": "Dfa", "Dfb": "Dfb", "Cold": "Cold"}
classes = ["A", "B", "C", "Dfa", "Dfb", "Cold"]
faces = [Patch(facecolor=color_map[key], edgecolor="none", alpha=0.6, label=legend_name_map[key]) for key in classes]
line_bnd = Line2D([0], [0], color="#111111", lw=2.4, ls=(0, (4, 1.5)), label="Warm/Cold boundary")
ax_na.legend(handles=[pt_flux, pt_npp, pt_tree] + faces + [line_bnd], loc="lower left", frameon=True, prop={"size": 18}, scatterpoints=1, borderpad=0.5, labelspacing=0.35, handletextpad=0.5, handlelength=1.5, markerscale=1.1, ncol=1)
ax_na.text(0.015, 0.985, "a", transform=ax_na.transAxes, ha="left", va="top", fontsize=35, fontweight="bold", zorder=10)
ax_eu.text(0.015, 0.985, "b", transform=ax_eu.transAxes, ha="left", va="top", fontsize=35, fontweight="bold", zorder=10)
plot_correlations(fig, ax_c, correlation_csv)
ax_c.set_ylim(top=1.0)
ax_c.set_yticks(np.round(np.arange(np.ceil(ax_c.get_ylim()[0] / 0.2) * 0.2, 1.001, 0.2), 2))
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.close(fig)
print("North America tree-ring count:", len(base_na))
print("Europe tree-ring count:", len(base_eu))
print("Tree-ring total displayed:", n_tree)
print("North America NPP site count:", len(extra_na))
print("Europe NPP site count:", len(extra_eu))
print("NPP site total:", n_npp)
print("North America flux tower count:", len(flux_na))
print("Europe flux tower count:", len(flux_eu))
print("Flux tower total:", n_flux)
print('✅ Saved figure:', out_png)