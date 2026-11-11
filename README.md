# Code and Figure Reproduction

**Persistent decline in the relative allocation of net primary productivity to xylem formation**

This document describes the local data, analysis scripts, and figure-reproduction workflow for the study. The project uses a Windows directory structure rooted at `D:/`. The current figure set contains four main figures, nine Extended Data figures, and six Supplementary figures.

## 1. Project organization

| Location | Contents |
|---|---|
| `D:/README.md` | Project documentation. |
| `D:/code/` | Nineteen figure scripts and `run_all.py`. |
| `D:/data/` | Nine numbered site and observation CSV tables. |
| `D:/data/Map/` | Geographic boundaries, land polygons, and climate-zone layers. |
| `D:/data/NPP and GPP/` | Productivity products and processing scripts. |
| `D:/data/Processed data/` | Tree-ring chronologies, gridded reconstructions, correlation products, and climate-analysis outputs. |
| `D:/results/` | Figure outputs. |

The productivity folders are `CMIP6 (NPP)`, `NPP (1-2)`, `S2_npp9-8`, `S2_npptree9-8`, `S3_gpp1-12`, `S3_npp1-12`, `S3_npp9-8`, and `transform`.

The processed-data folders are `Detrending`, `Difference_correlation`, `GPP and NPP`, `median`, `r_diff and r_raw`, `Sliding difference`, and `TRW`. The current local collection does not contain the `Detrending5` directory referenced in the earlier README, and none of the current figure scripts uses it.

## 2. Software and path configuration

The Python scripts use NumPy, pandas, SciPy, xarray, Matplotlib, GeoPandas, Shapely, Rasterio, netCDF4, cftime, scikit-learn, PyKrige, and joblib. Tree-ring detrending also requires R and the `dplR` package.

```powershell
python -m pip install numpy pandas scipy xarray matplotlib geopandas shapely rasterio netCDF4 cftime scikit-learn pykrige joblib
```

```r
install.packages("dplR")
```

The figure scripts use Times New Roman and save PNGs at 600 dpi. Package versions are not pinned; no environment lockfile is specified by this workflow.

Figure scripts generally define `ROOT = Path('D:/')`. The batch runner separately defines `ROOT = Path('D:/code')`. Several preprocessing scripts also define absolute input and output paths independently of `ROOT`. If the project is relocated, update these paths in each script, including the climate-data path on `H:/`.

The actual map directory is `data/Map/`, although some scripts refer to `data/map/`. These paths are equivalent on the usual Windows filesystem but must be made consistent on a case-sensitive filesystem. Keep the companion files for each Shapefile together.

## 3. Generate the figures

To run a single figure script:

```powershell
python "D:/code/Fig. 1.py"
```

To run all figure scripts:

```powershell
python "D:/code/run_all.py"
```

The runner recursively collects Python files from `D:/code/`, excludes itself and `__pycache__`, sorts the scripts by relative path, and executes them sequentially. It continues after a script fails, reports all failures at the end, and returns a nonzero exit status if any script failed. The order is alphabetical rather than manuscript order. Additional Python files placed in `code/` will also be executed.

Each figure script writes a PNG with the corresponding figure name to `D:/results/`, replacing an existing file with the same name. Preprocessing scripts under `data/` are not executed by the runner.

Most figure scripts read existing processed inputs. Supplementary Fig. 1 also repeats spatial interpolation for its robustness analysis, and Extended Data Fig. 9 performs kriging validation within the figure script. These two scripts require substantially more computation than plots based only on saved summaries.

## 4. Figure scripts and inputs

Each entry below corresponds to `code/<Figure>.py` and `results/<Figure>.png`. Paths beginning with `TRW/`, `Detrending/`, `median/`, `GPP and NPP/`, `r_diff and r_raw/`, or `Sliding difference/` refer to folders under `data/Processed data/`.

| Figure | Analysis or display | Main inputs |
|---|---|---|
| Fig. 1 | Tree-ring and productivity-observation sites, climate zones, and regional TRW–NPP correlation summaries. | Numbered site/observation tables; maps; `Detrending/fill/AgeDepSpline+fill.csv`; `TRW/AgeDepSpline (5+3).nc`; `median/Median_correlations.csv`. |
| Fig. 2 | Long-term standardized TRW/NPP changes and spatial trends in North America and Europe. | `TRW/AgeDepSpline (5+3).nc`; ISBA-CTRIP, CLASSIC, and iMAPLE S2 NPP; regional maps. |
| Fig. 3 | Spatial first-difference TRW–NPP correlations and regional moving-correlation comparisons. | AgeDepSpline `(5+3)` reconstruction; ISBA-CTRIP S2 NPP for the directly calculated correlations; maps; saved AgeDepSpline `(5+3)` regional moving-correlation CSVs. |
| Fig. 4 | Spatial climate associations and their temporal evolution. | `Sliding difference/Sliding difference1-1900-center.nc`; AgeDepSpline `(5+3)` reconstruction; maps; combined-region moving-correlation CSVs. |
| Extended Data Fig. 1 | Climate-zone moving correlations for the S2 NPP products. | AgeDepSpline `(5+3)` reconstruction; `data/NPP and GPP/S2_npp9-8/`; regional climate maps. |
| Extended Data Fig. 2 | TRW/NPP sensitivity to detrending method, spatial support, and the nppTree alternative. | ModNegExp, SFRCS, Spline, and AgeDepSpline reconstructions; selected S2 NPP products; ISBA-CTRIP nppTree; maps. |
| Extended Data Fig. 3 | Site-level TRW/NPP changes, spatial trends, plant-group comparisons, and the 12 most represented species. | Tree-ring site metadata; `Detrending/fill/AgeDepSpline+fill.csv`; the three selected S2 NPP products; world boundaries. |
| Extended Data Fig. 4 | Warm- and cold-region GPP, NPP, and NPP/GPP time series. | `GPP and NPP/warm_gpp.csv`, `warm_npp.csv`, `cold_gpp.csv`, and `cold_npp.csv`. |
| Extended Data Fig. 5 | Moving-correlation sensitivity to detrending, spatial support, and the nppTree configuration. | Six alternative configuration folders in `r_diff and r_raw/`. |
| Extended Data Fig. 6 | Climate-zone moving correlations for North America, Europe, and the combined region. | AgeDepSpline `(5+3)` regional moving-correlation CSVs. |
| Extended Data Fig. 7 | Temporal changes in climate-control effects alongside moving correlations. | Sliding-difference NetCDF; maps; AgeDepSpline `(5+3)/All r_diff/` CSVs. |
| Extended Data Fig. 8 | Climate-zone summaries of correlation changes after controlling for climate variables. | Sliding-difference NetCDF and regional climate maps. |
| Extended Data Fig. 9 | Site-pair correlations versus distance, variogram-model validation, and cumulative kriging weights. | Tree-ring site metadata; filled AgeDepSpline chronologies; `TRW/recon_mask (5+3).nc`. |
| Supplementary Fig. 1 | Robustness of gridded TRW to spatially stratified removal of 10% of sites, repeated five times for four detrending methods. | Site metadata; four filled chronology tables; four full-site `(5+3)` reconstructions; `TRW/recon_mask (5+3).nc`. |
| Supplementary Fig. 2 | North American spatial first-difference TRW–NPP correlations during 1981–2010. | AgeDepSpline `(5+3)` reconstruction; S2 NPP products; `World_countries.shp`. |
| Supplementary Fig. 3 | European spatial first-difference TRW–NPP correlations during 1981–2010. | AgeDepSpline `(5+3)` reconstruction; S2 NPP products; `World.shp`. |
| Supplementary Fig. 4 | Observation-site counts and temporal coverage within the study domains. | Site and observation tables; `Detrending/AgeDepSpline.csv`; AgeDepSpline reconstruction domain. |
| Supplementary Fig. 5 | Pooled observation–model GPP and NPP correlations. | Flux-tower coordinates and daytime GPP observations; NPP observations and coordinates; S3 GPP and NPP products. |
| Supplementary Fig. 6 | Site-level GPP validation for products selected separately using median daytime- and nighttime-method correlations. | Daytime and nighttime GPP tables; flux-tower coordinates; S3 GPP products. |

### Supplementary Fig. 1: site-removal robustness

The script covers 1900–2023 and evaluates AgeDepSpline, ModNegExp, SFRCS, and Spline. It removes 10% of eligible unique site coordinates using spatially stratified sampling with 5° strata, repeats the procedure five times, and reconstructs TRW with the remaining coordinates. Sites sharing a coordinate are handled together. The analysis uses spherical ordinary kriging, up to 100 neighboring coordinates, and five parallel jobs.

The reduced-site reconstruction is compared with the corresponding full-site NetCDF over paired valid grid-year values. Each panel reports the median Pearson correlation and NRMSE across the five repeats. NRMSE is RMSE divided by the sample standard deviation of the full-site values. Display sampling and axis trimming affect the plotted density only; metrics use all paired valid values.

The script checks consistency between the saved baseline reconstruction and the current inputs/settings before running the comparisons. Despite the internal folder name `TRW/leave-one-out`, this is a repeated 10% removal analysis. The current script creates that folder but keeps the reconstruction comparisons in memory and prints numerical summaries to the console; its saved deliverable is the figure PNG.

## 5. Data files

### Site and observation tables

| File in `data/` | Contents |
|---|---|
| `1. Basic information of global tree-ring sites.csv` | Tree-ring site coordinates, species, and plant classifications. |
| `2. Coordinates of Americas flux tower sites.csv` | Americas flux-tower coordinates. |
| `3. Coordinates of European ICOS flux tower sites.csv` | European ICOS flux-tower coordinates. |
| `4. Coordinates of North American and European flux tower sites.csv` | Combined flux-tower coordinates. |
| `5. NPP site coordinate table.csv` | NPP observation-site coordinates. |
| `6. Annual NPP values at observation sites.csv` | Site NPP observations and observation intervals. |
| `7. Carbon flux & remote sensing vs. tree-ring chronology median correlations.csv` | Separate supplied correlation-summary table. |
| `8. Annual GPP data of GPP_DT_VUT method.csv` | Annual daytime-method GPP observations. |
| `9. Annual GPP data of GPP_NT_VUT method.csv` | Annual nighttime-method GPP observations. |

Fig. 1 reads `Processed data/median/Median_correlations.csv`, which is distinct from numbered table 7. Extended Data Fig. 3 requires the metadata fields `name`, `latitude`, `longitude`, `species`, `Gymnosperm/Angiosperm`, and `Needle/broad`.

### Maps and productivity products

`data/Map/` includes world boundaries, land polygons, warm/cold-region boundaries, and regional climate layers. Common inputs include `World.shp`, `World_countries.shp`, `Europe 7.shp`, `Europe 7-class.shp`, `Koppen_1991_2020_NA_big7.shp`, and `Koppen_1991_2020_NA_big7_simple.shp`.

In productivity-directory names, `1-12` denotes January–December aggregation and `9-8` denotes September–August aggregation. Verify the time coordinate and units of each NetCDF before combining products. Directory names alone do not verify the aggregation of files produced by transformation scripts with shared output destinations (see Section 7).

### Processed products

| Folder | Contents and role |
|---|---|
| `Detrending/` | Site chronology outputs, consolidated method tables, and filled chronology tables in `fill/`. |
| `TRW/` | Reconstruction masks and 0.5° gridded TRW. Configurations `(3+3)`, `(4+3)`, and `(5+3)` denote 3°, 4°, and 5° spatial support with a minimum of three sites. |
| `median/` | Regional TRW–NPP correlation summaries used in product evaluation. |
| `Difference_correlation/` | First-difference TRW–NPP correlation products for 1981–2010. |
| `GPP and NPP/` | Warm- and cold-region productivity CSVs. |
| `r_diff and r_raw/` | Moving-correlation scripts and outputs for detrending/spatial configurations, with regional CSVs in `North America r_diff`, `Europe r_diff`, and `All r_diff`. |
| `Sliding difference/` | Climate-control script and `Sliding difference1-1900-center.nc`. |

## 6. Rebuild processed inputs

Preprocessing is organized by task and is not executed as a single end-to-end pipeline. Prepare the required source datasets and inspect each script's paths before rebuilding.

1. **Productivity preparation.** Use scripts under `data/NPP and GPP/transform/`, `CMIP6 (NPP)/`, and `NPP (1-2)/` as appropriate. `transform/1.nc.py` generates the common mask used by the product transformations. A dedicated S2 NPP transformation script is not present in `transform/`; the existing S2 NPP NetCDFs are downstream inputs.
2. **Tree-ring detrending.** Run `Processed data/Detrending/AgeDepSpline+Spline+ModNegExp+SFRCS.R`, then the corresponding `.py` script to consolidate the four chronology methods.
3. **Gap filling.** Run `Detrending/fill/1.fill.py` to produce the `+fill.csv` tables and climate-correlation summaries. `2.R and P.py` plots the summary statistics.
4. **Spatial reconstruction.** Generate the matching mask using a `TRW/recon_mask` script, then run `TRW/3+3.py`, `4+3.py`, or `5+3.py`. The current `5+3.py` processes all four detrending methods with spherical ordinary kriging, up to 100 neighboring coordinates, and five parallel jobs.
5. **Correlation and productivity summaries.** Run `median/Median_correlations.py`, `Difference_correlation/Difference_correlation.py`, `GPP and NPP/Extracting GPP and NPP.py`, and the relevant configuration scripts in `r_diff and r_raw/`.
6. **Climate-control analysis.** Run `Sliding difference/Sliding difference1-1900-center.py` with the TRW reconstruction, selected S2 NPP products, and climate input. The output uses `window_center_year` as its time dimension.
7. **Figure reproduction.** Run the individual figure scripts or `code/run_all.py` after their inputs are ready.

The regional moving-correlation preprocessing scripts and the climate-control script use 21-year windows on first-differenced series, with center-year labels. Some scripts request data through 2025 or 2026, whereas the manuscript analysis covers 1900–2023. Preserve the intended study period when regenerating or interpreting products.

## 7. Implementation details to check before rebuilding

| Item | Current implementation |
|---|---|
| Extended Data Fig. 1 window | Sets `WIN=20`, unlike the 21-year setting in moving-correlation preprocessing. |
| Fig. 3 productivity input | Directly calculated correlations use ISBA-CTRIP S2 NPP; moving curves are read from saved CSVs. These components should not be described as if they necessarily use identical product combinations. |
| nppTree moving-correlation configuration | Its preprocessing script lists ISBA-CTRIP nppTree together with SDGVM and JULES S2 NPP. It is therefore a mixed-product configuration. |
| Transformation output directories | `S2_nppTree_transform-1900.py` directs both `OUT_12` and `OUT_98` to `S2_npptree9-8/`; `S3_gpp_transform.py` directs both to `S3_gpp1-12/`. Separate destinations before retaining both aggregations, because identical filenames can overwrite each other. |
| Leading filename space | Extended Data Fig. 3 and the climate-control script include a leading space before `CLASSIC_S2_npp.nc`, with a fallback that strips it. The actual file is named without that space. |
| Mask-script filename | The `(3+3)` mask script is named `recon_mask (3+3) .py`, including a space before `.py`. |

These are descriptions of the inspected code. Rewriting this README does not modify the analysis scripts or regenerate their outputs.

## 8. Source data and external inputs

Tree-ring detrending reads original measurements from `D:/Tree rings/rwl/width/`. Gap filling and climate-control analysis read `H:/Climatic data/month/climate14.nc`. Productivity preprocessing also requires original source files; inspect the input definitions in the relevant transformation script rather than assuming that the processed NetCDFs contain all upstream inputs.

The original documentation identifies the following sources:

- Tree-ring measurements: International Tree-Ring Data Bank (ITRDB).
- NPP products: [National Earth System Science Data Center](https://www.nesdc.org.cn/sdo/detail?id=612f42ee7e28172cbed3d809) and [GLASS](https://www.glass.hku.hk/download.html).
- Model productivity: [Global Carbon Budget](https://mdosullivan.github.io/GCB/), with download commands in `data/NPP and GPP/transform/wget.sh`.
- CMIP6: [CMIP6 data search](https://aims2.llnl.gov/search/cmip6/).
- Geographic data: [Natural Earth 10 m land polygons](https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip).
- Climate datasets: descriptions and references in the manuscript's Supplementary Table 1.

This README was checked against the local file structure and script text. The complete figure and preprocessing pipelines were not rerun as part of this documentation update.
