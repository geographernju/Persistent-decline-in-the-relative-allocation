# Code and Figure Reproduction Guide

**Persistent decline in the relative allocation of net primary productivity to xylem formation**

This guide describes the files currently present in `D:/code`, `D:/data`, and `D:/results`. It was revised after inspecting the source code, directory structure, input references, and PNG metadata on 21 October 2026.

The inspection did not rerun the analyses or regenerate figures. Existing outputs establish that files are present; they do not establish that the current code reproduces them numerically or visually.

## 1. Project layout

```text
D:/
├── README.md
├── code/                         19 figure scripts and run_all.py
├── data/
│   ├── 1. ...csv through 9. ...csv
│   ├── Map/
│   ├── NPP and GPP/
│   │   ├── CMIP6 (NPP)/
│   │   ├── NPP (1-2)/
│   │   ├── S2_npp9-8/
│   │   ├── S2_npptree9-8/
│   │   ├── S3_gpp1-12/
│   │   ├── S3_npp1-12/
│   │   ├── S3_npp9-8/
│   │   └── transform/
│   └── Processed data/
│       ├── Detrending/
│       ├── Detrending5/
│       ├── Difference_correlation/
│       ├── GPP and NPP/
│       ├── median/
│       ├── r_diff and r_raw/
│       ├── Sliding difference/
│       └── TRW/
└── results/                      19 figure PNGs
```

There are four main figures, nine Extended Data figures, and six Supplementary figures. Each has a matching Python script in `code/` and PNG in `results/`. All 19 existing PNGs report approximately 600 dpi in their metadata.

## 2. Software and paths

Figure scripts use Python. Imports across the figure and preprocessing code include:

- NumPy, pandas, SciPy, xarray, and Matplotlib;
- GeoPandas, Shapely, and Rasterio for spatial operations;
- netCDF4 and cftime for NetCDF data and calendars;
- scikit-learn, PyKrige, and joblib for selected analyses.

Tree-ring detrending additionally requires R and `dplR`; `tools` is part of R. Scripts generally request Times New Roman, which should be installed for matching text appearance. Package versions are not pinned in this guide; an exact software environment was not reconstructed during this inspection.

Example Python package installation:

```powershell
python -m pip install numpy pandas scipy xarray matplotlib geopandas shapely rasterio netCDF4 cftime scikit-learn pykrige joblib
```

Example R package installation:

```r
install.packages("dplR")
```

Most figure scripts set `ROOT = Path("D:/")`; `run_all.py` separately sets `ROOT = Path("D:/code")`. Preprocessing scripts also contain absolute paths, including paths outside these three project directories. Changing a single `ROOT` does not update every path. When relocating the project, inspect all input and output settings.

Some scripts refer to `data/map`, whereas the actual directory is `data/Map`. Windows resolves this case difference; a case-sensitive system requires consistent capitalization. Keep shapefile components together and quote command-line paths containing spaces.

## 3. Generate figures from existing processed data

For the current Windows layout, run:

```powershell
python "D:/code/run_all.py"
```

The runner recursively discovers Python scripts under `D:/code`, excludes itself and `__pycache__`, sorts by relative path, and runs each sequentially with the current Python interpreter. Each subprocess uses its script directory as its working directory. The runner continues after individual failures, prints a summary, and exits with status 1 if any script fails.

The runner executes figure scripts only; it does not execute the preprocessing scripts under `data/`. Its alphabetical ordering is not manuscript figure order. Additional Python scripts placed under `code/` will also be discovered.

To generate one figure:

```powershell
python "D:/code/Fig. 1.py"
```

Outputs are saved to `D:/results/` and existing files with the same names can be overwritten. Spatial interpolation, model evaluation, and figure rendering can require substantial memory and time; no execution-time estimate was measured.

## 4. Current figure mapping

For every row below, the script is `code/<Figure>.py` and the output is `results/<Figure>.png`.

| Figure | Current content | Principal inputs |
|---|---|---|
| Fig. 1 | Site distributions, study domains and climate zones, and regional correlation summaries. | Site/observation CSVs, `Map/`, `TRW/AgeDepSpline (5+3).nc`, `median/Median_correlations.csv`. |
| Fig. 2 | Standardized TRW/NPP patterns and trends in North America and Europe. | `TRW/AgeDepSpline (5+3).nc`, S2 NPP for ISBA-CTRIP, CLASSIC, and iMAPLE, and regional maps. |
| Fig. 3 | TRW–NPP first-difference correlations, climate-region comparisons, and moving correlations. | AgeDepSpline reconstruction, ISBA-CTRIP S2 NPP, maps, and AgeDepSpline `(5+3)` moving-correlation CSVs. |
| Fig. 4 | Spatial and temporal climatic influences on moving correlations. | `Sliding difference1-1900-center.nc`, AgeDepSpline reconstruction, maps, and combined-region moving correlations. |
| Extended Data Fig. 1 | Climate-zone moving correlations calculated from reconstructed TRW and the S2 NPP ensemble. | AgeDepSpline `(5+3)`, `S2_npp9-8/`, and North American/European climate maps. |
| Extended Data Fig. 2 | Standardized TRW/NPP comparisons across detrending methods, reconstruction settings, and an nppTree alternative. | TRW reconstructions for ModNegExp, SFRCS, Spline, and AgeDepSpline; selected S2 NPP and nppTree products. |
| Extended Data Fig. 3 | Site-level TRW/NPP time series and trends, spatial patterns, and plant-type/species comparisons. | Site information, filled AgeDepSpline chronologies, selected S2 NPP products, and world boundaries. |
| Extended Data Fig. 4 | Warm/cold-region GPP, NPP, and NPP/GPP time series. | Four regional CSVs in `Processed data/GPP and NPP/`. |
| Extended Data Fig. 5 | Moving-correlation comparisons across detrending methods, reconstruction settings, and nppTree. | Six method/configuration directories in `r_diff and r_raw/`. |
| Extended Data Fig. 6 | AgeDepSpline moving correlations by climate zone for North America, Europe, and their combination. | AgeDepSpline `(5+3)` moving-correlation CSVs. |
| Extended Data Fig. 7 | Climate-factor effects on correlations through time, compared with combined-region moving correlations. | Sliding-difference NetCDF, maps, and AgeDepSpline `(5+3)/All r_diff/`. |
| Extended Data Fig. 8 | Climate-zone summaries of changes in correlation after controlling for climate factors. | Sliding-difference NetCDF and climate maps. |
| Extended Data Fig. 9 | Site-pair correlations versus distance, kriging cross-validation, and cumulative kriging weights. | Site information, filled AgeDepSpline chronologies, and the `(5+3)` reconstruction mask. |
| Supplementary Fig. 1 | North American spatial first-difference TRW–NPP correlations for 1981–2010. | AgeDepSpline `(5+3)`, the S2 NPP ensemble, and world boundaries. |
| Supplementary Fig. 2 | European spatial first-difference TRW–NPP correlations for 1981–2010. | AgeDepSpline `(5+3)`, the S2 NPP ensemble, and world boundaries. |
| Supplementary Fig. 3 | End-effect sensitivity: TRW/NPP comparisons using the supplied truncated-series reconstructions. | Four NetCDF files in `Detrending5/`, selected S2 NPP, and maps. |
| Supplementary Fig. 4 | Observation-site counts and temporal coverage in the study domains. | Site and observation CSVs, unfilled AgeDepSpline chronologies, and the reconstruction domain. |
| Supplementary Fig. 5 | Pooled observation–model GPP and NPP correlations. | Observation/site CSVs, `S3_gpp1-12/`, and `S3_npp1-12/`. |
| Supplementary Fig. 6 | Site-level GPP correlations for products selected separately by their highest median DT and NT correlations. | GPP DT/NT observation tables, flux-tower coordinates, and `S3_gpp1-12/`. |

Input paths abbreviated as `TRW/`, `median/`, `Detrending5/`, `Sliding difference/`, or `r_diff and r_raw/` are under `data/Processed data/`.

## 5. Data organization

### 5.1 Root CSV tables

| File prefix | Contents |
|---|---|
| `1. Basic information of global tree-ring sites.csv` | Tree-ring site information. |
| `2. Coordinates of Americas flux tower sites.csv` | Americas flux-tower coordinates. |
| `3. Coordinates of European ICOS flux tower sites.csv` | European ICOS flux-tower coordinates. |
| `4. Coordinates of North American and European flux tower sites.csv` | Combined flux-tower coordinates. |
| `5. NPP site coordinate table.csv` | NPP observation-site coordinates. |
| `6. Annual NPP values at observation sites.csv` | NPP observations and observation intervals. |
| `7. Carbon flux & remote sensing vs. tree-ring chronology median correlations.csv` | Supplied correlation-summary table. |
| `8. Annual GPP data of GPP_DT_VUT method.csv` | Annual daytime-method GPP observations. |
| `9. Annual GPP data of GPP_NT_VUT method.csv` | Annual nighttime-method GPP observations. |

The root File 7 and `Processed data/median/Median_correlations.csv` are separate files. The current Fig. 1 script reads the latter.

### 5.2 Maps and productivity datasets

`data/Map/` contains world boundaries, North American and European climate-zone polygons, warm/cold boundaries, and Natural Earth land polygons. Existing files include `Europe 7.shp`, `Europe 7-class.shp`, `Koppen_1991_2020_NA_big7.shp`, and `Koppen_1991_2020_NA_big7_simple.shp`; preserve their distinct names.

`data/NPP and GPP/` contains processed NetCDF products grouped by experiment and variable. The directory labels `1-12` and `9-8` denote calendar-year and September–August aggregation, respectively. The precise year labeling, units, and aggregation implemented by each script and NetCDF metadata should be checked before recomputing or combining products.

### 5.3 Processed analysis products

- `Detrending/`: site-level detrending outputs in `Detrending4/`, method-wide chronology tables, and filled chronologies in `fill/`.
- `TRW/`: reconstruction masks and gridded chronology NetCDFs. Labels `(3+3)`, `(4+3)`, and `(5+3)` correspond to mask search radii of 3°, 4°, and 5° with at least three sites. Grid spacing is 0.5°; interpolation has additional settings in its own scripts.
- `median/`: `Median_correlations.py` and `Median_correlations.csv`, comparing TRW with DGVM S2/S3, CMIP6, and NPP1-2 products for 1981–2010.
- `Difference_correlation/`: a script and supplied first-difference correlation NetCDF for 1981–2010.
- `GPP and NPP/`: `warm_gpp.csv`, `warm_npp.csv`, `cold_gpp.csv`, and `cold_npp.csv`, plus their extraction script.
- `r_diff and r_raw/`: seven method/configuration script directories and regional moving-correlation outputs. The supplied outputs are principally in `r_diff` subdirectories; the directory title does not imply a complete corresponding `r_raw` workflow.
- `Sliding difference/`: climate-control analysis code and `Sliding difference1-1900-center.nc`.
- `Detrending5/`: four supplied reconstruction NetCDFs used for end-effect sensitivity. The previous documentation describes removing the last 30 years of each source series; a dedicated script establishing that truncation and regenerating these four files was not found in the inspected directories.

## 6. Recompute analyses: dependencies and order

This is a dependency guide, not a single automated raw-data pipeline. Check each script's configuration before execution.

1. **Prepare original inputs.** Obtain raw tree-ring measurements, model products, and climate data. Several input locations are outside `data/`; see Section 7.
2. **Prepare the land mask and productivity products.** Run `data/NPP and GPP/transform/1.nc.py` to create the common mask if needed. The model transformation scripts and the scripts in `CMIP6 (NPP)/` and `NPP (1-2)/` depend on that mask. Review the shared-output-directory issue below before running the transformations.
3. **Detrend tree rings.** Run `AgeDepSpline+Spline+ModNegExp+SFRCS.R`, then the same-named `.py` script under `Processed data/Detrending/`. The Python script consolidates AgeDepSpline, Spline, ModNegExp, and SFRCS chronologies across sites for 1700–2025.
4. **Fill eligible chronology gaps.** Run `Detrending/fill/1.fill.py`; it requires the external climate file and produces method tables ending in `+fill.csv` and climate-correlation summaries. `2.R and P.py` plots those summary statistics.
5. **Create reconstruction masks, then interpolate.** Run the appropriate `recon_mask ... .py` script before `3+3.py`, `4+3.py`, or `5+3.py`. The actual `(3+3)` mask script is named `recon_mask (3+3) .py`, with a space before `.py`. Interpolation uses local ordinary kriging; the inspected `(5+3)` script specifies a spherical variogram, up to 100 neighbors, and five parallel jobs. Check method inputs: interpolation also lists RCS, whereas the chronology-consolidation script lists four other methods.
6. **Generate downstream products.** Run `median/Median_correlations.py`, `Difference_correlation/Difference_correlation.py`, and `GPP and NPP/Extracting GPP and NPP.py` as needed. Run the seven scripts under `r_diff and r_raw/` for the corresponding moving-correlation configurations. The AgeDepSpline `(5+3)` script uses a 21-year window; not all figures use identical window definitions (Extended Data Fig. 1 sets `WIN=20`).
7. **Compute climatic controls.** Run `Sliding difference/Sliding difference1-1900-center.py` after preparing TRW, selected S2 NPP, and climate inputs. It uses 21-year windows for 1900–2025 and writes the NetCDF consumed by Fig. 4 and Extended Data Figs. 7–8.
8. **Generate manuscript figures.** Run `code/run_all.py` after all required processed inputs are ready. Supplementary Fig. 3 additionally requires the supplied `Detrending5/` products.

There is no dedicated S2 NPP transformation script in the inspected `transform/` directory. Existing `S2_npp9-8/` files are supplied inputs; rebuilding them requires an appropriate upstream procedure that is not included as a separate script here.

## 7. External inputs and current reproduction limitations

The following input locations are coded in preprocessing scripts. Existence was checked during this inspection; these checks do not validate file contents.

| Location | Used for | Inspection status |
|---|---|---|
| `D:/Tree rings/rwl/width/` | Raw `.rwl` files for detrending. | Not found. |
| `D:/S2_npptree/` | Original nppTree inputs. | Not found. |
| `D:/S3_gpp/` | Original S3 GPP inputs. | Not found. |
| `D:/S3_npp/` | Original S3 NPP inputs. | Not found. |
| `D:/NPP (1-2)/` | Original paired NPP inputs. | Not found. |
| `D:/npp and tree ring/Model data/CMIP6/NPP_year/` | Original CMIP6 NPP inputs. | Directory exists outside the packaged data tree. |
| `H:/Climatic data/month/climate14.nc` | Gap filling and climate-control analyses. | File exists outside the packaged data tree. |

Two transformation scripts currently set their calendar-year and cross-year outputs to the same directory:

- `S2_nppTree_transform-1900.py`: both `OUT_12` and `OUT_98` use `S2_npptree9-8/`.
- `S3_gpp_transform.py`: both use `S3_gpp1-12/`.

Outputs with identical names may overwrite each other. Resolve the intended aggregation/output locations before rerunning these scripts. The existing folder name alone is insufficient to establish which aggregation remains on disk. This inspection did not modify those scripts.

`Extended Data Fig. 3.py` and `Supplementary Fig. 3.py` contain a leading space before `CLASSIC_S2_npp.nc` in one configured path. Both scripts include a fallback that strips the leading space before opening the file. The climate-control preprocessing script has a similar fallback.

A static check of 108 literal `data/...` references in figure scripts found the referenced files/directories present, apart from those two leading-space paths resolved by their fallbacks. This is not a complete validation of dynamically constructed filenames, NetCDF variables, or numerical reproducibility.

## 8. Source information recorded in the code

The following source links are retained from the existing documentation and source comments; their availability and dataset versions were not independently checked in this local inspection.

- Tree-ring measurements: International Tree-Ring Data Bank (ITRDB), named in the detrending script.
- NPP products: [National Earth System Science Data Center](https://www.nesdc.org.cn/sdo/detail?id=612f42ee7e28172cbed3d809) and [GLASS](https://www.glass.hku.hk/download.html).
- Model productivity: [Global Carbon Budget model-data page](https://mdosullivan.github.io/GCB/); download commands are stored in `data/NPP and GPP/transform/wget.sh`.
- CMIP6: [CMIP6 data-search address recorded in the processing script](https://aims2.llnl.gov/search/cmip6/).
- Land polygons: the mask script references [Natural Earth 10 m land](https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip), with an existing local copy in `data/Map/natural_earth_land/`.
- Climate inputs: the previous README refers to Supplementary Table 1 for source details. The local scripts establish the climate file path but do not by themselves provide a complete dataset provenance record.

## 9. Verification scope

The revised guide is based on the current script contents and on-disk products. The 19 script/output name pairs, PNG headers and resolution, principal input references, processing entry points, and external input locations were inspected. Neither a Code Ocean execution entry nor a full raw-data reproduction run was verified. Use the Windows commands above for the layout documented here.
