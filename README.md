# Code and Figure Reproduction

**Persistent decline in the relative allocation of net primary productivity to xylem formation**

This README describes the figure scripts, processed data, and input paths used in the study. The scripts use a Windows directory structure rooted at `D:/`. Figures can be generated from the corresponding processed inputs; rebuilding the datasets also requires the original tree-ring, climate, and model data described below.

## 1. Project structure

```text
D:/
├── README.md
├── code/                         Figure scripts and run_all.py
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
└── results/                      Figure PNGs
```

The figure set consists of four main figures, nine Extended Data figures, and six Supplementary figures. The corresponding Python scripts are in `code/`, with PNG outputs in `results/`.

## 2. Software and paths

Python packages used across the scripts include NumPy, pandas, SciPy, xarray, Matplotlib, GeoPandas, Shapely, Rasterio, netCDF4, cftime, scikit-learn, PyKrige, and joblib. The tree-ring detrending scripts also require R and `dplR`.

```powershell
python -m pip install numpy pandas scipy xarray matplotlib geopandas shapely rasterio netCDF4 cftime scikit-learn pykrige joblib
```

```r
install.packages("dplR")
```

Most plots use Times New Roman. Software versions are not pinned, so differences in package versions may affect results or rendering.

Most figure scripts use `ROOT = Path("D:/")`, while `run_all.py` uses `D:/code`. Other scripts contain separate absolute paths, including paths on `H:/`. Update the relevant paths if the data are stored elsewhere. Some scripts use `data/map` rather than `data/Map`; directory names must match on case-sensitive systems. Keep all companion files for each Shapefile together.

## 3. Generate figures

Run the complete figure set from existing processed inputs:

```powershell
python "D:/code/run_all.py"
```

The runner searches `D:/code` recursively for `.py` files (excluding itself and `__pycache__`), sorts them by relative path, and runs them sequentially. It reports failures but continues with the remaining scripts. Preprocessing scripts under `data/` are not included, and the run order is alphabetical rather than manuscript order. Newly added Python files under `code/` will also be picked up.

To regenerate one figure:

```powershell
python "D:/code/Fig. 1.py"
```

Figures are saved to `D:/results/`. Existing PNGs with the same names will be overwritten. Spatial interpolation and model evaluation can be memory-intensive.

## 4. Figure scripts

Each figure uses the corresponding `code/<Figure>.py` script and is saved as `results/<Figure>.png`. Paths such as `TRW/`, `median/`, `Detrending5/`, `Sliding difference/`, and `r_diff and r_raw/` in the table refer to subdirectories of `data/Processed data/`.

| Figure | Content | Main inputs |
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

## 5. Data files

### 5.1 Site and observation tables

| File | Contents |
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

`7. Carbon flux & remote sensing vs. tree-ring chronology median correlations.csv` and `Processed data/median/Median_correlations.csv` are different files. Fig. 1 uses the latter.

### 5.2 Maps and productivity products

`data/Map/` contains geographic boundaries and the North American and European climate-zone layers, including `Europe 7.shp`, `Europe 7-class.shp`, `Koppen_1991_2020_NA_big7.shp`, and `Koppen_1991_2020_NA_big7_simple.shp`.

`data/NPP and GPP/` contains the productivity NetCDF products and transformation scripts. In directory names, `1-12` indicates January–December aggregation and `9-8` indicates September–August aggregation. Check the time coordinate and units of individual products before combining data from different experiments.

### 5.3 Processed data

- `Detrending/`: site chronologies, combined method tables, and gap-filled chronologies in `fill/`.
- `TRW/`: reconstruction masks and gridded tree-ring chronologies. The `(3+3)`, `(4+3)`, and `(5+3)` configurations use 3°, 4°, and 5° spatial support, respectively, with at least three sites. Grid spacing is 0.5°.
- `median/`: TRW–NPP correlation summaries for 1981–2010, including DGVM S2/S3, CMIP6, and NPP1-2 products.
- `Difference_correlation/`: first-difference correlations for 1981–2010.
- `GPP and NPP/`: `warm_gpp.csv`, `warm_npp.csv`, `cold_gpp.csv`, and `cold_npp.csv`.
- `r_diff and r_raw/`: scripts and regional sliding-correlation outputs for different methods and spatial settings; outputs are primarily stored in `r_diff` subdirectories.
- `Sliding difference/`: climate-control analyses and the sliding-difference NetCDF used in the climate figures.
- `Detrending5/`: four reconstruction NetCDFs used for the sensitivity test that truncates the last 30 years of the tree-ring series.

## 6. Rebuild the analysis products

The preprocessing scripts are organized by task rather than as a single end-to-end pipeline. Run each step only after its inputs are available.

1. **Original data:** Prepare the tree-ring measurements, productivity products, geographic layers, and climate inputs. Some source datasets are outside the `D:/data` directory.
2. **Land mask and NPP processing:** Generate the common land mask using `data/NPP and GPP/transform/1.nc.py` when needed, followed by the relevant model processing scripts. Check the output directories noted below before rerunning transformations.
3. **Tree-ring detrending:** Run `AgeDepSpline+Spline+ModNegExp+SFRCS.R`, followed by the matching `.py` script in `Processed data/Detrending/`. The Python script combines chronologies produced by AgeDepSpline, Spline, ModNegExp, and SFRCS.
4. **Gap filling:** Run `Detrending/fill/1.fill.py` to generate the `+fill.csv` chronology tables and climate-correlation summaries. `2.R and P.py` plots the corresponding statistics.
5. **Spatial reconstruction:** Generate the reconstruction mask with the appropriate `recon_mask` script, then run `3+3.py`, `4+3.py`, or `5+3.py`. The `(5+3)` configuration uses ordinary kriging with a spherical variogram, up to 100 neighboring sites, and five parallel jobs.
6. **Correlation products:** Run `median/Median_correlations.py`, `Difference_correlation/Difference_correlation.py`, `GPP and NPP/Extracting GPP and NPP.py`, and the applicable scripts under `r_diff and r_raw/`.
7. **Climate analysis:** Run `Sliding difference/Sliding difference1-1900-center.py` using the selected TRW, S2 NPP, and climate inputs. The main sliding-correlation analysis uses a 21-year window; individual scripts may differ (for example, Extended Data Fig. 1 sets `WIN=20`).
8. **Figures:** Run `code/run_all.py` after preparing the necessary processed data. Supplementary Fig. 3 uses the separate `Detrending5/` sensitivity products.

The existing S2 NPP NetCDF files are inputs to downstream scripts; a dedicated transformation script for the S2 NPP data is not included in `transform/`. The supplied `Detrending5/` outputs likewise require their original truncation and reconstruction procedure for a complete rebuild.

## 7. External inputs and path checks

Several processing scripts refer to source data outside the project folders:

| Path | Purpose |
|---|---|
| `D:/Tree rings/rwl/width/` | Original `.rwl` measurements for detrending. |
| `D:/S2_npptree/` | Original nppTree files. |
| `D:/S3_gpp/` | Original S3 GPP files. |
| `D:/S3_npp/` | Original S3 NPP files. |
| `D:/NPP (1-2)/` | Original NPP1-2 files. |
| `D:/npp and tree ring/Model data/CMIP6/NPP_year/` | CMIP6 annual NPP data. |
| `H:/Climatic data/month/climate14.nc` | Climate data used for gap filling and climate-control analyses. |

These paths must point to the corresponding datasets before the preprocessing scripts can run. The processed inputs already present in `data/` allow figure generation without repeating every upstream step.

**Output paths to check:** `S2_nppTree_transform-1900.py` currently sets both `OUT_12` and `OUT_98` to `S2_npptree9-8/`, and `S3_gpp_transform.py` sets both to `S3_gpp1-12/`. Separate these destinations if both aggregation types are to be retained; otherwise identically named outputs may overwrite one another.

**Filename details:** `Extended Data Fig. 3.py` and `Supplementary Fig. 3.py` include a leading space before `CLASSIC_S2_npp.nc` in one input path, with a fallback that removes the space. The corresponding climate-control script uses a similar fallback. The `(3+3)` mask script is named `recon_mask (3+3) .py`, including the space before `.py`.

**Method consistency:** The spatial reconstruction scripts may refer to RCS alongside the four methods used by the chronology-consolidation script. Select matching chronology inputs when rerunning the sensitivity comparisons.

## 8. Data sources

- **Tree-ring measurements:** International Tree-Ring Data Bank (ITRDB).
- **NPP data:** [National Earth System Science Data Center](https://www.nesdc.org.cn/sdo/detail?id=612f42ee7e28172cbed3d809) and [GLASS](https://www.glass.hku.hk/download.html).
- **Model productivity data:** [Global Carbon Budget](https://mdosullivan.github.io/GCB/), with download commands in `data/NPP and GPP/transform/wget.sh`.
- **CMIP6:** [CMIP6 data search](https://aims2.llnl.gov/search/cmip6/).
- **Geographic data:** [Natural Earth 10 m land polygons](https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip).
- **Climate data:** See the study's Supplementary Table 1 for the dataset descriptions and source references.

Processed NetCDF files, masks, and chronology tables depend on the data versions and settings used when they were generated. Some upstream source-data preparation steps are not included in this repository.
