# Code and Figure Reproduction Guide

This repository provides the analysis code, supporting data, and figures for the manuscript:

**Persistent decline in the relative allocation of net primary productivity to xylem formation**

This README describes the file organization, major processing steps, figure-generation workflow, and data sources required to reproduce the analyses and figures.

## 1. Repository Structure

The project is organized into three main directories under the local root path `D:/`.

| Directory | Description |
|---|---|
| `code/` | 19 independent plotting scripts, including 4 main figures, 9 Extended Data figures, and 6 Supplementary figures. |
| `data/` | Basic observational data, maps, productivity datasets, intermediate processed data, and processing scripts. |
| `results/` | 19 PNG figures corresponding to the scripts in `code/`, saved at 600 dpi. |

## 2. Data Organization

### 2.1 Basic observations and site information

The `data/` root directory contains nine CSV files grouped by purpose:

1. **Tree-ring and flux-tower site information**  
   File 1 contains basic information for global tree-ring sites. Files 2–4 contain coordinates for flux-tower sites in the Americas, European ICOS network, and North America and Europe.

2. **NPP observations**  
   File 5 contains observation-site coordinates, and File 6 contains annual net primary productivity (NPP) observations.

3. **Correlation summaries**  
   File 7 contains median correlations between carbon-flux products, remote-sensing indices, and tree-ring chronologies.

4. **GPP observations**  
   Files 8 and 9 contain annual gross primary productivity (GPP) estimates derived using the GPP_DT_VUT and GPP_NT_VUT methods.

### 2.2 Maps and productivity data

`data/Map/` contains vector datasets for world boundaries, climate zones, and warm/cold-region boundaries. Associated shapefile components such as `.shp`, `.shx`, `.dbf`, and `.prj` are included.

`data/NPP and GPP/` contains NetCDF datasets for NPP, GPP, and `nppTree`, organized by S2 and S3 experiments and by temporal aggregation scheme. It also includes CMIP6 and NPP (1-2) datasets.

The directory naming convention is:

- `1-12`: January–December aggregation
- `9-8`: September of the current year to August of the following year

Scripts in `data/NPP and GPP/transform/` perform temporal aggregation, spatial organization, and masking of productivity datasets. `1.nc.py` generates the common mask. Scripts in `CMIP6 (NPP)/` and `NPP (1-2)/` process the corresponding datasets.

Productivity processing and tree-ring processing can be completed independently before the datasets are combined in later analyses.

## 3. Data Processing Workflow

### 3.1 Tree-ring chronology processing and gap filling

`data/Processed data/Detrending/` contains site chronologies processed using different detrending methods based on records from the International Tree-Ring Data Bank.

`AgeDepSpline+Spline+ModNegExp+SFRCS.R` applies four tree-ring standardization methods.

`AgeDepSpline+Spline+ModNegExp+SFRCS.py` summarizes the existing detrended outputs.

`fill/1.fill.py` fills eligible missing values using site and climate data. Output files end with `+fill.csv`.

`2.R and P.py` summarizes correlation coefficients and significance results.

### 3.2 Valid-domain definition and gridded TRW reconstruction

Scripts in `TRW/` first generate reconstruction masks based on site distribution and valid chronology coverage.

`3+3.py`, `4+3.py`, and `5+3.py` then perform local ordinary kriging interpolation and produce gridded tree-ring width (TRW) data at 0.5° resolution.

The labels `3+3`, `4+3`, and `5+3` represent different search-radius and minimum-site settings. Exact parameters are defined in the corresponding scripts.

### 3.3 Regional productivity time series

Scripts in `GPP and NPP/` aggregate model GPP and NPP for warm and cold climate regions and generate:

- `warm_gpp.csv`
- `warm_npp.csv`
- `cold_gpp.csv`
- `cold_npp.csv`

These files are used for regional time-series analyses and NPP/GPP calculations.

### 3.4 Difference correlation and moving-correlation analyses

`Median/` contains median statistics of first-difference correlations between tree-ring data and multiple NPP products.

`Difference_correlation/` contains spatial outputs of first-difference correlations between TRW and model NPP.

`r_diff and r_raw/` contains moving-correlation results organized by detrending method, reconstruction setting, North America, Europe, and combined regions. Existing climate-region outputs are mainly stored in the `r_diff` subdirectories.

`Sliding difference/` contains differences between correlations before and after controlling for climate variables and is used to evaluate climatic influences.

### 3.5 End-effect sensitivity analysis

`Detrending5/` contains gridded TRW datasets produced after removing the final 30 years from each original tree-ring series for different standardization methods.

These datasets are used to evaluate potential end effects.

## 4. Figure Scripts and Outputs

Each plotting script is named according to its corresponding figure and saves its output to `results/`.

| Script | Main content |
|---|---|
| `Fig. 1` | Site distributions, study regions, climate zones, and summarized correlations between tree-ring and productivity datasets. |
| `Fig. 2` | Long-term changes in TRW and NPP, the TRW/NPP metric, and its spatial and temporal trends. |
| `Fig. 3` | Spatial patterns of first-difference TRW–NPP correlations, regional comparisons, and moving-correlation changes. |
| `Fig. 4` | Climatic controls on correlations, their spatial distribution, and temporal changes in warm and cold regions. |
| `Extended Data Fig. 1, 4, 5` | Comparisons among detrending methods, reconstruction settings, or climate regions. |
| `Extended Data Fig. 2` | Site-level TRW/NPP trends and contributions, including comparisons among plant types and tree species. |
| `Extended Data Fig. 3` | Model GPP, NPP, and NPP/GPP time series in warm and cold regions. |
| `Extended Data Fig. 6` | Moving-correlation comparisons between tree rings and multiple NPP models across climate regions. |
| `Extended Data Fig. 7, 8` | Climate-region comparisons of correlations before and after controlling for climatic variables. |
| `Extended Data Fig. 9` | Spatial correlations among sites, kriging cross-validation, and interpolation weights as a function of distance. |
| `Supplementary Fig. 1, 2` | Spatial comparisons of first-difference correlations between tree-ring data and different model NPP products. |
| `Supplementary Fig. 3, 4` | Comparisons of TRW/NPP among detrending methods, and site-number and observational-coverage analyses. |
| `Supplementary Fig. 5, 6` | Validation of model productivity against NPP observations and flux-tower GPP observations. |

## 5. Environment and Reproduction

### 5.1 Software environment

The current analysis and plotting scripts are mainly written in Python.

Major dependencies include:

- NumPy
- pandas
- SciPy
- xarray
- Matplotlib
- GeoPandas
- Shapely
- Rasterio
- netCDF4

Some analyses additionally use:

- scikit-learn
- PyKrige
- joblib
- cftime

Install the required Python packages and spatial-data dependencies before running the scripts.

### 5.2 Generate figures from existing processed data

Individual scripts in `code/` can be run separately.

To execute all figure scripts sequentially, run:

```bash
python code/run_all.py
```

### 5.3 Recompute from raw data

The current scripts use local absolute paths.

When moving the project to another environment, check the `ROOT` setting in the scripts. In most cases, only the root directory needs to be updated.

## 6. Data Sources

Some datasets are provided only as example data. Full datasets should be downloaded from their official sources.

### NPP (1-2)

National Earth System Science Data Center:

https://www.nesdc.org.cn/sdo/detail?id=612f42ee7e28172cbed3d809

GLASS:

https://www.glass.hku.hk/download.html

### Model productivity data

https://mdosullivan.github.io/GCB/

The exact download addresses are stored in:

```text
data/NPP and GPP/transform/wget.sh
```

### Climate data

The climate dataset used in:

```text
H:/Climatic data/month/climate14.nc
```

is derived from the data sources listed in Supplementary Table 1.

## 7. Notes

- Figure outputs are saved at 600 dpi.
- File names in `results/` correspond directly to scripts in `code/`.
- Shapefile components should remain in the same directory.
- Processed datasets are organized by analysis stage to support reproducibility and figure-level tracing.
