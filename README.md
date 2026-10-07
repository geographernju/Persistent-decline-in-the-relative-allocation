# Persistent-decline-in-the-relative-allocation
Code and Figure Reproduction Guide
This project provides the analysis code, supporting data, and figures for the paper “Persistent decline in the relative allocation of net primary productivity to xylem formation”. This document describes the file organization according to data use and analysis steps, helping readers locate the scripts corresponding to each figure, understand the processing workflow, and reproduce the figures using the existing processed results.
1 Directory Organization
The current files are organized into three peer directories: code, data, and results. The local root path is D:\. The directory names below are given relative to this root path.
Directory	Contents and purpose
code/	19 independent plotting scripts, including 4 main figures, 9 Extended Data figures, and 6 Supplementary figures. The scripts are stored directly in this directory.
data/	Basic observation tables, maps, productivity data, and intermediate data and processing scripts organized by analysis step.
results/	19 PNG figures, with filenames corresponding one-to-one with the scripts in code/. Figures are saved at 600 dpi.
2 Data Categories
Basic Observations and Site Information
The root data/ directory contains nine CSV files, which can be grouped into four categories by purpose:
① Tree-ring and flux-tower site information: No. 1 contains basic information for global tree-ring sites; Nos. 2–4 provide coordinates for flux-tower sites in the Americas, European ICOS, and North America and Europe.
② NPP observations: No. 5 contains observation-site coordinates, and No. 6 contains annual net primary productivity (NPP) at the observation sites.
③ Correlation summary: No. 7 contains median correlations among carbon fluxes, remote-sensing indices, and tree-ring chronologies.
④ GPP observations: Nos. 8 and 9 contain annual gross primary productivity (GPP) derived using the GPP_DT_VUT and GPP_NT_VUT methods, respectively.
Map and Productivity Data
data/Map/ stores vector data including the world basemap, climate zones, and warm- and cold-region boundaries. The associated .shp, .shx, .dbf, .prj, and other companion files with matching names are included.
data/NPP and GPP/ stores NetCDF data for NPP, GPP, nppTree, and related variables. The files are organized by S2 and S3 experiments and temporal aggregation method, and also include CMIP6 and NPP (1-2) data. The labels 1-12 and 9-8 refer to calendar-year aggregation and September-to-August aggregation, respectively.
Scripts in data/NPP and GPP/transform/ perform temporal aggregation, spatial organization, and masking of productivity data; 1.nc.py is used to generate the common mask. Scripts in the peer directories CMIP6 (NPP)/ and NPP (1-2)/ process the corresponding data sources. This stage can be completed independently of tree-ring processing, after which the outputs are used together in subsequent analyses.
3 Data Processing Workflow
data/Processed data/ organizes processed data and scripts by analysis step.
1 Tree-ring Series Preparation and Missing-Value Filling
Detrending/ stores site chronologies prepared using different detrending methods from the International Tree-Ring Data Bank. The outputs are stored in data/Processed data/Detrending/Detrending4/.
AgeDepSpline+Spline+ModNegExp+SFRCS.R applies four detrending methods to the raw tree-ring data.
AgeDepSpline+Spline+ModNegExp+SFRCS.py summarizes the existing detrending results.
fill/1.fill.py uses site and climate data to fill missing values that meet the specified criteria, producing files ending in +fill.csv; 2.R and P.py summarizes correlation coefficients and significance results.
2 Valid Extent and Gridded Tree-Ring Reconstruction
The recon_mask scripts in TRW/ first generate reconstruction masks based on site distribution and valid chronologies. The scripts 3+3.py, 4+3.py, and 5+3.py then apply local ordinary kriging interpolation using the corresponding masks to produce tree-ring data on a 0.5° grid. The labels 3+3, 4+3, and 5+3 represent different search-radius and minimum-site settings; detailed parameters are provided in the respective scripts.
3 Regional Productivity Time Series
Extraction scripts in GPP and NPP/ aggregate model GPP and NPP for warm and cold regions and generate warm_gpp.csv, warm_npp.csv, cold_gpp.csv, and cold_npp.csv for regional time-series and NPP/GPP analyses.
4 Differenced Correlation and Moving-Correlation Analysis
Median/ contains median statistics of first-difference correlations between tree rings and different NPP products.
Difference_correlation/ stores spatial results of first-difference correlations between tree rings and model NPP.
r_diff and r_raw/ organizes moving-correlation results by detrending method, reconstruction setting, North America, Europe, and the combined region; current regional outputs are mainly stored in the r_diff subdirectories under each method.
Sliding difference/ stores differences in correlations before and after controlling for climate variables, for comparison of climatic influences.
5
Detrending5/ contains gridded TRW generated using different standardization methods after removing the final 30 years from each original tree-ring series.

4 Correspondence Between Plotting Scripts and Results
Each plotting script is named by figure number and writes the corresponding PNG file to results/.
Script number	Main content
Fig. 1	Site distribution, study regions and climate zones, and a summary of correlations between tree rings and productivity.
Fig. 2	Changes in tree rings and NPP, the TRW/NPP metric, and its spatial and temporal trends.
Fig. 3	Spatial patterns of first-difference correlations between tree rings and NPP, regional comparisons, and changes in moving correlations.
Fig. 4	Effects of climate factors on correlations and their spatial distribution, together with temporal changes in warm and cold regions.
Extended Data Fig. 1, 4, 5	Comparison of results under different detrending methods, reconstruction settings, or climate zones.
Extended Data Fig. 2	Site-level TRW/NPP trends and contributions, with comparisons among plant types and tree species.
Extended Data Fig. 3	Model GPP, NPP, and NPP/GPP time series in warm and cold regions.
Extended Data Fig. 6	Comparison of moving correlations between tree rings and multi-model NPP across climate zones.
Extended Data Fig. 7, 8	Comparison of correlations before and after controlling for climate variables across climate zones.
Extended Data Fig. 9	Spatial correlations among sites, kriging cross-validation, and changes in interpolation weights with distance.
Supplementary Fig. 1, 2	Spatial comparison of first-difference correlations between tree rings and different model NPP products.
Supplementary Fig. 3, 4	Comparison of TRW/NPP under different detrending methods, together with site numbers and observational coverage.
Supplementary Fig. 5, 6	Correlation comparison and validation between model productivity and observed NPP and flux-tower GPP.

5 Running and Reproduction
Runtime Environment
The current plotting and processing scripts are primarily written in Python. Major dependencies include NumPy, pandas, SciPy, xarray, Matplotlib, GeoPandas, Shapely, Rasterio, and netCDF4; some analyses also use scikit-learn, PyKrige, joblib, and cftime. Before running the scripts, install the required libraries and their spatial-data dependencies, and prepare the fonts specified by the scripts.
Generating Figures Using Existing Data
Individual scripts in code/ can be run separately, or code/run_all.py can be used to run all scripts.
Recalculating from Raw Data
The current scripts use local absolute paths. When migrating to a different environment, check the ROOT setting; only the ROOT path needs to be changed.
Raw Data Sources
Only example data are provided for some datasets. Complete datasets should be downloaded from the official sources.
The source portals for the NPP (1-2) data are the National Ecosystem Science Data Center and the GLASS data website:
https://www.nesdc.org.cn/sdo/detail?id=612f42ee7e28172cbed3d809
https://www.glass.hku.hk/download.html
The source portal for model productivity data is https://mdosullivan.github.io/GCB/. Specific download addresses are stored in data/NPP and GPP/transform/wget.sh.
The data in H:\Climatic data\month\climate14.nc are derived from the sources listed in Supplementary Table 1, which provides the corresponding data sources.

