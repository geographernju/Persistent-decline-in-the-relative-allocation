import pandas as pd
from pathlib import Path
root=Path('D:/data/Processed data/Detrending')
input_dir=Path('D:/data/Processed data/Detrending/Detrending4')
variables=("Spline","ModNegExp","SFRCS","AgeDepSpline")
years=pd.Index(range(1700,2026),name="year")
data={variable:{} for variable in variables}
files=sorted(input_dir.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")
for filepath in files:
    try:
        df=pd.read_csv(filepath)
        if "year" not in df.columns:
            print(f"[Skipped] {filepath.name}: missing year")
            continue
        year=pd.to_numeric(df["year"],errors="coerce")
        for variable in variables:
            if variable not in df.columns:
                continue
            values=pd.to_numeric(df[variable],errors="coerce")
            series=pd.Series(values.to_numpy(),index=year.to_numpy())
            series=series[series.index.notna()]
            series=series[~series.index.duplicated(keep="first")]
            data[variable][filepath.stem]=series.reindex(years)
    except Exception as e:
        print(f"[Error] {filepath.name}: {e}")
for variable in variables:
    if not data[variable]:
        raise ValueError(f"No data extracted for {variable}")
    result=pd.DataFrame(data[variable],index=years).reset_index()
    output_file=root/f"{variable}.csv"
    result.to_csv(output_file,index=False,encoding="utf-8-sig")
    print(f"Saved: {output_file} | Files: {len(data[variable])} | Years: {len(result)}")