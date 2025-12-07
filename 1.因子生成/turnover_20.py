import pandas as pd
import os
import glob
import numpy as np
from pandas.tseries.offsets import MonthEnd

base_dir = r"..\data"
result_dir = r"..\raw_factors"
market_path = os.path.join(base_dir, "TRD_Dalyr")
factor_name = "turnover_20"
output_file = os.path.join(result_dir, f"{factor_name}.csv")

file_pattern = os.path.join(market_path, '*.csv')
use_cols = ['Stkcd', 'Trddt', 'Dnvaltrd', 'Dsmvosd']

all_data = []
for file_path in glob.glob(file_pattern):
    try:
        df_check = pd.read_csv(file_path, nrows=1)
        valid_cols = [c for c in use_cols if c in df_check.columns]
        
        if 'Stkcd' not in valid_cols or 'Trddt' not in valid_cols: continue
        df_temp = pd.read_csv(file_path, usecols=valid_cols)
        df_temp['Trddt'] = pd.to_datetime(df_temp['Trddt'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Trddt', 'Stkcd'])
        all_data.append(df_temp)
    except Exception as e:
        print(f"跳过 {os.path.basename(file_path)}: {e}")

df = pd.concat(all_data, ignore_index=True)
df = df.sort_values(by=['Stkcd', 'Trddt'])

df['daily_turnover'] = df['Dnvaltrd'] / df['Dsmvosd']
df.loc[np.isinf(df['daily_turnover']), 'daily_turnover'] = np.nan

# 计算20日均值
df[factor_name] = df.groupby('Stkcd')['daily_turnover'].transform(
    lambda x: x.rolling(window=20, min_periods=15).mean()
)

df['year_month'] = df['Trddt'].dt.to_period('M')
df_month = df.sort_values('Trddt').groupby(['Stkcd', 'year_month']).tail(1).copy()

df_month['Trddt'] = df_month['Trddt'] + MonthEnd(0)
output_df = df_month[['Stkcd', 'Trddt', factor_name]].copy()

output_df['Stkcd'] = output_df['Stkcd'].astype(int).astype(str).str.zfill(6)
output_df = output_df.dropna().sort_values(['Trddt', 'Stkcd'])

output_df.to_csv(output_file, index=False)
