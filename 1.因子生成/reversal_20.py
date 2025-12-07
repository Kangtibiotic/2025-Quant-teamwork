import pandas as pd
import os
import glob
import numpy as np
from pandas.tseries.offsets import MonthEnd

base_dir = r"..\data"
result_dir = r"..\raw_factors"
market_path = os.path.join(base_dir, "TRD_Dalyr")
factor_name = "reversal_20"
output_file = os.path.join(result_dir, f"{factor_name}.csv")

use_cols = ['Stkcd', 'Trddt', 'Clsprc', 'CumulateFwardFactor']

all_data = []
for file_path in glob.glob(os.path.join(market_path, '*.csv')):
    try:
        df_check = pd.read_csv(file_path, nrows=1)
        valid_cols = [c for c in use_cols if c in df_check.columns]
        
        if 'Stkcd' not in valid_cols or 'Trddt' not in valid_cols: continue
        
        df_temp = pd.read_csv(file_path, usecols=valid_cols)
        df_temp['Trddt'] = pd.to_datetime(df_temp['Trddt'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Trddt', 'Stkcd'])
        all_data.append(df_temp)
    except Exception:
        pass

df = pd.concat(all_data, ignore_index=True)
df = df.sort_values(by=['Stkcd', 'Trddt'])

if 'CumulateFwardFactor' in df.columns:
    df['adj_close'] = df['Clsprc'] * df['CumulateFwardFactor']
else:
    print("警告: 未找到复权因子，使用原始收盘价计算。")
    df['adj_close'] = df['Clsprc']

df[factor_name] = df.groupby('Stkcd')['adj_close'].transform(
    lambda x: -x.pct_change(periods=20)
)
df['year_month'] = df['Trddt'].dt.to_period('M')
df_month = df.sort_values('Trddt').groupby(['Stkcd', 'year_month']).tail(1).copy()

df_month['Trddt'] = df_month['Trddt'] + MonthEnd(0)

output_df = df_month[['Stkcd', 'Trddt', factor_name]].copy()
output_df['Stkcd'] = output_df['Stkcd'].astype(int).astype(str).str.zfill(6)
output_df = output_df.dropna().sort_values(['Trddt', 'Stkcd'])

output_df.to_csv(output_file, index=False)
