import pandas as pd
import os
import glob
import numpy as np
from pandas.tseries.offsets import MonthEnd

# 下行波动率函数
def calc_downside_std(x):
    downside_ret = np.where(x < 0, x, 0)
    return np.std(downside_ret)

base_dir = r"..\data"
result_dir = r"..\raw_factors"
market_path = os.path.join(base_dir, "TRD_Dalyr")
factor_name = "downside_vol"
output_file = os.path.join(result_dir, f"{factor_name}.csv")

use_cols = ['Stkcd', 'Trddt', 'Clsprc', 'CumulateFwardFactor']
all_data = []
for file_path in glob.glob(os.path.join(market_path, '*.csv')):
    try:
        df_temp = pd.read_csv(file_path) 
        valid = [c for c in use_cols if c in df_temp.columns]
        if 'Stkcd' not in valid or 'Trddt' not in valid: continue
        df_temp = df_temp[valid]
        df_temp['Trddt'] = pd.to_datetime(df_temp['Trddt'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Trddt', 'Stkcd'])
        all_data.append(df_temp)
    except: pass

df = pd.concat(all_data, ignore_index=True).sort_values(['Stkcd', 'Trddt'])

if 'CumulateFwardFactor' in df.columns:
    df['adj_close'] = df['Clsprc'] * df['CumulateFwardFactor']
else:
    print("警告：没有前复权，使用原始价格")
    df['adj_close'] = df['Clsprc']

df['ret'] = df.groupby('Stkcd')['adj_close'].pct_change()

df[factor_name] = df.groupby('Stkcd')['ret'].transform(
    lambda x: x.rolling(window=20, min_periods=15).apply(calc_downside_std, raw=True)
)

df['year_month'] = df['Trddt'].dt.to_period('M')
df_month = df.sort_values('Trddt').groupby(['Stkcd', 'year_month']).tail(1).copy()
df_month['Trddt'] = df_month['Trddt'] + MonthEnd(0)

output_df = df_month[['Stkcd', 'Trddt', factor_name]].copy().dropna().sort_values(['Trddt', 'Stkcd'])
output_df['Stkcd'] = output_df['Stkcd'].astype(int).astype(str).str.zfill(6)
output_df.to_csv(output_file, index=False)
