import pandas as pd
import os
import glob
import numpy as np
from pandas.tseries.offsets import MonthEnd
from scipy import stats

# 计算R方
def calc_r2(y):
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


base_dir = r"processed_data"
result_dir = r"raw_factors"
market_path = os.path.join(base_dir, "TRD_Dalyr")
factor_name = "trend_strength"
output_file = os.path.join(result_dir, f"{factor_name}.csv")

use_cols = ['Stkcd', 'Trddt', 'Clsprc', 'CumulateFwardFactor']
all_data = []
for file_path in glob.glob(os.path.join(market_path, '*.csv')):
    try:
        df_check = pd.read_csv(file_path, nrows=1)
        if not set(['Stkcd', 'Trddt']).issubset(df_check.columns): continue
        df_temp = pd.read_csv(file_path, usecols=lambda x: x in use_cols)
        df_temp['Trddt'] = pd.to_datetime(df_temp['Trddt'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Trddt', 'Stkcd'])
        all_data.append(df_temp)
    except Exception: pass

df = pd.concat(all_data, ignore_index=True).sort_values(['Stkcd', 'Trddt'])

if 'CumulateFwardFactor' in df.columns:
    df['adj_close'] = df['Clsprc'] * df['CumulateFwardFactor']
else:
    print("警告：没有前复权，只能使用原始价格")
    df['adj_close'] = df['Clsprc']

df[factor_name] = df.groupby('Stkcd')['adj_close'].transform(
    lambda x: x.rolling(window=20, min_periods=15).apply(calc_r2, raw=True)
)

df['year_month'] = df['Trddt'].dt.to_period('M')
df_month = df.sort_values('Trddt').groupby(['Stkcd', 'year_month']).tail(1).copy()
df_month['Trddt'] = df_month['Trddt'] + MonthEnd(0)

output_df = df_month[['Stkcd', 'Trddt', factor_name]].copy().dropna().sort_values(['Trddt', 'Stkcd'])
output_df['Stkcd'] = output_df['Stkcd'].astype(int).astype(str).str.zfill(6)
output_df.to_csv(output_file, index=False)
