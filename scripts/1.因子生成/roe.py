import pandas as pd
import os
import glob
import numpy as np
from pandas.tseries.offsets import MonthEnd

# 财务数据延迟函数
def get_pub_date(d):
    if d.month == 3: return pd.Timestamp(d.year, 4, 30)
    elif d.month == 6: return pd.Timestamp(d.year, 8, 31)
    elif d.month == 9: return pd.Timestamp(d.year, 10, 31)
    elif d.month == 12: return pd.Timestamp(d.year + 1, 4, 30)
    return pd.NaT

base_dir = r"processed_data"
result_dir = r"raw_factors"
ins_path = r"raw_data\FS_Comins.csv"
bas_path = r"raw_data\FS_Combas.csv"
market_path = os.path.join(base_dir, "TRD_Dalyr")
factor_name = "roe"
output_file = os.path.join(result_dir, f"{factor_name}.csv")

# 财务数据读取
ins = pd.read_csv(ins_path, usecols=['Stkcd', 'Accper', 'B002000101']).rename(columns={'B002000101': 'NetIncome'})
bas = pd.read_csv(bas_path, usecols=['Stkcd', 'Accper', 'A003100000']).rename(columns={'A003100000': 'Equity'})
ins['Accper'] = pd.to_datetime(ins['Accper'])
bas['Accper'] = pd.to_datetime(bas['Accper'])

df_fin = pd.merge(ins, bas, on=['Stkcd', 'Accper'], how='inner').sort_values(['Stkcd', 'Accper'])

df_fin['NetIncome_q'] = df_fin.groupby(['Stkcd', df_fin['Accper'].dt.year])['NetIncome'].diff()
is_q1 = df_fin['Accper'].dt.month == 3
df_fin.loc[is_q1, 'NetIncome_q'] = df_fin.loc[is_q1, 'NetIncome']
df_fin['NetIncome_TTM'] = df_fin.groupby('Stkcd')['NetIncome_q'].rolling(4, min_periods=4).sum().reset_index(0, drop=True)

df_fin['Equity_lag4'] = df_fin.groupby('Stkcd')['Equity'].shift(4)
df_fin['Avg_Equity'] = (df_fin['Equity'] + df_fin['Equity_lag4']) / 2

df_fin[factor_name] = df_fin['NetIncome_TTM'] / df_fin['Avg_Equity']

# 财务数据滞后处理
df_fin['PublicationDate'] = df_fin['Accper'].apply(get_pub_date)
df_fin = df_fin[['Stkcd', 'PublicationDate', factor_name]].dropna().sort_values(['PublicationDate', 'Stkcd'])
df_fin = df_fin.drop_duplicates(subset=['Stkcd', 'PublicationDate'], keep='last')

market_files = glob.glob(os.path.join(market_path, '*.csv'))
use_cols = ['Stkcd', 'Trddt'] 
all_mkt = []
for file_path in glob.glob(os.path.join(market_path, '*.csv')):
    try:
        df_check = pd.read_csv(file_path, nrows=1)
        valid_cols = [c for c in use_cols if c in df_check.columns]
        if 'Stkcd' not in valid_cols or 'Trddt' not in valid_cols: continue
        
        df_temp = pd.read_csv(file_path, usecols=valid_cols)
        df_temp['Trddt'] = pd.to_datetime(df_temp['Trddt'], errors='coerce')
        df_temp = df_temp.dropna(subset=['Trddt', 'Stkcd'])
        all_mkt.append(df_temp)
    except Exception:
        pass

df_mkt = pd.concat(all_mkt, ignore_index=True)
df_mkt['month_str'] = df_mkt['Trddt'].dt.strftime('%Y-%m')
df_mkt = df_mkt.sort_values('Trddt').groupby(['Stkcd', 'month_str']).tail(1).copy()

df_mkt = df_mkt.drop(columns=['month_str'])

final = pd.merge_asof(
    df_mkt,
    df_fin,
    left_on='Trddt',
    right_on='PublicationDate',
    by='Stkcd',
    direction='backward'
)

output = final[['Stkcd', 'Trddt', factor_name]].copy()
output['Trddt'] = output['Trddt'] + MonthEnd(0)
output['Stkcd'] = output['Stkcd'].astype(int).astype(str).str.zfill(6)
output = output.dropna().sort_values(['Trddt', 'Stkcd'])

output.to_csv(output_file, index=False)
