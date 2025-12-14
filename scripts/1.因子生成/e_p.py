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
market_path = os.path.join(base_dir, "TRD_Dalyr")
factor_name = "e_p"
output_file = os.path.join(result_dir, f"{factor_name}.csv")

# 取财务数据
ins = pd.read_csv(ins_path, usecols=['Stkcd', 'Accper', 'B002000101'])
ins = ins.rename(columns={'B002000101': 'NetIncome'})
ins['Accper'] = pd.to_datetime(ins['Accper'])
ins = ins.sort_values(['Stkcd', 'Accper'])

ins['NetIncome_q'] = ins.groupby(['Stkcd', ins['Accper'].dt.year])['NetIncome'].diff()
is_q1 = ins['Accper'].dt.month == 3
ins.loc[is_q1, 'NetIncome_q'] = ins.loc[is_q1, 'NetIncome']

ins['NetIncome_TTM'] = ins.groupby('Stkcd')['NetIncome_q'].rolling(window=4, min_periods=4).sum().reset_index(level=0, drop=True)

# 财务数据延迟
ins['PublicationDate'] = ins['Accper'].apply(get_pub_date)
df_fin = ins[['Stkcd', 'PublicationDate', 'NetIncome_TTM']].dropna().sort_values(['PublicationDate', 'Stkcd'])
df_fin = df_fin.drop_duplicates(subset=['Stkcd', 'PublicationDate'], keep='last')

market_files = glob.glob(os.path.join(market_path, '*.csv'))
use_cols = ['Stkcd', 'Trddt', 'Dsmvosd'] 
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
df_mkt = df_mkt.rename(columns={'Dsmvosd': 'TotalMarketCap'})
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

final[factor_name] = final['NetIncome_TTM'] / final['TotalMarketCap']
final.replace([np.inf, -np.inf], np.nan, inplace=True)

output = final[['Stkcd', 'Trddt', factor_name]].copy()
output['Trddt'] = output['Trddt'] + MonthEnd(0)
output['Stkcd'] = output['Stkcd'].astype(int).astype(str).str.zfill(6)
output = output.dropna().sort_values(['Trddt', 'Stkcd'])

output.to_csv(output_file, index=False)
