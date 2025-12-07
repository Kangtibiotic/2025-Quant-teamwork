import os
import pandas as pd
from tqdm import tqdm


raw_data_dir = r'..\raw_data'
data_dir = r'.\data'

# 将复权因子合并到日频数据中
TRD_dir = os.path.join(raw_data_dir, 'TRD_Dalyr')
ohlc_files = os.listdir(TRD_dir)

ohlc_dfs = []
for file in tqdm(ohlc_files, desc='Loading trade data'):
    daily_ohlc = pd.read_csv(os.path.join(TRD_dir, file),
                             usecols=['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd',
                                    'Dnvaltrd', 'Dsmvosd', 'Dsmvtll','LimitDown', 'LimitUp',
                                    'LimitStatus'],
                             dtype={'Stkcd': str})
    ohlc_dfs.append(daily_ohlc)
daily_ohlc = pd.concat(ohlc_dfs)
daily_ohlc['Trddt'] = pd.to_datetime(daily_ohlc['Trddt'])
daily_ohlc = daily_ohlc.sort_values(['Trddt', 'Stkcd'])

adj_factors = pd.read_csv(os.path.join(data_dir, 'TRD_AdjustFactor.csv'), dtype={'Symbol': str}, 
                          usecols=['Symbol', 'TradingDate', 'CumulateFwardFactor']
                          )
adj_factors.rename(columns={'Symbol': 'Stkcd', 'TradingDate': 'Trddt'}, inplace=True)
adj_factors['Trddt'] = pd.to_datetime(adj_factors['Trddt'])

daily_ohlc_merged = daily_ohlc.merge(adj_factors, how='left', on=['Stkcd', 'Trddt'])
daily_ohlc_merged['CumulateFwardFactor'] = daily_ohlc_merged.groupby('Stkcd')[
    'CumulateFwardFactor'].bfill()
daily_ohlc_merged['CumulateFwardFactor'] = daily_ohlc_merged['CumulateFwardFactor'].fillna(1)

for month, group in tqdm(daily_ohlc_merged.groupby(daily_ohlc_merged['Trddt'].dt.to_period('M'))):
    group.to_csv(os.path.join(data_dir, fr'TRD_Dalyr\{month}.csv'), index=False)
    
    
#  筛选财报数据
income = pd.read_csv(
    os.path.join(data_dir, "FS_Comins.csv"),
    skiprows=[1, 2],
    parse_dates=["Accper"],
    dtype={"Stkcd": str},
)
income = income[(income["Accper"].dt.month != 1) & (income["Typrep"] == "A")]
income.to_csv(os.path.join(data_dir, "FS_Comins.csv"), index=False)

balance = pd.read_csv(
    os.path.join(data_dir, "FS_Combas.csv"),
    skiprows=[1, 2],
    parse_dates=["Accper"],
    dtype={"Stkcd": str},
)
balance = balance[(balance["Accper"].dt.month != 1) & (balance["Typrep"] == "A")]
balance.to_csv(os.path.join(data_dir, "FS_Combas.csv"), index=False)
