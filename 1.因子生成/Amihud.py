import os
import pandas as pd
import numpy as np
from tqdm import tqdm

data_dir = r'..\data\TRD_Dalyr'
daily_ohlc_files = os.listdir(data_dir)
daily_ohlc_list = []
for file in tqdm(daily_ohlc_files, desc='Loading data'):
    daily_ohlc = pd.read_csv(os.path.join(data_dir, file),
                             dtype={'Stkcd': str})
    daily_ohlc_list.append(daily_ohlc)
daily_ohlc = pd.concat(daily_ohlc_list)
daily_ohlc['Trddt'] = pd.to_datetime(daily_ohlc['Trddt'])
daily_ohlc.sort_values(['Trddt', 'Stkcd'], inplace=True)


# Amihud非流动性因子
daily_ohlc['AdjClsprc'] = daily_ohlc['Clsprc'] * daily_ohlc['CumulateFwardFactor']
daily_ohlc['Return_1d'] = daily_ohlc.groupby('Stkcd')['AdjClsprc'].pct_change()
daily_ohlc['Amihud'] = daily_ohlc['Return_1d'].abs() / daily_ohlc['Dnvaltrd'] * 1e10
factor = daily_ohlc.groupby([daily_ohlc['Trddt'].dt.to_period('M'), daily_ohlc['Stkcd']]).agg(
    {'Amihud': 'mean'}
).reset_index().dropna()

output_path = r'..\raw_factors\Amihud.csv'
# os.makedirs(output_path)
factor.to_csv(output_path, index=False)