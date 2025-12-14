import os
import pandas as pd
import numpy as np
from tqdm import tqdm


data_dir = r'processed_data/TRD_Dalyr'
daily_ohlc_files = os.listdir(data_dir)
daily_ohlc_list = []
for file in tqdm(daily_ohlc_files, desc='Loading data'):
    daily_ohlc = pd.read_csv(os.path.join(data_dir, file),
                             dtype={'Stkcd': str})
    daily_ohlc_list.append(daily_ohlc)
daily_ohlc = pd.concat(daily_ohlc_list)
daily_ohlc['Trddt'] = pd.to_datetime(daily_ohlc['Trddt'])
daily_ohlc.sort_values(['Trddt', 'Stkcd'], inplace=True)


# 换手率波动性因子
daily_ohlc['turnover'] = daily_ohlc['Dnshrtrd'] * daily_ohlc['Clsprc'] \
    / daily_ohlc['Dsmvosd'] * 1e-3
factor = daily_ohlc.groupby([daily_ohlc['Trddt'].dt.to_period('M'), daily_ohlc['Stkcd']]).agg(
    {'turnover': 'std'}
).reset_index()
factor['UTD'] = factor.groupby('Stkcd')['turnover'].transform(
    lambda x: x.rolling(6).apply(lambda y: y.std() / y.mean())
)
factor = factor.drop(columns='turnover').dropna()

output_path = r'raw_factors/UTD.csv'
os.makedirs(output_path, exist_ok=True)
factor.to_csv(output_path, index=False)
