import os
import pandas as pd
import numpy as np
from tqdm import tqdm


data_dir = r'..\data\TRD_Dalyr'
daily_ohlc_files = os.listdir(data_dir)
daily_ohlc_list = []
for file in tqdm(daily_ohlc_files, desc='Loading data'):
    daily_ohlc = pd.read_csv(os.path.join(data_dir, file),
                             dtype={'Stkcd': str},
                             usecols=['Stkcd', 'Trddt', 'Dsmvosd'])
    daily_ohlc_list.append(daily_ohlc)
daily_ohlc = pd.concat(daily_ohlc_list)
daily_ohlc['Trddt'] = pd.to_datetime(daily_ohlc['Trddt'])
daily_ohlc.sort_values(['Trddt', 'Stkcd'], inplace=True)


# 对数市值因子
daily_ohlc['logSize'] = np.log(daily_ohlc['Dsmvosd'])
factor = daily_ohlc.groupby([daily_ohlc['Trddt'].dt.to_period('M'), daily_ohlc['Stkcd']]).agg(
    {'logSize': 'mean'}
).reset_index().dropna()
output_path = r'..\raw_factors\logSize.csv'
#os.makedirs(output_path, exist_ok=True)
factor.to_csv(output_path, index=False)
