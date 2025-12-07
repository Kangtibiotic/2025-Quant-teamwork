import os
import pandas as pd
import numpy as np
from tqdm import tqdm

input_dir = r'..\data\TRD_Dalyr'
output_file = r'..\raw_factors\OVB_SLP.csv'
volume_col = 'Dnshrtrd'

for file_name in tqdm(sorted(os.listdir(input_dir))):
    if not file_name.endswith('.csv'):
        continue

    year, month = file_name.replace('.csv', '').split('-')
    trddt = f"{int(year)}/{int(month)}/1"
    file_path = os.path.join(input_dir, file_name)

    df = pd.read_csv(file_path)
    df['Trddt'] = pd.to_datetime(df['Trddt'])
    df = df.sort_values(['Stkcd', 'Trddt'])

    df['AdjClose'] = df['Clsprc'] * df['CumulateFwardFactor']

    def obv_slope(g):
        prices = g['AdjClose'].values
        volumes = g[volume_col].values
        N = len(prices)
        if N < 2:
            return np.nan

        obv = np.zeros(N)
        for t in range(1, N):
            if prices[t] > prices[t - 1]:
                obv[t] = obv[t - 1] + volumes[t]
            elif prices[t] < prices[t - 1]:
                obv[t] = obv[t - 1] - volumes[t]
            else:
                obv[t] = obv[t - 1]

        x = np.arange(N)
        slope = np.polyfit(x, obv, 1)[0]
        return slope

    slope_series = df.groupby('Stkcd').apply(obv_slope, include_groups=False)

    result = pd.DataFrame({
        'Trddt': trddt,
        'Stkcd': slope_series.index,
        'OVB_SLP': slope_series.values
    })

    write_header = not os.path.exists(output_file)
    result.to_csv(output_file, index=False, mode='a', header=write_header)
