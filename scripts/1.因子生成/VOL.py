import os
import pandas as pd
import numpy as np
from tqdm import tqdm

input_dir = r'processed_data/TRD_Dalyr'
output_file = r'raw_factors/VOL.csv'

for file_name in tqdm(sorted(os.listdir(input_dir))):
    if not file_name.endswith('.csv'):
        continue

    year, month = file_name.replace('.csv', '').split('-')
    trddt = f"{int(year)}/{int(month)}/1"
    file_path = os.path.join(input_dir, file_name)

    df = pd.read_csv(file_path, dtype={'Stkcd': str})
    df['Trddt'] = pd.to_datetime(df['Trddt'])
    df = df.sort_values(['Stkcd', 'Trddt'])

    df['AdjClose'] = df['Clsprc'] * df['CumulateFwardFactor']

    vol_series = (
        df.groupby('Stkcd')['AdjClose']
        .apply(lambda x: np.log(x / x.shift(1)).dropna().std())
    )

    result = pd.DataFrame({
        'Trddt': trddt,
        'Stkcd': vol_series.index,
        'VOL': vol_series.values
    })

    write_header = not os.path.exists(output_file)
    result.to_csv(output_file, index=False, mode='a', header=write_header)