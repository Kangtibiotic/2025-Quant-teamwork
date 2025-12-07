import os
import pandas as pd
from tqdm import tqdm

input_dir = r'..\data\TRD_Dalyr'
output_file = r'..\raw_factors\mom_1m.csv'

# 遍历所有月度文件
for file_name in tqdm(sorted(os.listdir(input_dir))):
    if not file_name.endswith('.csv'):
        continue

    # 解析年月
    year, month = file_name.replace('.csv', '').split('-')
    trddt = f"{int(year)}/{int(month)}/1"
    file_path = os.path.join(input_dir, file_name)

    # 读取数据
    df = pd.read_csv(file_path, dtype={'Stkcd': str})
    df = df.sort_values(by=["Stkcd", "Trddt"])
    df["Trddt"] = pd.to_datetime(df["Trddt"])

    df['AdjClose'] = df['Clsprc'] * df['CumulateFwardFactor']

    # 动量：最后价格 / 第一个价格 - 1
    first_close = df.groupby("Stkcd").first()["AdjClose"]
    last_close = df.groupby("Stkcd").last()["AdjClose"]

    momentum = (last_close / first_close - 1).reset_index()
    momentum.columns = ["Stkcd", "mom_1m"]
    momentum["Trddt"] = trddt

    # 追加写入
    write_header = not os.path.exists(output_file)
    momentum.to_csv(output_file, index=False, mode='a', header=write_header)
