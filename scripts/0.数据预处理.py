import os
import pandas as pd
from tqdm import tqdm


raw_data_dir = r'raw_data'
processed_data_dir = r'processed_data'

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

adj_factors = pd.read_csv(os.path.join(raw_data_dir, 'TRD_AdjustFactor.csv'),
                          dtype={'Symbol': str},
                          usecols=['Symbol', 'TradingDate', 'CumulateFwardFactor'])
adj_factors.rename(columns={'Symbol': 'Stkcd', 'TradingDate': 'Trddt'}, inplace=True)
adj_factors['Trddt'] = pd.to_datetime(adj_factors['Trddt'])

daily_ohlc_merged = daily_ohlc.merge(adj_factors, how='left', on=['Stkcd', 'Trddt'])
daily_ohlc_merged['CumulateFwardFactor'] = daily_ohlc_merged.groupby('Stkcd')[
    'CumulateFwardFactor'].bfill()
daily_ohlc_merged['CumulateFwardFactor'] = daily_ohlc_merged['CumulateFwardFactor'].fillna(1)

# 输出按月拆分的 OHLC 数据
TRD_processed_dir = os.path.join(processed_data_dir, 'TRD_Dalyr')
os.makedirs(TRD_processed_dir, exist_ok=True)
for month, group in tqdm(daily_ohlc_merged.groupby(daily_ohlc_merged['Trddt'].dt.to_period('M'))):
    group.to_csv(os.path.join(TRD_processed_dir, f'{month}.csv'), index=False)


# 筛选财报数据
income = pd.read_csv(
    os.path.join(raw_data_dir, "FS_Comins.csv"),
    skiprows=[1, 2],
    parse_dates=["Accper"],
    dtype={"Stkcd": str},
)
income = income[(income["Accper"].dt.month != 1) & (income["Typrep"] == "A")]
income.to_csv(os.path.join(processed_data_dir, "FS_Comins.csv"), index=False)

balance = pd.read_csv(
    os.path.join(raw_data_dir, "FS_Combas.csv"),
    skiprows=[1, 2],
    parse_dates=["Accper"],
    dtype={"Stkcd": str},
    encoding='gbk'
)
balance = balance[(balance["Accper"].dt.month != 1) & (balance["Typrep"] == "A")]
balance.to_csv(os.path.join(processed_data_dir, "FS_Combas.csv"), index=False)


# 生成月度无风险利率
nrrate = pd.read_excel(
    os.path.join(raw_data_dir, "TRD_Nrrate.xlsx"),
    skiprows=[1, 2],
    parse_dates=["Clsdt"],
)
nrrate["month"] = nrrate["Clsdt"].dt.to_period("M").dt.to_timestamp()

nrrate["daily"] = nrrate["Nrrdaydt"] / 100
monthly = (
    nrrate.groupby("month")["daily"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index(name="nrrate")
)
monthly.to_csv(os.path.join(processed_data_dir, "nrrate.csv"), index=False)


# 处理FF5因子
FF5 = pd.read_excel(
    os.path.join(raw_data_dir, "STK_MKT_FIVEFACMONTH.xlsx"),
    skiprows=[1, 2],
    parse_dates=["TradingMonth"],
)
portfolio_type = 1# 选择 portfolio，每个值代表计算因子强度时采取的划分方式

FF5["TradingMonth"] = pd.to_datetime(FF5["TradingMonth"])

FF5["MarketNum"] = FF5["MarkettypeID"].str.extract(r'(\d+)').astype(int) # 提取 MarkettypeID 数字部分

max_mkt = ( # 每个月选取 MarketNum 最大的 MarkettypeID，确保覆盖当前的全A股
    FF5.groupby(FF5["TradingMonth"])["MarketNum"]
      .transform(lambda x: x == x.max())
)
FF5_max_mkt = FF5[max_mkt].copy()
FF5_max_mkt_port = FF5_max_mkt[FF5_max_mkt["Portfolios"] == portfolio_type].copy()

all_months = FF5["TradingMonth"].drop_duplicates().sort_values() # 检查缺失月份
existing_months = FF5_max_mkt_port["TradingMonth"].drop_duplicates()

missing_months = sorted(set(all_months) - set(existing_months))

if missing_months:
    print("以下月份在最大 MarkettypeID 下不存在指定 portfolio 的数据：")
    for m in missing_months:
        print(m.strftime("%Y-%m"))
else:
    print("FF5因子无缺失月份。")

FF5_max_mkt_port["month"] = FF5_max_mkt_port["TradingMonth"].dt.to_period("M").dt.to_timestamp() # 生成最终输出表

output_cols = ["month", "RiskPremium1", "SMB1", "HML1", "RMW1", "CMA1"]
FF5_out = FF5_max_mkt_port[output_cols].sort_values("month")

output_path = os.path.join(processed_data_dir, f"FF5.csv")
FF5_out.to_csv(output_path, index=False)

