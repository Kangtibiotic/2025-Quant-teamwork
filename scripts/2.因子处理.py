import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import factor_utils as fu


def process_factor(
        factor_series: pd.Series,
        industry_series: pd.Series = None,
        market_cap_series: pd.Series = None,
        mad_n: float = 5.0,
) -> pd.Series:
    """
    对单个因子执行完整预处理流程：
    1) 按行业中位数填充缺失
    2) MAD 缩尾
    3) Z-score 标准化
    4) 中性化（行业 + 可选市值）
    5) 再 Z-score 标准化
    """

    factor = factor_series.copy()

    # 若传入行业序列，则生成行业哑变量
    industry_dummies = None
    if industry_series is not None:
        industry_dummies = get_industry_dummies(industry_series)

    # ===== Step 1: 缺失值填充（按行业中位数） =====
    if industry_series is not None:
        # 按行业取中位数填充
        median_by_ind = factor.groupby(industry_series).transform("median")
        factor = factor.fillna(median_by_ind)

    # 若仍有缺失（行业缺失或没有行业列），用全局中位数填
    factor = factor.fillna(factor.median())

    # ===== Step 2: MAD 缩尾 =====
    factor = fu.winsorize_by_mad(factor, n=mad_n)

    # ===== Step 3: 标准化 =====
    factor = fu.standardize_zscore(factor)

    # ===== Step 4: 中性化（行业 + 市值） =====
    if industry_dummies is not None:
        factor = fu.neutralize(
            factor_series=factor,
            industry_dummies=industry_dummies,
            market_cap_series=market_cap_series
        )

    # ===== Step 5: 再标准化 =====
    factor = fu.standardize_zscore(factor)

    return factor


def get_industry_dummies(series):
    ind_dummies = pd.get_dummies(series, prefix="ind", drop_first=False)
    ind_dummies.index = series.index
    return ind_dummies


if __name__ == "__main__":
    # 读取因子
    factor_dir = r"raw_factors"
    factor_file_list = os.listdir(factor_dir)

    factor = pd.DataFrame()
    for file in tqdm(factor_file_list, desc="Reading factor files"):
        if factor.empty:
            factor = pd.read_csv(os.path.join(factor_dir, file), dtype={"Stkcd": str})
            factor["Trddt"] = (
                    pd.to_datetime(factor["Trddt"]) + pd.tseries.offsets.MonthEnd(0)
            )
            earliest_time = factor["Trddt"].min()
        else:
            new_factor = pd.read_csv(
                os.path.join(factor_dir, file), dtype={"Stkcd": str}
            )
            new_factor["Trddt"] = (
                    pd.to_datetime(new_factor["Trddt"]) + pd.tseries.offsets.MonthEnd(0)
            )
            new_earliest_time = new_factor["Trddt"].min()
            if new_earliest_time > earliest_time:
                earliest_time = new_earliest_time
                factor = factor[factor["Trddt"] >= earliest_time]
            else:
                new_factor = new_factor[new_factor["Trddt"] >= earliest_time]

            factor = factor.merge(new_factor, on=["Stkcd", "Trddt"], how="outer")

    non_factor_cols = ["Stkcd", "Trddt"]
    factor_cols = [col for col in factor.columns if col not in non_factor_cols]

    # 标准化、中性化
    # 读取行业数据，合并
    industry_file_path = r"processed_data/ind_zhongxin.csv"
    df_ind = pd.read_csv(
        industry_file_path, index_col=0, dtype={"in_date": str, "out_date": str}
    )
    df_ind["in_date"] = pd.to_datetime(df_ind["in_date"])
    df_ind["out_date"] = pd.to_datetime(df_ind["out_date"].str[:8], errors="coerce")
    df_ind["out_date"] = df_ind["out_date"].fillna(pd.Timestamp.max)
    df_ind["Stkcd"] = df_ind["ts_code"].str[:6]
    df_ind.sort_values(["Stkcd", "in_date"], inplace=True)

    factor_ind = factor.merge(
        df_ind[["Stkcd", "l1_code", "in_date", "out_date"]], how="left"
    )
    factor_ind = factor_ind[
        (factor_ind["Trddt"] >= factor_ind["in_date"])
        & (factor_ind["Trddt"] <= factor_ind["out_date"])
        ]
    factor_ind.drop_duplicates(subset=["Stkcd", "Trddt"], keep="last", inplace=True)
    factor_ind.drop(columns=["in_date", "out_date"], inplace=True)

    # 读取流通市值数据，合并
    data_dir = r"processed_data/TRD_Dalyr"
    daily_ohlc_files = os.listdir(data_dir)
    daily_ohlc_list = []
    for file in tqdm(daily_ohlc_files, desc="Loading size data"):
        daily_ohlc = pd.read_csv(
            os.path.join(data_dir, file),
            dtype={"Stkcd": str},
            usecols=["Stkcd", "Trddt", "Dsmvosd"],
        )
        daily_ohlc_list.append(daily_ohlc)
    daily_ohlc = pd.concat(daily_ohlc_list)
    daily_ohlc["Trddt"] = pd.to_datetime(daily_ohlc["Trddt"])
    daily_ohlc.sort_values(["Trddt", "Stkcd"], inplace=True)
    size = (
        daily_ohlc.groupby([daily_ohlc["Trddt"].dt.to_period("M"), daily_ohlc["Stkcd"]])
        .agg({"Dsmvosd": "last"})
        .reset_index()
        .dropna()
    )
    size["Trddt"] = size["Trddt"].dt.to_timestamp() + pd.tseries.offsets.MonthEnd(0)
    factor_ind_size = factor_ind.merge(size, how="left")

    processed_list = []
    for dt, group in tqdm(factor_ind_size.groupby("Trddt"), desc="Processing factors"):
        group = group.copy()
        industry_dummies = get_industry_dummies(group)
        mktcap_series = group["Dsmvosd"]

        # 对每个因子进行处理
        for fac in factor_cols:
            if fac == "logSize":
                group[fac] = process_factor(
                    factor_series=group[fac],
                    industry_series=group['l1_code'],
                    mad_n=5.0,
                )
            else:
                group[fac] = process_factor(
                    factor_series=group[fac],
                    industry_series=group['l1_code'],
                    market_cap_series=mktcap_series,
                    mad_n=5.0,
                )

        processed_list.append(group)

    # 合并处理结果
    factor_processed = pd.concat(processed_list, ignore_index=True)
    factor_processed = factor_processed[["Stkcd", "Trddt"] + factor_cols]
    factor_processed.to_csv(
        os.path.join('processed_data/processed_factors.csv'), index=False
    )
