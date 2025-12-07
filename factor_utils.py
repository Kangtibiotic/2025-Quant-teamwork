import pandas as pd
import numpy as np
import statsmodels.api as sm

"""
因子预处理工具函数库
包含以下功能：
1. MAD缩尾
2. 市值与行业中性化
3. Z-Score标准化
"""

def winsorize_by_mad(factor_series: pd.Series, n: float = 5.0) -> pd.Series:
    """
    参数:
    factor_series : pd.Series，输入的原始因子序列（截面数据）。
    n : float, 默认为5
    返回:
    pd.Series，经过 MAD 缩尾处理后的因子序列。
    """
    if not isinstance(factor_series, pd.Series):
        raise TypeError("输入pandas Series。")
    data = factor_series.dropna()
    
    median = data.median()
    
    dev_from_median = (data - median).abs()
    mad = dev_from_median.median()
    
    scaled_mad = mad
    
    if scaled_mad == 0:
        return factor_series
        
    upper_bound = median + n * scaled_mad
    lower_bound = median - n * scaled_mad
    
    clipped_series = factor_series.clip(lower=lower_bound, upper=upper_bound)
    
    return clipped_series


def neutralize(
    factor_series: pd.Series,
    industry_dummies: pd.DataFrame,
    market_cap_series: pd.Series = None
) -> pd.Series:
    """
    对因子进行行业中性化，可选市值中性化。
    
    参数:
    factor_series : pd.Series
        输入的因子序列，索引为股票代码
    industry_dummies : pd.DataFrame
        行业虚拟变量矩阵，索引为股票代码
    market_cap_series : pd.Series or None
        市值序列；若为 None，则只做行业中性化

    返回:
    pd.Series
        中性化处理后的因子序列
    """

    if not isinstance(factor_series, pd.Series):
        raise TypeError("factor_series 必须是 pandas Series。")
    if not isinstance(industry_dummies, pd.DataFrame):
        raise TypeError("industry_dummies 必须是 pandas DataFrame。")

    # ===== 构造自变量 X =====
    X_parts = []

    # 市值中性化
    if market_cap_series is not None:
        if not isinstance(market_cap_series, pd.Series):
            raise TypeError("market_cap_series 必须是 pandas Series。")
        log_mcap = np.log(market_cap_series).rename("log_mcap")
        X_parts.append(log_mcap)

    # 行业哑变量
    X_parts.append(industry_dummies.astype(float))

    # 合并 X
    X = pd.concat(X_parts, axis=1)
    Y = factor_series.rename("factor")

    # 清洗 NA 并对齐
    data_all = pd.concat([Y, X], axis=1).dropna()
    Y_clean = data_all["factor"]
    X_clean = data_all[X.columns]

    # 自变量数量 > 样本数 ⇒ 无法回归
    if Y_clean.shape[0] < X_clean.shape[1]:
        print("数据点少于特征数，无法进行中性化，返回原始因子。")
        return factor_series

    # 回归
    model = sm.OLS(Y_clean, X_clean).fit()
    residuals = model.resid

    # 输出序列与原 index 对齐
    neutralized = pd.Series(np.nan, index=factor_series.index, name="neutralized_factor")
    neutralized.loc[residuals.index] = residuals

    return neutralized



def standardize_zscore(factor_series: pd.Series) -> pd.Series:
    """
    对因子数据进行 Z-Score 标准化。
    参数:
    factor_series : pd.Series,输入的因子序列

    返回:
    pd.Series,经过 Z-Score 标准化后的因子序列
    """
    
    if not isinstance(factor_series, pd.Series):
        raise TypeError("输入的数据必须是 pandas Series。")
        
    data = factor_series.dropna()

    mean = data.mean()
    std = data.std()
    
    if std == 0:
        return pd.Series(0.0, index=factor_series.index)
        
    zscore_series = (factor_series - mean) / std
    
    return zscore_series