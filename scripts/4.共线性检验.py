import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

os.makedirs("results/plots", exist_ok=True)

def corr(df, factor_cols):
    # 计算相关性与绘制热力图
    correlation_matrix = df[factor_cols].corr(method='spearman')  # 使用spearman
    print(correlation_matrix.to_string(float_format="%.4f"))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )

    plt.title("Factor Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/plots/factor_correlation_heatmap.png")
    plt.show()


    # 共线性检查
    df_factors = df[factor_cols]
    df_factors_clean = df_factors.dropna()
    remaining_rows = len(df_factors_clean)

    if remaining_rows < len(factor_cols) + 1:
        print("删除 NaN 后剩余的数据太少，无法计算 VIF。")
        exit()
    X = add_constant(df_factors_clean)
    # 计算 VIF
    vif_data = pd.DataFrame()
    vif_data["Factor"] = X.columns[1:]

    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(1, X.shape[1])  # 从 1 开始，跳过 const
    ]
    # print(vif_data.to_string(float_format="%.4f"))
    return vif_data


df = pd.read_csv("processed_data/processed_factors.csv", dtype={'Stkcd': str})
factor_cols = [col for col in df.columns if col not in ['Stkcd', 'Trddt']]

vif = corr(df, factor_cols)

# 因子合并与剔除
# 直接保留['turnover_20', 'Amihud', 'cvp_20', 'OVB_SLP', 'UTD', 'trend_strength']
# 剔除['revenue_growth',  'trend_strength', 'log_Size']
new_df = df[['Stkcd', 'Trddt', 'turnover_20', 'Amihud', 'cpv_20', 'OVB_SLP', 'UTD']].copy()
# 1. 合并e_p和roe
new_df['earnings_ability'] = 0.5 * (df['roe'] + df['e_p'])
# 2. 合并VOL和downside_vol
new_df['VOL'] = 0.5 * (df['VOL'] + df['downside_vol'])
# 3. 合并reversal_20和mom_1m
new_df['mom'] = 0.5 * (df['reversal_20'] - df['mom_1m'])

new_factor_cols = [col for col in new_df.columns if col not in ['Stkcd', 'Trddt']]

new_vif = corr(new_df, new_factor_cols)

new_df.to_csv('processed_data/final_factors.csv', index=False)
