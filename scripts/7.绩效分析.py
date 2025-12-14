import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# 0.路径管理与依赖

# 无风险利率文件路径
path_rf = "processed_data/nrrate.csv"

# FF5 因子文件路径
path_ff5 = "processed_data/FF5.csv"

# 策略收益率文件路径
returns_paths_dict = {
    "lightGBM_train": "results/train_results/lightGBM_returns.csv",
    "lightGBM_test":  "results/test_results/lightGBM_returns.csv",

    "NeuralNetwork_train": "results/train_results/NeuralNetwork_returns.csv",
    "NeuralNetwork_test":  "results/test_results/NeuralNetwork_returns.csv",

    "Ridge_train": "results/train_results/Ridge_returns.csv",
    "Ridge_test":  "results/test_results/Ridge_returns.csv"
}

output_dir = "results/performance_analysis/"
os.makedirs(output_dir, exist_ok=True)

# 1.数据加载与对齐
def load_and_align_data(path_rf, path_ff5, path_returns):
    """
    输入：
        path_rf: 无风险利率 CSV 文件路径
        path_ff5: FF5 因子 CSV 文件路径
        returns_paths_dict: dict, 每个策略收益率 CSV 文件路径
    输出：
        rf: DataFrame, datetime索引, RF列
        ff5: DataFrame, datetime索引, MKT/SMB/HML/RMW/CMA列
        returns_dict: dict, 每个策略收益率 DataFrame, datetime索引, 多列组合
    """
    # 读取无风险利率
    rf = pd.read_csv(path_rf, index_col=0)
    rf.index = pd.to_datetime(rf.index, errors="coerce")
    rf = rf[rf.index.notnull()].sort_index()
    rf = rf.ffill()  # 替代 fillna(method="ffill")
    rf.rename(columns={"nrrate": "RF"}, inplace=True)
    # 读取FF5因子并重命名列
    ff5 = pd.read_csv(path_ff5, index_col=0)
    ff5.index = pd.to_datetime(ff5.index, errors="coerce")
    ff5 = ff5[ff5.index.notnull()].sort_index()
    ff5 = ff5.rename(columns={
        "RiskPremium1": "MKT",
        "SMB1": "SMB",
        "HML1": "HML",
        "RMW1": "RMW",
        "CMA1": "CMA"
    })
    ff5 = ff5.ffill()

    # 读取各策略收益率
    returns_dict = {}
    for name, path in returns_paths_dict.items():
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notnull()].sort_index()
        df = df.ffill()
        returns_dict[name] = df

    return rf, ff5, returns_dict

# 2.计算超额收益率
def compute_excess_returns(returns_dict, rf):
    """
    计算每个策略每个组合的超额收益率，保证索引对齐。
    输入：
        returns_dict: dict, 每个值是策略收益率 DataFrame (n_dates × n组合)
        rf: DataFrame, 无风险利率, datetime索引, RF列
    输出：
        excess_dict: dict, 每个值是超额收益率 DataFrame
    """
    excess_dict = {}
    for name, df in returns_dict.items():
        # 取收益率与rf的公共日期
        common_index = df.index.intersection(rf.index)
        df_aligned = df.loc[common_index]
        rf_aligned = rf.loc[common_index]

        # 超额收益率
        excess = df_aligned.subtract(rf_aligned["RF"], axis=0)
        excess_dict[name] = excess

    return excess_dict

# 3.五因子回归
def run_ff5_regression(excess_dict, ff5, use_newey_west=True, lags=3):
    """
    对每个策略的超额收益率的6个组合进行 FF5 回归。
    输入：
        excess_dict: dict，每个值是 DataFrame (n_dates × 6) 超额收益率
        ff5: DataFrame (n_dates × 5) 因子值
        use_newey_west: bool，是否使用 Newey-West 调整标准误
        lags: int，Newey-West 滞后阶数

    输出：
        results: dict，每个策略对应 6 个组合的回归结果
                 {strategy: {P1: res1, P2: res2, ..., P6: res6}}
    """

    results = {}
    factors = ["MKT","SMB","HML","RMW","CMA"]

    for strat, df in excess_dict.items():
        strat_res = {}
        for col in df.columns:
            y = df[col]
            X = ff5[factors]

            # 强制取公共日期
            common_index = y.index.intersection(X.index)
            y_aligned = y.loc[common_index]
            X_aligned = X.loc[common_index]

            # 添加常数项
            X_aligned = sm.add_constant(X_aligned)

            # OLS 回归
            model = sm.OLS(y_aligned, X_aligned).fit(cov_type='HAC', cov_kwds={'maxlags': lags} if use_newey_west else {})

            strat_res[col] = model
        results[strat] = strat_res

    return results


# 4.显著性检验与回归结果整理
def summarize_results(results):
    """
       输入：
           results: dict，每个策略对应每个组合的回归结果
                    {strategy: {P1: res1, P2: res2, ..., P6: res6}}
       输出：
           summary_df: DataFrame 汇总表格
                       列：['Strategy','Portfolio','Alpha','Alpha_t','Alpha_p',
                             'MKT','MKT_t','MKT_p', ..., 'CMA','CMA_t','CMA_p','R2','Adj_R2']
       """
    summary_list = []

    factors = ["MKT", "SMB", "HML", "RMW", "CMA"]

    for strat, combos in results.items():
        for combo_name, res in combos.items():
            record = {
                "Strategy": strat,
                "Portfolio": combo_name,
                "Alpha": res.params["const"],
                "Alpha_t": res.tvalues["const"],
                "Alpha_p": res.pvalues["const"],
                "R2": res.rsquared,
                "Adj_R2": res.rsquared_adj
            }

            for factor in factors:
                record[factor] = res.params[factor]
                record[f"{factor}_t"] = res.tvalues[factor]
                record[f"{factor}_p"] = res.pvalues[factor]

            summary_list.append(record)

    summary_df = pd.DataFrame(summary_list)
    summary_df["Strategy_Portfolio"] = summary_df["Strategy"] + "_" + summary_df["Portfolio"]


    return summary_df

# 5：因子贡献计算
def compute_factor_contributions(excess_dict, ff5, results):
    """
    计算每个策略每个组合的 alpha + 因子贡献时间序列。

    输入：
        excess_dict: dict, 超额收益率 (n_dates × 6)
        ff5: DataFrame, 因子时间序列 (n_dates × 5)
        results: dict, 模块3回归结果 {strategy: {P1: res, ...}}

    输出：
        contributions_dict: dict,
            {strategy: {P1: DataFrame, ...}}, 每个DataFrame (n_dates × 6列: alpha+5因子)
    """
    factors = ["MKT", "SMB", "HML", "RMW", "CMA"]
    contributions_dict = {}

    for strat, combos in results.items():
        strat_dict = {}
        for combo_name, res in combos.items():
            common_index = excess_dict[strat][combo_name].index.intersection(ff5.index)
            df = pd.DataFrame(index=common_index)
            # alpha 贡献
            df["Alpha"] = res.params["const"]
            # 各因子贡献
            for factor in factors:
                df[factor] = ff5.loc[common_index, factor] * res.params[factor]
            strat_dict[combo_name] = df
        contributions_dict[strat] = strat_dict

    return contributions_dict


# 6.绘制单一收益率文件堆叠图
def plot_factor_contributions_stack_all(contributions_dict, strategy):
    factors = ["Alpha", "MKT", "SMB", "HML", "RMW", "CMA"]
    combos = list(contributions_dict[strategy].keys())

    # 汇总每个组合的总贡献
    summary = {factor: [] for factor in factors}
    for combo in combos:
        df = contributions_dict[strategy][combo]
        for factor in factors:
            # 按时间区间求和，保留正负
            summary[factor].append(df[factor].sum())

    summary_df = pd.DataFrame(summary, index=combos)

    # 绘制堆叠柱状图
    summary_df.plot(kind="bar", stacked=True, figsize=(12,6))
    plt.title(f"Factor Contributions (Total) - {strategy}")
    plt.xlabel("Portfolio")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Total Contribution")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{strategy}_stack.png", dpi=300)
    plt.show()

# 7.绘制综合收益率堆叠图
def plot_factor_contributions_stack_compare(contrib_dict, strategies):
    factors = ["Alpha", "MKT", "SMB", "HML", "RMW", "CMA"]

    # 找到所有组合名称（取第一个策略作为参考）
    combos = list(contrib_dict[strategies[0]].keys())

    plot_data = []

    for combo in combos:
        for strat in strategies:
            df = contrib_dict[strat][combo]
            # 时间区间总贡献
            row = [df[factor].sum() for factor in factors]
            plot_data.append([f"{combo}_{strat}"] + row)

    plot_df = pd.DataFrame(plot_data, columns=["Portfolio"] + factors)
    plot_df.set_index("Portfolio", inplace=True)

    # 绘制堆叠柱状图
    plot_df.plot(kind="bar", stacked=True, figsize=(14,7))
    plt.title("Factor Contributions Comparison Across Strategies")
    plt.xlabel("Portfolio_Strategy")
    plt.ylabel("Total Contribution")
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"{output_dir}/strategies_comparison_stack.png", dpi=300)
    plt.show()

# 8.绘制因子暴露热力图
def plot_factor_exposures_heatmap(results, strategy):
    factors = ["const", "MKT", "SMB", "HML", "RMW", "CMA"]
    combos = list(results[strategy].keys())

    data = {combo: [results[strategy][combo].params[factor] for factor in factors] for combo in combos}
    df = pd.DataFrame(data, index=factors)
    df.rename(index={"const": "Alpha"}, inplace=True)

    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, cmap="RdBu_r", center=0, linewidths=0.5)
    plt.title(f"Factor Exposures Heatmap - {strategy}")
    plt.ylabel("Factors")
    plt.xlabel("Portfolio")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{strategy}_heatmap.png", dpi=300)
    plt.show()


def plot_train_test_r2_longshort(results_dict):
    """
    绘制三模型样本内外 R² 对比柱状图，仅使用 long_short 组合
    输入：
        results_dict: summarize_results 或回归结果字典
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # 如果是回归结果字典，则先转换为 summary_df
    if isinstance(results_dict, dict):
        summary_df = summarize_results(results_dict)
    else:
        summary_df = results_dict.copy()

    # 只保留 long_short 组合，并创建副本避免 SettingWithCopyWarning
    df_ls = summary_df[summary_df["Portfolio"].str.contains("long_short", case=False)].copy()

    # 提取模型和样本类型
    df_ls.loc[:, "Model"] = df_ls["Strategy"].str.replace(r"_(train|test)$", "", regex=True)
    df_ls.loc[:, "Period"] = df_ls["Strategy"].str.extract(r"_(train|test)$")[0]

    # 排序方便绘图
    models = df_ls["Model"].unique()
    periods = ["train", "test"]

    # 准备数据
    r2_data = []
    for model in models:
        for period in periods:
            df_filtered = df_ls[(df_ls["Model"]==model) & (df_ls["Period"]==period)]
            r2_value = df_filtered.loc[df_filtered["Portfolio"].str.contains("long_short"), "R2"].values[0]
            r2_data.append([model, period, r2_value])

    r2_df = pd.DataFrame(r2_data, columns=["Model", "Period", "R2"])

    # 绘图
    fig, ax = plt.subplots(figsize=(8,6))
    width = 0.35
    x = np.arange(len(models))

    train_r2 = r2_df[r2_df["Period"]=="train"]["R2"].values
    test_r2 = r2_df[r2_df["Period"]=="test"]["R2"].values

    ax.bar(x - width/2, train_r2, width, label="Train R²")
    ax.bar(x + width/2, test_r2, width, label="Test R²")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("R²")
    ax.set_title("Train vs Test R² for Long-Short Portfolio")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/train_test_r2_longshort.png", dpi=300)
    plt.show()




def main():
    rf, ff5, returns_dict = load_and_align_data(path_rf, path_ff5, returns_paths_dict)

    excess_dict = compute_excess_returns(returns_dict, rf)

    results = run_ff5_regression(excess_dict, ff5, use_newey_west=True, lags=3)

    summarize_results(results).to_csv(f"{output_dir}/regression_summary.csv")

    contrib_dict = compute_factor_contributions(excess_dict, ff5, results)

    strategies = list(excess_dict.keys())  # 所有策略文件名称
    for strat in strategies:
        # 因子贡献堆叠图
        plot_factor_contributions_stack_all(contrib_dict, strategy=strat)

        # 因子暴露热力图
        plot_factor_exposures_heatmap(results, strategy=strat)

    plot_factor_contributions_stack_compare(contrib_dict, strategies)

    plot_train_test_r2_longshort(summarize_results(results))

if __name__ == "__main__":
    main()

