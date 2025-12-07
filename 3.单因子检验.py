import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats


def rank_ic(df_merged, factor_col, return_col):
    """
    计算因子值与未来收益率的Rank IC（信息系数）

    Parameters:
    -----------
    df_factor : DataFrame
        因子数据，包含['Trddt', 'Stkcd', 'ret_t+1']和因子值列
    factor_col : str, optional
        因子列名，如果不指定则自动识别
    return_col : str, optional
        收益率列名，如果不指定则自动识别

    Returns:
    --------
    ic_series : Series
        按日期排序的Rank IC序列
    ic_stats : dict
        IC统计信息
    """

    # 按日期分组计算Rank IC
    ic_results = []

    for date, group in df_merged.groupby("Trddt"):
        if len(group) < 5:  # 至少需要5个观测值
            continue
        
        # 计算Spearman秩相关系数
        ic = group[factor_col].corr(group[return_col], method='spearman')

        # if not np.isnan(ic):
        ic_results.append({"Trddt": date, "IC": ic})

    # 转换为DataFrame
    ic_df = pd.DataFrame(ic_results)

    if len(ic_df) == 0:
        print("警告: 没有计算出有效的IC值")
        return pd.Series(), {}

    # 设置日期索引
    ic_df = ic_df.sort_values("Trddt").set_index("Trddt")
    ic_series = ic_df["IC"]

    # 计算IC统计量
    ic_stats = {
        "factor_name": factor_col,
        "IC_mean": ic_series.mean(),
        "IC_std": ic_series.std(),
        "IC_IR": ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0,
    }

    return ic_series, ic_stats


def plot_ic_analysis(ic_series, ic_stats, factor_name, figsize=(15, 12)):
    """
    可视化IC序列和统计信息

    Parameters:
    -----------
    ic_series : Series
        IC时间序列
    ic_stats : dict
        IC统计信息
    figsize : tuple, optional
        图形大小
    """

    if len(ic_series) == 0:
        print("警告: IC序列为空，无法绘图")
        return

    # 设置中文字体和样式
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid")

    # 创建图形和子图布局
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    # 子图1: IC时间序列
    ax1 = plt.subplot(gs[0, :])
    dates = ic_series.index.to_timestamp()
    values = ic_series.values

    # 绘制IC序列
    line = ax1.plot(dates, values, "b-", alpha=0.7, linewidth=1.5, label="Daily IC")

    # 绘制移动平均线
    window = min(20, len(ic_series) // 4)  # 自适应窗口
    if window > 1:
        ic_ma = ic_series.rolling(window=window, center=True).mean()
        ax1.plot(dates, ic_ma, "r-", linewidth=2, label=f"{window}day MA")

    # 绘制零线和均值线
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.axhline(
        y=ic_stats["IC_mean"],
        color="green",
        linestyle="--",
        alpha=0.8,
        label=f'Mean: {ic_stats["IC_mean"]:.4f}',
    )

    # 填充正负区域
    ax1.fill_between(
        dates, values, 0, where=values >= 0, facecolor="red", alpha=0.2, label="IC > 0"
    )
    ax1.fill_between(
        dates, values, 0, where=values < 0, facecolor="green", alpha=0.2, label="IC < 0"
    )

    ax1.set_title(
        f'Rank IC Time Series - {ic_stats.get("factor_name", "factor")}',
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("IC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: IC分布直方图
    ax2 = plt.subplot(gs[1, 0])
    n, bins, patches = ax2.hist(
        values, bins=30, alpha=0.7, color="steelblue", edgecolor="black", density=True
    )

    # 添加正态分布曲线
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, ic_stats["IC_mean"], ic_stats["IC_std"])
    ax2.plot(x, p, "k-", linewidth=2, label="Normal Distribution")

    # 添加均值和标准差线
    ax2.axvline(
        ic_stats["IC_mean"],
        color="red",
        linestyle="--",
        label=f'Mean: {ic_stats["IC_mean"]:.4f}',
    )
    ax2.axvline(
        ic_stats["IC_mean"] + ic_stats["IC_std"],
        color="orange",
        linestyle=":",
        alpha=0.7,
        label="±1 std",
    )
    ax2.axvline(
        ic_stats["IC_mean"] - ic_stats["IC_std"],
        color="orange",
        linestyle=":",
        alpha=0.7,
    )

    ax2.set_title("IC Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("IC")
    ax2.set_ylabel("Density")
    ax2.legend()

    # 子图3: IC累积曲线
    ax3 = plt.subplot(gs[1, 1])
    ic_cumulative = ic_series.cumsum()
    ax3.plot(dates, ic_cumulative, "purple", linewidth=2)
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.set_title("IC Cumulative Curve", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Cumulative IC")
    ax3.grid(True, alpha=0.3)

    # 子图4: 统计信息表格
    ax4 = plt.subplot(gs[2, 1])
    ax4.axis("off")

    # 准备统计信息
    stats_data = [
        ["Factor Name", ic_stats.get("factor_name", "Unknown")],
        ["Observation Periods", len(ic_series)],
        ["IC Mean", f"{ic_stats['IC_mean']:.4f}"],
        ["IC Std", f"{ic_stats['IC_std']:.4f}"],
        ["ICIR", f"{ic_stats['IC_IR']:.4f}"],
        ["IC > 0 Ratio", f"{(ic_series > 0).mean():.2%}"],
        ["IC Abs Mean", f"{ic_series.abs().mean():.4f}"],
        ["IC Abs Median", f"{ic_series.abs().median():.4f}"],
        [
            "Max Positive IC",
            f"{ic_series[ic_series > 0].max():.4f}" if (ic_series > 0).any() else "N/A",
        ],
        [
            "Min Negative IC",
            f"{ic_series[ic_series < 0].min():.4f}" if (ic_series < 0).any() else "N/A",
        ],
    ]

    # 创建表格
    table = ax4.table(
        cellText=stats_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        bbox=[0.1, 0.1, 0.8, 0.8],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # 设置表格样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 标题行
            cell.set_facecolor("#4F81BD")
            cell.set_text_props(color="white", weight="bold")
        elif i % 2 == 1:  # 奇数行
            cell.set_facecolor("#F2F2F2")

    ax4.set_title("IC Statistics", fontsize=12, fontweight="bold", y=0.95)

    plt.tight_layout()
    plt.savefig(fr'.\单因子测试results\{factor_name}_ic.png')
    plt.show()


def grouping_test(df_merged, factor_col, return_col, n_groups=5):
    """
    因子分组测试 - 检验因子在不同分组中的收益率表现

    Parameters:
    -----------
    df_merged : DataFrame
        因子数据，包含['Trddt', 'Stkcd']、ret_t+1和因子值列
    factor_col : str
        因子列名
    return_col : str
        收益率列名
    n_groups : int, optional
        分组数量，默认分为5组

    Returns:
    --------
    group_results : dict
        包含分组测试结果的字典
    """

    # 按日期进行分组
    df_merged = df_merged.sort_values(["Trddt", factor_col])
    df_merged["group"] = df_merged.groupby("Trddt")[factor_col].transform(
        lambda x: pd.qcut(x, n_groups, labels=False, duplicates="drop") + 1
    )

    # 计算各组的收益率统计
    group_stats = []
    long_short_returns = []

    for date, date_group in df_merged.groupby("Trddt"):
        if len(date_group) < n_groups:  # 确保有足够股票分组
            continue

        date_returns = {}

        # 计算各组的等权平均收益率
        for group_num in range(1, n_groups + 1):
            group_data = date_group[date_group["group"] == group_num]
            if len(group_data) > 0:
                date_returns[f"group_{group_num}"] = group_data[return_col].mean()
            else:
                date_returns[f"group_{group_num}"] = np.nan

        # 计算多空组合收益（最高组 - 最低组）
        if not np.isnan(date_returns[f"group_{n_groups}"]) and not np.isnan(
            date_returns["group_1"]
        ):
            long_short_return = (
                date_returns[f"group_{n_groups}"] - date_returns["group_1"]
            )
            date_returns["long_short"] = long_short_return
            long_short_returns.append(
                {"Trddt": date, "long_short_return": long_short_return}
            )

        date_returns["Trddt"] = date
        group_stats.append(date_returns)

    # 转换为DataFrame
    group_stats_df = pd.DataFrame(group_stats).set_index("Trddt")
    long_short_df = pd.DataFrame(long_short_returns).set_index("Trddt")

    # 计算分组累计收益
    cumulative_returns = {}
    for i in range(1, n_groups + 1):
        col_name = f"group_{i}"
        cumulative_returns[col_name] = (1 + group_stats_df[col_name]).cumprod() - 1

    cumulative_returns["long_short"] = (
        1 + long_short_df["long_short_return"]
    ).cumprod() - 1

    # 计算分组表现统计
    performance_stats = {}
    for i in range(1, n_groups + 1):
        col_name = f"group_{i}"
        returns = group_stats_df[col_name].dropna()
        if len(returns) > 0:
            performance_stats[col_name] = {
                "mean_return": returns.mean(),
                "std_return": returns.std(),
                "sharpe_ratio": (
                    returns.mean() / returns.std() * np.sqrt(252)
                    if returns.std() > 0
                    else 0
                ),
                "win_rate": (returns > 0).mean(),
                "obs_count": len(returns),
            }

    # 多空组合统计
    ls_returns = long_short_df["long_short_return"].dropna()
    if len(ls_returns) > 0:
        performance_stats["long_short"] = {
            "mean_return": ls_returns.mean(),
            "std_return": ls_returns.std(),
            "sharpe_ratio": (
                ls_returns.mean() / ls_returns.std() * np.sqrt(252)
                if ls_returns.std() > 0
                else 0
            ),
            "win_rate": (ls_returns > 0).mean(),
            "ic_mean": ls_returns.mean(),  # 多空收益可以看作IC的一种表现形式
            "ir": ls_returns.mean() / ls_returns.std() if ls_returns.std() > 0 else 0,
            "obs_count": len(ls_returns),
        }

    # 计算分组换手率（粗略估计）
    turnover_stats = calculate_group_turnover(df_merged, n_groups)

    # 返回完整结果
    results = {
        "group_returns": group_stats_df,
        "long_short_returns": long_short_df,
        "cumulative_returns": pd.DataFrame(cumulative_returns),
        "performance_stats": performance_stats,
        "turnover_stats": turnover_stats,
        "factor_col": factor_col,
        "return_col": return_col,
        "n_groups": n_groups,
    }

    return results


def calculate_group_turnover(df_merged, n_groups):
    """
    计算分组换手率
    """
    turnover_stats = {}

    # 按日期排序，为计算换手率做准备
    df_merged = df_merged.sort_values(["Stkcd", "Trddt"])

    for group_num in range(1, n_groups + 1):
        group_turnovers = []

        # 获取每个日期该组的股票
        dates = sorted(df_merged["Trddt"].unique())

        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i - 1]

            current_stocks = set(
                df_merged[
                    (df_merged["Trddt"] == current_date)
                    & (df_merged["group"] == group_num)
                ]["Stkcd"]
            )
            prev_stocks = set(
                df_merged[
                    (df_merged["Trddt"] == prev_date)
                    & (df_merged["group"] == group_num)
                ]["Stkcd"]
            )

            if len(prev_stocks) > 0:
                # 换手率 = 新进入的股票数量 / 总股票数量
                new_stocks = current_stocks - prev_stocks
                turnover = len(new_stocks) / len(current_stocks)
                group_turnovers.append(turnover)

        if len(group_turnovers) > 0:
            turnover_stats[f"group_{group_num}"] = {
                "mean_turnover": np.mean(group_turnovers),
                "std_turnover": np.std(group_turnovers),
            }

    return turnover_stats


def plot_grouping_results(results, figsize=(14, 10)):
    """
    绘制分组测试结果

    Parameters:
    -----------
    results : dict
        分组测试结果字典
    figsize : tuple, optional
        图表大小
    """
    # 获取分组数量
    n_groups = results["n_groups"]

    # 生成彩虹色系
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n_groups))
    # 多空组合使用黑色
    ls_color = "black"

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 分组累计收益
    cumulative_returns = results["cumulative_returns"]
    cumulative_returns.index = cumulative_returns.index.to_timestamp()

    # 绘制各分组累计收益
    for i in range(1, n_groups + 1):
        col_name = f"group_{i}"
        if col_name in cumulative_returns.columns:
            axes[0, 0].plot(
                cumulative_returns.index,
                cumulative_returns[col_name],
                color=rainbow_colors[i - 1],
                linewidth=2,
                label=f"Group {i}",
            )

    # 绘制多空组合累计收益
    if "long_short" in cumulative_returns.columns:
        axes[0, 0].plot(
            cumulative_returns.index,
            cumulative_returns["long_short"],
            color=ls_color,
            linewidth=3,
            linestyle="--",
            label="Long-Short",
        )

    axes[0, 0].set_title("Cumulative Returns", fontsize=14, fontweight="bold")
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # 分组平均收益率柱状图
    performance_stats = results["performance_stats"]
    group_means = []
    group_labels = []

    for i in range(1, n_groups + 1):
        group_key = f"group_{i}"
        if group_key in performance_stats:
            group_means.append(performance_stats[group_key]["mean_return"])
            group_labels.append(f"G{i}")

    bars = axes[0, 1].bar(group_labels, group_means, color=rainbow_colors)
    axes[0, 1].set_title("Average Returns", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    for bar, value in zip(bars, group_means):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 多空组合收益分布
    ls_returns = results["long_short_returns"]["long_short_return"].dropna()
    if len(ls_returns) > 0:
        n, bins, patches = axes[1, 0].hist(
            ls_returns, bins=50, alpha=0.8, color="skyblue", edgecolor="black"
        )

        # 为直方图添加渐变色效果
        cmap = plt.cm.Blues
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)

        for c, p in zip(col, patches):
            plt.setp(p, "facecolor", cmap(c))

        axes[1, 0].axvline(
            ls_returns.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {ls_returns.mean():.4f}",
        )
        axes[1, 0].axvline(0, color="black", linestyle="-", alpha=0.5)
        axes[1, 0].set_title("Long-Short Returns", fontsize=14, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 分组换手率柱状图
    turnover_stats = results["turnover_stats"]
    if turnover_stats:
        turnovers = []
        turn_labels = []
        for i in range(1, n_groups + 1):
            group_key = f"group_{i}"
            if group_key in turnover_stats:
                turnovers.append(turnover_stats[group_key]["mean_turnover"])
                turn_labels.append(f"G{i}")

        bars_turn = axes[1, 1].bar(turn_labels, turnovers, color=rainbow_colors)
        axes[1, 1].set_title("Average Turnover Rate", fontsize=14, fontweight="bold")

        axes[1, 1].grid(True, alpha=0.3)

        # 在换手率柱状图上添加数值标签
        for bar, value in zip(bars_turn, turnovers):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()

    factor_name = results.get("factor_col", "factor")
    plt.suptitle(
        f"Grouping Test Results - {factor_name}", fontsize=16, fontweight="bold", y=1.02
    )
    
    plt.savefig(fr'.\单因子测试results\{factor_name}_grouping.png')
    plt.show()


if __name__ == "__main__":
    daily_ohlc_files = os.listdir("data\TRD_Dalyr")
    daily_ohlc_list = []
    for file in tqdm(daily_ohlc_files, desc="Loading data"):
        daily_ohlc = pd.read_csv(
            os.path.join("data\TRD_Dalyr", file), dtype={"Stkcd": str}
        )
        daily_ohlc_list.append(daily_ohlc)
    daily_ohlc = pd.concat(daily_ohlc_list)
    daily_ohlc["Trddt"] = pd.to_datetime(daily_ohlc["Trddt"])
    daily_ohlc.sort_values(["Trddt", "Stkcd"], inplace=True)
    monthly_ohlc = (
        daily_ohlc.groupby(["Stkcd", daily_ohlc["Trddt"].dt.to_period("M")])
        .agg({"Opnprc": "first", "CumulateFwardFactor": "first"})
        .reset_index()
    )
    monthly_ohlc["adj_open"] = (
        monthly_ohlc["Opnprc"] * monthly_ohlc["CumulateFwardFactor"]
    )
    monthly_ohlc["ret_t+1"] = monthly_ohlc.groupby("Stkcd")["adj_open"].transform(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )

    # 读取因子
    factors = pd.read_csv("processed_factors.csv", dtype={"Stkcd": str})
    factors["Trddt"] = pd.to_datetime(factors["Trddt"]).dt.to_period("M")
    factor_list = [x for x in factors.columns if x not in ["Stkcd", "Trddt"]]

    # 选取2023年之前的因子为测试集
    factors = factors[factors["Trddt"] < pd.to_datetime("20230101").to_period("M")]
    factors = factors.merge(monthly_ohlc[["Stkcd", "Trddt", "ret_t+1"]])

    summary_list = []
    n_groups = 10
    for factor_col in factor_list:
        print(f"\n====== 正在分析因子: {factor_col} ======")

        # --- 1. IC 测试 ---
        ic_series, ic_stats = rank_ic(factors[['Stkcd', 'Trddt', 'ret_t+1', factor_col]], factor_col, 'ret_t+1')
        plot_ic_analysis(ic_series, ic_stats, factor_col)
        if ic_stats["IC_mean"] < 0:
            print(f"IC均值为负，反转因子 {factor_col}")
            factors[factor_col] = -factors[factor_col]
            ic_series, ic_stats = rank_ic(
                factors, factor_col, "ret_t+1"
            )  # 重新计算一次

        # --- 2. 分组测试 ---
        group_results = grouping_test(factors[['Stkcd', 'Trddt', 'ret_t+1', factor_col]], factor_col, "ret_t+1", n_groups=n_groups)
        plot_grouping_results(group_results)

        # --- 3. 汇总关键统计结果 ---
        long_short_stats = group_results["performance_stats"].get("long_short", {})
        summary_list.append(
            {
                "factor_name": factor_col,
                # IC统计
                "IC_mean": ic_stats.get("IC_mean", np.nan),
                "IC_std": ic_stats.get("IC_std", np.nan),
                "IC_IR": ic_stats.get("IC_IR", np.nan),
                "IC_abs_mean": ic_series.abs().mean() if len(ic_series) > 0 else np.nan,
                # 多空收益
                "long_short_mean_ret": long_short_stats.get("mean_return", np.nan),
                "long_short_std_ret": long_short_stats.get("std_return", np.nan),
                "long_short_sharpe": long_short_stats.get("sharpe_ratio", np.nan),
                "long_short_winrate": long_short_stats.get("win_rate", np.nan),
                # 分组间平均收益差异
                "group1_mean_ret": group_results["performance_stats"]
                .get("group_1", {})
                .get("mean_return", np.nan),
                f"group{n_groups}_mean_ret": group_results["performance_stats"]
                .get(f"group_{n_groups}", {})
                .get("mean_return", np.nan),
                "group_spread": group_results["performance_stats"]
                .get(f"group_{n_groups}", {})
                .get("mean_return", 0)
                - group_results["performance_stats"]
                .get("group_1", {})
                .get("mean_return", 0),
                # 分组换手率
                "avg_turnover": (
                    np.mean(
                        [
                            v["mean_turnover"]
                            for v in group_results["turnover_stats"].values()
                        ]
                    )
                    if group_results["turnover_stats"]
                    else np.nan
                ),
            }
        )

    # ==================== 保存汇总结果 ====================
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values(by="IC_mean", ascending=False)

    # 保留两位小数（只对数值列）
    for col in summary_df.select_dtypes(include=[np.number]).columns:
        summary_df[col] = summary_df[col].round(4)

    output_path = r"单因子测试results\单因子测试results.csv"
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 所有因子测试完毕，结果已保存到：{output_path}")
