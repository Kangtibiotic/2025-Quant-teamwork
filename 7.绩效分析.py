import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# ====== 文件路径 ======
portfolio_file = 'data/portfolio_returns.csv'
index_file = 'data/IDX_Idxtrdmth.xlsx'
fivefac_file = 'data/STK_MKT_FIVEFACMONTH.xlsx'
img_folder = 'images'
os.makedirs(img_folder, exist_ok=True)

# ====== 可调参数 ======
selected_index = '000001' # 选定用于与各投资组合对照的指数，编码方式见IDX_Idxinfo.xlsx
selected_market = 'P9706' # 五因子模型的股票市场类型编码，编码方式见STK_MKT_FIVEFACMONTH[DES][xlsx].txt，下同
selected_portfolio_number = 3 # 五因子模型的投资组合类型
window_length = 12 # 滚动回归的窗口大小
step = 1 # 滚动回归的步长

# ====== 1. 读取组合收益和指数 ======
portfolio_df = pd.read_csv(portfolio_file, parse_dates=['month'])
index_df = pd.read_excel(index_file, parse_dates=['Month'])
index_df = index_df.rename(columns={'Month':'month','Idxrtn':'return'})
index_df['month'] = pd.to_datetime(index_df['month'], format='%Y-%m')
selected_index_df = index_df[index_df['Indexcd']==selected_index][['month','return']].rename(columns={'return':'Index_'+selected_index})
portfolio_df = portfolio_df.merge(selected_index_df, on='month', how='left')

combo_cols = [col for col in portfolio_df.columns if col != 'month']

# ====== 2. 读取五因子数据并选择 ======
fivefac_df = pd.read_excel(fivefac_file, parse_dates=['TradingMonth'])
fivefac_df = fivefac_df.rename(columns={'TradingMonth':'month'})
factor_cols = ['RiskPremium1','SMB1','HML1','RMW1','CMA1']
factor_df = fivefac_df[(fivefac_df['MarkettypeID']==selected_market)&(fivefac_df['Portfolios']==selected_portfolio_number)][['month']+factor_cols]

# ====== 3. 对齐组合收益与因子 ======
reg_df = portfolio_df.merge(factor_df, on='month', how='inner')

# ====== 4. 因子回归函数 ======
def factor_regression(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    return {'alpha': model.params['const'],
            'beta': model.params.drop('const'),
            'residuals': model.resid,
            'rsquared': model.rsquared}

# ====== 5. 因子回归分析 ======
results = {}
for col in combo_cols:
    y = reg_df[col]
    X = reg_df[factor_cols]
    results[col] = factor_regression(y,X)

# 整理 alpha, beta, R²
alpha_df = pd.DataFrame({k:v['alpha'] for k,v in results.items()}, index=['Alpha']).T
beta_df = pd.DataFrame({k:v['beta'] for k,v in results.items()}).T
beta_df.columns = factor_cols
rsquared_df = pd.DataFrame({k:v['rsquared'] for k,v in results.items()}, index=['R_squared']).T

# 因子贡献
factor_contrib_df = beta_df.multiply(reg_df[factor_cols].mean(), axis=1)
factor_contrib_df['Total_Factor_Contribution'] = factor_contrib_df.sum(axis=1)

# 归因表
attribution_df = pd.concat([alpha_df, factor_contrib_df, rsquared_df], axis=1)
attribution_df.to_csv('portfolio_factor_attribution.csv')

# ====== 6. 绩效分析函数 ======
def performance_metrics(returns, freq=12):
    ann_return = returns.mean()*freq
    ann_vol = returns.std()*np.sqrt(freq)
    sharpe = ann_return/ann_vol if ann_vol!=0 else np.nan
    cum_ret = (1+returns).cumprod()
    max_dd = (cum_ret/cum_ret.cummax()-1).min()
    return pd.Series({'Annualized Return':ann_return,
                      'Annualized Volatility':ann_vol,
                      'Sharpe Ratio':sharpe,
                      'Max Drawdown':max_dd})

# ====== 7. 绩效表 ======
performance_df = pd.DataFrame(index=combo_cols)
for col in combo_cols:
    perf = performance_metrics(portfolio_df[col])
    performance_df.loc[col, perf.index] = perf.values
# 年化 alpha
performance_df['Annualized Alpha'] = alpha_df['Alpha']
performance_df.to_csv('portfolio_performance.csv')

# ====== 8. 可视化 ======
# ====== 堆叠柱状图（正负分开叠加 + alpha） ======
plt.figure(figsize=(12,6))
plt.axhline(0, color='black', linewidth=0.8)  # x轴基线

bottoms_pos = np.zeros(len(attribution_df))  # 正值累加
bottoms_neg = np.zeros(len(attribution_df))  # 负值累加

for i, fac in enumerate(factor_cols):
    vals = attribution_df[fac].values
    pos = np.where(vals>0, vals, 0)
    neg = np.where(vals<0, vals, 0)
    # 为每个因子统一颜色
    color = plt.get_cmap('tab10')(i)
    plt.bar(attribution_df.index, pos, bottom=bottoms_pos, color=color, label=fac)
    plt.bar(attribution_df.index, neg, bottom=bottoms_neg, color=color, label='_nolegend_')
    bottoms_pos += pos
    bottoms_neg += neg

# alpha 统一灰色，正负方向与值相符
alpha_vals = attribution_df['Alpha'].values
alpha_pos = np.where(alpha_vals>0, alpha_vals, 0)
alpha_neg = np.where(alpha_vals<0, alpha_vals, 0)
plt.bar(attribution_df.index, alpha_pos, bottom=bottoms_pos, color='gray', label='Alpha')
plt.bar(attribution_df.index, alpha_neg, bottom=bottoms_neg, color='gray', label='_nolegend_')

plt.title('Portfolio Factor Attribution (Positive/Negative by Direction)')
plt.ylabel('Monthly Return Contribution')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_folder,'factor_attribution.png'))
plt.close()

# 热力图（beta暴露）
plt.figure(figsize=(10,6))
sns.heatmap(beta_df, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Portfolio Factor Exposure Heatmap (Beta)')
plt.ylabel('Portfolio')
plt.tight_layout()
plt.savefig(os.path.join(img_folder,'factor_exposure_heatmap.png'))
plt.close()

# ====== 9. 滚动回归 alpha 和 beta ======
rolling_alphas = pd.DataFrame(index=reg_df['month'][window_length-1:], columns=combo_cols)
rolling_betas = {fac: pd.DataFrame(index=reg_df['month'][window_length-1:], columns=combo_cols) for fac in factor_cols}

# 计算滚动 alpha 和 beta
for col in combo_cols:
    for start in range(0, len(reg_df)-window_length+1, step):
        end = start + window_length
        y = reg_df[col].iloc[start:end]
        X = reg_df[factor_cols].iloc[start:end]
        res = factor_regression(y, X)
        rolling_alphas.at[reg_df['month'].iloc[end-1], col] = res['alpha']
        for fac in factor_cols:
            rolling_betas[fac].at[reg_df['month'].iloc[end-1], col] = res['beta'][fac]

# ====== 绘制滚动 alpha（每组合一张图） ======
plt.figure(figsize=(12,6))
for col in combo_cols:
    plt.plot(rolling_alphas.index, rolling_alphas[col].astype(float), label=col)
plt.title(f'Rolling Alpha (Window={window_length} months)')
plt.xlabel('Month')
plt.ylabel('Alpha')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(img_folder,'rolling_alpha.png'))
plt.close()

# ====== 绘制滚动 beta（组合-因子交叉：每组合一张图） ======
for col in combo_cols:
    plt.figure(figsize=(12,6))
    for fac in factor_cols:
        plt.plot(rolling_betas[fac].index, rolling_betas[fac][col].astype(float), label=fac)
    plt.title(f'Rolling Beta over Time for {col} (All Factors)')
    plt.xlabel('Month')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder,f'rolling_beta_combo_{col}.png'))
    plt.close()

# ====== 绘制滚动 beta（因子-组合交叉：每因子一张图） ======
for fac in factor_cols:
    plt.figure(figsize=(12,6))
    for col in combo_cols:
        plt.plot(rolling_betas[fac].index, rolling_betas[fac][col].astype(float), label=col)
    plt.title(f'Rolling Beta over Time for Factor {fac} (All Portfolios)')
    plt.xlabel('Month')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder,f'rolling_beta_factor_{fac}.png'))
    plt.close()

# ====== 10. 基本绩效图 ======
plt.figure(figsize=(10,6))
portfolio_df[combo_cols].boxplot()
plt.title('Portfolio Monthly Returns Boxplot')
plt.ylabel('Monthly Return')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(img_folder,'portfolio_boxplot.png'))
plt.close()

plt.figure(figsize=(10,6))
for col in combo_cols:
    plt.plot(portfolio_df['month'], (1+portfolio_df[col]).cumprod(), label=col)
plt.title('Cumulative Returns')
plt.xlabel('Month')
plt.ylabel('Cumulative Return')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_folder,'portfolio_cum_returns.png'))
plt.close()

print("分析完成，指标保存至 'portfolio_performance.csv'，归因保存至 'portfolio_factor_attribution.csv'，图像保存在 images/ 文件夹")
