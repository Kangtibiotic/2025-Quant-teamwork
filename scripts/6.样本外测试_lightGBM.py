import os
import pandas as pd
from tqdm import tqdm
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 读取交易数据
data_dir = r'processed_data/TRD_Dalyr'
daily_ohlc_files = os.listdir(data_dir)
monthly_ohlc_list = []
for file in tqdm(daily_ohlc_files, desc='Loading data'):
    daily_ohlc = pd.read_csv(os.path.join(data_dir, file),
                             dtype={'Stkcd': str},
                             usecols=['Stkcd', 'Trddt', 'Opnprc', 'CumulateFwardFactor'],
                             parse_dates=['Trddt']
                             )
    daily_ohlc['month'] = daily_ohlc['Trddt'].dt.to_period('M')
    monthly_ohlc = daily_ohlc.groupby(['Stkcd', 'month'], as_index=False).agg('first')
    monthly_ohlc_list.append(monthly_ohlc)
monthly_ohlc = pd.concat(monthly_ohlc_list)
monthly_ohlc.sort_values(['month', 'Stkcd'], inplace=True)
monthly_ohlc['adj_open'] = monthly_ohlc['Opnprc'] * monthly_ohlc['CumulateFwardFactor']
monthly_ohlc['ret'] = monthly_ohlc.groupby('Stkcd')['adj_open'].transform(
    lambda x: x.shift(-1) / x - 1)


# 读取因子
factors = pd.read_csv(r'processed_data/final_factors.csv', dtype={'Stkcd': str}, parse_dates=['Trddt'])
factors['use_month'] = factors['Trddt'].dt.to_period('M') + 1

df_merged = pd.merge(factors, monthly_ohlc[['Stkcd', 'month', 'ret']], left_on=['Stkcd', 'use_month'], right_on=['Stkcd', 'month'])

# 数据准备
feature_cols = [col for col in factors.columns if col not in ['Stkcd', 'Trddt', 'use_month']]
df_clean = df_merged.dropna(subset=feature_cols + ['ret']).copy()
df_clean = df_clean.sort_values('use_month')

df_clean = df_clean[df_clean['use_month'] >= '2021-01']

unique_months = df_clean['use_month'].unique()
unique_months = sorted(unique_months)

# 参数设置
window_size = 24
n_groups = 5

# lightGBM参数
n_estimators = 50
learning_rate = 0.05
max_depth = 3
num_leaves = 15
random_state = 42
n_jobs = -1
verbose = -1

# 模型滚动训练
results_list = []

for i in tqdm(range(window_size, len(unique_months)), desc="Rolling lightGBM"):
    predict_month = unique_months[i]                 
    train_months = unique_months[i-window_size:i] 
    train_data = df_clean[df_clean['use_month'].isin(train_months)]
    predict_data = df_clean[df_clean['use_month'] == predict_month].copy()
    
    if train_data.empty or predict_data.empty:
        print(f"警告：在{unique_months[i]}找不到训练或预测数据")
        continue
        
    X_train = train_data[feature_cols]
    y_train = train_data['ret']
    X_predict = predict_data[feature_cols]
    
    # 训练
    '''
    参数说明：
    n_estimators: 树的数量，类似于迭代次数
    learning_rate: 学习率，较小的值通常更稳健但需要更多的树
    max_depth: 树的最大深度，限制深度防止过拟合
    num_leaves: 叶子节点数，控制模型复杂度
    random_state: 随机种子，保证结果可复现
    n_jobs: 并行线程数，-1 表示使用所有核心
    verbose: -1 关闭训练日志输出
    '''
    model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
    model.fit(X_train, y_train)
    
    # 预测
    predict_data['pred_score'] = model.predict(X_predict)
    
    try:
        predict_data['group'] = pd.qcut(predict_data['pred_score'], n_groups, labels=False)
        
        # 计算每组的平均真实收益率
        group_ret = predict_data.groupby('group')['ret'].mean()
        group_ret.name = predict_month 
        results_list.append(group_ret)
        
    except ValueError:
        print(f"警告：跳过{predict_month},因为没有足够数据")
        continue

if results_list:
    final_results = pd.concat(results_list, axis=1).T
    final_results.index.name = 'month'
    final_results.columns = [f'Group_{i+1}' for i in range(n_groups)]

    # 保存结果
    os.makedirs('results/test_results', exist_ok=True)
    final_results['long_short'] = final_results[f'Group_{n_groups}'] - final_results['Group_1']
    output_filename = 'results/test_results/lightGBM_returns.csv'
    final_results.to_csv(output_filename)


    # 绘制累计收益图
    df_cumulative = (1 + final_results).cumprod()
    df_cumulative.index = df_cumulative.index.to_timestamp()

    plt.figure(figsize=(15, 8)) 
    colors = cm.get_cmap('viridis', n_groups)

    for i in range(n_groups):
        col_name = f'Group_{i+1}'
        if col_name in df_cumulative.columns:
            color = colors(i)
            
            plt.plot(df_cumulative.index, 
                     df_cumulative[col_name], 
                     label=col_name,      
                     color=color,          
                     linewidth=2.5 if i in [0, n_groups - 1] else 1.5, 
                     alpha=0.9)

    # 绘制多空组合
    if 'long_short' in final_results.columns:
        long_short_cumulative = (1 + final_results['long_short']).cumprod()
        plt.plot(df_cumulative.index, long_short_cumulative, 
                label='Long-Short', color='red', linewidth=3, linestyle='--', alpha=0.9)

    plt.title('Decile Portfolios Cumulative Returns (After-2023 lightGBM Regression)', 
              fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Net Asset Value (NAV)', fontsize=14)

    plt.legend(title='Portfolios (Low to High Score)', 
               bbox_to_anchor=(1.02, 1), 
               loc='upper left', 
               fontsize=12,
               frameon=True, 
               shadow=True)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()

    os.makedirs('results/plots', exist_ok=True)
    output_img = 'results/plots/lightGBM_test_cumulative_returns_plot.png'
    plt.savefig(output_img, dpi=300)
    plt.show()
    
    # 计算分组收益统计
    print("\n分组收益统计:")
    print(final_results.describe())
    
    # 计算所有分组和多空组合的年化收益和夏普比率
    performance_stats = []
    
    for col in final_results.columns:
        returns = final_results[col]
        
        # 年化收益率
        annual_return = returns.mean() * 12
        
        # 年化标准差
        annual_std = returns.std() * np.sqrt(12)
        
        # 夏普比率 (假设无风险利率为0)
        sharpe_ratio = annual_return / annual_std if annual_std != 0 else 0
        
        performance_stats.append({
            'Portfolio': col,
            'Annualized_Return': annual_return,
            'Annualized_Std': annual_std,
            'Sharpe_Ratio': sharpe_ratio
        })
    
    # 创建性能统计DataFrame
    performance_df = pd.DataFrame(performance_stats)
    performance_df = performance_df.set_index('Portfolio')
    
    print("\n分组绩效统计:")
    print(performance_df)
    
    # 保存性能统计
    performance_df.to_csv('results/test_results/lightGBM_performance_stats.csv')
    print("\n性能统计已保存至: results/test_results/lightGBM_performance_stats.csv")
    
else:
    print("没有生成有效结果")
