import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import copy

# 读取交易数据
data_dir = r'data\TRD_Dalyr'
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
factors = pd.read_csv(r'final_factors.csv', dtype={'Stkcd': str}, parse_dates=['Trddt'])
factors['use_month'] = factors['Trddt'].dt.to_period('M') + 1

df_merged = pd.merge(factors, monthly_ohlc[['Stkcd', 'month', 'ret']], 
                     left_on=['Stkcd', 'use_month'], right_on=['Stkcd', 'month'])

# 数据准备
feature_cols = [col for col in factors.columns if col not in ['Stkcd', 'Trddt', 'use_month']]
df_clean = df_merged.dropna(subset=feature_cols + ['ret']).copy()
df_clean = df_clean.sort_values('use_month')

df_clean = df_clean[df_clean['use_month'] < '2023-01']

unique_months = df_clean['use_month'].unique()
unique_months = sorted(unique_months)

# 神经网络参数
window_size = 24
n_groups = 5

# 神经网络超参数
hidden_layers = [32, 16]  # 隐藏层神经元数量
learning_rate = 0.001
batch_size = 128
# 训练参数：第一次训练用更多epoch，后续微调用较少epoch
first_epochs = 30
fine_tune_epochs = 10
random_state = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义神经网络模型
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # 添加dropout防止过拟合
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# 设置随机种子
torch.manual_seed(random_state)
np.random.seed(random_state)

# 模型滚动训练
results_list = []
scaler = StandardScaler()
previous_model = None  # 保存上一个月的模型

for i in tqdm(range(window_size, len(unique_months)), desc="Rolling Neural Network"):
    predict_month = unique_months[i]                 
    train_months = unique_months[i-window_size:i] 
    train_data = df_clean[df_clean['use_month'].isin(train_months)]
    predict_data = df_clean[df_clean['use_month'] == predict_month].copy()
    
    if train_data.empty or predict_data.empty:
        print(f"警告：在{unique_months[i]}找不到训练或预测数据")
        continue
        
    X_train = train_data[feature_cols].values
    y_train = train_data['ret'].values.reshape(-1, 1)
    X_predict = predict_data[feature_cols].values
    
    # 数据标准化
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_predict_tensor = torch.FloatTensor(X_predict_scaled).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    input_size = len(feature_cols)
    model = MLPRegressor(input_size, hidden_layers).to(device)
    
    # 如果是第一次训练，使用之前的模型参数进行初始化
    if previous_model is not None:
        model.load_state_dict(previous_model)
        print("使用之前训练的模型参数初始化")
        current_epochs = fine_tune_epochs  # 后续训练用较少的epoch
        current_lr = learning_rate * 0.1  # 微调时使用更小的学习率
    else:
        current_epochs = first_epochs  # 第一次训练用较多的epoch
        current_lr = learning_rate
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-5)
    
    # 训练模型
    model.train()
    best_epoch_loss = float('inf')
    
    for epoch in range(current_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # 打印训练信息
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Month {predict_month}, Epoch [{epoch+1}/{current_epochs}], Loss: {avg_epoch_loss:.6f}')
        
        # 更新最佳损失（只记录，不用于早停）
        if avg_epoch_loss < best_epoch_loss:
            best_epoch_loss = avg_epoch_loss
    
    print(f'Month {predict_month} 训练完成, 最佳损失: {best_epoch_loss:.6f}, 训练轮数: {epoch+1}')
    
    # 保存当前模型状态用于下一次初始化
    previous_model = copy.deepcopy(model.state_dict())
    
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_predict_tensor).cpu().numpy().flatten()
    
    predict_data['pred_score'] = predictions
    
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
    os.makedirs('train_results', exist_ok=True)
    output_filename = 'train_results/NeuralNetwork_returns.csv'
    final_results.to_csv(output_filename)
    final_results['long_short'] = final_results[f'Group_{n_groups}'] - final_results['Group_1']

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

    plt.title('Decile Portfolios Cumulative Returns (Pre-2023 Neural Network Regression)', 
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

    os.makedirs('plots', exist_ok=True)
    output_img = 'plots/NeuralNetwork_train_cumulative_returns_plot.png'
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
    performance_df.to_csv('train_results/NeuralNetwork_performance_stats.csv')
    print("\n性能统计已保存至: train_results/NeuralNetwork_performance_stats.csv")
    
else:
    print("没有生成有效结果")