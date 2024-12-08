# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import platform
import warnings
warnings.filterwarnings('ignore')

# 方案1：使用系统字体
import matplotlib as mpl
# 设置中文字体为系统默认的中文字体
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

# 方案2：如果方案1不生效，可以尝试手动加载字体文件
try:
    # Windows系统字体路径
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 也可以使用 msyh.ttf (微软雅黑)
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'SimHei'
except:
    print("无法加载指定字体，将使用系统默认字体")

# 设置seaborn样式
sns.set_style("whitegrid")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class TemperatureDataset(Dataset):
    def __init__(self, data, seq_length, is_test=False):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.is_test = is_test

    def __len__(self):
        if self.is_test:
            # 返回实际可用的预测点数量
            return len(self.data) - self.seq_length
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # 获取输入序列
        X = self.data[idx:idx+self.seq_length]
        
        # 获取目标值
        if idx + self.seq_length < len(self.data):
            y = self.data[idx+self.seq_length:idx+self.seq_length+1, 0]
        else:
            # 如果到达数据末尾，使用最后一个值
            y = self.data[-1:, 0]
            
        return X, y

class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 增加LSTM层的复杂度
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 增加全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2是因为使用了双向LSTM
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2是因为双向
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_and_process_data(file_path, seq_length=12, target_date='2024-03-30'):
    """加载和处理数据"""
    print(f"正在读取文件: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"原始数据形状: {df.shape}")
    
    # 时间处理
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index('时间', inplace=True)
    
    # 添加时间特征
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # 处理缺失值
    df['环境温度'] = df['环境温度'].interpolate(method='time')
    df['环境温度'] = df['环境温度'].fillna(method='ffill').fillna(method='bfill')
    
    print(f"处理后的数据形状: {df.shape}")
    
    # 获取目标日期的数据作为测试集
    target_date = pd.to_datetime(target_date).date()
    test_data = df[df.index.date == target_date]
    
    # 确保测试数据包含完整的24小时
    if len(test_data) < 24:
        print(f"警告：测试数据不足24小时，当前只有{len(test_data)}小时")
    else:
        # 只取当天的24小时数据
        test_data = test_data.iloc[:24]
    
    # 获取训练数据（不包括测试日期）
    train_data = df[df.index.date < target_date]
    
    print(f"目���预测日期: {target_date}")
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    if len(test_data) == 0:
        raise ValueError(f"未找到{target_date}的数据！")
    
    # 数据标准化
    scaler = MinMaxScaler()
    train_normalized = scaler.fit_transform(train_data[['环境温度', 'hour_sin', 'hour_cos']].values)
    test_normalized = scaler.transform(test_data[['环境温度', 'hour_sin', 'hour_cos']].values)
    
    # 创建数据集
    train_dataset = TemperatureDataset(train_normalized, seq_length, is_test=False)
    test_dataset = TemperatureDataset(test_normalized, seq_length, is_test=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader, scaler, test_data.index, test_data['环境温度']

def train_model(model, train_loader, num_epochs=500, learning_rate=0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期延长的倍数
        eta_min=1e-6  # 最小学习率
    )
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 30
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 保存最佳模型和相关参数
    model_info = {
        'state_dict': model.state_dict(),
        'input_size': model.lstm.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'best_loss': best_loss,
        'train_losses': train_losses,
        'epoch': epoch
    }
    torch.save(model_info, 'temperature_model_full.pth')
    print(f'模型已保存到 temperature_model_full.pth')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_model(model, test_loader, scaler, full_day_prediction=False):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    predictions = []
    actuals = []
    
    print(f"使用设备: {device}")
    print(f"测试数据批次数量: {len(test_loader)}")  # 添加这行来调试
    
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            try:
                X, y = X.to(device), y.to(device)
                output = model(X)
                
                # 将预测值添加到完整的特征向量中进行反标准化
                pred_full = np.zeros((output.shape[0], 3))
                pred_full[:, 0] = output.detach().cpu().numpy().flatten()
                pred_temp = scaler.inverse_transform(pred_full)[:, 0]
                
                predictions.append(pred_temp)
                actuals.append(y.cpu().numpy())
                
            except Exception as e:
                print(f"处理批次 {batch_idx} 时发生错误: {str(e)}")
                continue
    
    if not predictions:
        print("警告：没有生成预测值")
        return None, None
    
    try:
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        # 反标准化实际值
        actuals_full = np.zeros((actuals.shape[0], 3))
        actuals_full[:, 0] = actuals.flatten()
        actuals = scaler.inverse_transform(actuals_full)[:, 0].reshape(-1, 1)
        
        print(f'预测值形状: {predictions.shape}')
        print(f'实际值形状: {actuals.shape}')
        
        return predictions, actuals
        
    except Exception as e:
        print(f"处理终结果时发生错误: {str(e)}")
        return None, None

def plot_comparison(predictions, actuals, timestamps, actual_temps):
    """绘制详细对比图"""
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    plt.subplot(2, 1, 1)
    
    # 使用可用的数据长度
    available_hours = min(len(timestamps), len(predictions), len(actuals))
    predictions = predictions[:available_hours]
    actuals = actuals[:available_hours]
    timestamps = timestamps[:available_hours]
    
    # 绘制预测值
    plt.plot(timestamps, predictions, label='预测温度', 
            color='red', linestyle='--', linewidth=1.5, marker='x')
    
    # 只绘制有效的实际值
    valid_mask = ~np.isnan(actuals)
    if np.any(valid_mask):
        valid_timestamps = timestamps[valid_mask]
        valid_actuals = actuals[valid_mask]
        plt.plot(valid_timestamps, valid_actuals, 
                label='实际温度', color='blue', linewidth=1.5, marker='o')
    
    plt.title(f'温度对比 ({timestamps[0].date()})', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 计算误差统计（只对实际值的部分）
    if np.any(valid_mask):
        valid_predictions = predictions[valid_mask]
        errors = valid_actuals - valid_predictions
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mean_squared_error(valid_actuals, valid_predictions))
        max_error = np.max(np.abs(errors))
        
        # 在图上添加统计信息
        stats_text = f'平均绝对误差: {mae:.2f}°C\n均方根误差: {rmse:.2f}°C\n最大误差: {max_error:.2f}°C'
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细的小时预测对比
    print("\n每小时详细对比:")
    print("时间".ljust(20), "实际温度".ljust(10), "预测温度".ljust(10), "误差".ljust(10))
    print("-" * 50)
    
    # 使用原始温度数据进行对比
    for i, timestamp in enumerate(timestamps):
        actual_temp = actual_temps.loc[timestamp] if timestamp in actual_temps.index else "N/A"
        pred = predictions[i] if i < len(predictions) else "N/A"
        error = actual_temp - pred if isinstance(actual_temp, (int, float)) and isinstance(pred, (int, float)) else "N/A"
        
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M')}".ljust(20),
              f"{actual_temp if isinstance(actual_temp, str) else f'{actual_temp:.2f}°C'}".ljust(10),
              f"{pred if isinstance(pred, str) else f'{pred:.2f}°C'}".ljust(10),
              f"{error if isinstance(error, str) else f'{error:.2f}°C'}".ljust(10))

def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Linux':
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
    elif system == 'Darwin':  # macOS
        font_list = ['Arial Unicode MS', 'Heiti TC']
    else:
        font_list = ['DejaVu Sans']

    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except:
            continue
    print("警告：未能找到合适的中文字体")

def main():
    try:
        # 设置中文字体
        setup_chinese_font()
        
        # 加载数据
        print("正在加载数据...")
        train_loader, test_loader, scaler, timestamps, actual_temps = load_and_process_data(
            'proj_week5/temperature_hourly.csv', 
            seq_length=12,
            target_date='2024-03-29'
        )
        
        # 创建和训练模型
        print("正在创建和训练模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTM().to(device)
        
        # 检查是否存在已保存的模型
        try:
            model_info = torch.load('temperature_model_full.pth', map_location=device)
            model.load_state_dict(model_info['state_dict'])
            print(f"已加载保存的模型，最佳损失值: {model_info['best_loss']:.6f}")
            print(f"模型训练轮次: {model_info['epoch'] + 1}")
        except FileNotFoundError:
            print("未找到已保存的模型，开始训练新模型...")
            model = train_model(model, train_loader)
        
        # 评估模型
        print("正在评估模型...")
        predictions, actuals = evaluate_model(model, test_loader, scaler)
        
        if predictions is None or actuals is None:
            print("��型评估失败，无法生成预测结果")
            return
            
        # 绘制详细对比图
        print("正在生成可视化结果...")
        plot_comparison(predictions.flatten(), actuals.flatten(), timestamps, actual_temps)
        
        # 打印当天所有小时的实际温度
        target_date = pd.to_datetime('2024-03-29').date()  # 使用实际的目标日期
        print(f"\n{target_date}全天温度数据:")
        print("时间".ljust(20), "实际温度".ljust(10))
        print("-" * 30)
        for time, temp in zip(actual_temps.index, actual_temps):
            print(f"{time.strftime('%Y-%m-%d %H:%M')}".ljust(20), 
                  f"{temp:.2f}°C".ljust(10))
        
    except Exception as e:
        print(f"程序执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 