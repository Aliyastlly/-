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
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib as mpl
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

try:
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'SimHei'
except:
    print("无法加载指定字体，将使用系统默认字体")

# 设置样式和随机种子
sns.set_style("whitegrid")
torch.manual_seed(42)
np.random.seed(42)

class TemperatureDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of bounds")
        return (self.data[idx:idx+self.seq_length], 
                self.data[idx+self.seq_length:idx+self.seq_length+1])

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_and_process_data(file_path, seq_length=24):
    # 读取数据
    df = pd.read_csv(file_path)
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index('时间', inplace=True)
    df['环境温度'] = df['环境温度'].fillna(method='ffill').fillna(method='bfill')
    
    # 分割训练集和测试集
    last_date = df.index.date[-1]
    test_data = df[df.index.date == last_date]
    train_data = df[df.index.date < last_date]
    
    if len(test_data) < seq_length + 1:
        print("警告：测试数据太短，将包含更多训练数据到测试集中")
        test_data = df.iloc[-(seq_length + 24):]
        train_data = df.iloc[:-(seq_length + 24)]
    
    print(f"\n目标日期: {last_date}")
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    # 数据标准化
    scaler = MinMaxScaler()
    train_normalized = scaler.fit_transform(train_data[['环境温度']].values)
    test_normalized = scaler.transform(test_data[['环境温度']].values)
    
    # 创建数据加载器
    train_dataset = TemperatureDataset(train_normalized, seq_length)
    test_dataset = TemperatureDataset(test_normalized, seq_length)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader, scaler, test_data.index[-24:]

def train_model(model, train_loader, num_epochs=200, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=10, verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 20
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
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'轮次 [{epoch+1}/{num_epochs}], 损失: {avg_loss:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f'早停于轮次 {epoch+1}')
            break
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.title('训练过程中的损失变化', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return model

def evaluate_model(model, test_loader, scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            predictions.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())
    
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    print(f'测试MSE: {mse:.2f}°C')
    
    return predictions[-24:], actuals[-24:]

def plot_comparison(predictions, actuals, timestamps):
    plt.figure(figsize=(15, 10))
    
    # 温度对比图
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, actuals, label='实际温度', color='blue', linewidth=1.5, marker='o')
    plt.plot(timestamps, predictions, label='预测温度', color='red', 
            linestyle='--', linewidth=1.5, marker='x')
    plt.title(f'温度对比 ({timestamps[0].date()})', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # 计算误差统计
    errors = actuals - predictions
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    max_error = np.max(np.abs(errors))
    
    stats_text = f'平均绝对误差: {mae:.2f}°C\nRMSE: {rmse:.2f}°C\n最大误差: {max_error:.2f}°C'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    # 误差分析图
    plt.subplot(2, 1, 2)
    plt.bar(timestamps, errors, color='green', alpha=0.6, label='误差')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('每小时预测误差', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('误差 (°C)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细对比
    print("\n每小时详细对比:")
    print("时间".ljust(20), "实际温度".ljust(10), "预测温度".ljust(10), "误差".ljust(10))
    print("-" * 50)
    for i, timestamp in enumerate(timestamps):
        print(f"{timestamp.strftime('%Y-%m-%d %H:%M')}".ljust(20),
              f"{actuals[i]:.2f}°C".ljust(10),
              f"{predictions[i]:.2f}°C".ljust(10),
              f"{errors[i]:.2f}°C".ljust(10))
    
    # 打印统计摘要
    print("\n预测统计:")
    print(f"平均绝对误差: {mae:.2f}°C")
    print(f"均方根误差: {rmse:.2f}°C")
    print(f"最大绝对误差: {max_error:.2f}°C")

def main():
    # 加载数据
    train_loader, test_loader, scaler, timestamps = load_and_process_data('proj_week5/temperature_hourly.csv')
    
    # 创建和训练模型
    model = LSTM()
    model = train_model(model, train_loader)
    
    # 评估模型
    predictions, actuals = evaluate_model(model, test_loader, scaler)
    
    # 绘制详细对比图
    plot_comparison(predictions.flatten(), actuals.flatten(), timestamps)

if __name__ == "__main__":
    main() 