import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_condensate_flow(df, window_size=10, pv_threshold=0.04):
    """
    分析主凝结水流量的稳态特性
    
    参数:
    df: DataFrame, 包含 'k_ts' 和 '主凝结水流量' 列的数据框
    window_size: int, 滑动窗口大小（默认10）
    pv_threshold: float, 峰谷差比例阈值（默认0.04，即4%）
    
    返回:
    tuple: (stable_mask, peak_valley_ratios, timestamps, flow_data)
    """
    # 创建数据副本
    data = df.copy()
    flow_data = data['主凝结水流量'].values
    
    # 计算峰谷差和标准差
    peak_valley_ratios = []
    timestamps = []
    
    # 遍历计算指标
    for i in range(len(flow_data) - window_size + 1):
        window = flow_data[i:i + window_size]
        mean_value = np.mean(window)
        
        if mean_value != 0:
            # 计算峰谷差比例
            peak_valley = np.max(window) - np.min(window)
            pv_ratio = peak_valley / mean_value
            peak_valley_ratios.append(pv_ratio)
            timestamps.append(data['k_ts'].iloc[i + window_size // 2])
    
    # 转换为numpy数组
    peak_valley_ratios = np.array(peak_valley_ratios)
    timestamps = np.array(timestamps)
    
    # 计算稳定掩码
    stable_mask = peak_valley_ratios <= pv_threshold
    
    return stable_mask, peak_valley_ratios, timestamps, flow_data

def plot_flow_analysis(data, stable_mask, peak_valley_ratios, timestamps, flow_data, window_size):
    """
    绘制流量分析图表
    """
    # 创建主图表
    plt.figure(figsize=(15, 12))
    
    # 1. 流量变化曲线与稳定工况标记
    plt.subplot(2, 1, 1)
    plt.plot(data['k_ts'], data['主凝结水流量'], 'b-', label='流量变化')
    plt.title('主凝结水流量变化 (红点表示稳定工况)')
    plt.xlabel('时间')
    plt.ylabel('流量(t/h)')
    plt.grid(True)
    
    # 标记稳定点
    stable_times = timestamps[stable_mask]
    stable_flows = data['主凝结水流量'].iloc[window_size//2:len(peak_valley_ratios)+window_size//2][stable_mask].values
    plt.scatter(stable_times, stable_flows, color='red', s=20, label='稳定工况')
    plt.legend()
    
    # 2. 峰谷差比例变化
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, peak_valley_ratios, 'g-')
    plt.axhline(y=0.04, color='r', linestyle='--', label='稳定阈值(4%)')
    plt.title('流量峰谷差比例变化')
    plt.xlabel('时间')
    plt.ylabel('峰谷差/平均值')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()

def plot_flow_distribution(stable_flow_data):
    """
    绘制流量分布图
    """
    plt.figure(figsize=(15, 6))
    sns.histplot(data=stable_flow_data, kde=True)
    plt.title('稳定工况下主凝结水流量分布')
    plt.xlabel('流量(t/h)')
    plt.ylabel('频数')
    plt.grid(True)
    return plt.gcf()

def print_flow_statistics(data, flow_data, stable_mask, peak_valley_ratios, window_size):
    """
    打印流量统计信息
    """
    print("\n主凝结水流量稳定性分析统计:")
    print(f"总数据点数: {len(data)}")
    print(f"\n峰谷差法 (阈值4%):")
    print(f"稳定工况点数: {np.sum(stable_mask)}")
    print(f"稳定比例: {np.sum(stable_mask) / len(peak_valley_ratios) * 100:.2f}%")
    
    # 计算基本统计特征
    print("\n主凝结水流量统计特征:")
    print(f"平均值: {np.mean(flow_data):.2f} t/h")
    print(f"标准差: {np.std(flow_data):.2f} t/h")
    print(f"最大值: {np.max(flow_data):.2f} t/h")
    print(f"最小值: {np.min(flow_data):.2f} t/h")
    print(f"中位数: {np.median(flow_data):.2f} t/h")
    
    # 计算稳定工况下的统计特征
    stable_flow_data = flow_data[window_size//2:len(peak_valley_ratios)+window_size//2][stable_mask]
    print("\n稳定工况下的统计特征:")
    print(f"平均值: {np.mean(stable_flow_data):.2f} t/h")
    print(f"标准差: {np.std(stable_flow_data):.2f} t/h")
    print(f"最大值: {np.max(stable_flow_data):.2f} t/h")
    print(f"最小值: {np.min(stable_flow_data):.2f} t/h")
    print(f"中位数: {np.median(stable_flow_data):.2f} t/h")

def main():
    """
    主函数
    """
    try:
        # 读取数据
        df = pd.read_csv('C:/Users/Li/PycharmProjects/动手深度学习/最大出力预测/file/half_year_data_with_names.csv')  # 替换为实际的数据文件路径
        
        # 分析稳态
        window_size = 10
        stable_mask, peak_valley_ratios, timestamps, flow_data = analyze_condensate_flow(df, window_size)
        
        # 绘制主要分析图
        fig1 = plot_flow_analysis(df, stable_mask, peak_valley_ratios, timestamps, flow_data, window_size)
        plt.show()
        
        # 打印统计信息
        print_flow_statistics(df, flow_data, stable_mask, peak_valley_ratios, window_size)
        
        # 绘制分布图
        stable_flow_data = flow_data[window_size//2:len(peak_valley_ratios)+window_size//2][stable_mask]
        fig2 = plot_flow_distribution(stable_flow_data)
        plt.show()
        
        # 可选：保存结果
        df['is_flow_stable'] = False
        stable_indices = np.where(stable_mask)[0] + window_size//2
        df.loc[stable_indices, 'is_flow_stable'] = True
        # df.to_csv('condensate_flow_analysis.csv', index=False)
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()