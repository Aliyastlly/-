import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_steady_state(df, window_size=10, pv_threshold=0.01, std_threshold=0.005):
    """
    分析时间序列数据的稳态特性
    
    参数:
    df: DataFrame, 包含 'k_ts' 和 '发电机有功功率1' 列的数据框
    window_size: int, 滑动窗口大小
    pv_threshold: float, 峰谷差比例阈值（默认0.01，即1%）
    std_threshold: float, 标准差比例阈值（默认0.005，即0.5%）
    
    返回:
    tuple: (stable_points_pv, stable_points_std, peak_valley_ratios, std_ratios, timestamps)
    """
    peak_valley_ratios = []
    std_ratios = []
    timestamps = []
    stable_points_pv = []
    stable_points_std = []

    for i in range(len(df) - window_size + 1):
        window = df['发电机有功功率1'].iloc[i:i + window_size]
        mean_value = window.mean()
        
        if mean_value != 0:
            # 计算峰谷差比例
            peak_valley = window.max() - window.min()
            pv_ratio = peak_valley / mean_value
            peak_valley_ratios.append(pv_ratio)
            
            # 计算标准差比例
            std_ratio = window.std() / mean_value
            std_ratios.append(std_ratio)
            
            # 记录时间戳
            timestamps.append(df['k_ts'].iloc[i + window_size // 2])
            
            # 判断是否稳定
            if pv_ratio <= pv_threshold:
                stable_points_pv.append(i + window_size // 2)
            if std_ratio <= std_threshold:
                stable_points_std.append(i + window_size // 2)
    
    return stable_points_pv, stable_points_std, peak_valley_ratios, std_ratios, timestamps

def plot_steady_state_analysis(df, stable_points_pv, stable_points_std):
    """
    绘制稳态分析图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 1. 峰谷差分析图
    ax1.plot(df['k_ts'], df['发电机有功功率1'], 'b-', label='功率变化', alpha=0.6)
    if stable_points_pv:
        stable_times = df['k_ts'].iloc[stable_points_pv]
        stable_powers = df['发电机有功功率1'].iloc[stable_points_pv]
        ax1.scatter(stable_times, stable_powers, color='red', s=20, label='稳定工况(峰谷差≤1%)')
    ax1.set_title('功率变化曲线 (峰谷差法)')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('功率(MW)')
    ax1.grid(True)
    ax1.legend()

    # 2. 标准差分析图
    ax2.plot(df['k_ts'], df['发电机有功功率1'], 'b-', label='功率变化', alpha=0.6)
    if stable_points_std:
        stable_times = df['k_ts'].iloc[stable_points_std]
        stable_powers = df['发电机有功功率1'].iloc[stable_points_std]
        ax2.scatter(stable_times, stable_powers, color='red', s=20, label='稳定工况(标准差≤0.5%)')
    ax2.set_title('功率变化曲线 (标准差法)')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('功率(MW)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    return fig

def print_statistics(df, stable_points_pv, stable_points_std):
    """
    打印稳态分析统计信息
    """
    print("\n稳定性分析统计:")
    print(f"总数据点数: {len(df)}")
    print(f"\n峰谷差法:")
    print(f"稳定工况点数: {len(stable_points_pv)}")
    print(f"稳定比例: {len(stable_points_pv) / len(df) * 100:.2f}%")
    print(f"\n标准差法:")
    print(f"稳定工况点数: {len(stable_points_std)}")
    print(f"稳定比例: {len(stable_points_std) / len(df) * 100:.2f}%")

def main():
    """
    主函数
    """
    try:
        # 读取数据
        df = pd.read_csv('C:/Users/Li/PycharmProjects/动手深度学习/最大出力预测/file/half_year_data_with_names.csv')  # 替换为实际的数据文件路径
        
        # 分析稳态
        stable_points_pv, stable_points_std, peak_valley_ratios, std_ratios, timestamps = \
            analyze_steady_state(df)
        
        # 绘制图形
        fig = plot_steady_state_analysis(df, stable_points_pv, stable_points_std)
        plt.show()
        
        # 打印统计信息
        print_statistics(df, stable_points_pv, stable_points_std)
        
        # 可选：保存结果
        df['is_stable_pv'] = False
        df['is_stable_std'] = False
        df.loc[stable_points_pv, 'is_stable_pv'] = True
        df.loc[stable_points_std, 'is_stable_std'] = True
        df.to_csv('steady_state_results.csv', index=False)
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main() 