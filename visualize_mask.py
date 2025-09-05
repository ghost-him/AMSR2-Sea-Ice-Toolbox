# file: visualize_mask.py

import argparse
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

def visualize_mask(input_file, output_file):
    """
    可视化由 create_statistical_land_mask.py 生成的NetCDF掩码文件。

    Args:
        input_file (str):输入的 .nc 掩码文件路径。
        output_file (str):输出的 .png 图像文件路径。
    """
    print(f"--- 开始可视化任务 ---")
    print(f"正在读取输入文件: {input_file}")

    try:
        with nc.Dataset(input_file, 'r') as ds:
            # 读取数据
            land_mask = ds.variables['land_mask'][:]
            nan_ratio = ds.variables['nan_ratio'][:]
            # 尝试读取元数据以用于标题
            try:
                threshold = ds.nan_frequency_threshold
                title_threshold = f" (Threshold ≥ {threshold:.2f})"
            except AttributeError:
                title_threshold = "" # 如果文件中没有阈值属性

    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{input_file}'")
        return
    except KeyError as e:
        print(f"错误: 输入文件中缺少必需的变量: {e}。")
        print("请确保输入文件是由 'create_statistical_land_mask.py' 脚本生成的。")
        return

    print("数据读取成功，正在生成图像...")

    # --- 开始绘图 ---
    # 创建一个1行2列的子图布局
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 1. 左侧子图: 最终的陆地掩码 (Binary) ---
    ax1 = axes[0]
    im1 = ax1.imshow(land_mask, cmap='viridis', interpolation='nearest')
    ax1.set_title('Final Land Mask (1 = Land)', fontsize=14)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    
    # 为左侧子图添加颜色条
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('0: Non-Land, 1: Land')
    cbar1.set_ticks([0, 1]) # 确保颜色条只显示0和1

    # --- 2. 右侧子图: NaN 频率比率 (Continuous) ---
    ax2 = axes[1]
    # 使用 vmin 和 vmax 确保颜色条范围是 0-1
    im2 = ax2.imshow(nan_ratio, cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title('NaN Frequency Ratio', fontsize=14)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('') # 避免与左图的Y轴标签重叠

    # 为右侧子图添加颜色条
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Ratio (0.0 = always valid, 1.0 = always NaN)')

    # --- 整体设置 ---
    # 添加一个总标题
    fig.suptitle(f'Land Mask Visualization for\n{input_file.split("/")[-1]}{title_threshold}', fontsize=16)

    # 调整布局以防止标题重叠
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 保存图像 ---
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"--- 任务完成 ---")
        print(f"图像已成功保存到: {output_file}")
    except Exception as e:
        print(f"错误: 保存文件时发生错误: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化由 create_statistical_land_mask.py 生成的陆地掩码。")
    parser.add_argument('--input_file', type=str, required=True, 
                        help='输入的 NetCDF 掩码文件路径 (例如: land_mask_stat_85pct.nc)。')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='输出的 PNG 图像文件路径 (例如: land_mask_visualization.png)。')

    args = parser.parse_args()
    
    visualize_mask(
        input_file=args.input_file,
        output_file=args.output_file
    )