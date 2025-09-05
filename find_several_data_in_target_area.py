# file: find_several_data_in_target_area.py

import os
import re
import argparse
import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import patches
import numpy as np
import numpy.ma as ma # 导入掩码数组模块
import netCDF4 as nc
from tqdm import tqdm
import matplotlib.pyplot as plt # 导入绘图库

def GenTimeList(start_time, end_time):
    """生成日期列表。"""
    Times = []
    current = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    while current <= end:
        Times.append(int(current.strftime("%Y%m%d")))
        current += relativedelta(days=1)
    return Times
def visualize_sample_roi(file_path, roi_bounds):
    """
    可视化指定文件中ROI在完整图像中的位置。
    会显示完整的图像，并在其上用红框标出ROI的位置。
    """
    print("\n--- 开始生成ROI样本可视化图像 ---")
    print(f"使用文件进行预览: {os.path.basename(file_path)}")
    
    ymin, ymax = roi_bounds['ymin'], roi_bounds['ymax']
    xmin, xmax = roi_bounds['xmin'], roi_bounds['xmax']
    
    try:
        with nc.Dataset(file_path, 'r') as src_nc:
            if 'z' not in src_nc.variables:
                print(f"错误: 文件 {file_path} 中找不到变量 'z'。")
                return

            # 1. 读取完整的数据用于背景显示
            full_data = src_nc.variables['z'][:, :]
            
            # 2. 单独读取ROI数据用于统计缺失值
            roi_data = full_data[ymin:ymax, xmin:xmax]
            
            # 3. 统计ROI内的缺失值
            missing_count = ma.count_masked(roi_data)
            total_pixels = roi_data.size
            missing_percent = (missing_count / total_pixels) * 100 if total_pixels > 0 else 0
            
            # --- 开始绘图 ---
            # 使用 plt.subplots() 可以更方便地获取到 axes 对象 ax
            fig, ax = plt.subplots(figsize=(10, 12)) 
            
            # 4. 显示完整的背景图像
            # 使用一个简单的灰度图(cmap='gray')可以让红框更突出
            im = ax.imshow(full_data, cmap='gray', origin='upper') 
            fig.colorbar(im, ax=ax, shrink=0.6, label='Sea Ice Concentration (Full Image)')
            
            # 5. 创建并添加红色矩形框来标记ROI
            rect = patches.Rectangle(
                (xmin, ymin),           # 矩形左下角的坐标 (x, y)
                xmax - xmin,            # 矩形的宽度
                ymax - ymin,            # 矩形的高度
                linewidth=2,            # 框线宽度
                edgecolor='red',        # 框线颜色
                facecolor='none'        # 矩形内部颜色 ('none' 表示透明)
            )
            ax.add_patch(rect)
            
            ax.set_title(f"ROI Location in Full Image\nFile: {os.path.basename(file_path)}")
            ax.set_xlabel("X Index (Column)")
            ax.set_ylabel("Y Index (Row)")
            
            # 6. 在图像下方添加统计信息文本
            stats_text = (f"ROI Bounds: Y({ymin}:{ymax}), X({xmin}:{xmax})\n"
                          f"Missing pixels inside RED BOX: {missing_count} ({missing_percent:.2f}%)")
            plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10, 
                        bbox={"facecolor":"wheat", "alpha":0.5, "pad":5})

            print("图像已生成。请查看弹出的窗口。")
            print("关闭图像窗口后，主程序将继续执行...")
            plt.show() # 显示图像
            
    except Exception as e:
        print(f"\n生成可视化预览时出错: {e}")
        print("将跳过预览并继续主任务。")


def find_missing_data_dates_in_roi(data_root_path, start_time, end_time, roi_bounds, threshold_percent=0.1):
    """
    在一个指定的感兴趣区域（ROI）内，查找哪些日期的海冰数据出现了缺失值（masked values）。

    Args:
        data_root_path (str): 存放每日 .nc 文件的根目录。
        start_time (int): 要处理的开始日期 (YYYYMMDD)。
        end_time (int): 要处理的结束日期 (YYYYMMDD)。
        roi_bounds (dict): 感兴趣区域的数组索引范围。
                           格式: {'ymin': int, 'ymax': int, 'xmin': int, 'xmax': int}
        threshold_percent (float): 触发报告的缺失像素百分比阈值。
                                     默认为0.1，意味着ROI中只要有超过0.1%的像素是缺失的就会被报告。
                                     设置为0可以在出现任何缺失值时都报告。
    """
    ymin, ymax = roi_bounds['ymin'], roi_bounds['ymax']
    xmin, xmax = roi_bounds['xmin'], roi_bounds['xmax']

    print("--- 开始查找海冰区内的缺失值日期任务 ---")
    print(f"扫描目录: {data_root_path}")
    print(f"处理的时间范围: {start_time} to {end_time}")
    print(f"感兴趣的区域 (ROI) 索引:")
    print(f"  Y (行) 范围: {ymin} to {ymax}")
    print(f"  X (列) 范围: {xmin} to {xmax}")
    print(f"报告阈值: ROI内缺失像素占比 >= {threshold_percent}%")

    # 1. 查找文件
    file_map = {}
    pattern = re.compile(r"asi-AMSR2-n3125-(\d{8})-v5\.4\.nc")
    for root, _, files in os.walk(data_root_path):
        for file in files:
            match = pattern.match(file)
            if match:
                date_int = int(match.group(1))
                file_map[date_int] = os.path.join(root, file)

    # 2. 筛选在时间范围内的文件
    full_time_range = GenTimeList(start_time, end_time)
    available_files = []
    missing_dates_count = 0
    for date in full_time_range:
        if date in file_map:
            available_files.append((date, file_map[date])) # 同时保存日期和路径
        else:
            missing_dates_count += 1
    
    if missing_dates_count > 0:
        print(f"警告: 在请求的时间范围内有 {missing_dates_count} 天的数据文件缺失。")
        
    if not available_files:
        print("错误: 在指定的时间范围内没有找到任何可用的数据文件。任务终止。")
        return

    print(f"将检查 {len(available_files)} 个文件。")

    # --- 新增步骤：可视化第一个文件的ROI ---
    first_file_path = available_files[0][1]
    visualize_sample_roi(first_file_path, roi_bounds)

    # --- 核心逻辑：在ROI内查找缺失值 ---
    found_anomalies = []

    # 3. 遍历所有文件，检查ROI
    for date, file_path in tqdm(available_files, desc="正在检查每日数据"):
        try:
            with nc.Dataset(file_path, 'r') as src_nc:
                roi_data = src_nc.variables['z'][ymin:ymax, xmin:xmax]
                
                if roi_data.size == 0:
                    print(f"\n错误: 在文件 {file_path} 中定义的ROI区域大小为零。请检查索引范围。")
                    return

                # **核心修改：使用 np.ma.count_masked 来计算被掩码（缺失）的像素数**
                missing_count_in_roi = ma.count_masked(roi_data)

                if missing_count_in_roi > 0:
                    total_pixels_in_roi = roi_data.size
                    missing_percentage = (missing_count_in_roi / total_pixels_in_roi) * 100
                    
                    if missing_percentage >= threshold_percent:
                        found_anomalies.append({
                            "date": date,
                            "file_path": file_path,
                            "missing_count": missing_count_in_roi,
                            "missing_percentage": missing_percentage
                        })
        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {e}")

    # 4. 报告结果
    print("\n--- 任务完成 ---")
    if not found_anomalies:
        print("在指定的时间范围和ROI内，没有发现任何包含缺失值的异常日期。")
    else:
        print(f"发现 {len(found_anomalies)} 个异常日期，其ROI内存在缺失值：")
        print("-" * 80)
        print(f"{'日期':<12} | {'ROI内缺失像素数':<20} | {'ROI内缺失占比':<20} | {'文件路径':<40}")
        print("-" * 80)
        for anomaly in found_anomalies:
            print(f"{anomaly['date']:<12} | {anomaly['missing_count']:<20} | {f'{anomaly['missing_percentage']:.2f}%':<20} | {os.path.basename(anomaly['file_path']):<40}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="在每日NetCDF数据的一个指定区域(ROI)中，查找出现缺失值(masked values)的日期。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--source_dir', type=str, required=True, help='存放每日 .nc 文件的源根目录。')
    parser.add_argument('--start_date', type=int, required=True, help='要处理的开始日期 (格式: YYYYMMDD)。')
    parser.add_argument('--end_date', type=int, required=True, help='要处理的结束日期 (格式: YYYYMMDD)。')
    
    parser.add_argument('--ymin', type=int, required=True, help='感兴趣区域(ROI)的最小Y轴索引(起始行)。')
    parser.add_argument('--ymax', type=int, required=True, help='感兴趣区域(ROI)的最大Y轴索引(结束行，不包含)。')
    parser.add_argument('--xmin', type=int, required=True, help='感兴趣区域(ROI)的最小X轴索引(起始列)。')
    parser.add_argument('--xmax', type=int, required=True, help='感兴趣区域(ROI)的最大X轴索引(结束列，不包含)。')
    
    parser.add_argument(
        '--threshold_percent', type=float, default=0.1, 
        help='报告异常的缺失像素百分比阈值。\n'
             '例如: 输入 1.0 表示当ROI内有1%%或更多的像素是缺失时才报告。\n'
             '(默认值: 0.1)'
    )
    
    args = parser.parse_args()
    
    if args.ymin >= args.ymax or args.xmin >= args.xmax:
        raise ValueError("索引范围无效: ymin必须小于ymax，xmin必须小于xmax。")
        
    roi_bounds = {
        'ymin': args.ymin,
        'ymax': args.ymax,
        'xmin': args.xmin,
        'xmax': args.xmax
    }

    find_missing_data_dates_in_roi(
        data_root_path=args.source_dir,
        start_time=args.start_date,
        end_time=args.end_date,
        roi_bounds=roi_bounds,
        threshold_percent=args.threshold_percent
    )