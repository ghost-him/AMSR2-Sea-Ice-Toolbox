# file: create_mask.py

import os
import re
import argparse
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import netCDF4 as nc
from tqdm import tqdm

def GenTimeList(start_time, end_time):
    """生成日期列表。"""
    Times = []
    current = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    while current <= end:
        Times.append(int(current.strftime("%Y%m%d")))
        current += relativedelta(days=1)
    return Times

def create_statistical_land_mask(data_root_path, start_time, end_time, output_path, threshold=0.85):
    """
    通过统计nan值的频率来创建陆地掩码。
    如果一个像素点是nan的次数占比超过指定阈值，则被认为是陆地。

    Args:
        data_root_path (str): 存放每日 .nc 文件的根目录。
        start_time (int): 要处理的开始日期 (YYYYMMDD)。
        end_time (int): 要处理的结束日期 (YYYYMMDD)。
        output_path (str): 输出的掩码文件的路径。
        threshold (float): 判断为陆地的nan频率阈值 (0.0 to 1.0)。
    """
    print("--- 开始创建基于统计的陆地掩码任务 ---")
    print(f"扫描目录: {data_root_path}")
    print(f"处理的时间范围: {start_time} to {end_time}")
    print(f"使用的NaN频率阈值: {threshold:.2f} ({threshold*100}%)")

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
            available_files.append(file_map[date])
        else:
            missing_dates_count += 1
    
    if missing_dates_count > 0:
        print(f"警告: 在请求的时间范围内有 {missing_dates_count} 天的数据文件缺失。")
        
    if not available_files:
        print("错误: 在指定的时间范围内没有找到任何可用的数据文件。任务终止。")
        return

    total_files_processed = len(available_files)
    print(f"将使用 {total_files_processed} 个文件来生成掩码。")

    # --- 核心逻辑：统计NaN频率 ---
    
    # 3. 初始化nan计数器
    print("正在初始化NaN计数器...")
    with nc.Dataset(available_files[0], 'r') as first_file:
        x_coords = first_file.variables['x'][:]
        y_coords = first_file.variables['y'][:]
        ny, nx = len(y_coords), len(x_coords)
        # 使用int32类型，足以应对几十年的日数据计数
        nan_count_map = np.zeros((ny, nx), dtype=np.int32)

    # 4. 遍历所有文件，累加nan出现的次数
    for file_path in tqdm(available_files, desc="正在统计NaN次数"):
        with nc.Dataset(file_path, 'r') as src_nc:
            data = src_nc.variables['z'][:]
            current_nan_mask = np.isnan(data)
            # 布尔数组可以直接用于数学运算，True=1, False=0
            nan_count_map += current_nan_mask
            
    # 5. 计算NaN频率并应用阈值生成最终陆地掩码
    print("计算NaN频率并应用阈值...")
    # 为避免除以零的警告（虽然在这里total_files_processed > 0）
    nan_ratio = nan_count_map / total_files_processed
    land_mask = (nan_ratio >= threshold)

    # 6. 保存结果到NetCDF文件
    print(f"正在将最终掩码写入到: {output_path}")
    with nc.Dataset(output_path, 'w', format='NETCDF4') as dest_nc:
        dest_nc.createDimension('y', ny)
        dest_nc.createDimension('x', nx)
        
        ys = dest_nc.createVariable('y', 'f4', ('y',))
        xs = dest_nc.createVariable('x', 'f4', ('x',))
        # 保存最终的0/1陆地掩码
        mask_var = dest_nc.createVariable('land_mask', 'u1', ('y', 'x'), zlib=True)
        # 同时保存nan频率图，这对于调试和分析非常有用
        ratio_var = dest_nc.createVariable('nan_ratio', 'f4', ('y', 'x'), zlib=True)
        
        ys[:] = y_coords
        xs[:] = x_coords
        mask_var[:] = land_mask.astype(np.uint8)
        ratio_var[:] = nan_ratio
        
        dest_nc.description = (
            "Land mask generated from time-series data based on NaN frequency. "
            f"A pixel is marked as land if its NaN frequency is >= {threshold}."
        )
        dest_nc.source_directory = data_root_path
        dest_nc.time_range_start = str(start_time)
        dest_nc.time_range_end = str(end_time)
        dest_nc.num_files_processed = total_files_processed
        dest_nc.nan_frequency_threshold = threshold
        
        mask_var.long_name = "land_mask"
        mask_var.units = "1 = land, 0 = non-land"
        mask_var.comment = f"Generated where nan_ratio >= {threshold}"
        
        ratio_var.long_name = "NaN frequency"
        ratio_var.units = "ratio"
        ratio_var.comment = "The ratio of days a pixel was NaN over the total processed days."

    land_pixels = np.sum(land_mask)
    total_pixels = ny * nx
    land_percentage = (land_pixels / total_pixels) * 100

    print(f"--- 任务完成 ---")
    print(f"成功创建掩码文件: {output_path}")
    print(f"掩码维度 (Height, Width): {land_mask.shape}")
    print(f"被识别为陆地的像素点数: {land_pixels} ({land_percentage:.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="根据每日NetCDF数据中nan值的频率创建陆地掩码。")
    parser.add_argument('--source_dir', type=str, required=True, help='存放每日 .nc 文件的源根目录。')
    parser.add_argument('--start_date', type=int, required=True, help='要处理的开始日期 (格式: YYYYMMDD)。')
    parser.add_argument('--end_date', type=int, required=True, help='要处理的结束日期 (格式: YYYYMMDD)。')
    parser.add_argument('--output_file', type=str, required=True, help='输出的掩码文件的路径和名称。')
    parser.add_argument('--threshold', type=float, default=0.85, help='判断为陆地的NaN频率阈值 (0.0到1.0之间)，默认为0.85。')
    
    args = parser.parse_args()
    
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("阈值必须在0.0和1.0之间。")

    create_statistical_land_mask(
        data_root_path=args.source_dir,
        start_time=args.start_date,
        end_time=args.end_date,
        output_path=args.output_file,
        threshold=args.threshold
    )