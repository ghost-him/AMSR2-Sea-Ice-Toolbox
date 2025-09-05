# file: consolidate_data.py

import os
import re
import argparse
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import netCDF4 as nc
from tqdm import tqdm
import numba

# ==============================================================================
# 时空插值算法 (Spatio-Temporal Interpolation Algorithm)
# ==============================================================================

@numba.jit(nopython=True, parallel=True)
def spatio_temporal_interpolate(data, time_window, space_window, alpha, sigma):
    """
    使用时空加权平均算法对3D数据立方体中的NaN值进行插值。
    该函数经过Numba JIT编译以获得高性能。

    参数:
    data (np.ndarray): 输入的三维数组 (time, height, width)，包含 np.nan 作为缺失值。
    time_window (int): 时间搜索半径。
    space_window (int): 空间搜索半径。
    alpha (float): 时空平衡因子。
    sigma (float): 高斯核的带宽。

    返回:
    np.ndarray: 插值后的三维数组。
    """
    T, H, W = data.shape
    interpolated_data = data.copy()
    
    missing_indices = []
    for t in range(T):
        for y in range(H):
            for x in range(W):
                if np.isnan(data[t, y, x]):
                    missing_indices.append((t, y, x))

    # Numba 可以很好地并行化这个外层循环
    for i in numba.prange(len(missing_indices)):
        t_m, y_m, x_m = missing_indices[i]
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        t_start = max(0, t_m - time_window)
        t_end = min(T, t_m + time_window + 1)
        y_start = max(0, y_m - space_window)
        y_end = min(H, y_m + space_window + 1)
        x_start = max(0, x_m - space_window)
        x_end = min(W, x_m + space_window + 1)
        
        for t_k in range(t_start, t_end):
            for y_k in range(y_start, y_end):
                for x_k in range(x_start, x_end):
                    if not np.isnan(data[t_k, y_k, x_k]):
                        dist_sq = (y_m - y_k)**2 + (x_m - x_k)**2 + alpha * (t_m - t_k)**2
                        weight = np.exp(-dist_sq / (2 * sigma**2))
                        weighted_sum += weight * data[t_k, y_k, x_k]
                        total_weight += weight
        
        if total_weight > 1e-6: # 增加一个小的阈值防止除零
            interpolated_data[t_m, y_m, x_m] = weighted_sum / total_weight
            
    return interpolated_data

# ==============================================================================
# 数据合并主逻辑 (Main Data Consolidation Logic)
# ==============================================================================

def GenTimeList(start_time, end_time):
    """生成日期列表 (与您提供的代码相同)"""
    Times = []
    current = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    while current <= end:
        Times.append(int(current.strftime("%Y%m%d")))
        current += relativedelta(days=1)
    return Times

def create_consolidated_file_with_interpolation(data_root_path, start_time, end_time, output_path,
                                                batch_size=30, time_window=2, space_window=5,
                                                alpha=10.0, sigma=5.0):
    """
    扫描、插值处理并合并每日的 .nc 文件到一个大的 NetCDF 文件中。
    该版本采用分块处理策略，以支持需要时间上下文的插值算法。

    Args:
        data_root_path (str): 存放每日 .nc 文件的根目录。
        start_time (int): 要处理的开始日期 (YYYYMMDD)。
        end_time (int): 要处理的结束日期 (YYYYMMDD)。
        output_path (str): 输出的合并文件的路径。
        batch_size (int): 一次读入内存进行处理的帧数（天数）。
        time_window (int): 插值算法的时间搜索半径。
        space_window (int): 插值算法的空间搜索半径。
        alpha (float): 插值算法的时空平衡因子。
        sigma (float): 插值算法的高斯核带宽。
    """
    print("--- 开始数据合并与插值任务 ---")
    print(f"扫描目录: {data_root_path}")
    print(f"请求的时间范围: {start_time} to {end_time}")
    print("\n插值参数:")
    print(f"  - 批处理大小 (Batch Size): {batch_size} 天")
    print(f"  - 时间窗口半径 (Time Window): {time_window} 天")
    print(f"  - 空间窗口半径 (Space Window): {space_window} 像素")
    print(f"  - 时空因子 (Alpha): {alpha}")
    print(f"  - 高斯核带宽 (Sigma): {sigma}\n")

    # 1. 找到所有相关文件并建立日期到路径的映射
    file_map = {}
    pattern = re.compile(r"asi-AMSR2-n3125-(\d{8})-v5\.4\.nc")
    for root, _, files in os.walk(data_root_path):
        for file in files:
            match = pattern.match(file)
            if match:
                date_int = int(match.group(1))
                file_map[date_int] = os.path.join(root, file)

    # 2. 获取所需时间范围内的文件路径，并记录缺失的日期
    full_time_range = GenTimeList(start_time, end_time)
    available_files = []
    available_dates = []
    missing_dates = []

    for date in full_time_range:
        if date in file_map:
            available_files.append(file_map[date])
            available_dates.append(date)
        else:
            missing_dates.append(date)
            # print(f"警告: 日期 {date} 的数据文件缺失，将跳过此日期。")
    
    num_available = len(available_files)
    if num_available == 0:
        print("错误: 在指定的时间范围内没有找到任何可用的数据文件。任务终止。")
        return

    print(f"在请求的 {len(full_time_range)} 天中，成功定位 {num_available} 个文件，缺失 {len(missing_dates)} 个文件。")

    # 3. 从第一个可用文件中读取元数据，并创建输出文件结构
    with nc.Dataset(available_files[0], 'r') as first_file:
        x_coords = first_file.variables['x'][:]
        y_coords = first_file.variables['y'][:]
        ny, nx = len(y_coords), len(x_coords)
        
        with nc.Dataset(output_path, 'w', format='NETCDF4') as dest_nc:
            print("正在创建输出文件结构...")
            dest_nc.createDimension('time', None)
            dest_nc.createDimension('y', ny)
            dest_nc.createDimension('x', nx)
            
            times = dest_nc.createVariable('time', 'i4', ('time',))
            ys = dest_nc.createVariable('y', 'f4', ('y',))
            xs = dest_nc.createVariable('x', 'f4', ('x',))
            sea_ice_conc = dest_nc.createVariable('sea_ice_conc', 'f4', ('time', 'y', 'x'), zlib=True)
            
            ys[:] = y_coords
            xs[:] = x_coords
            dest_nc.description = "Consolidated and Interpolated AMSR2 Sea Ice Concentration Data"

            # 4. 分块读取、插值、写入
            time_idx_written = 0
            
            pbar = tqdm(total=num_available, desc="处理并写入数据")
            for i in range(0, num_available, batch_size):
                # a. 确定当前批次和需要读取的范围 (包含padding)
                batch_start_idx = i
                batch_end_idx = min(i + batch_size, num_available)
                
                read_start_idx = max(0, batch_start_idx - time_window)
                read_end_idx = min(num_available, batch_end_idx + time_window)
                
                # b. 读取数据块到内存
                chunk_data_list = []
                for file_idx in range(read_start_idx, read_end_idx):
                    with nc.Dataset(available_files[file_idx], 'r') as src_nc:
                        data_masked = src_nc.variables['z'][:]
                        if isinstance(data_masked, np.ma.MaskedArray):
                            data = data_masked.filled(np.nan)
                        else:
                            data = np.array(data_masked)
                        chunk_data_list.append(data.astype(np.float32))
                
                data_chunk = np.stack(chunk_data_list, axis=0)

                # c. 执行时空插值
                interpolated_chunk = spatio_temporal_interpolate(
                    data_chunk, time_window, space_window, alpha, sigma
                )
                
                # d. 提取出当前批次对应的有效结果
                # 计算在插值后的大块中，我们需要的有效数据的起始和结束索引
                valid_start_in_chunk = batch_start_idx - read_start_idx
                valid_end_in_chunk = batch_end_idx - read_start_idx
                processed_batch = interpolated_chunk[valid_start_in_chunk:valid_end_in_chunk]

                # e. 对插值后的数据进行最后的处理（归一化和nan值填充）
                # 插值后可能仍有nan（如果某点周围全是nan），用0作为最终的fallback
                processed_batch = np.nan_to_num(processed_batch, nan=0.0)
                processed_batch /= 100.0
                
                # f. 将处理好的批次写入文件
                num_in_batch = processed_batch.shape[0]
                sea_ice_conc[time_idx_written : time_idx_written + num_in_batch, :, :] = processed_batch
                
                dates_for_batch = available_dates[batch_start_idx:batch_end_idx]
                times[time_idx_written : time_idx_written + num_in_batch] = dates_for_batch
                
                time_idx_written += num_in_batch
                pbar.update(num_in_batch)
            
            pbar.close()
            final_shape = sea_ice_conc.shape

    print(f"--- 任务完成 ---")
    print(f"成功创建合并文件: {output_path}")
    print(f"最终数据维度 (Time, Height, Width): {final_shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将每日海冰密集度 NetCDF 文件合并并进行时空插值。")
    # --- 基础参数 ---
    parser.add_argument('--source_dir', type=str, required=True, help='存放每日 .nc 文件的源根目录。')
    parser.add_argument('--start_date', type=int, required=True, help='要处理的开始日期 (格式: YYYYMMDD)。')
    parser.add_argument('--end_date', type=int, required=True, help='要处理的结束日期 (格式: YYYYMMDD)。')
    parser.add_argument('--output_file', type=str, required=True, help='输出的合并文件的路径和名称。')
    
    # --- 性能与插值参数 ---
    parser.add_argument('--batch_size', type=int, default=30, help='一次读入内存处理的帧数 (天数)，影响内存占用。默认: 30')
    parser.add_argument('--time_window', type=int, default=3, help='插值时的时间搜索半径 (天)。默认: 3')
    parser.add_argument('--space_window', type=int, default=7, help='插值时的空间搜索半径 (像素)。默认: 7')
    parser.add_argument('--alpha', type=float, default=15.0, help='插值时的时空平衡因子。默认: 15.0')
    parser.add_argument('--sigma', type=float, default=7.0, help='插值时的高斯核带宽。默认: 7.0')

    args = parser.parse_args()

    create_consolidated_file_with_interpolation(
        data_root_path=args.source_dir,
        start_time=args.start_date,
        end_time=args.end_date,
        output_path=args.output_file,
        batch_size=args.batch_size,
        time_window=args.time_window,
        space_window=args.space_window,
        alpha=args.alpha,
        sigma=args.sigma
    )