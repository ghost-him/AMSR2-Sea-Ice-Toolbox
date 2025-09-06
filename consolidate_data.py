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
        
        if total_weight > 1e-6:
            interpolated_data[t_m, y_m, x_m] = weighted_sum / total_weight
            
    return interpolated_data

# ==============================================================================
# 数据合并主逻辑 (Main Data Consolidation Logic)
# ==============================================================================

def GenTimeList(start_time, end_time):
    """生成日期列表"""
    Times = []
    current = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    while current <= end:
        Times.append(int(current.strftime("%Y%m%d")))
        current += relativedelta(days=1)
    return Times

def create_consolidated_file_with_interpolation(data_root_path, start_time, end_time, output_path,
                                                land_mask_path=None, batch_size=30, time_window=2,
                                                space_window=5, alpha=10.0, sigma=5.0):
    """
    扫描、插值处理并合并每日的 .nc 文件到一个大的 NetCDF 文件中。
    该版本采用分块处理策略，并可以选择性地应用陆地掩码。

    Args:
        data_root_path (str): 存放每日 .nc 文件的根目录。
        start_time (int): 要处理的开始日期 (YYYYMMDD)。
        end_time (int): 要处理的结束日期 (YYYYMMDD)。
        output_path (str): 输出的合并文件的路径。
        land_mask_path (str, optional): 指向陆地掩码NetCDF文件的路径。默认为 None。
        batch_size (int): 一次读入内存进行处理的帧数（天数）。
        ... (其他插值参数)
    """
    print("--- 开始数据合并与插值任务 ---")
    print(f"扫描目录: {data_root_path}")
    print(f"请求的时间范围: {start_time} to {end_time}")
    
    if land_mask_path:
        print(f"将使用陆地掩码文件: {land_mask_path}")
    else:
        print("未提供陆地掩码文件，将不应用掩码。")
    
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

    # 2. 获取所需时间范围内的文件路径
    full_time_range = GenTimeList(start_time, end_time)
    available_files = [file_map[date] for date in full_time_range if date in file_map]
    available_dates = [date for date in full_time_range if date in file_map]
    num_available = len(available_files)
    
    if num_available == 0:
        print("错误: 在指定的时间范围内没有找到任何可用的数据文件。任务终止。")
        return

    print(f"在请求的 {len(full_time_range)} 天中，成功定位 {num_available} 个文件，缺失 {len(full_time_range) - num_available} 个文件。")

    # 3. (新增) 如果提供了路径，则加载陆地掩码
    land_mask = None
    if land_mask_path:
        if not os.path.exists(land_mask_path):
            print(f"错误: 掩码文件未找到于: {land_mask_path}")
            return
        try:
            with nc.Dataset(land_mask_path, 'r') as mask_nc:
                if 'land_mask' not in mask_nc.variables:
                    print(f"错误: 掩码文件 {land_mask_path} 中未找到名为 'land_mask' 的变量。")
                    return
                land_mask = mask_nc.variables['land_mask'][:].astype(bool)
                print("陆地掩码加载成功。")
        except Exception as e:
            print(f"错误: 加载陆地掩码文件时出错: {e}")
            return

    # 4. 从第一个可用文件中读取元数据，并创建输出文件结构
    with nc.Dataset(available_files[0], 'r') as first_file:
        x_coords = first_file.variables['x'][:]
        y_coords = first_file.variables['y'][:]
        ny, nx = len(y_coords), len(x_coords)
        
        # (新增) 验证掩码维度
        if land_mask is not None and land_mask.shape != (ny, nx):
            print(f"错误: 陆地掩码的维度 {land_mask.shape} 与数据维度 {(ny, nx)} 不匹配。")
            return
        
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
            dest_nc.description = "Consolidated, Interpolated, and Masked AMSR2 Sea Ice Concentration Data"

            # 5. 分块读取、插值、写入
            time_idx_written = 0
            pbar = tqdm(total=num_available, desc="处理并写入数据")
            for i in range(0, num_available, batch_size):
                # a-d. (与之前相同) 读取、插值、提取有效批次
                batch_start_idx = i
                batch_end_idx = min(i + batch_size, num_available)
                read_start_idx = max(0, batch_start_idx - time_window)
                read_end_idx = min(num_available, batch_end_idx + time_window)
                
                chunk_files = available_files[read_start_idx:read_end_idx]
                chunk_data_list = []
                for file_path in chunk_files:
                    with nc.Dataset(file_path, 'r') as src_nc:
                        data_masked = src_nc.variables['z'][:]
                        data = data_masked.filled(np.nan) if isinstance(data_masked, np.ma.MaskedArray) else np.array(data_masked)
                        chunk_data_list.append(data.astype(np.float32))
                data_chunk = np.stack(chunk_data_list, axis=0)

                interpolated_chunk = spatio_temporal_interpolate(data_chunk, time_window, space_window, alpha, sigma)
                
                valid_start_in_chunk = batch_start_idx - read_start_idx
                valid_end_in_chunk = batch_end_idx - read_start_idx
                processed_batch = interpolated_chunk[valid_start_in_chunk:valid_end_in_chunk]

                # e. 对插值后的数据进行最后的处理
                processed_batch = np.nan_to_num(processed_batch, nan=0.0)
                processed_batch /= 100.0
                
                # f. (新增) 如果提供了陆地掩码，则应用它
                if land_mask is not None:
                    # land_mask是2D(H, W), processed_batch是3D(T, H, W)。
                    # NumPy广播机制会自动将2D掩码应用到每个时间切片上。
                    # 将掩码为True(陆地)的位置的值设为0.0。
                    processed_batch[:, land_mask] = 0.0

                # g. 将处理好的批次写入文件
                num_in_batch = processed_batch.shape[0]
                sea_ice_conc[time_idx_written : time_idx_written + num_in_batch, :, :] = processed_batch
                
                dates_for_batch = available_dates[batch_start_idx:batch_end_idx]
                times[time_idx_written : time_idx_written + num_in_batch] = dates_for_batch
                
                time_idx_written += num_in_batch
                pbar.update(num_in_batch)
            
            pbar.close()
            final_shape = sea_ice_conc.shape

    print(f"\n--- 任务完成 ---")
    print(f"成功创建合并文件: {output_path}")
    print(f"最终数据维度 (Time, Height, Width): {final_shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将每日海冰密集度 NetCDF 文件合并并进行时空插值。")
    # --- 基础参数 ---
    parser.add_argument('--source_dir', type=str, required=True, help='存放每日 .nc 文件的源根目录。')
    parser.add_argument('--start_date', type=int, required=True, help='要处理的开始日期 (格式: YYYYMMDD)。')
    parser.add_argument('--end_date', type=int, required=True, help='要处理的结束日期 (格式: YYYYMMDD)。')
    parser.add_argument('--output_file', type=str, required=True, help='输出的合并文件的路径和名称。')
    parser.add_argument('--land_mask_file', type=str, default=None, 
                        help='(可选) 指向陆地掩码 NetCDF 文件的路径。如果提供，插值后的陆地区域将被设置为0。')

    # --- 性能与插值参数 ---
    parser.add_argument('--batch_size', type=int, default=30, help='一次读入内存处理的帧数 (天数)，影响内存占用。默认: 30')
    parser.add_argument('--time_window', type=int, default=3, help='插值时的时间搜索半径 (天)。默认: 3')
    parser.add_argument('--space_window', type=int, default=7, help='插值时的空间搜索半径 (像素)。默认: 7')
    parser.add_argument('--alpha', type=float, default=25.0, help='插值时的时空平衡因子。值越大越强调空间邻近性。推荐: 25.0')
    parser.add_argument('--sigma', type=float, default=3.0, help='插值时的高斯核带宽。推荐: 3.0')

    args = parser.parse_args()

    # 注意：我根据上一轮的建议，调整了alpha和sigma的默认值，使其更适合海冰数据
    create_consolidated_file_with_interpolation(
        data_root_path=args.source_dir,
        start_time=args.start_date,
        end_time=args.end_date,
        output_path=args.output_file,
        land_mask_path=args.land_mask_file,  # 传递新参数
        batch_size=args.batch_size,
        time_window=args.time_window,
        space_window=args.space_window,
        alpha=args.alpha,
        sigma=args.sigma
    )