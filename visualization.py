# file: visualization.py
# -*- coding: utf-8 -*-
"""
一个用于可视化单个海冰密集度（SIC）NetCDF文件的高级脚本。

该脚本使用 Cartopy 库进行精确的地理空间投影，生成一幅高质量的、
带有地理参考的地图。用户只需提供一个原始数据文件即可。

核心功能:
1. 读取单个包含海冰密集度数据的 NetCDF 文件。
2. 使用北极立体投影（Polar Stereographic）进行准确的地理可视化。
3. 在地图上叠加陆地、海岸线和经纬网格线，提供地理上下文。
4. 自动从文件名提取日期用于图表标题。
5. 通过命令行参数接收输入文件，易于使用和集成。

命令行使用示例:
    python your_script_name.py --input_file ./data/asi-AMSR2-n3125-20250715-v5.4.nc
"""

import os
import argparse
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Tuple, Optional

def read_geospatial_netcdf(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    从单个 NetCDF 文件中读取海冰密集度数据及其 x, y 坐标。

    Args:
        file_path (str): NetCDF 文件路径。

    Returns:
        一个包含 (数据, x坐标, y坐标) 的元组，如果出错则返回 None。
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 at '{file_path}'")
        return None

    print(f"正在读取文件: {os.path.basename(file_path)}")
    try:
        with nc.Dataset(file_path, 'r') as ds:
            # 检查必要的变量是否存在
            if 'z' not in ds.variables:
                print(f"错误: 在文件中找不到变量 'z'。请检查文件内容。")
                return None
            if 'x' not in ds.variables or 'y' not in ds.variables:
                print(f"错误: 在文件中找不到坐标变量 'x' 或 'y'。")
                return None

            x_coords = ds.variables['x'][:]
            y_coords = ds.variables['y'][:]
            data = ds.variables['z'][:]
        
        # 将 MaskedArray 转换为普通 NumPy 数组，保留 nan 作为缺失值
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
            
        return data.astype(np.float32), x_coords, y_coords

    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生严重错误: {e}")
        return None


def visualize_sic_map(input_file: str) -> None:
    """
    主函数，负责读取数据并生成地理空间可视化图像。

    Args:
        input_file (str): 输入的 .nc 文件路径。
    """
    # --- 1. 读取数据 ---
    result = read_geospatial_netcdf(input_file)
    if result is None:
        print("因数据读取失败，程序终止。")
        return

    sic_data, x_coords, y_coords = result

    # --- 2. 定义投影和地理范围 ---
    # 这是标准的 NSIDC EASE-Grid North 投影参数
    PROJECTION = ccrs.Stereographic(
        central_latitude=90.0, central_longitude=-45.0, true_scale_latitude=70.0,
        globe=ccrs.Globe(semimajor_axis=6378137.0, inverse_flattening=298.257223563)
    )
    
    # 根据坐标数据计算地图的显示范围
    GRID_EXTENT = [
        x_coords.min(), x_coords.max(),
        y_coords.min(), y_coords.max()
    ]
    
    # --- 3. 数据方向处理 ---
    data_oriented = sic_data[:]

    # --- 4. 创建图表 ---
    # 设置一个适合极地投影的图形尺寸
    fig, ax = plt.subplots(1, 1, figsize=(10, 12), subplot_kw={'projection': PROJECTION})
    
    # 定义颜色映射和范围
    cmap = plt.get_cmap('Blues_r')
    norm = colors.Normalize(vmin=0, vmax=100)
    
    ax.set_extent(GRID_EXTENT, crs=PROJECTION)
    ax.set_facecolor('lightgray') # 缺失数据显示为灰色

    # 绘制海冰数据
    im = ax.imshow(data_oriented, cmap=cmap, norm=norm, 
                   extent=GRID_EXTENT, transform=PROJECTION, zorder=1, origin='lower')
    
    # 添加地理特征
    land_feature = cfeature.LAND.with_scale('50m')
    ax.add_feature(land_feature, zorder=2, edgecolor='black', facecolor='#D2B48C') # 棕褐色陆地
    ax.coastlines(resolution='50m', zorder=3, linewidth=0.8)

    # 添加网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}
    
    # --- 5. 美化图表 ---
    # 从文件名中提取日期用于标题
    try:
        # 适用于 asi-AMSR2-n3125-20250715-v5.4.nc 这样的格式
        date_str = os.path.basename(input_file).split('-')[3]
        title = f'Sea Ice Concentration on {date_str}'
    except IndexError:
        title = 'Sea Ice Concentration'
    
    ax.set_title(title, fontsize=16, pad=20)
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.04, pad=0.05)
    cbar.set_label('Sea Ice Concentration (%)', fontsize=12)
    
    # --- 6. 保存并显示图像 ---
    output_filename = os.path.basename(input_file).replace('.nc', '.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"\n可视化图像已成功保存为: '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="使用 Cartopy 高质量可视化单个海冰密集度（SIC）NetCDF 文件。"
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True, 
        help='需要可视化的单个原始 .nc 文件的路径。'
    )
    
    args = parser.parse_args()
    visualize_sic_map(args.input_file)