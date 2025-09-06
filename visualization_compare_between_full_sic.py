# file: visualization_compare_between_full_sic.py

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse

def find_time_index(nc_file, target_date):
    """在 NetCDF 文件中根据日期 (YYYYMMDD) 查找对应的时间索引。"""
    time_var = nc_file.variables.get('time')
    if time_var is None:
        print("错误: 文件中未找到 'time' 变量。")
        return None
    try:
        all_dates = time_var[:]
        match_indices = np.where(all_dates == target_date)[0]
        if len(match_indices) > 0:
            return match_indices[0]
        else:
            return None
    except Exception as e:
        print(f"查找时间索引时出错: {e}")
        return None

# <<< MODIFIED: 函数签名增加了 no_land 参数
def visualize_comparison_v4(original_file_path, consolidated_file_path, target_date, no_land=False):
    try:
        with nc.Dataset(original_file_path, 'r') as src_nc:
            data_masked = src_nc.variables['z'][:]
            if isinstance(data_masked, np.ma.MaskedArray):
                original_data = data_masked.filled(np.nan)
            else:
                original_data = np.array(data_masked)
            x_coords = src_nc.variables['x'][:]
            y_coords = src_nc.variables['y'][:]
    except Exception as e:
        print(f"无法读取原始文件 '{original_file_path}': {e}")
        return
    try:
        with nc.Dataset(consolidated_file_path, 'r') as dest_nc:
            time_idx = find_time_index(dest_nc, target_date)
            if time_idx is None:
                print(f"错误: 在合并文件 '{consolidated_file_path}' 中找不到日期 {target_date}。")
                return
            interpolated_data = dest_nc.variables['sea_ice_conc'][time_idx, :, :] * 100.0
    except Exception as e:
        print(f"无法读取合并文件 '{consolidated_file_path}': {e}")
        return

    PROJECTION = ccrs.Stereographic(
        central_latitude=90.0, central_longitude=-45.0, true_scale_latitude=70.0,
        globe=ccrs.Globe(semimajor_axis=6378137.0, inverse_flattening=298.257223563)
    )
    pixel_size_x = abs(x_coords[1] - x_coords[0])
    pixel_size_y = abs(y_coords[1] - y_coords[0])
    GRID_EXTENT = [
        x_coords.min() - pixel_size_x / 2.0, x_coords.max() + pixel_size_x / 2.0,
        y_coords.min() - pixel_size_y / 2.0, y_coords.max() + pixel_size_y / 2.0
    ]

    original_data_flipped = original_data
    interpolated_data_oriented = interpolated_data

    print(f"正在为日期 {target_date} 生成高质量对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': PROJECTION})

    cmap = plt.get_cmap('Blues_r')
    norm = colors.Normalize(vmin=0, vmax=100)

    # <<< MODIFIED: 内部函数签名增加了 no_land 参数
    def plot_map(ax, data, title, no_land_flag):
        ax.set_extent(GRID_EXTENT, crs=PROJECTION)

        ax.set_facecolor('lightgray')
        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, cmap=cmap, norm=norm, 
                       extent=GRID_EXTENT, transform=PROJECTION, zorder=1, origin='lower')

        # <<< MODIFIED: 使用 if 条件判断是否绘制陆地和经纬线
        if not no_land_flag:
            land_feature = cfeature.LAND.with_scale('50m')
            ax.add_feature(land_feature, zorder=2, edgecolor='black', facecolor='#D2B48C') # Tan land
            ax.coastlines(resolution='50m', zorder=3, linewidth=0.8)

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10, 'color': 'gray'}
            gl.ylabel_style = {'size': 10, 'color': 'gray'}

        ax.set_title(title, fontsize=14, pad=20)
        return im

    # <<< MODIFIED: 在调用 plot_map 时传入 no_land 参数
    im1 = plot_map(axes[0], original_data_flipped, f'Original Data ({target_date})\n(Missing data is gray)', no_land)
    im2 = plot_map(axes[1], interpolated_data_oriented, f'Interpolated Data ({target_date})', no_land)

    fig.colorbar(im2, ax=axes, orientation='horizontal', fraction=0.04, pad=0.05, label='Sea Ice Concentration (%)')
    plt.suptitle(f'Interpolation Comparison for {target_date}', fontsize=18, y=0.98)

    output_filename = f'interpolation_comparison_{target_date}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图像已保存为: {output_filename}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化海冰密集度插值前后的高质量对比结果。")
    parser.add_argument('--original_file', type=str, required=True, help='单个原始每日 .nc 文件的路径。')
    parser.add_argument('--consolidated_file', type=str, required=True, help='合并并插值后的大 .nc 文件的路径。')
    parser.add_argument('--date', type=int, required=True, help='要可视化的日期 (格式: YYYYMMDD)。')
    # <<< ADDED: 添加新的命令行参数 --no_land
    parser.add_argument('--no_land', action='store_true', help='取消显示陆地和经纬线，只显示海冰数据。')

    args = parser.parse_args()

    # <<< MODIFIED: 将新的 no_land 参数传递给可视化函数
    visualize_comparison_v4(
        original_file_path=args.original_file,
        consolidated_file_path=args.consolidated_file,
        target_date=args.date,
        no_land=args.no_land
    )