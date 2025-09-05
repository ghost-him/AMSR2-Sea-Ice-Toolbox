# file: get_dims.py

import os
import re
import netCDF4 as nc
import sys

def get_first_nc_file_dimensions(data_root_path):
    """
    查找指定目录下的第一个匹配的NetCDF文件，并输出其'z'变量的维度。
    
    Args:
        data_root_path (str): 存放 .nc 文件的根目录。
    """
    first_file_path = None
    # 使用与您脚本中相同的正则表达式来查找文件
    pattern = re.compile(r"asi-AMSR2-n3125-(\d{8})-v5\.4\.nc")

    # 遍历目录，找到第一个匹配的文件就停止
    print(f"正在目录 '{data_root_path}' 中搜索数据文件...")
    for root, _, files in os.walk(data_root_path):
        for file in files:
            if pattern.match(file):
                first_file_path = os.path.join(root, file)
                break  # 找到后退出内层循环
        if first_file_path:
            break  # 找到后退出外层循环
            
    if not first_file_path:
        print(f"错误: 在目录 '{data_root_path}' 中没有找到任何匹配 'asi-AMSR2-n3125-*.nc' 格式的文件。")
        sys.exit(1) # 退出程序

    print(f"找到文件: {first_file_path}")
    print("正在读取维度信息...")

    try:
        with nc.Dataset(first_file_path, 'r') as src_nc:
            # 检查文件中是否存在名为 'z' 的变量
            if 'z' in src_nc.variables:
                # 获取 'z' 变量的维度信息
                dimensions = src_nc.variables['z'].shape
                
                # .shape 返回一个元组，通常是 (长度, 宽度) 或 (行数, 列数)
                height = dimensions[0]
                width = dimensions[1]
                
                print("\n" + "="*40)
                print("      数据变量 'z' 的完整尺寸信息")
                print("="*40)
                print(f"  长度 (Height / Rows): {height}")
                print(f"  宽度 (Width / Columns): {width}")
                print("="*40)
                print(f"\n提示: 您的 ymax 值必须小于 {height}，xmax 值必须小于 {width}。")

            else:
                print(f"错误: 在文件 {first_file_path} 中没有找到名为 'z' 的变量。")

    except Exception as e:
        print(f"读取或处理文件 {first_file_path} 时发生错误: {e}")


if __name__ == '__main__':
    source_directory = "./data" 
    get_first_nc_file_dimensions(source_directory)