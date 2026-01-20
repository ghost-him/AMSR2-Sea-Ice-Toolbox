# file: download.py

import os
import argparse
import requests
from datetime import date, timedelta, datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# --- 配置区 ---
# URL 模板，{year} 和 {date_str} 是将要被替换的占位符
URL_TEMPLATE = "https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n3125/netcdf/{year}/asi-AMSR2-n3125-{date_str}-v5.4.nc"

def download_single_file(current_date, output_dir, force_overwrite):
    """
    下载单个日期的数据文件。
    """
    year = current_date.strftime("%Y")
    date_str_format = current_date.strftime("%Y%m%d")
    
    # 构建完整 URL 和本地文件名
    url = URL_TEMPLATE.format(year=year, date_str=date_str_format)
    filename = f"asi-AMSR2-n3125-{date_str_format}-v5.4.nc"
    filepath = os.path.join(output_dir, filename)
    
    # 检查文件是否已存在 (除非强制覆盖)
    if not force_overwrite and os.path.exists(filepath):
        print(f"文件 {filename} 已存在，跳过下载。")
        return True
    
    # print(f"正在下载 {filename} ...")
    
    # 发送下载请求
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 如果状态码不是 2xx，则抛出 HTTPError 异常

        # 保存文件
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✅ 下载成功: {filename}")
        return True

    except requests.exceptions.HTTPError as e:
        # 更优雅地处理 HTTP 错误
        if e.response.status_code == 404:
            print(f"❌ 下载失败: {filename} (文件未在服务器上找到 - 404 Not Found)")
        else:
            print(f"❌ 下载失败: {filename} (HTTP错误: {e})")
        return False
    except requests.exceptions.RequestException as e:
        # 捕获网络连接、超时等其他错误
        print(f"❌ 下载失败: {filename} (网络错误: {e})")
        return False

def download_files(start_date_str: str, end_date_str: str, output_dir: str = '.', force_overwrite: bool = False, max_workers: int = 4):
    """
    根据给定的开始和结束日期，下载指定范围内的海冰数据文件。

    :param start_date_str: 开始日期字符串，格式为 'YYYYMMDD'
    :param end_date_str: 结束日期字符串，格式为 'YYYYMMDD'
    :param output_dir: 文件保存的目录，默认为当前目录
    :param force_overwrite: 是否覆盖已存在的文件，默认为 False
    :param max_workers: 并行下载的任务数
    """
    # 1. 解析日期字符串为 date 对象
    try:
        start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
        end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
    except ValueError:
        print("错误：日期格式不正确。请输入 YYYYMMDD 格式，例如：20230101")
        return

    # 确保开始日期不晚于结束日期
    if start_date > end_date:
        print("错误：开始日期不能晚于结束日期。")
        return

    # 2. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"文件将保存到: {os.path.abspath(output_dir)}")
    print(f"使用 {max_workers} 个并行进程进行下载...")

    # 3. 构造日期列表
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    # 4. 使用进程池进行下载
    download_func = partial(download_single_file, output_dir=output_dir, force_overwrite=force_overwrite)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_func, date_list)


def main():
    """
    主函数，用于解析命令行参数并启动下载进程。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="海冰数据批量下载脚本。根据指定的日期范围下载数据文件。",
        epilog="示例: python download.py 20230101 20230110 -o ./data"
    )
    
    # 添加必须的位置参数：开始日期和结束日期
    parser.add_argument(
        "--start_date",
        type=str,
        help="下载开始日期，格式为 YYYYMMDD (例如: 20230101)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="下载结束日期，格式为 YYYYMMDD (例如: 20230110)"
    )
    
    # 添加可选参数：输出目录
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=".",
        help="保存文件的目录 (默认为当前目录)"
    )
    
    # 添加可选参数：是否强制覆盖
    parser.add_argument(
        "-f", "--force",
        action="store_true", # 当出现此参数时，其值为 True
        help="如果文件已存在，则强制覆盖重新下载"
    )

    # 添加可选参数：进程数
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=5,
        help="并行下载的进程数 (默认为 5)"
    )

    # 解析参数
    args = parser.parse_args()
    
    print("--- 海冰数据批量下载脚本 ---")
    print("数据来源: data.seaice.uni-bremen.de")
    
    # 执行下载函数
    download_files(args.start_date, args.end_date, args.output_dir, args.force, args.processes)
    
    print("\n--- 所有下载任务已完成 ---")


if __name__ == "__main__":
    main()