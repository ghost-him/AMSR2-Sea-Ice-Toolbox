# AMSR2 海冰数据预处理工具箱 🧊
[English Document](readme-en.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

一个专为 **AMSR2 海冰密集度数据** 设计的高效、易用的预处理流水线。从数据下载、时空插值、到生成可直接用于深度学习的 PyTorch `DataLoader`，一站式解决所有预处理难题。

---

## ✨ 效果一览：从原始数据到可用特征

| **核心功能：高性能时空插值** | **辅助功能：陆地掩码生成** |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="asset/interpolation_comparison_20240417.png" alt="插值效果对比" width="450"> | <img src="asset/land_mask.png" alt="陆地掩码" width="450"> |
| *左：原始数据 (含大量缺失值)；右：插值后数据* | *基于长时间序列数据统计生成的精准陆地/海洋掩码* |

---

## 🎯 目标数据源

本工具箱**专门**为以下特定数据集进行优化和设计：

-   **数据中心**: [University of Bremen](https://data.seaice.uni-bremen.de/amsr2/)
-   **产品**: `asi_daygrid_swath`
-   **分辨率**: `n3125`
-   **格式**: `netcdf`

直接访问链接：[uni-bremen.de/amsr2/asi_daygrid_swath/n3125/netcdf/](https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n3125/netcdf/)

---

## 🚀 快速开始 (3步完成)

### 1. 环境准备

克隆本仓库并安装依赖项：

```bash
git clone https://github.com/ghost-him/AMSR2-Sea-Ice-Toolbox.git
cd AMSR2-Sea-Ice-Toolbox
pip install numpy xarray netcdf4 torch torchvision matplotlib cartopy tqdm numba requests python-dateutil
mkdir data
```

### 2. 数据处理

下载并整合数据。此步骤将自动下载、合并、并使用高性能插值算法填充缺失值。

```bash
# 下载指定日期范围的数据 (例如: 2024年4月)
python download.py --start_date 20240401 --end_date 20240430 -o data

# 将下载的日度数据整合为一个文件，并进行插值
python consolidate_data.py \
    --source_dir ./data \
    --start_date 20240401 \
    --end_date 20240430 \
    --output_file ./sea_ice_concentration_202404.nc
```

### 3. 制作陆地掩码 (可选)

为了在训练中忽略陆地像素，您可以根据长时间序列的数据创建一个掩码。数据时间跨度越长，掩码越精确。

```bash
# 使用下载的数据创建掩码，阈值为0.85
python create_mask.py \
    --source_dir ./data \
    --start_date 20240401 \
    --end_date 20240430 \
    --output_file ./land_mask.nc \
    --threshold 0.85
```

### 4.数据可视化

提供了多种可视化工具，帮助您直观地检查数据质量、掩码效果和插值结果。

#### 可视化单日原始数据

使用 `visualization.py` 脚本，可以对单个 `netcdf` 文件进行地理投影，并将其绘制成带有海岸线和经纬网格的地图。

```bash
python visualization.py --input_file ./data/asi-AMSR2-n3125-20240417-v5.4.nc
```

#### 可视化陆地掩码

使用 `visualize_mask.py` 脚本，可以将 `create_mask.py` 生成的二值化掩码和 `NaN` 频率图进行可视化。

```bash
python visualize_mask.py \
    --input_file ./land_mask.nc \
    --output_file ./land_mask_visualization.png
```

#### 可视化插值前后对比

使用 `visualization_compare_between_full_sic.py` 脚本，可以直观地对比同一天原始数据和插值后数据的差异，以评估插值效果。

```bash
python visualization_compare_between_full_sic.py \
    --original_file ./data/asi-AMSR2-n3125-20240417-v5.4.nc \
    --consolidated_file ./sea_ice_concentration_202404.nc \
    --date 20240417
```



### 5. 加载到 PyTorch (含掩码应用)

在 `DataLoader` 中加载数据，并应用之前生成的陆地掩码。

```python
import torch
import xarray as xr
from utils import ConsolidatedSICDataset
from torch.utils.data import DataLoader

def load_mask_from_nc(mask_path):
    """从netCDF文件加载land_mask并转换为PyTorch张量。"""
    with xr.open_dataset(mask_path) as ds_mask:
        # 提取'land_mask'变量，转换为float32类型的torch tensor
        land_mask_tensor = torch.from_numpy(ds_mask['land_mask'].values).float()
    print(f"成功加载掩码: {mask_path}, 形状: {land_mask_tensor.shape}")
    return land_mask_tensor

# 1. 加载掩码
# 假设掩码文件在 ./land_mask.nc
try:
    mask_tensor = load_mask_from_nc('./land_mask.nc')
    # 将掩码移动到合适的设备 (例如 'cuda' 或 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_tensor = mask_tensor.to(device)
except FileNotFoundError:
    print("警告: 掩码文件 './land_mask.nc' 未找到。将不使用掩码。")
    mask_tensor = None

# 2. 创建数据集实例
dataset = ConsolidatedSICDataset(
    consolidated_file_path='./sea_ice_concentration_202404.nc',
    start_time=20240401,
    end_time=20240430,
    input_length=5,   # 使用过去5天的数据
    pred_length=3     # 预测未来3天的数据
)

# 3. 创建DataLoader(实际工作中推荐设置num_workers=8)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 4. 迭代数据并应用掩码
inputs, targets = next(iter(data_loader))
inputs, targets = inputs.to(device), targets.to(device)

print(f"应用掩码前 - 输入形状: {inputs.shape}")
print(f"应用掩码前 - 目标形状: {targets.shape}")

if mask_tensor is not None:
    # 使用广播机制将掩码应用到输入和目标数据
    # 掩码形状: (H, W), 数据形状: (B, T, C, H, W)
    # unsqueeze将掩码变为 (1, 1, 1, H, W) 以匹配数据维度
    inputs = inputs * mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    targets = targets * mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    print("\n掩码已成功应用！")

print(f"\n应用掩码后 - 输入形状: {inputs.shape}")
print(f"应用掩码后 - 目标形状: {targets.shape}")

# 现在，inputs 和 targets 中的陆地部分已经被置零
# 可以将它们输入到您的模型中进行训练
```

---

## 🛠️ 工具箱详解

### 核心功能

#### 1. `consolidate_data.py` - 数据整合与时空插值
将分散的日度 NetCDF 文件合并为单一时间序列文件，并采用 **Numba JIT 加速** 的时空高斯加权算法填充缺失值。

-   **高性能**: C 语言级别的插值计算速度。
-   **内存友好**: 分批处理机制，轻松处理数十年数据。
-   **可配置**: 自由调整时空搜索窗口、权重等参数。

<details>
<summary><b>查看详细参数</b></summary>

```bash
python consolidate_data.py \
    --source_dir ./data \
    --start_date 20120812 \
    --end_date 20120831 \
    --output_file ./full_sic.nc \
    --batch_size 30 \
    --time_window 3 \
    --space_window 7 \
    --alpha 15.0 \
    --sigma 7.0
```
- `batch_size`: 一次处理的天数，影响内存占用。
- `time_window`: 时间搜索半径（天）。
- `space_window`: 空间搜索半径（像素）。
- `alpha`: 时空平衡因子。
- `sigma`: 高斯核带宽。

</details>

#### 2. `utils.py` - 深度学习数据加载器
提供 `ConsolidatedSICDataset` 类，一个专为时空预测任务设计的 PyTorch Dataset。

-   **灵活**: 自定义输入/输出序列长度、时间间隔等。
-   **高效**: 支持数据预加载到内存或实时从磁盘读取。
-   **健壮**: 自动过滤含过多缺失值的无效样本。

<details>
<summary><b>查看 `ConsolidatedSICDataset` 使用示例</b></summary>

```python
from utils import ConsolidatedSICDataset

dataset = ConsolidatedSICDataset(
    consolidated_file_path='./full_sic.nc',
    start_time=20120812,
    end_time=20120816,
    input_length=2,     # 输入序列长度
    input_gap=1,        # 输入序列中每帧的时间间隔
    pred_gap=1,         # 预测序列中每帧的时间间隔
    pred_shift=1,       # 预测序列相对于输入的起始偏移
    pred_length=2,      # 预测序列长度
    samples_gap=1,      # 样本之间的时间步长
    preload_data=False  # False: 实时读取, True: 预加载到内存
)
```

</details>

### 辅助工具

<details>
<summary><b>`download.py` - 数据下载</b></summary>

自动从 [University of Bremen](https://data.seaice.uni-bremen.de/amsr2/) 数据中心批量下载指定日期范围的 `asi_daygrid_swath` NetCDF 数据。

-   **支持断点续传**: 如果文件已存在，则默认跳过，不会重复下载。
-   **自动创建目录**: 将数据保存在指定的输出目录中。
-   **错误处理**: 能优雅处理文件未找到 (404) 或其他网络问题。

**使用方法:**
```bash
# 下载2023年1月1日至1月10日的数据到 ./data 目录
python download.py --start_date 20230101 --end_date 20230110 --output-dir ./data

# 如果需要强制覆盖已存在的文件
python download.py --start_date 20230101 --end_date 20230110 -o ./data -f
```
</details>

<details>
<summary><b>`create_mask.py` - 陆地掩码生成</b></summary>

通过统计长时间序列数据中每个像素点出现缺失值（NaN）的频率，来生成一个高精度的陆地/海洋掩码。当一个点的缺失频率超过设定的阈值时，它被标记为陆地。这里推荐使用全部的数据来制作，数据量越大，则越精确。

-   **统计驱动**: 基于真实数据分布，比通用固定掩码更精确。
-   **可配置阈值**: 灵活调整判断为陆地的标准。
-   **双重输出**: 同时保存最终的二进制掩码（0/1）和原始的NaN频率图，便于分析。

**使用方法:**
```bash
# 基于 ./data 目录中2013年至2023年的数据，创建掩码
# 使用默认阈值 0.85 (即一个点在85%以上的时间里是NaN，则视为陆地)
python create_mask.py \
    --source_dir ./data \
    --start_date 20130101 \
    --end_date 20231231 \
    --output_file ./land_mask_85pct.nc \
    --threshold 0.85
```
</details>

<details>
<summary><b>`visualize_mask.py` - 掩码可视化</b></summary>

将 `create_mask.py` 生成的 `.nc` 掩码文件可视化，同时展示最终的二进制陆地掩码和NaN频率比率图。

**使用方法:**
```bash
python visualize_mask.py \
    --input_file ./land_mask_85pct.nc \
    --output_file ./land_mask_visualization.png
```
</details>

<details>
<summary><b>`visualization.py` - 单日数据可视化</b></summary>

使用 **Cartopy** 库对单个原始数据文件进行地理空间投影（北极立体投影），生成带有海岸线、经纬网格的高质量地图。

**使用方法:**
```bash
python visualization.py --input_file ./data/asi-AMSR2-n3125-20240417-v5.4.nc
```
</details>

<details>
<summary><b>`visualization_compare_between_full_sic.py` - 插值效果对比</b></summary>

并排比较同一天的原始数据（含缺失值）和经过 `consolidate_data.py` 插值后的数据，直观评估插值算法的效果。

**使用方法:**
```bash
python visualization_compare_between_full_sic.py \
    --original_file ./data/asi-AMSR2-n3125-20240417-v5.4.nc \
    --consolidated_file ./sea_ice_concentration_202404.nc \
    --date 20240417
```
</details>

<details>
<summary><b>`get_dims.py` - 获取数据维度</b></summary>

快速读取指定目录下第一个 `.nc` 文件的 `z` 变量（海冰密集度数据）的维度（高度和宽度），这在确定ROI（感兴趣区域）的索引范围时非常有用。

**使用方法:**
```bash
# 脚本默认会查找 ./data 目录
python get_dims.py
```
</details>

<details>
<summary><b>`find_several_data_in_target_area.py` - ROI数据质量分析</b></summary>

在给定的时间范围和指定的矩形区域（ROI）内，逐日检查数据，并报告哪些天在ROI内出现了缺失值。

-   **精确定位**: 帮助识别特定区域的数据质量问题。
-   **可视化辅助**: 会自动生成一幅图像，在全图中用红框标出你所定义的ROI位置，以供核对。
-   **量化报告**: 详细列出每个问题日期的缺失像素数量和百分比。

**使用方法:**
```bash
# 检查在2024年4月，Y轴索引100-200，X轴索引300-400的区域内的数据缺失情况
python find_several_data_in_target_area.py \
    --source_dir ./data \
    --start_date 20240401 \
    --end_date 20240430 \
    --ymin 100 --ymax 200 \
    --xmin 300 --xmax 400
```
</details>

---

## 💡 技术亮点

-   **🚀 高性能计算**: 插值核心算法使用 **Numba JIT** 编译，大幅提升处理速度。
-   **🌍 精准地理可视化**: 采用 **Cartopy** 进行北极立体投影，自动绘制海岸线、经纬网。
-   **🧠 深度学习就绪**: 无缝集成的 PyTorch `Dataset` 和 `DataLoader`，专注于模型本身。
-   **💾 内存优化**: 智能的批处理和数据加载策略，从容应对海量数据。

---

## 📜 许可证

本项目遵循 [MIT License](LICENSE)。

## 🙏 致谢

-   **数据来源**: [University of Bremen Sea Ice Remote Sensing Group](https://seaice.uni-bremen.de/start/)
-   **AI 辅助开发**: Google Gemini 2.5 Pro, Anthropic Claude 4 Sonnet
-   **核心依赖**: NumPy, xarray, PyTorch, Cartopy, Numba 等优秀的开源项目。

