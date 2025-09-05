# file: utils.py

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime
from tqdm import tqdm
import torch.nn.functional as F
import traceback


def unfold_StackOverChannel(img, patch_size):
    """
    将图像切分成多个小块并沿着通道堆叠
    Args:
        img (N, *, C, H, W): 最后两个维度必须是空间维度
        patch_size(k_h,k_w): 长度为2的元组，就是configs.patch_size
    Returns:
        output (N, *, C*k_h*k_w, H/k_h, W/k_w)
    """
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5
    if patch_size[0] == 1 and patch_size[1] == 1:
        return img

    pt = img.unfold(-2, size=patch_size[0], step=patch_size[0])
    pt = pt.unfold(-2, size=patch_size[1], step=patch_size[1]).flatten(-2)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert pt.size(-3) == img.size(-3) * patch_size[0] * patch_size[1]
    return pt


def fold_tensor(tensor, output_size, patch_size):
    """
    用non-overlapping的块重建图像
    Args:
        input tensor shape (N, *, C*k_h*k_w, h, w)
        output_size: (H, W)，要重建的原始图像的大小
        patch_size: (k_h, k_w)
        请注意，对于non-overlapping的滑动窗口，通常stride等于patch_size
    Returns:
        output (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    n_dim = len(tensor.shape)
    assert n_dim == 4 or n_dim == 5

    if patch_size[0] == 1 and patch_size[1] == 1:
        return tensor

    # 展平输入
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor

    # 使用 F.fold 函数进行重建
    folded = F.fold(
        f.flatten(-2),
        output_size=output_size,
        kernel_size=patch_size,
        stride=patch_size,
    )

    if n_dim == 5:
        folded = folded.view(tensor.size(0), tensor.size(1), *folded.shape[1:])

    return folded


def prepare_inputs_targets(
    len_time, input_gap, input_length, pred_shift, pred_gap, pred_length, samples_gap
):
    """
    计算输入和目标序列的索引 (滑动窗口).
    """
    input_span = input_gap * (input_length - 1) + 1
    pred_span = pred_gap * (pred_length - 1) + 1
    total_span_days = input_span + pred_shift + pred_span - 1

    # 检查请求的序列总天数是否超过了可用的时间长度
    if total_span_days > len_time:
        return np.array([]), np.array([])
    
    max_n_sample = len_time - total_span_days + 1
    if max_n_sample <= 0:
        return np.array([]), np.array([])

    all_indices = []
    # 我们不能再用简单的numpy广播，因为需要逐个验证样本的有效性
    for i in range(0, max_n_sample, samples_gap):
        input_ind = i + np.arange(0, input_span, input_gap)
        target_ind = i + input_span + pred_shift -1 + np.arange(0, pred_span, pred_gap)
        all_indices.append(np.concatenate([input_ind, target_ind]))

    if not all_indices:
        return np.array([]), np.array([])

    all_indices = np.array(all_indices)
    idx_inputs = all_indices[:, :input_length]
    idx_targets = all_indices[:, input_length:]
    
    return idx_inputs, idx_targets


class ConsolidatedSICDataset(Dataset):
    """
    一个从预先合并好的单个 NetCDF 文件中高效读取海冰密集度数据的 PyTorch 数据集类。

    该版本能够识别并跳过那些因原始数据缺失而导致不完整的样本。

    """
    def __init__(
        self,
        consolidated_file_path,
        start_time,
        end_time,
        input_length,
        pred_length,
        input_gap = 1,
        pred_shift = 1,
        pred_gap = 1,
        samples_gap = 1,
        preload_data = False,
    ):
        """
        Args:
            consolidated_file_path (str): 预处理合并后的大 .nc 文件路径.
            start_time (int): 数据集的开始日期 (YYYYMMDD).
            end_time (int): 数据集的结束日期 (YYYYMMDD).
            ... (其他参数同前)
            preload_data (bool): 如果为 True，将数据全部加载到内存中以加快访问速度。
                                 如果为 False，数据将按需从磁盘读取，以节省内存。
        """
        super().__init__()
        self.preload_data = preload_data
        self.consolidated_file_path = consolidated_file_path
        self.data_handle = None # 用于持有文件句柄
        self.input_length = input_length
        self.pred_length = pred_length

        self.preload_data = preload_data

        # 1. 打开 .nc 文件。注意：我们不再使用 'with' 语句，以便在需要时保持文件打开。
        print(f"正在打开文件: '{consolidated_file_path}'")
        full_data = xr.open_dataset(self.consolidated_file_path)
        
        # 如果不是预加载模式，需要保持文件句柄
        if not self.preload_data:
            self.data_handle = full_data

        # 加载缺失日期列表
        if 'missing_dates' in full_data.variables:
            self.missing_dates = set(full_data['missing_dates'].values)
        else:
            self.missing_dates = set()
            
        # 生成请求的完整日期序列
        self.full_requested_times = self._generate_full_time_list(start_time, end_time)
        
        # 筛选需要的时间片
        time_slice = slice(start_time, end_time)
        dataset_slice = full_data.sel(time=time_slice)
        
        if self.preload_data:
            print(f"预加载模式: 正在从 '{consolidated_file_path}' 中加载 {start_time} 到 {end_time} 的数据到内存...")
            self.data_or_accessor = dataset_slice.sea_ice_conc.values
            print(f"数据加载完成。可用数据形状: {self.data_or_accessor.shape}")
            # 数据已在内存，可以关闭文件句柄
            full_data.close()
        else:
            print(f"实时读取模式: 数据将按需从磁盘 '{consolidated_file_path}' 读取。")
            # 只保留对 xarray DataArray 的引用（accessor），不实际加载数据
            self.data_or_accessor = dataset_slice.sea_ice_conc
            # 注意：此时 self.data_handle 仍然持有打开的文件句柄

        # 以下部分逻辑不变，因为它们只依赖于时间坐标，不依赖于庞大的海冰数据
        self.available_times = dataset_slice.time.values
        self.available_times_set = set(self.available_times)

        # 2. 生成所有潜在的样本索引
        potential_idx_inputs, potential_idx_targets = self._generate_potential_samples(
            input_gap, input_length, pred_shift, pred_gap, pred_length, samples_gap
        )

        if potential_idx_inputs.shape[0] == 0:
            self.idx_inputs, self.idx_targets = np.array([]), np.array([])
        else:
            # 3. 过滤掉包含缺失日期的无效样本
            print("正在过滤无效样本...")
            valid_indices = self._filter_invalid_samples(potential_idx_inputs, potential_idx_targets)

            if not valid_indices:
                self.idx_inputs, self.idx_targets = np.array([]), np.array([])
            else:
                self.idx_inputs = potential_idx_inputs[valid_indices]
                self.idx_targets = potential_idx_targets[valid_indices]

        # 如果是预加载模式，此时文件已经关闭。如果不是，文件还开着。
        # 这里为了逻辑统一，把full_data的关闭放在这里，但实际上如果是预加载，前面已经关了。
        if self.preload_data:
            pass # 已经关闭
        else:
            # 在非预加载模式下，dataset_slice 及其引用的 full_data (self.data_handle) 必须保持打开
            # 我们将在 close() 方法中关闭它
            pass

        if self.__len__() == 0:
            print("警告: 当前参数配置下，无法生成任何有效样本。请检查时间范围、序列长度和数据缺失情况。")
        else:
            print(f"数据集初始化成功，共生成 {self.__len__()} 个有效样本。")

    def _generate_full_time_list(self, start, end):
        start_dt = datetime.datetime.strptime(str(start), "%Y%m%d")
        end_dt = datetime.datetime.strptime(str(end), "%Y%m%d")
        return [int(d.strftime("%Y%m%d")) for d in [start_dt + datetime.timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]]

    def _generate_potential_samples(self, input_gap, input_length, pred_shift, pred_gap, pred_length, samples_gap):
        len_time = len(self.full_requested_times)
        return prepare_inputs_targets(
            len_time, input_gap, input_length, pred_shift, pred_gap, pred_length, samples_gap
        )
    
    def _filter_invalid_samples(self, potential_idx_inputs, potential_idx_targets):
        valid_sample_indices = []
        num_potential_samples = len(potential_idx_inputs)
        for i in tqdm(range(num_potential_samples), desc="过滤样本"):
            all_indices = np.concatenate([potential_idx_inputs[i], potential_idx_targets[i]])
            required_dates = [self.full_requested_times[idx] for idx in all_indices]
            if all(date in self.available_times_set for date in required_dates):
                valid_sample_indices.append(i)
        print(f"过滤完成。在 {num_potential_samples} 个潜在样本中，有 {len(valid_sample_indices)} 个是有效的。")
        return valid_sample_indices

    def __len__(self):
        return len(self.idx_inputs)

    def __getitem__(self, index):
        # 1 & 2. 获取样本日期
        input_dates = [self.full_requested_times[i] for i in self.idx_inputs[index]]
        target_dates = [self.full_requested_times[i] for i in self.idx_targets[index]]

        # 3. 在 self.available_times 中找到这些日期的实际位置索引
        input_indices_in_available = np.searchsorted(self.available_times, input_dates)
        target_indices_in_available = np.searchsorted(self.available_times, target_dates)

        # 4. 根据模式从内存或磁盘获取数据
        if self.preload_data:
            # 直接从内存中的 NumPy 数组切片
            inputs_np = self.data_or_accessor[input_indices_in_available]
            targets_np = self.data_or_accessor[target_indices_in_available]
        else:
            # 使用 xarray 的 isel (integer selection) 从磁盘读取
            # .values 会触发实际的磁盘IO
            inputs_np = self.data_or_accessor.isel(time=input_indices_in_available).values
            targets_np = self.data_or_accessor.isel(time=target_indices_in_available).values

        # 增加通道维度 (T, C, H, W)，其中 C=1
        inputs_np = inputs_np[:, np.newaxis, :, :]
        targets_np = targets_np[:, np.newaxis, :, :]

        return torch.from_numpy(inputs_np).float(), torch.from_numpy(targets_np).float()

    def close(self):
        """
        如果数据集处于实时读取模式，则关闭 NetCDF 文件句柄。
        在完成数据集的使用后调用此方法是个好习惯。
        """
        if self.data_handle is not None:
            print(f"正在关闭文件句柄: {self.consolidated_file_path}")
            self.data_handle.close()
            self.data_handle = None
    def GetInputs(self):
        """
        获取数据集中所有样本的输入数据。
        此方法会遍历整个数据集，如果数据未预加载到内存，可能会比较耗时。

        Returns:
            np.ndarray: 一个 NumPy 数组，形状为 (num_samples, input_length, 1, H, W)。
        """
        if self.__len__() == 0:
            # 如果数据集为空，返回一个具有正确形状但样本数为0的空数组
            h, w = self.data_or_accessor.shape[-2:]
            return np.empty((0, self.input_length, 1, h, w), dtype=np.float32)

        # 通过遍历数据集来收集所有输入样本
        all_inputs = [self.__getitem__(i)[0].numpy() for i in tqdm(range(len(self)), desc="正在提取所有输入样本")]

        # 将样本列表堆叠成一个单一的Numpy数组
        return np.stack(all_inputs, axis=0)

    def GetTargets(self):
        """
        获取数据集中所有样本的目标数据。
        此方法会遍历整个数据集，如果数据未预加载到内存，可能会比较耗时。

        Returns:
            np.ndarray: 一个 NumPy 数组，形状为 (num_samples, pred_length, 1, H, W)。
        """
        if self.__len__() == 0:
            # 如果数据集为空，返回一个具有正确形状但样本数为0的空数组
            h, w = self.data_or_accessor.shape[-2:]
            return np.empty((0, self.pred_length, 1, h, w), dtype=np.float32)

        # 通过遍历数据集来收集所有目标样本
        all_targets = [self.__getitem__(i)[1].numpy() for i in tqdm(range(len(self)), desc="正在提取所有目标样本")]

        # 将样本列表堆叠成一个单一的Numpy数组
        return np.stack(all_targets, axis=0)

    def GetTimes(self):
        """
        获取数据集中每个样本对应的输入和目标日期序列。

        Returns:
            np.ndarray: 一个 NumPy 数组，形状为 (num_samples, input_length + pred_length)，
                        其中的值为 YYYYMMDD 格式的整数日期。
        """
        if self.__len__() == 0:
            # 如果数据集为空，返回一个具有正确形状但样本数为0的空数组
            return np.empty((0, self.input_length + self.pred_length), dtype=int)
        # 将完整的日期请求列表转换为 NumPy 数组以便于高级索引
        times_array = np.array(self.full_requested_times)

        # 对于非空数据集，索引数组保证是2D的
        all_indices = np.concatenate([self.idx_inputs, self.idx_targets], axis=1)

        # 使用索引从时间数组中选取出所有样本对应的日期
        return times_array[all_indices]


# --- 使用示例 ---
if __name__ == '__main__':

    # --- 演示2: 实时读取模式 (preload_data=False)，新行为 ---
    print("\n--- 演示2: 实时读取模式 (内存占用低，访问有IO开销) ---")
    dataset_realtime = ConsolidatedSICDataset(
        consolidated_file_path='./full_sic.nc',
        start_time=20120812,
        end_time=20120816,
        input_length=2,
        input_gap=1,
        pred_gap=1,
        pred_shift=1,
        pred_length=2,
        samples_gap=1,
        preload_data=False # 明确设置为 False
    )

    if len(dataset_realtime) > 0:
        # 在多进程加载(num_workers > 0)时，xarray句柄可能存在序列化问题
        # 通常建议在实时读取模式下使用 num_workers=0
        loader_realtime = DataLoader(dataset_realtime, batch_size=2, shuffle=False, num_workers=0)
        batch_inputs, batch_targets = next(iter(loader_realtime))
        print(batch_inputs)
        print(f"\n成功从 [实时读取] DataLoader 获取一个批次！")
        print(f"输入批次的形状 (B, T_in, C, H, W): {batch_inputs.shape}")
        print(f"输出批次的形状 (B, T_out, C, H, W): {batch_targets.shape}")
    else:
        print("\n未能生成任何样本。")

    # !! 重要：在实时读取模式下，使用完毕后应手动关闭文件句柄
    dataset_realtime.close()
    
    print("\n--- 演示结束 ---")