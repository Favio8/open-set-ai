import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb
from sklearn.preprocessing import OneHotEncoder


def augment_ecg(x, augmentation_type='default'):
    """
    ECG信号增强函数
    支持多种增强策略，用于生成多视图用于对比学习
    
    Args:
        x: ECG信号 (C, L) - C个导联，L为信号长度
        augmentation_type: 增强类型 ('default', 'strong', 'weak')
    """
    # 复制信号
    x = x.copy()
    
    if augmentation_type == 'strong':
        # 强增强策略
        # 1. 幅度缩放（更大范围）
        scale = np.random.uniform(0.6, 1.4)
        x = x * scale
        
        # 2. 高斯噪声（更大方差）
        noise = np.random.normal(0, 0.02, x.shape)
        x = x + noise
        
        # 3. 时间偏移（更大范围）
        shift = np.random.randint(0, max(1, x.shape[1] // 5))
        x = np.roll(x, shift, axis=1)
        
        # 4. 导联随机缺失（更高概率）
        if np.random.rand() < 0.5:
            num_leads_to_drop = np.random.randint(1, max(2, x.shape[0] // 2))
            leads_to_drop = np.random.choice(x.shape[0], num_leads_to_drop, replace=False)
            x[leads_to_drop] = 0
            
    elif augmentation_type == 'weak':
        # 弱增强策略
        # 1. 小幅度缩放
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale
        
        # 2. 小幅度高斯噪声
        noise = np.random.normal(0, 0.005, x.shape)
        x = x + noise
        
        # 3. 小幅度时间偏移
        shift = np.random.randint(0, max(1, x.shape[1] // 20))
        x = np.roll(x, shift, axis=1)
        
    else:  # default
        # 原始增强策略
        # 1. 幅度缩放
        scale = np.random.uniform(0.8, 1.2)
        x = x * scale
        
        # 2. 高斯噪声
        noise = np.random.normal(0, 0.01, x.shape)
        x = x + noise
        
        # 3. 时间偏移
        shift = np.random.randint(0, max(1, x.shape[1] // 10))
        x = np.roll(x, shift, axis=1)
        
        # 4. 导联随机缺失
        if np.random.rand() < 0.3:
            lead = np.random.randint(0, x.shape[0])
            x[lead] = 0
    
    return x


def onehot_label(labels):
    labels = labels.reshape(-1, 1)
    # 兼容新版本sklearn：sparse_output替代sparse
    try:
        onehot_encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(sparse=False)
    onehot_vector = onehot_encoder.fit_transform(labels)
    tmp_list = []
    for j in range(onehot_vector.shape[0]):
        tmp_list.append(onehot_vector[j, :])
    return tmp_list


class ECGDataset_unseen(Dataset):
    def __init__(self, phase, data_dir, label_csv, leads, length):
        super(ECGDataset_unseen, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        labels_int = df['label'].values
        label_types = list(set(labels_int))
        onehot_vector = onehot_label(labels_int)
        df["label_vec"] = onehot_vector
        df = df.loc[df['split'] == phase]
        
        # -------- 过滤掉不存在的文件 --------
        valid_indices = []
        for idx, row in df.iterrows():
            file_path = os.path.join(data_dir, row['Recording'])
            if os.path.exists(file_path):
                valid_indices.append(idx)
        df = df.loc[valid_indices]
        print(f"ECGDataset_unseen ({phase}): 保留 {len(df)} 个有效样本 (out of {len(df) + len([i for i in df.index if i not in valid_indices])})")
        
        self.data_dir = data_dir
        self.labels = df
        self.nleads = leads
        data_df = pd.DataFrame()
        data_df["Recording"], data_df["label_vec"] = df["Recording"], df["label_vec"]
        self.data = data_df.values
        self.length = length

    def __getitem__(self, index: int):
        file_name, onehot_label = self.data[index]
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path, sep=",")
        ecg_data = df.values[:]
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-self.length:, :]
        result = np.zeros((self.length, self.nleads))
        result[-nsteps:, :] = ecg_data

        # -------- 返回两个增强视图用于对比学习 --------
        view1 = result.transpose()  # (C, L)
        # 应用随机增强生成第二个视图
        view2 = augment_ecg(view1, augmentation_type='default')
        
        return (torch.from_numpy(view1).float(), 
                torch.from_numpy(view2).float(), 
                torch.from_numpy(onehot_label).float())

    def __len__(self):
        return len(self.labels)


class ECGDataset_unseen_MHL_stage2(Dataset):
    def __init__(self, phase, data_dir, label_csv, leads, length):
        super(ECGDataset_unseen_MHL_stage2, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        labels_int = df['label'].values
        label_types = list(set(labels_int))
        onehot_vector = onehot_label(labels_int)
        df["label_vec"] = onehot_vector
        assert phase == 'train_valid'
        trn_indx_stage2 = df[df['split'] == 'train'].index.tolist()
        val_indx_stage2 = df[df['split'] == 'valid'].index.tolist()
        trn_val_indx = []
        trn_val_indx.extend(trn_indx_stage2)
        trn_val_indx.extend(val_indx_stage2)
        df = df.loc[trn_val_indx]
        
        # -------- 过滤掉不存在的文件 --------
        valid_indices = []
        for idx, row in df.iterrows():
            file_path = os.path.join(data_dir, row['Recording'])
            if os.path.exists(file_path):
                valid_indices.append(idx)
        df = df.loc[valid_indices]
        print(f"ECGDataset_unseen_MHL_stage2: 保留 {len(df)} 个有效样本")
        
        self.data_dir = data_dir
        self.labels = df
        self.nleads = leads
        data_df = pd.DataFrame()
        data_df["Recording"], data_df["label_vec"] = df["Recording"], df["label_vec"]
        self.data = data_df.values
        self.length = length

    def __getitem__(self, index: int):
        file_name, onehot_label = self.data[index]
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path, sep=",")
        ecg_data = df.values[:]
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-self.length:, :]
        result = np.zeros((self.length, self.nleads))
        result[-nsteps:, :] = ecg_data

        ecg = result.transpose()  # (C, L)

        if self.phase == 'train_valid':
            # 为多视图对比学习生成两个不同增强的视图
            # View 1: 使用弱增强
            view1 = augment_ecg(ecg.copy(), augmentation_type='weak')
            # View 2: 使用强增强
            view2 = augment_ecg(ecg.copy(), augmentation_type='strong')

            return (
                torch.from_numpy(view1).float(),
                torch.from_numpy(view2).float(),
                torch.from_numpy(onehot_label).float()
            )
        else:
            return (
                torch.from_numpy(ecg).float(),
                torch.from_numpy(onehot_label).float()
            )

    def __len__(self):
        return len(self.labels)