"""
生成 Stage1 标签文件，跳过增强文件检查
"""
import os
import pandas as pd
import numpy as np
import copy
from collections import Counter

def gen_label_csv_simple(label_file, output_csv, unseen_class_index=5, trn_ratio=0.7, val_ratio=0.1, seed=42):
    """
    简化版标签生成，跳过增强文件检查
    """
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        return False
    
    df_label = pd.read_csv(label_file)
    org_y = df_label['label'].values
    data_num = org_y.shape[0]
    
    np.random.seed(seed)
    
    # 分割数据
    seen_indices = np.where(org_y != unseen_class_index)[0]
    unseen_indices = np.where(org_y == unseen_class_index)[0]
    
    # 为每个类别分割数据
    class_list = list(set(org_y))
    trn_indx = []
    val_indx = []
    
    for class_k in class_list:
        class_k_indices = np.where(org_y == class_k)[0]
        np.random.shuffle(class_k_indices)
        class_k_trn_num = int(len(class_k_indices) * trn_ratio)
        class_k_val_num = int(len(class_k_indices) * val_ratio)
        trn_indx.extend(class_k_indices[:class_k_trn_num].tolist())
        val_indx.extend(class_k_indices[class_k_trn_num:class_k_trn_num + class_k_val_num].tolist())
    
    test_indx = list(set(range(data_num)) - set(trn_indx) - set(val_indx))
    
    # 添加 split 列
    df_label['split'] = 'train'
    df_label.loc[trn_indx, 'split'] = 'train'
    df_label.loc[val_indx, 'split'] = 'valid'
    df_label.loc[test_indx, 'split'] = 'test'
    
    # 添加 aug_label (与 label 相同)
    df_label['aug_label'] = df_label['label']
    df_label['org_label'] = df_label['label']
    
    # 保存
    df_label.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    print(f"Train: {len(trn_indx)}, Valid: {len(val_indx)}, Test: {len(test_indx)}")
    return True


if __name__ == "__main__":
    # CPSC18
    label_file = "./data_path/data_dir_szhou/CPSC18_label_all.csv"
    output_file = "./OpenMax/CPSC18_resnet34_label_scaling_up_Stage1_1.csv"
    gen_label_csv_simple(label_file, output_file, unseen_class_index=5, seed=1)
    
    print("\nDone! Now run:")
    print("python run_1_train.py --dataset 'CPSC18' --model 'resnet34' --transform_type 'scaling_up' --seed 1 --data_path './data_path/data_dir_szhou/'")
