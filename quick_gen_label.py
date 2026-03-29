"""
快速生成训练所需的标签文件
"""
import os
import pandas as pd
import numpy as np

def quick_gen_label_csv(label_file, output_csv, unseen_class_index=5, 
                        trn_ratio=0.7, val_ratio=0.1, seed=42):
    """
    直接生成 Stage1 标签文件，不添加增强数据行
    """
    df = pd.read_csv(label_file)
    data_num = len(df)
    
    np.random.seed(seed)
    
    # 按类别分层分割
    class_list = sorted(df['label'].unique())
    trn_indx = []
    val_indx = []
    test_indx = []
    
    for class_k in class_list:
        class_indices = df[df['label'] == class_k].index.tolist()
        np.random.shuffle(class_indices)
        
        n_trn = int(len(class_indices) * trn_ratio)
        n_val = int(len(class_indices) * val_ratio)
        
        trn_indx.extend(class_indices[:n_trn])
        val_indx.extend(class_indices[n_trn:n_trn + n_val])
        test_indx.extend(class_indices[n_trn + n_val:])
    
    # 创建 split 列
    df['split'] = 'train'
    df.loc[trn_indx, 'split'] = 'train'
    df.loc[val_indx, 'split'] = 'valid'
    df.loc[test_indx, 'split'] = 'test'
    
    # 添加必要的列
    df['aug_label'] = df['label']
    df['org_label'] = df['label']
    
    # 保存
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"生成完成: {output_csv}")
    print(f"  Train: {len(trn_indx)}, Valid: {len(val_indx)}, Test: {len(test_indx)}")
    print(f"  Classes: {class_list}")
    return True


if __name__ == "__main__":
    # 为 scaling_up 生成标签文件
    label_file = "./data_path/data_dir_szhou/CPSC18_label_all.csv"
    output_file = "./OpenMax/CPSC18_resnet34_label_scaling_up_Stage1_1.csv"
    quick_gen_label_csv(label_file, output_file, seed=1)
    
    # 也生成 Stage2 的标签文件
    output_file2 = "./OpenMax/CPSC18_resnet34_label_scaling_up_1.csv"
    quick_gen_label_csv(label_file, output_file2, seed=1)
    
    print("\n标签文件生成完成！现在运行:")
    print("python run_1_train.py --dataset 'CPSC18' --model 'resnet34' --transform_type 'scaling_up' --seed 1 --data_path './data_path/data_dir_szhou/'")
