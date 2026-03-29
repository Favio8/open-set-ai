# generate_augmented_files.py
import os
import numpy as np
import wfdb
import pandas as pd



def generate_simple_augmentation(data_path, dataset='CPSC18', transform_type='scaling_up'):
    """生成简单的数据增强文件"""
    label_file = os.path.join(data_path, f'{dataset}_label_all.csv')
    data_dir = os.path.join(data_path, f'{dataset}_szhou_all')

    if not os.path.exists(label_file):
        print(f"标签文件不存在: {label_file}")
        return

    df = pd.read_csv(label_file)

    # 只处理训练和验证集
    train_val_indices = df[df['split'].isin(['train', 'valid'])].index

    for idx in train_val_indices:
        row = df.iloc[idx]
        original_file = row['Recording']
        original_path = os.path.join(data_dir, original_file)

        if not os.path.exists(original_path):
            continue

        # 生成增强文件名
        augmented_file = f"{transform_type}_{original_file}"
        augmented_path = os.path.join(data_dir, augmented_file)

        if os.path.exists(augmented_path):
            continue

        try:
            # 读取原始信号
            record = wfdb.rdrecord(original_path.replace('.mat', ''))
            signal = record.p_signal

            # 应用简单变换
            if transform_type == 'scaling_up':
                # 放大信号
                augmented_signal = signal * 1.1
            elif transform_type == 'scaling_down':
                # 缩小信号
                augmented_signal = signal * 0.9
            elif transform_type == 'reverse':
                # 反转信号
                augmented_signal = signal[::-1]
            else:
                # 默认不改变
                augmented_signal = signal

            # 保存增强后的信号
            wfdb.wrsamp(
                record_name=augmented_path.replace('.mat', ''),
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=augmented_signal,
                comments=record.comments
            )

            print(f"生成增强文件: {augmented_file}")

        except Exception as e:
            print(f"处理文件 {original_file} 时出错: {e}")
            # 创建空文件作为占位
            with open(augmented_path, 'wb') as f:
                np.save(f, np.array([]))

    print(f"{transform_type} 增强文件生成完成")


if __name__ == "__main__":
    data_path = "D:/Open_World_ECG_Classification-main/data/"
    generate_simple_augmentation(data_path, dataset='CPSC18', transform_type='scaling_up')