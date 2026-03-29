# create_hardneg_files.py
import os
import pandas as pd
import numpy as np


def create_hardneg_placeholder_files(data_path, dataset='CPSC18'):
    """为硬负样本创建空的占位文件"""
    label_file = os.path.join(data_path, f'{dataset}_label_all.csv')
    data_dir = os.path.join(data_path, f'{dataset}_szhou_all')

    if not os.path.exists(label_file):
        print(f"标签文件不存在: {label_file}")
        return

    df = pd.read_csv(label_file)

    # 只复制训练和验证集的原始文件作为硬负样本占位
    for idx, row in df.iterrows():
        if row.get('split', None) in ['train', 'valid']:
            original_file = row['Recording']
            hardneg_file = f"hardneg_{original_file}"
            original_path = os.path.join(data_dir, original_file)
            hardneg_path = os.path.join(data_dir, hardneg_file)

            # 如果原始文件存在但硬负样本文件不存在，创建硬链接（或复制）
            if os.path.exists(original_path) and not os.path.exists(hardneg_path):
                try:
                    # 创建硬链接（不会占用额外磁盘空间）
                    os.link(original_path, hardneg_path)
                    print(f"创建硬链接: {hardneg_file}")
                except:
                    # 如果硬链接失败，复制文件
                    import shutil
                    shutil.copy2(original_path, hardneg_path)
                    print(f"复制文件: {hardneg_file}")

    print("硬负样本占位文件创建完成")


if __name__ == "__main__":
    data_path = "D:/Open_World_ECG_Classification-main/data_dir_szhou/"
    create_hardneg_placeholder_files(data_path, dataset='CPSC18')