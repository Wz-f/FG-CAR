"""
FG-CARE Utility Functions (DGL Version)
"""
import os
import random
import numpy as np
import pandas as pd
import torch

import config
from dataset import FGCAREDataset


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_feature_files(df, root_dir, image_vec_dir, text_vec_dir, 
                        image_id_col, text_id_col):
    """检查特征文件是否存在"""
    valid_indices = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        file_name_image = str(row[image_id_col]).split(".")[0]
        file_name_text = str(row[text_id_col])
        
        img_global_path = f'{root_dir}{image_vec_dir}{file_name_image}_full_image.npy'
        text_global_path = f'{root_dir}{text_vec_dir}{file_name_text}_full_text.npy'
        text_local_path = f'{root_dir}{text_vec_dir}{file_name_text}.npy'
        
        if (os.path.exists(img_global_path) and 
            os.path.exists(text_global_path) and 
            os.path.exists(text_local_path)):
            valid_indices.append(idx)
    
    return df.iloc[valid_indices].reset_index(drop=True)


def set_up_mediaeval2015():
    """加载 MediaEval 2015 数据集"""
    df_train = pd.read_csv(
        f'{config.root_dir}{config.me15_train_csv_name}',
        sep='\t', engine='python', on_bad_lines='skip'
    )
    df_train = df_train.dropna().reset_index(drop=True)
    df_train = df_train[df_train['label'].isin(['real', 'fake'])]
    df_train = df_train[~df_train['clean_image_id'].str.contains(',', na=False)]
    
    df_test = pd.read_csv(
        f'{config.root_dir}{config.me15_test_csv_name}',
        sep='\t', engine='python', on_bad_lines='skip'
    )
    df_test = df_test.dropna().reset_index(drop=True)
    df_test = df_test[df_test['label'].isin(['real', 'fake'])]
    df_test = df_test[~df_test['clean_image_id'].str.contains(',', na=False)]
    
    print(f"原始训练集: {len(df_train)}, 原始测试集: {len(df_test)}")
    
    df_train = check_feature_files(
        df_train, config.root_dir, config.me15_image_vec_dir, 
        config.me15_text_vec_dir, "clean_image_id", "tweetId"
    )
    df_test = check_feature_files(
        df_test, config.root_dir, config.me15_image_vec_dir,
        config.me15_text_vec_dir, "clean_image_id", "tweetId"
    )
    
    print(f"过滤后训练集: {len(df_train)}, 过滤后测试集: {len(df_test)}")
    
    train_dataset = FGCAREDataset(
        df_train, config.root_dir, "clean_image_id", "tweetId",
        config.me15_image_vec_dir, config.me15_text_vec_dir
    )
    test_dataset = FGCAREDataset(
        df_test, config.root_dir, "clean_image_id", "tweetId",
        config.me15_image_vec_dir, config.me15_text_vec_dir
    )
    
    return train_dataset, test_dataset


def set_up_weibo():
    """加载 Weibo 数据集"""
    df_train = pd.read_csv(f'{config.root_dir}{config.we_train_csv_name}', sep='\t')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.we_test_csv_name}', sep='\t')
    df_test = df_test.dropna().reset_index(drop=True)
    
    print(f"原始训练集: {len(df_train)}, 原始测试集: {len(df_test)}")
    
    df_train = check_feature_files(
        df_train, config.root_dir, config.we_image_vec_dir,
        config.we_text_vec_dir, "clean_image_id", "tweetId"
    )
    df_test = check_feature_files(
        df_test, config.root_dir, config.we_image_vec_dir,
        config.we_text_vec_dir, "clean_image_id", "tweetId"
    )
    
    print(f"过滤后训练集: {len(df_train)}, 过滤后测试集: {len(df_test)}")
    
    train_dataset = FGCAREDataset(
        df_train, config.root_dir, "clean_image_id", "tweetId",
        config.we_image_vec_dir, config.we_text_vec_dir
    )
    test_dataset = FGCAREDataset(
        df_test, config.root_dir, "clean_image_id", "tweetId",
        config.we_image_vec_dir, config.we_text_vec_dir
    )
    
    return train_dataset, test_dataset


def print_metrics(metrics, prefix=""):
    """打印评估指标"""
    print(f"\n{prefix} Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall: {metrics['recall']:.2f}%")
    print(f"  F1-Score: {metrics['f1']:.2f}%")
    print(f"  Avg Alpha: {metrics['alpha']:.4f}")