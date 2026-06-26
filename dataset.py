"""
FG-CARE Dataset (DGL Version)
返回原始特征，图构建在模型中进行
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class FGCAREDataset(Dataset):
    """
    FG-CARE 数据集
    返回：文本全局特征、文本局部特征、图像全局特征、图像局部特征、标签
    """
    
    def __init__(self, df, root_dir, image_id_col, text_id_col,
                 image_vec_dir, text_vec_dir):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.image_id_col = image_id_col
        self.text_id_col = text_id_col
        self.image_vec_dir = image_vec_dir
        self.text_vec_dir = text_vec_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        file_name_image = str(row[self.image_id_col]).split(".")[0]
        file_name_text = str(row[self.text_id_col])
        
        # ========== 加载图像特征 ==========
        img_global_path = f'{self.root_dir}{self.image_vec_dir}{file_name_image}_full_image.npy'
        img_global = np.load(img_global_path)  # (1, 2048)
        
        img_local_path = f'{self.root_dir}{self.image_vec_dir}{file_name_image}.npy'
        try:
            img_local = np.load(img_local_path)  # (N, 2048)
        except:
            img_local = img_global.copy()

        # 确保 img_local 是2D
        if img_local.ndim == 1:
            img_local = img_local.reshape(1, -1)
        
        # ========== 加载文本特征 ==========
        text_global_path = f'{self.root_dir}{self.text_vec_dir}{file_name_text}_full_text.npy'
        text_global = np.load(text_global_path)  # (1, 768)
        
        text_local_path = f'{self.root_dir}{self.text_vec_dir}{file_name_text}.npy'
        text_local = np.load(text_local_path)  # (M, 768)

        # 确保 text_local 是2D
        if text_local.ndim == 1:
            text_local = text_local.reshape(1, -1)
        
        # ========== 转为 Tensor ==========
        text_global = torch.tensor(text_global, dtype=torch.float32).squeeze(0)  # (768,)
        text_local = torch.tensor(text_local, dtype=torch.float32)  # (M, 768)
        img_global = torch.tensor(img_global, dtype=torch.float32).squeeze(0)  # (2048,)
        img_local = torch.tensor(img_local, dtype=torch.float32)  # (N, 2048)
        
        # ========== 标签 ==========
        label_str = row['label']
        if label_str == 'real' or label_str == 0:
            label = 0
        else:
            label = 1
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'text_global': text_global,
            'text_local': text_local,
            'img_global': img_global,
            'img_local': img_local,
            'num_text_nodes': text_local.size(0),
            'num_img_nodes': img_local.size(0),
            'label': label
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    text_globals = torch.stack([item['text_global'] for item in batch])
    img_globals = torch.stack([item['img_global'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    text_locals = [item['text_local'] for item in batch]
    img_locals = [item['img_local'] for item in batch]
    num_text_nodes = [item['num_text_nodes'] for item in batch]
    num_img_nodes = [item['num_img_nodes'] for item in batch]
    
    return {
        'text_global': text_globals,
        'img_global': img_globals,
        'text_local': text_locals,
        'img_local': img_locals,
        'num_text_nodes': num_text_nodes,
        'num_img_nodes': num_img_nodes,
        'label': labels
    }