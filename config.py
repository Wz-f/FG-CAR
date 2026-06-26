## Base data folder
root_dir = "../"



######## MediaEval 2015 ########
me15_train_csv_name = "mediaeval2015/train_tweets_new.txt"
me15_test_csv_name = "mediaeval2015/test_tweets_new.txt"
me15_image_vec_dir = "image_feature_2/"
me15_text_vec_dir = "text_feature_2/"

####### Weibo ########
we_train_csv_name = "weibo_dataset/df_train2_converted_1.txt"
we_test_csv_name = "weibo_dataset/df_test2_converted_1.txt"
we_image_vec_dir = "weibo_data/image_data_1/"
we_text_vec_dir = "weibo_data/text_data_1/"

######## Model Hyperparameters ########
text_feat_dim = 768
image_feat_dim = 2048
hidden_dim = 256
proj_dim = 128

# 图构建参数
top_k = 5
syntax_window = 3
sim_threshold = 0.5

# GAT 参数
gat_heads = 4
gat_layers = 2
gat_dropout = 0.3


# 图增强参数
aug_base_keep_ratio = 0.9        # 提高保留比例，减少破坏
random_drop_ratio = 0.1          # 降低随机丢弃比例
feat_noise_std = 0.05            # 节点特征高斯噪声
feat_mask_ratio = 0.05           # 节点特征遮蔽比例

# 对比学习参数
temperature = 0.1                # 提高温度，使梯度更平滑
cl_margin = 0.1                  # 对比学习 margin
hard_negative_weight = 0.3       # 硬负样本权重

# 损失权重（基础值）
lambda_graph = 0.01               # 降低图对比损失权重
lambda_global = 0.4             # λ_global


# 训练参数
batch_size = 32
epochs = 30
lr = 2e-4
weight_decay = 1e-5

# 分类
num_classes = 2

def get_image_feat_dim(dataset_name):
    """
    根据数据集自动调整图像特征维度
    
    Args:
        dataset_name: 数据集名称 ('me15' 或 'we')
    
    Returns:
        image_feat_dim: 图像特征维度
    """
    if dataset_name == 'we':
        return 1024  # Weibo 数据集使用 1024 维
    else:
        return 2048  # MediaEval2015 数据集使用 2048 维




# ============================================================
# 消融实验配置
# ============================================================
# 可选值:
#   'full'           - 完整模型 (默认)
#   'no_semantic_aug' - 移除语义感知增强，使用随机 DropEdge
#   'no_graph_cl'    - 移除图对比损失 (λ1 = 0)
#   'no_global_cl'   - 移除全局对比损失 (λ2 = 0)
#   'no_fine_grained' - 移除细粒度输入，仅用全局特征构建退化图
#   'no_caf'         - 移除 CAF 门控，使用直接拼接融合
# ============================================================
ablation_mode = 'full'


def get_loss_weights(ablation_mode):
    """
    根据消融模式返回正确的损失权重
    
    Args:
        ablation_mode: 消融模式字符串
        
    Returns:
        lambda_graph: 图对比损失权重
        lambda_global: 全局对比损失权重
    """
    if ablation_mode == 'no_graph_cl':
        # 变体 B: w/o Graph CL - 移除图对比损失
        return 0.0, lambda_global
    elif ablation_mode == 'no_global_cl':
        # 变体 C: w/o Global CL - 移除全局对比损失
        return lambda_graph, 0.0
    else:
        # 其他模式使用默认权重
        return lambda_graph, lambda_global


def get_ablation_description(ablation_mode):
    """获取消融模式的描述"""
    descriptions = {
        'full': 'Full Model (完整模型)',
        'no_semantic_aug': 'w/o Semantic Aug (移除语义感知增强)',
        'no_graph_cl': 'w/o Graph CL (移除图对比损失)',
        'no_global_cl': 'w/o Global CL (移除全局对比损失)',
        'no_fine_grained': 'w/o Fine-Grained (移除细粒度输入)',
        'no_caf': 'w/o CAF (移除冲突感知融合)'
    }
    return descriptions.get(ablation_mode, 'Unknown Mode')