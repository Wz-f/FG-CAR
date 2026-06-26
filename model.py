"""
FG-CARE: Fine-Grained Cross-Modal Attention Reasoning Model
DGL Version with Ablation Study Support

支持的消融变体:
- 'full': 完整模型
- 'no_semantic_aug': 移除语义感知增强
- 'no_graph_cl': 移除图对比损失
- 'no_global_cl': 移除全局对比损失
- 'no_fine_grained': 移除细粒度输入
- 'no_caf': 移除 CAF 门控融合
"""
import dgl
from dgl.nn.pytorch import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ============================================================
# 第一步：细粒度跨模态图构建模块
# ============================================================

class GraphConstructor(nn.Module):
    """
    Cross-Modal Reasoning Graph Construction (Section 3.2)
    支持 'no_fine_grained' 消融变体
    """
    
    def __init__(self, text_dim=768, img_dim=2048, hidden_dim=256,
                 top_k=5, syntax_window=3, sim_threshold=0.5):
        super().__init__()
        self.top_k = top_k
        self.syntax_window = syntax_window
        self.sim_threshold = sim_threshold
        self.hidden_dim = hidden_dim
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        
        # 用于 no_fine_grained 变体的全局特征投影
        self.text_global_proj = nn.Linear(text_dim, hidden_dim)
        self.img_global_proj = nn.Linear(img_dim, hidden_dim)
        
    def forward(self, text_local, img_local, 
                text_global=None, img_global=None, ablation_mode='full'):
        """
        Args:
            text_local: (M, 768) 文本局部特征
            img_local: (N, 2048) 图像局部特征
            text_global: (768,) 文本全局特征 (用于 no_fine_grained)
            img_global: (2048,) 图像全局特征 (用于 no_fine_grained)
            ablation_mode: 消融模式
            
        Returns:
            g: DGL图
        """
        device = text_local.device
        
        # ================================================================
        # 变体 D: w/o Fine-Grained - 仅使用全局特征构建退化图
        # ================================================================
        if ablation_mode == 'no_fine_grained':
            assert text_global is not None and img_global is not None, \
                "no_fine_grained mode requires global features"
            
            # 投影全局特征到隐藏维度
            text_feat = self.text_global_proj(text_global.unsqueeze(0))  # (1, hidden_dim)
            img_feat = self.img_global_proj(img_global.unsqueeze(0))     # (1, hidden_dim)
            
            node_features = torch.cat([text_feat, img_feat], dim=0)  # (2, hidden_dim)
            
            # 构建退化图：2个节点，互相连接 + 自环
            edges_src = [0, 1, 0, 1]
            edges_dst = [1, 0, 0, 1]
            edge_weights = [1.0, 1.0, 1.0, 1.0]
            edge_types = [2, 2, 0, 1]  # 跨模态边 + 自环
            
            g = dgl.graph((edges_src, edges_dst), num_nodes=2)
            g = g.to(device)
            
            g.ndata['feat'] = node_features
            g.ndata['modality'] = torch.tensor([0, 1], dtype=torch.long, device=device)
            g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float, device=device)
            g.edata['type'] = torch.tensor(edge_types, dtype=torch.long, device=device)
            
            return g
        
        # ================================================================
        # 完整模式：使用细粒度局部特征构建图
        # ================================================================
        M = text_local.size(0)
        N = img_local.size(0)
        
        text_feat = self.text_proj(text_local)  # (M, hidden_dim)
        img_feat = self.img_proj(img_local)     # (N, hidden_dim)
        
        node_features = torch.cat([text_feat, img_feat], dim=0)
        
        edges_src = []
        edges_dst = []
        edge_weights = []
        edge_types = []
        
        # 1. 跨模态边 (Cross-Modal Edges)
        if M > 0 and N > 0:
            text_norm = F.normalize(text_feat, dim=1)
            img_norm = F.normalize(img_feat, dim=1)
            sim_matrix = torch.mm(text_norm, img_norm.t())
            
            k_text = min(self.top_k, N)
            _, top_img_indices = torch.topk(sim_matrix, k_text, dim=1)
            for i in range(M):
                for j in top_img_indices[i]:
                    j_val = j.item()
                    edges_src.extend([i, M + j_val])
                    edges_dst.extend([M + j_val, i])
                    w = sim_matrix[i, j_val].item()
                    edge_weights.extend([w, w])
                    edge_types.extend([2, 2])
            
            k_img = min(self.top_k, M)
            _, top_text_indices = torch.topk(sim_matrix.t(), k_img, dim=1)
            existing_edges = set(zip(edges_src, edges_dst))
            for j in range(N):
                for i in top_text_indices[j]:
                    i_val = i.item()
                    if (i_val, M + j) not in existing_edges:
                        edges_src.extend([i_val, M + j])
                        edges_dst.extend([M + j, i_val])
                        w = sim_matrix[i_val, j].item()
                        edge_weights.extend([w, w])
                        edge_types.extend([2, 2])
        
        # 2. 文本模态内边
        for i in range(M):
            for j in range(max(0, i - self.syntax_window), 
                          min(M, i + self.syntax_window + 1)):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_weights.append(1.0)
                    edge_types.append(0)
        
        # 3. 图像模态内边
        if N > 1:
            img_local_norm = F.normalize(img_feat, dim=1)
            img_sim = torch.mm(img_local_norm, img_local_norm.t())
            
            for i in range(N):
                for j in range(i + 1, N):
                    if img_sim[i, j] >= self.sim_threshold:
                        edges_src.extend([M + i, M + j])
                        edges_dst.extend([M + j, M + i])
                        w = img_sim[i, j].item()
                        edge_weights.extend([w, w])
                        edge_types.extend([1, 1])
        
        # 构建 DGL 图
        num_nodes = M + N
        if len(edges_src) == 0:
            for i in range(num_nodes):
                edges_src.append(i)
                edges_dst.append(i)
                edge_weights.append(1.0)
                edge_types.append(0 if i < M else 1)
        
        g = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
        g = g.to(device)
        
        g.ndata['feat'] = node_features
        g.ndata['modality'] = torch.cat([
            torch.zeros(M, dtype=torch.long, device=device),
            torch.ones(N, dtype=torch.long, device=device)
        ])
        g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float, device=device)
        g.edata['type'] = torch.tensor(edge_types, dtype=torch.long, device=device)
        
        return g


# ============================================================
# 第二步：图增强模块 - 支持消融变体
# ============================================================

class SemanticAwareAugmentor(nn.Module):
    """
    Semantic-Aware Graph Augmentation
    支持 'no_semantic_aug' 消融变体
    """
    
    def __init__(self, base_keep_ratio=0.8, random_drop_ratio=0.2):
        super().__init__()
        self.base_keep_ratio = base_keep_ratio
        self.random_drop_ratio = random_drop_ratio  # 用于消融变体
    
    def forward(self, g, ablation_mode='full'):
        """
        Args:
            g: DGL图
            ablation_mode: 消融模式
            
        Returns:
            g_aug: 增强后的 DGL 图
        """
        device = g.device
        edge_types = g.edata['type']
        edge_weights = g.edata['weight']
        num_edges = g.num_edges()
        
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        cross_mask = (edge_types == 2)
        
        if cross_mask.sum() > 0:
            # ================================================================
            # 变体 A: w/o Semantic Aug - 使用随机均匀 DropEdge
            # ================================================================
            if ablation_mode == 'no_semantic_aug':
                # 不使用语义感知，随机均匀采样
                random_vals = torch.rand(cross_mask.sum(), device=device)
                cross_keep = random_vals > self.random_drop_ratio
            else:
                # ============================================================
                # 完整模式：语义感知扰动 - 保留概率与相似度成正比
                # ============================================================
                cross_weights = edge_weights[cross_mask]
                
                min_w = cross_weights.min()
                max_w = cross_weights.max()
                if max_w > min_w:
                    normalized_weights = (cross_weights - min_w) / (max_w - min_w + 1e-8)
                else:
                    normalized_weights = torch.ones_like(cross_weights)
                
                keep_probs = self.base_keep_ratio + (1 - self.base_keep_ratio) * normalized_weights
                keep_probs = keep_probs.clamp(0.3, 1.0)
                
                random_vals = torch.rand_like(keep_probs)
                cross_keep = random_vals < keep_probs
            
            cross_indices = torch.where(cross_mask)[0]
            keep_mask[cross_indices] = cross_keep
        
        kept_edges = torch.where(keep_mask)[0]
        g_aug = dgl.edge_subgraph(g, kept_edges, preserve_nodes=True)
        
        return g_aug


class GATEncoder(nn.Module):
    """Graph Attention Network Encoder"""
    
    def __init__(self, in_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(in_dim, hidden_dim // num_heads, num_heads,
                   feat_drop=dropout, attn_drop=dropout, activation=F.elu,
                   allow_zero_in_degree=True)  
        )
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, num_heads,
                       feat_drop=dropout, attn_drop=dropout, activation=F.elu,
                       allow_zero_in_degree=True)
            )
        
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, g, features):
        h = features
        for gat in self.gat_layers:
            h = gat(g, h)
            h = h.flatten(1)
        
        node_emb = h
        gate_scores = self.gate_nn(node_emb)
        gate_scores = F.softmax(gate_scores, dim=0)
        graph_emb = (gate_scores * node_emb).sum(dim=0)
        
        return node_emb, graph_emb


class GraphContrastiveLoss(nn.Module):
    """Graph Contrastive Loss (InfoNCE)"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, h_orig, h_aug):
        h_orig = F.normalize(h_orig, dim=1)
        h_aug = F.normalize(h_aug, dim=1)
        
        batch_size = h_orig.size(0)
        sim_matrix = torch.mm(h_orig, h_aug.t()) / self.temperature
        labels = torch.arange(batch_size, device=h_orig.device)
        
        return F.cross_entropy(sim_matrix, labels)


# ============================================================
# 第三步：全局对齐与融合模块
# ============================================================

class GlobalAlignmentModule(nn.Module):
    """Global Modality Alignment"""
    
    def __init__(self, text_dim=768, img_dim=2048, proj_dim=128):
        super().__init__()
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim * 2, proj_dim)
        )
        
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim * 2, proj_dim)
        )
        
    def forward(self, text_global, img_global):
        z_T = self.text_proj(text_global)
        z_V = self.img_proj(img_global)
        return z_T, z_V


class GlobalContrastiveLoss(nn.Module):
    """Global Contrastive Loss"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_T, z_V):
        z_T = F.normalize(z_T, dim=1)
        z_V = F.normalize(z_V, dim=1)
        
        batch_size = z_T.size(0)
        sim_matrix = torch.mm(z_T, z_V.t()) / self.temperature
        labels = torch.arange(batch_size, device=z_T.device)
        
        loss_t2v = F.cross_entropy(sim_matrix, labels)
        loss_v2t = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_t2v + loss_v2t) / 2


class ConflictAwareFusion(nn.Module):
    """
    Conflict-Aware Adaptive Fusion (CAF)
    支持 'no_caf' 消融变体
    """
    
    def __init__(self, text_dim=768, img_dim=2048, graph_dim=256, hidden_dim=256):
        super().__init__()
        
        self.text_align = nn.Linear(text_dim, hidden_dim)
        self.img_align = nn.Linear(img_dim, hidden_dim)
        self.graph_align = nn.Linear(graph_dim, hidden_dim)
        
        context_dim = hidden_dim * 3
        self.W_context = nn.Linear(context_dim, hidden_dim)
        self.W_gate = nn.Linear(hidden_dim, 1)
        self.W_fusion = nn.Linear(hidden_dim, hidden_dim)
        
        # ================================================================
        # 用于 no_caf 变体：直接拼接后降维
        # ================================================================
        self.concat_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, H_T, H_V, h_graph, ablation_mode='full'):
        """
        Args:
            H_T: (B, 768) 全局文本特征
            H_V: (B, 2048) 全局图像特征
            h_graph: (B, 256) 图表示
            ablation_mode: 消融模式
            
        Returns:
            Z_final: (B, hidden_dim) 最终融合表示
            alpha: (B, 1) 门控权重 (no_caf 模式下返回 None)
        """
        H_T_prime = self.text_align(H_T)
        H_V_prime = self.img_align(H_V)
        h_prime = self.graph_align(h_graph)
        
        # ================================================================
        # 变体 E: w/o CAF - 跳过门控，直接拼接融合
        # ================================================================
        if ablation_mode == 'no_caf':
            # 全局融合特征
            H_global_fused = self.W_fusion(H_T_prime + H_V_prime)
            
            # 直接拼接图特征和全局融合特征
            concat_feat = torch.cat([h_prime, H_global_fused], dim=1)
            Z_final = self.concat_fusion(concat_feat)
            
            return Z_final, None  # 返回 None 表示没有 alpha
        
        # ================================================================
        # 完整模式：CAF 门控融合
        # ================================================================
        D_diff = H_T_prime - H_V_prime
        D_prod = H_T_prime * H_V_prime
        F_disc = torch.cat([D_diff, D_prod], dim=1)
        
        Context = torch.cat([F_disc, h_prime], dim=1)
        alpha = torch.sigmoid(
            self.W_gate(torch.tanh(self.W_context(Context)))
        )
        
        H_global_fused = self.W_fusion(H_T_prime + H_V_prime)
        H_global_fused = self.dropout(H_global_fused)
        
        Z_final = alpha * h_prime + (1 - alpha) * H_global_fused
        
        return Z_final, alpha


# ============================================================
# 第四步：完整的 FG-CARE 模型（支持消融实验）
# ============================================================

class FGCARE(nn.Module):
    """
    FG-CARE: Fine-Grained Cross-Modal Attention Reasoning Model
    DGL Version with Ablation Study Support
    
    消融变体:
    - 'full': 完整模型
    - 'no_semantic_aug': 移除语义感知增强，使用随机 DropEdge
    - 'no_graph_cl': 移除图对比损失 (λ1 = 0)
    - 'no_global_cl': 移除全局对比损失 (λ2 = 0)
    - 'no_fine_grained': 移除细粒度输入，仅用全局特征构建退化图
    - 'no_caf': 移除 CAF 门控，使用直接拼接融合
    """
    
    def __init__(self,
                 text_dim=768,
                 img_dim=2048,
                 hidden_dim=256,
                 proj_dim=128,
                 num_classes=2,
                 top_k=5,
                 syntax_window=3,
                 sim_threshold=0.5,
                 gat_heads=4,
                 gat_layers=2,
                 gat_dropout=0.3,
                 aug_keep_ratio=0.8,
                 random_drop_ratio=0.2,
                 temperature=0.07,
                 lambda_graph=0.3,
                 lambda_global=0.3,
                 ablation_mode='full'):
        super().__init__()
        
        # 存储消融模式
        self.ablation_mode = ablation_mode
        
        # 根据消融模式调整损失权重
        self.lambda_graph, self.lambda_global = get_loss_weights(
            ablation_mode, lambda_graph, lambda_global
        )
        
        # 图构建器
        self.graph_constructor = GraphConstructor(
            text_dim=text_dim,
            img_dim=img_dim,
            hidden_dim=hidden_dim,
            top_k=top_k,
            syntax_window=syntax_window,
            sim_threshold=sim_threshold
        )
        
        # 图增强器
        self.augmentor = SemanticAwareAugmentor(
            base_keep_ratio=aug_keep_ratio,
            random_drop_ratio=random_drop_ratio
        )
        
        # GAT 编码器
        self.gat_encoder = GATEncoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=gat_dropout
        )
        
        # 图对比损失
        self.graph_cl_loss = GraphContrastiveLoss(temperature=temperature)
        
        # 全局对齐模块
        self.global_align = GlobalAlignmentModule(
            text_dim=text_dim,
            img_dim=img_dim,
            proj_dim=proj_dim
        )
        self.global_cl_loss = GlobalContrastiveLoss(temperature=temperature)
        
        # 冲突感知融合
        self.caf = ConflictAwareFusion(
            text_dim=text_dim,
            img_dim=img_dim,
            graph_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, text_global, text_local_list, img_global, img_local_list, 
                labels=None, training=True):
        """
        Args:
            text_global: (B, 768) 全局文本特征
            text_local_list: List of (M_i, 768) 局部文本特征
            img_global: (B, 2048) 全局图像特征
            img_local_list: List of (N_i, 2048) 局部图像特征
            labels: (B,) 标签
            training: bool
            
        Returns:
            logits: (B, num_classes)
            total_loss: scalar (if training)
            loss_dict: dict of sub-losses
        """
        device = text_global.device
        batch_size = text_global.size(0)
        ablation_mode = self.ablation_mode
        
        # ========== Step 1: 构建图并编码 ==========
        h_orig_list = []
        h_aug_list = []
        
        for i in range(batch_size):
            text_local = text_local_list[i].to(device)
            img_local = img_local_list[i].to(device)


            # 确保都是2D张量
            while text_local.dim() > 2:
                text_local = text_local.squeeze(0)
            while img_local.dim() > 2:
                img_local = img_local.squeeze(0)
                
            if text_local.dim() == 1:
                text_local = text_local.unsqueeze(0)
            if img_local.dim() == 1:
                img_local = img_local.unsqueeze(0)

            # 调试：打印形状
            # if i == 0:
            #     print(f"DEBUG: text_local.shape={text_local.shape}, img_local.shape={img_local.shape}")

            # ============================================================
            # 变体 D: no_fine_grained - 使用全局特征构建退化图
            # ============================================================
            if ablation_mode == 'no_fine_grained':
                g = self.graph_constructor(
                    text_local, img_local,
                    text_global=text_global[i],
                    img_global=img_global[i],
                    ablation_mode=ablation_mode
                )
            else:
                g = self.graph_constructor(text_local, img_local)
            
            # GAT 编码
            _, h_orig = self.gat_encoder(g, g.ndata['feat'])
            h_orig_list.append(h_orig)
            
            # ============================================================
            # 图增强（用于图对比学习）
            # 变体 B (no_graph_cl): 跳过增强（因为不需要计算图对比损失）
            # 变体 A (no_semantic_aug): 使用随机均匀 DropEdge
            # ============================================================
            if training and ablation_mode != 'no_graph_cl':
                g_aug = self.augmentor(g, ablation_mode=ablation_mode)
                _, h_aug = self.gat_encoder(g_aug, g_aug.ndata['feat'])
                h_aug_list.append(h_aug)
        
        h_orig = torch.stack(h_orig_list, dim=0)  # (B, hidden_dim)
        
        # ========== Step 2: 图对比损失 ==========
        # 变体 B: no_graph_cl - 设置 λ1 = 0，跳过计算
        L_graph = torch.tensor(0.0, device=device)
        if training and self.lambda_graph > 0 and len(h_aug_list) > 0:
            h_aug = torch.stack(h_aug_list, dim=0)
            L_graph = self.graph_cl_loss(h_orig, h_aug)
        
        # ========== Step 3: 全局对齐 ==========
        z_T, z_V = self.global_align(text_global, img_global)
        
        # 变体 C: no_global_cl - 设置 λ2 = 0，跳过计算
        L_global = torch.tensor(0.0, device=device)
        if training and self.lambda_global > 0:
            L_global = self.global_cl_loss(z_T, z_V)
        
        # ========== Step 4: CAF 融合 ==========
        # 变体 E: no_caf - 使用直接拼接替代门控
        Z_final, alpha = self.caf(text_global, img_global, h_orig, 
                                   ablation_mode=ablation_mode)
        
        # ========== Step 5: 分类 ==========
        logits = self.classifier(Z_final)
        
        # ========== 计算损失 ==========
        loss_dict = {
            'L_graph': L_graph.item() if isinstance(L_graph, torch.Tensor) else L_graph,
            'L_global': L_global.item() if isinstance(L_global, torch.Tensor) else L_global,
            'alpha_mean': alpha.mean().item() if alpha is not None else 0.0,
            'ablation_mode': ablation_mode
        }
        
        total_loss = None
        if training and labels is not None:
            L_cls = F.cross_entropy(logits, labels)
            total_loss = L_cls + self.lambda_graph * L_graph + self.lambda_global * L_global
            loss_dict['L_cls'] = L_cls.item()
            loss_dict['total_loss'] = total_loss.item()
        
        return logits, total_loss, loss_dict


# ============================================================
# 辅助函数
# ============================================================

def get_loss_weights(ablation_mode, base_lambda_graph=0.3, base_lambda_global=0.3):
    """
    根据消融模式返回正确的损失权重
    
    Args:
        ablation_mode: 消融模式字符串
        base_lambda_graph: 基础图对比损失权重
        base_lambda_global: 基础全局对比损失权重
        
    Returns:
        lambda_graph: 图对比损失权重
        lambda_global: 全局对比损失权重
    """
    if ablation_mode == 'no_graph_cl':
        # ============================================================
        # 变体 B: w/o Graph CL - 移除图对比损失
        # ============================================================
        return 0.0, base_lambda_global
    
    elif ablation_mode == 'no_global_cl':
        # ============================================================
        # 变体 C: w/o Global CL - 移除全局对比损失
        # ============================================================
        return base_lambda_graph, 0.0
    
    else:
        # 其他模式（包括 full, no_semantic_aug, no_fine_grained, no_caf）
        # 使用默认权重
        return base_lambda_graph, base_lambda_global


def get_ablation_description(ablation_mode):
    """获取消融模式的描述"""
    descriptions = {
        'full': 'Full Model (完整模型)',
        'no_semantic_aug': 'w/o Semantic Aug (移除语义感知增强，使用随机DropEdge)',
        'no_graph_cl': 'w/o Graph CL (移除图对比损失，λ1=0)',
        'no_global_cl': 'w/o Global CL (移除全局对比损失，λ2=0)',
        'no_fine_grained': 'w/o Fine-Grained (移除细粒度输入，仅用全局特征)',
        'no_caf': 'w/o CAF (移除门控融合，使用直接拼接)'
    }
    return descriptions.get(ablation_mode, 'Unknown Mode')