import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class MCMILoss(nn.Module):
    """
    多视图对比学习结合最大互信息损失函数
    论文：Multi-view Contrastive Learning with Maximal Mutual Information 
         for Continual Generalized Category Discovery
    作者：Zihao Zhao, Xiao Li, et al.
    期刊：Expert Systems With Applications 266 (2025) 125994
    
    核心创新：
    1. 实例级 + 类别级双视图对比学习
    2. OpenMax样本分离（已知/新样本）
    3. 最大互信息损失：L_MI = H(Y) - H(Y|Z)
    4. 伪监督对比学习
    """
    
    def __init__(self, temp=0.07, alpha=1.0, beta=0.4, num_classes=None):
        super().__init__()
        self.temp = temp  # 温度系数（论文：τ=0.07）
        self.alpha = alpha  # 实例级和类别级对比损失权重（论文：α=1）
        self.beta = beta  # 互信息损失权重（论文：β=0.4）
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def convert_vec_to_int(self, label_vec):
        """将one-hot标签转换为整数标签"""
        return torch.argmax(label_vec, -1)

    def supervised_contrastive_loss(self, embeddings, labels, tau=None):
        """
        论文Section 3.2: 监督对比损失 L_SUP-ins
        
        L_SUP-ins = -∑_{i∈B} 1/|N(i)| ∑_{p∈N(i)} log [ exp(φ(z_i)·φ(z_p)/τ) / 
                   ∑_n 1[n=i] exp(φ(z_i)·φ(z_n)/τ) ]
        """
        if tau is None:
            tau = self.temp
        
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        # 规范化嵌入
        embeddings = F.normalize(embeddings, dim=-1)
        
        # 创建掩码矩阵：同类样本为True
        labels = labels.reshape(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线（不与自己比较）
        mask = mask * (1 - torch.eye(batch_size, device=device))
        
        # 计算相似度矩阵
        exp_logits = torch.exp(torch.matmul(embeddings, embeddings.T) / tau)
        
        # 计算正样本对的指数相似度
        pos = torch.sum(exp_logits * mask, dim=1)
        
        # 计算所有正样本对的个数
        pos_count = torch.sum(mask, dim=1)
        
        # 避免除以0
        pos_count = torch.clamp(pos_count, min=1)
        
        # 计算分母
        denom = torch.sum(exp_logits, dim=1)
        
        # 防止log(0)
        loss = -torch.mean(torch.log(pos / (denom + 1e-8) + 1e-8))
        
        return loss

    def pseudo_supervised_contrastive_loss(self, embeddings, pseudo_labels, tau=None):
        """
        论文Section 3.3.1: 伪监督对比损失
        
        L_PSUP-ins 和 L_PSUP-cls 结构相同
        L_PSUP = -∑_{j∈B} 1/|N(j)| ∑_{q∈N(j)} log [ exp(φ(z_j)·φ(z_q)/τ) / 
                 ∑_n 1[n=j] exp(φ(z_j)·φ(z_n)/τ) ]
        """
        if tau is None:
            tau = self.temp
        
        device = embeddings.device
        batch_size = embeddings.size(0)
        
        # 规范化嵌入
        embeddings = F.normalize(embeddings, dim=-1)
        
        # 创建掩码矩阵：同伪标签为True
        pseudo_labels = pseudo_labels.reshape(-1, 1)
        mask = torch.eq(pseudo_labels, pseudo_labels.T).float().to(device)
        
        # 移除对角线
        mask = mask * (1 - torch.eye(batch_size, device=device))
        
        # 计算相似度矩阵
        exp_logits = torch.exp(torch.matmul(embeddings, embeddings.T) / tau)
        
        # 计算损失
        pos = torch.sum(exp_logits * mask, dim=1)
        pos_count = torch.sum(mask, dim=1)
        pos_count = torch.clamp(pos_count, min=1)
        denom = torch.sum(exp_logits, dim=1)
        
        loss = -torch.mean(torch.log(pos / (denom + 1e-8) + 1e-8))
        
        return loss

    def maximal_mutual_information_loss(self, predictions, labels):
        """
        论文Section 3.3.2: 最大互信息损失
        
        L_MI = H(Y) - H(Y|Z)
        
        其中：
        - H(Y) = -∑_r P(Y=r) log P(Y=r) 
          P(Y=r)为小批次中类别级投影头预测为第r类的均值
          促使投影头输出多样标签
          
        - H(Y|Z) = -1/|B| ∑_j ∑_r p_{j,r} log p_{j,r}
          p_{j,r}为投影头对x_j的第r类预测值
          可最小化投影头输出熵，提升预测置信度
        """
        # predictions: [batch_size, num_classes] (sigmoid或softmax之后)
        # labels: [batch_size, num_classes] (one-hot或概率)
        
        batch_size = predictions.size(0)
        
        # 确保预测是概率分布
        predictions = F.softmax(predictions, dim=1)
        
        # 计算边缘分布熵 H(Y)
        # P(Y=r) = 小批次中预测为第r类的平均概率
        p_y = torch.mean(predictions, dim=0)
        # 防止log(0)
        p_y = torch.clamp(p_y, min=1e-8)
        h_y = -torch.sum(p_y * torch.log(p_y))
        
        # 计算条件分布熵 H(Y|Z)
        # 对每个样本求所有类别预测的交叉熵
        h_y_given_z = -torch.mean(
            torch.sum(predictions * torch.log(torch.clamp(predictions, min=1e-8)), dim=1)
        )
        
        # 互信息 = 边缘熵 - 条件熵
        # 论文目标：最大化互信息，所以损失为负的互信息
        mutual_information = h_y - h_y_given_z
        
        # 返回负的互信息作为损失（最小化损失 = 最大化互信息）
        mi_loss = -mutual_information
        
        return mi_loss

    def forward_initial_stage(self, emb_ins, out_cls, targets):
        """
        论文Section 3.2: 初始阶段（Session 0）
        
        L_0 = L_SUP-ins + α L_CE-cls
        
        Args:
            emb_ins: 实例级投影头的嵌入 [batch_size, emb_dim]
            out_cls: 类别级投影头的输出（logits）[batch_size, num_classes]
            targets: 标签 (one-hot或整数格式)
        """
        # 转换标签格式
        if targets.dim() == 2:  # one-hot
            targets_int = self.convert_vec_to_int(targets)
        else:
            targets_int = targets
        
        # L_SUP-ins: 实例级监督对比损失
        sup_ins_loss = self.supervised_contrastive_loss(emb_ins, targets_int)
        
        # L_CE-cls: 类别级交叉熵损失
        ce_cls_loss = self.ce(out_cls, targets_int)
        
        # 总损失
        total_loss = sup_ins_loss + self.alpha * ce_cls_loss
        
        return total_loss, sup_ins_loss, ce_cls_loss

    def forward_continuous_stage(self, emb_ins, emb_cls, out_cls, 
                                pseudo_labels_ins, pseudo_labels_cls):
        """
        论文Section 3.3-3.4: 连续发现阶段
        
        L_t = L_PSUP-ins + α L_PSUP-cls + β L_MI
        
        Args:
            emb_ins: 实例级嵌入 [batch_size, emb_dim_ins]
            emb_cls: 类别级嵌入 [batch_size, emb_dim_cls]
            out_cls: 类别级分类输出（logits）[batch_size, num_classes]
            pseudo_labels_ins: 实例级伪标签 [batch_size]
            pseudo_labels_cls: 类别级伪标签 [batch_size]
        """
        # L_PSUP-ins: 实例级伪监督对比损失
        psup_ins_loss = self.pseudo_supervised_contrastive_loss(emb_ins, pseudo_labels_ins)
        
        # L_PSUP-cls: 类别级伪监督对比损失
        psup_cls_loss = self.pseudo_supervised_contrastive_loss(emb_cls, pseudo_labels_cls)
        
        # L_MI: 最大互信息损失
        mi_loss = self.maximal_mutual_information_loss(out_cls, None)
        
        # 总损失 (论文Table 3: α=1, β=0.4)
        total_loss = psup_ins_loss + self.alpha * psup_cls_loss + self.beta * mi_loss
        
        return total_loss, psup_ins_loss, psup_cls_loss, mi_loss

    def forward(self, emb_ins, emb_cls, out_cls, targets, 
                pseudo_labels_ins=None, pseudo_labels_cls=None, stage='initial'):
        """
        前向传播
        
        Args:
            emb_ins: 实例级嵌入
            emb_cls: 类别级嵌入
            out_cls: 类别级分类输出
            targets: 真实标签（初始阶段）
            pseudo_labels_ins: 伪标签（连续阶段）
            pseudo_labels_cls: 伪标签（连续阶段）
            stage: 'initial' 或 'continuous'
        """
        if stage == 'initial':
            return self.forward_initial_stage(emb_ins, out_cls, targets)
        elif stage == 'continuous':
            return self.forward_continuous_stage(emb_ins, emb_cls, out_cls, 
                                                pseudo_labels_ins, pseudo_labels_cls)
        else:
            raise ValueError(f"Unknown stage: {stage}")

class OpenMaxSeparation(nn.Module):
    """
    论文Section 3.3.1: OpenMax样本分离机制
    
    使用OpenMax通过Weibull分布拟合已知类别特征分布，
    计算样本的未知类别得分，实现已知/新样本分离。
    
    OpenMax基于极值理论（EVT）：
    1. 计算未知类别得分 score_novel
    2. 计算校正权重向量 W_correction
    3. 计算开放集预测向量 p_novel
    4. 通过掩码mask_novel标记新样本
    """
    
    def __init__(self, num_known_classes=None):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.weibull_params = {}  # 存储Weibull参数
        
    def compute_weibull_params(self, features, labels, num_classes=None):
        """
        根据训练特征计算Weibull分布参数
        
        Args:
            features: [N, dim] 训练集特征
            labels: [N] 训练集标签
            num_classes: 已知类别数
        """
        if num_classes is None:
            num_classes = self.num_known_classes
        
        features = features.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        # 对每个已知类别拟合Weibull分布
        all_distances = []
        for c in range(num_classes):
            class_features = features[labels == c]
            
            # 计算到类别中心的距离
            center = np.mean(class_features, axis=0)
            distances = np.linalg.norm(class_features - center, axis=1)
            all_distances.extend(distances)
            
            # 简单的Weibull参数估计（使用距离统计量）
            # 实际实现中可用scipy.stats进行更精确拟合
            self.weibull_params[c] = {
                'center': center,
                'mean_dist': np.mean(distances),
                'std_dist': np.std(distances)
            }
        
        # 计算全局统计量用于自适应阈值
        all_distances = np.array(all_distances)
        self.global_mean_dist = np.mean(all_distances)
        self.global_std_dist = np.std(all_distances)
    
    def forward(self, features, predictions, known_labels, num_known_classes):
        """
        论文Algorithm: 无标签样本分离
        
        Args:
            features: [B, dim] 样本特征
            predictions: [B, num_classes] 分类预测
            known_labels: 已知类别的伪标签
            num_known_classes: 已知类别数
            
        Returns:
            mask_novel: [B] 新样本掩码 (True=新, False=已知)
            corrected_probs: [B, num_classes] 校正后的预测概率
        """
        batch_size = features.size(0)
        device = features.device
        
        # 获取预测概率
        probs = F.softmax(predictions, dim=1)  # [B, num_classes]
        
        # 计算样本到已知类别中心的距离
        features_np = features.cpu().detach().numpy()
        
        novel_scores = np.zeros(batch_size)
        
        for b in range(batch_size):
            feat = features_np[b]
            min_distance = float('inf')
            
            # 找到最近的已知类别中心
            for c in range(num_known_classes):
                if c in self.weibull_params:
                    center = self.weibull_params[c]['center']
                    distance = np.linalg.norm(feat - center)
                    min_distance = min(min_distance, distance)
            
            # 基于全局统计量的未知类别得分计算
            # 如果样本远离所有已知类别中心，则认为是新样本
            if hasattr(self, 'global_mean_dist'):
                # 使用自适应阈值：mean + 1.5*std（论文推荐）
                adaptive_threshold = self.global_mean_dist + 1.5 * self.global_std_dist
                novel_scores[b] = min_distance / (adaptive_threshold + 1e-8)
            else:
                # 降级方案：使用各类别的平均距离阈值
                mean_threshold = np.mean([v['mean_dist'] for v in self.weibull_params.values()])
                novel_scores[b] = min_distance / (mean_threshold + 1e-8)
        
        # 生成掩码：距离阈值标记新样本
        # 论文采用adaptively threshold，超过1.0则认为是新样本
        threshold = 1.0
        mask_novel = torch.from_numpy(novel_scores > threshold).bool().to(device)
        
        # 返回掩码和原始预测
        return mask_novel, probs


class PseudoLabelGenerator(nn.Module):
    """
    伪标签生成模块
    
    论文Section 3.3.1:
    1. OpenMax分离已知和新样本
    2. 对新样本进行k-means聚类得到伪标签
    3. 已知样本伪标签通过argmax得到
    """
    
    def __init__(self, n_clusters=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans = None
        
    def generate_pseudo_labels(self, features, mask_novel, predictions, 
                              num_known_classes, known_predictions=None):
        """
        根据OpenMax掩码生成伪标签
        
        Args:
            features: [B, dim] 样本特征
            mask_novel: [B] 新样本掩码  
            predictions: [B, num_classes] 分类预测
            num_known_classes: 已知类别数
            known_predictions: [B, num_known_classes] 已知类别预测（可选）
            
        Returns:
            pseudo_labels: [B] 伪标签
        """
        device = features.device
        batch_size = features.size(0)
        
        pseudo_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        probs = F.softmax(predictions, dim=1)
        
        # 已知样本：直接使用argmax
        known_mask = ~mask_novel
        if known_mask.sum() > 0:
            pseudo_labels[known_mask] = torch.argmax(probs[known_mask, :num_known_classes], dim=1)
        
        # 新样本：k-means聚类
        if mask_novel.sum() > 0:
            novel_features = features[mask_novel]
            novel_features_np = novel_features.cpu().detach().numpy()
            
            # 确定新类别数量（假设与已知类别数相同或通过启发式方法确定）
            n_novel_classes = self.n_clusters if self.n_clusters is not None else num_known_classes
            
            if len(novel_features_np) > n_novel_classes:
                # 进行k-means聚类
                kmeans = KMeans(n_clusters=n_novel_classes, random_state=42, n_init=10)
                novel_pseudo_labels = kmeans.fit_predict(novel_features_np)
                
                # 新样本的伪标签偏移（从num_known_classes开始）
                novel_pseudo_labels = torch.from_numpy(novel_pseudo_labels).long().to(device)
                pseudo_labels[mask_novel] = novel_pseudo_labels + num_known_classes
            else:
                # 样本数过少，直接分配
                pseudo_labels[mask_novel] = torch.randint(0, n_novel_classes, 
                                                          (mask_novel.sum(),), dtype=torch.long, device=device) + num_known_classes
        
        return pseudo_labels


class CELoss(nn.Module):
    """基础交叉熵损失"""
    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
    
    def convert_vec_to_int(self, label_vec):
        label_int = torch.argmax(label_vec, -1)
        return label_int
    
    def forward(self, embeddings, output_vec, targets):
        targets = self.convert_vec_to_int(targets)
        return self.xent_loss(output_vec, targets)


class SupConLoss(nn.Module):
    """
    监督对比学习损失函数
    结合有监督信息和对比学习的思想
    改进自 "Supervised Contrastive Learning" https://arxiv.org/abs/2004.11362
    """
    def __init__(self, alpha=0.5, temp=0.1):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        """
        改进的NT-Xent损失，更好的处理同类样本
        """
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            # 创建同类样本掩码
            mask = torch.eq(labels, labels.transpose(0, 1))
            # 删除对角线元素（不与自己比较）
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        
        # 计算logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        
        # 删除对角线元素
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        
        # 数值稳定性：减去最大值
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        
        # 计算概率
        exp_logits = torch.exp(logits)
        
        # 计算log概率（仅考虑同类样本）
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # 处理没有正样本的情况
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        # 计算平均log似然
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        
        return loss

    def convert_vec_to_int(self, label_vec):
        label_int = torch.argmax(label_vec, -1)
        return label_int

    def forward(self, embeddings, output_vec, targets):
        targets = self.convert_vec_to_int(targets)
        
        # 规范化嵌入
        normed_cls_feats = F.normalize(embeddings, dim=-1)
        
        # 分类损失
        ce_loss = (1 - self.alpha) * self.xent_loss(output_vec, targets)
        
        # 对比学习损失
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        
        return ce_loss + cl_loss
