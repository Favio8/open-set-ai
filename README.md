# Open-World ECG Classification via Multi-View Contrastive Maximal Mutual Information

## 1. Introduction

本项目将两篇论文的方法结合：

1. **原论文** (Zhou et al., Neural Networks 2024): [Open-World Electrocardiogram Classification via Domain Knowledge-Driven Contrastive Learning](https://www.sciencedirect.com/science/article/pii/S0893608024004751)

2. **MCMI 论文** (Zhao et al., Expert Systems with Applications 2025): Multi-View Contrastive Learning with Maximal Mutual Information for Continual Generalized Category Discovery

**核心思想**：在开集 ECG 分类中，先用领域知识驱动的对比学习训练已知类，再用 MCMI 的连续学习策略发现新类。

---

## 2. 算法详解

### 2.1 两阶段训练流程

| 阶段 | 原论文方法 | MCMI 改进 |
|------|-----------|-----------|
| **Stage 1** (已知类学习) | Supervised Contrastive + Cross-Entropy + Hard Negative 增强 | 相同 |
| **Stage 2** (新类发现) | Multi-Hypersphere Learning（学习紧凑球体） | **MCMI 连续学习**：pseudo-supervised contrastive + Mutual Information |

### 2.2 Stage 1: 领域知识驱动的监督对比学习

**损失函数**：
```
L = (1 - α) × L_CE + α × L_CL
```

- **L_CL**: Supervised Contrastive Loss — 拉近同类样本嵌入，推远异类样本
- **L_CE**: Cross-Entropy Loss — 标准分类损失
- **Hard Negative 生成**：利用临床知识修改 ECG 的诊断区域，生成难分负样本

**关键洞察**：临床专家知道每类 ECG 的诊断特征（如 AF 的 P 波消失、STE 的 ST 段抬升），通过修改这些诊断区域生成 hard negative，引导模型学习可区分的诊断特征。

### 2.3 Stage 2: MCMI 连续学习（新类发现）

**原论文方法**：Multi-Hypersphere Learning — 学习紧凑的球体表示，样本离球体中心太远则判定为新类

**MCMI 改进**（本项目采用）：

```
L_Stage2 = L_psup_ins + α × L_psup_cls + β × L_MI
```

- **L_psup_ins**: Pseudo-supervised Instance Contrastive — 在实例嵌入空间进行对比学习
- **L_psup_cls**: Pseudo-supervised Class Contrastive — **新增**：在类嵌入空间进行对比学习（这是 MCMI 的核心贡献）
- **L_MI**: Mutual Information Loss — **新增**：最大化输入特征与预测标签之间的互信息，解决对已知类的偏置问题

**为什么需要 Mutual Information？**

当发现新类时，模型倾向于将未知样本错误归类到已知类（因为已知类训练数据更多）。MCMI 通过最大化 I(x; y) ≈ H(y) - H(y|x) 鼓励类级投影头输出多样化的类别分布，从而缓解这一偏置。

### 2.4 开集识别：OpenMax Separation

使用与 MCMI 论文相同的 OpenMax 机制区分已知类和未知类：

1. **原型距离得分**：计算样本到各类原型的距离，指数衰减评分
2. **分类置信度**：最大 softmax 概率
3. **协议得分**：预测类别与最近原型类别是否一致

```
known_score = 0.55 × proto_score + 0.35 × cls_conf + 0.10 × agreement
novel_mask = (distance > threshold) OR (known_score < min_known_score)
```

### 2.5 多视图数据增强

- **View 1（弱增强）**：amplitude scale 0.95-1.05，gaussian noise 0.005，temporal shift 3%
- **View 2（强增强 hardneg）**：amplitude scale 0.80-1.20，gaussian noise 0.02，temporal shift 12%，lead dropout 50%，temporal mask 10%

---

## 3. 评估流程

### 3.1 评估逻辑 (`run_2_results.py`)

```
1. 加载 checkpoint（model_state_dict, separator_state, split_meta）
         ↓
2. 从 checkpoint 重建模型结构
         ↓
3. 从 checkpoint 加载 OpenMaxSeparation
         ↓
4. 使用 stage2_split_csv 加载测试数据集
         ↓
5. 推理：从测试集提取 feat、logits
         ↓
6. OpenMaxSeparation.predict_open_labels(feat, logits)
   → 返回：pred_known, pred_open, novel_mask, known_score, novel_score 等
         ↓
7. compute_open_set_metrics(y_true, y_pred, unknown_label, novel_scores)
   → 返回：accuracy, macro_f1, micro_f1, weighted_f1, old_macro_f1, new_f1, auroc
         ↓
8. 保存结果到 results_final/
```

### 3.2 评估指标

| 指标 | 描述 |
|------|------|
| `accuracy` | 整体分类准确率 |
| `macro_f1` | 所有类的宏平均 F1 |
| `micro_f1` | 微平均 F1 |
| `weighted_f1` | 按类样本数加权的 F1 |
| `old_macro_f1` | 仅已知类的宏平均 F1 |
| `new_f1` | 新类/未知类检测的 F1 |
| `auroc_known_vs_unknown` | 已知类 vs 未知类二分类的 AUROC |

### 3.3 预测输出

每个测试样本的预测详情：
- `pred_known`：已知类预测结果
- `pred_open`：含未知标签的开放预测
- `known_score`：已知类置信度（0~1）
- `novel_score`：1 - known_score
- `novel_mask`：预测为新类的布尔掩码
- `nearest_known`：最近的已知类原型索引
- `nearest_dist`：到最近原型的距离

---

## 4. 使用方法

### 4.1 环境配置

```bash
pip install -r requirements.txt
```

### 4.2 数据准备

1. 联系 shuang.zhou@connect.polyu.hk 获取数据集访问密码
2. 从 OneDrive 下载数据集：https://1drv.ms/f/c/f8be98f7ec1588fa/EhIXx0LLh1pAgdgRW14va0oBl_6x52s9fhRFL5Vk4omxGA?e=Z4ozg3
3. 将数据解压到项目 `data_path/` 目录下

### 4.3 训练模型

训练入口：`run_1_train.py`

**最小命令示例**：
```bash
python run_1_train.py --dataset CPSC18 --model resnet34 --transform_type hardneg --seed 1
```

**指定未知类**：
```bash
python run_1_train.py --dataset CPSC18 --model resnet34 --unseen_classes 4 --novel_clusters 1 --seed 1
```

**多未知类实验**：
```bash
python run_1_train.py --dataset YourDataset --model resnet34 --unseen_classes 4,6,8 --novel_clusters 3
```

### 4.4 评估模型

评估入口：`run_2_results.py`

```bash
python run_2_results.py --dataset CPSC18 --model resnet34 --transform_type hardneg --seed 1
```

### 4.5 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--unseen_classes` | 逗号分隔的未知类原始标签 | 数据集预设 |
| `--novel_clusters` | Stage 2 对新类样本聚成多少个伪簇 | 1 |
| `--distance_scale` | OpenMax 分离阈值参数 | 2.0 |
| `--min_known_score` | OpenMax 最小已知类得分 | 0.35 |
| `--normalize_signal` | 是否对每条 ECG 做按导联 z-score | False |

---

## 5. 引用

```bib
@article{zhou2024openecg,
  title={Open-world electrocardiogram classification via domain knowledge-driven contrastive learning},
  author={Zhou, Shuang and Huang, Xiao and Liu, Ninghao and Zhang, Wen and Zhang, Yuan-Ting and Chung, Fu-Lai},
  journal={Neural Networks},
  volume={179},
  pages={106551},
  year={2024},
  publisher={Elsevier}
}
```
