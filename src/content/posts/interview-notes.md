---
title: 面试八股与项目复盘
published: 2026-11-19
tags: 
  - 机器学习
  - Transformer
  - LLM
  - GRPO
category: 技术学习
draft: true
---

> 面试重点：基础知识、技术原理、项目深度、技术选型、前沿技术。
>
> 准备策略：不仅要知道技术是什么，还要能够解释为什么选择它、有什么替代方案、不同方案的优缺点。

---

# 第一部分：机器学习与深度学习基础

## 1. 机器学习基础

### 1.1 机器学习有哪些常见范式？

**监督学习（Supervised Learning）**

使用带标签的数据训练模型，学习输入 X 到输出 Y 的映射。

- 分类：预测离散类别，例如 SKU 是否同款。
- 回归：预测连续数值，例如商品价格。

**无监督学习（Unsupervised Learning）**

没有明确标签，通过数据自身结构发现规律。

- 聚类：K-Means、DBSCAN。
- 降维：PCA、t-SNE。
- 表征学习：自监督学习。

**半监督学习（Semi-supervised Learning）**

少量有标签数据 + 大量无标签数据，通过伪标签、一致性正则化等方式提升模型效果。

**自监督学习（Self-supervised Learning）**

通过数据本身构造监督信号，例如：

- BERT：Masked Language Modeling。
- GPT：Next Token Prediction。
- 对比学习：构造正负样本学习表征。

### 1.2 Bias 和 Variance

- Bias（偏差）：模型过于简单，无法充分拟合数据，容易欠拟合。
- Variance（方差）：模型对训练数据过于敏感，容易过拟合。

解决方法：

| 问题  | 解决方案                        |
| --- | --------------------------- |
| 高偏差 | 增加模型复杂度、增加特征、减少过强正则化        |
| 高方差 | 数据增强、Dropout、L2 正则化、早停、增加数据 |

### 1.3 过拟合和欠拟合

**过拟合：** 训练集表现很好，验证集表现差。

原因：

- 模型复杂度过高。
- 训练数据不足。
- 数据噪声较多。
- 训练时间过长。

解决：

- Dropout。
- Weight Decay。
- Early Stopping。
- 数据增强。
- 交叉验证。

**欠拟合：** 训练集和测试集效果都不好。

解决：

- 增加模型容量。
- 增加有效特征。
- 适当增加训练轮数。
- 减少正则化。

### 1.4 常见损失函数

**交叉熵损失（Cross Entropy Loss）**

用于分类任务。

二分类：

$ L = -[y log(p) + (1-y)log(1-p)] $

多分类：

$ L = -Σ y_i log(p_i) $

交叉熵衡量真实分布与预测分布之间的差异。

**MSE（Mean Squared Error）**

$ L_{\mathrm{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2 $

主要用于回归任务，对较大的预测误差惩罚更重。

**Focal Loss**

用于类别不平衡场景，降低容易分类样本的损失权重，重点关注困难样本。

FL(p_t) = -α(1-p_t)^γ log(p_t)

其中：

- α：类别平衡权重。
- γ：困难样本调节参数。

**Triplet Loss**

用于度量学习。

目标是让 Anchor 与 Positive 的距离小于 Anchor 与 Negative 的距离。

L = max(0, d(a,p)-d(a,n)+margin)

常用于人脸识别、图像检索、商品匹配。

### 1.5 常见优化器

**SGD**

θ = θ - η∇L

优点：简单、泛化表现有时较好。

缺点：对学习率敏感，收敛速度可能较慢。

**Momentum**

在 SGD 中加入历史梯度动量，减少震荡，加快收敛。

**Adam**

结合一阶动量和二阶动量：

- 一阶矩：梯度的指数移动平均。
- 二阶矩：梯度平方的指数移动平均。

优点：自适应学习率，训练稳定，适合深度学习。

**AdamW**

将 Weight Decay 从 Adam 的梯度更新中解耦，改善权重正则化效果。

LLM 训练中经常使用 AdamW。

### 1.6 Batch Size、Learning Rate、Epoch

- Batch Size：一次参数更新使用的样本数量。
- Iteration：一次参数更新。
- Epoch：完整遍历训练集一次。

关系：

Steps per Epoch ≈ N / Batch Size

Batch Size 增大：

- 梯度估计更加平滑。
- 显存消耗增加。
- 通常需要调整学习率。

Learning Rate 过大可能导致训练不稳定，过小则收敛慢。

---

# 第二部分：深度学习与 Transformer

## 2.1 神经网络基本结构

神经网络由输入层、隐藏层、输出层组成。

核心计算：

z = Wx+b

a = f(z)

常见激活函数：

**Sigmoid**

将输出映射到 (0,1)，常用于二分类输出层。

缺点：容易产生梯度消失。

**Tanh**

输出范围 (-1,1)，零中心化，但也存在梯度消失。

**ReLU**

f(x)=max(0,x)

优点：计算简单、缓解梯度消失。

缺点：可能出现 Dead ReLU。

**GELU**

结合输入值和概率分布进行平滑激活，常用于 Transformer。

## 2.2 前向传播与反向传播

前向传播：

输入经过网络计算，得到预测结果和 Loss。

反向传播：

利用链式法则计算 Loss 对每个参数的梯度。

参数更新：

θ = θ - η∇θL

本质是通过梯度下降最小化损失函数。

## 2.3 Transformer 整体架构

![Transformer 原始架构图](/images/tech/transformer-all-you-need-is-attention.png)

Transformer 最初采用 Encoder-Decoder 架构。

**Encoder：**

- Self-Attention。
- Feed Forward Network。
- Residual Connection。
- Layer Normalization。

**Decoder：**

- Masked Self-Attention。
- Cross-Attention。
- Feed Forward Network。
- Residual Connection。
- Layer Normalization。

常见模型：

| 模型   | 架构              | 主要任务     |
| ---- | --------------- | -------- |
| BERT | Encoder-only    | 理解、分类、编码 |
| GPT  | Decoder-only    | 自回归生成    |
| T5   | Encoder-Decoder | 文本到文本任务  |

现在主流生成式大模型大多采用 Decoder-only 架构。

## 2.4 Self-Attention 原理

输入经过三个线性变换：

Q = XWq

K = XWk

V = XWv

注意力计算：

Attention(Q,K,V) = Softmax(QKᵀ/√d_k)V

含义：

1. Q：当前 token 想寻找什么信息。
2. K：每个 token 提供什么匹配特征。
3. V：每个 token 实际携带的信息。
4. QKᵀ：计算 token 之间的相关程度。
5. Softmax：归一化注意力权重。
6. 与 V 相乘：对信息进行加权聚合。

为什么除以 √d_k？

随着维度增加，QK 内积的方差会增大，导致 Softmax 进入饱和区域，梯度变小。除以 √d_k 可以稳定数值分布。

## 2.5 Multi-Head Attention

将 Q、K、V 分成多个 Head 并行计算注意力。

不同 Head 可以学习不同的关系，例如：

- 语义关联。
- 位置关联。
- 长距离依赖。
- 局部上下文关系。

最后将各个 Head 的结果拼接，再经过线性映射。

## 2.6 Causal Mask

GPT 是自回归模型，预测当前 token 时不能看到未来 token。

例如：

Token 1 可以看到自己。

Token 2 可以看到 Token 1、2。

Token 3 可以看到 Token 1、2、3。

通过将未来位置的 Attention Score 设置为负无穷，实现因果掩码。

## 2.7 RoPE 位置编码

RoPE（Rotary Position Embedding）通过旋转矩阵将位置信息注入 Q 和 K。

核心特点：

- 将绝对位置信息编码到向量旋转中。
- Attention 内积能够体现相对位置关系。
- 支持一定程度的长度外推设计。

与传统绝对位置编码相比，RoPE 更适合现代 Decoder-only LLM。

## 2.8 LayerNorm 与 RMSNorm

LayerNorm：

对一个样本的特征维度计算均值和方差，再进行归一化。

RMSNorm：

不减去均值，只根据均方根进行缩放。

RMSNorm 计算更简单，很多现代 LLM 使用 RMSNorm。

## 2.9 Residual Connection

残差连接：

Output = F(x)+x

作用：

- 改善深层网络梯度传播。
- 缓解梯度消失。
- 使模型更容易学习恒等映射。

## 2.10 Transformer 计算复杂度

对于序列长度 N、隐藏维度 d：

Self-Attention 的主要计算复杂度约为：

O(N²d)

注意力矩阵的空间复杂度约为 O(N²)。

当上下文长度增大时，Attention 计算和显存开销都会明显增加。

---

# 第三部分：大模型训练与后训练

## 3.1 LLM 预训练、SFT、RLHF 的区别

**预训练（Pretraining）**

通过大量无监督或自监督文本进行 Next Token Prediction，让模型学习语言规律、知识和基础能力。

**SFT（Supervised Fine-Tuning）**

使用指令和高质量回答进行监督微调，让模型学会按照指令完成任务。

**RLHF（Reinforcement Learning from Human Feedback）**

通过人类偏好数据构建奖励信号，再使用强化学习优化模型输出。

典型流程：

Pretraining → SFT → Reward Modeling → PPO

现代方法也包括 DPO、GRPO 等，不一定都需要传统 PPO 流程。

## 3.2 SFT 原理

SFT 本质是监督学习。

训练数据：

(x, y)

模型根据输入 x 自回归生成 y。

目标：

最大化正确回答的条件概率：

L = -Σ log P(y_t | x,y\_\<t)

训练时通常采用 Teacher Forcing：

模型在预测当前位置时，输入前面的真实 Token，而不是自己上一步生成的 Token。

### 为什么使用 SFT？

- 让模型学习业务规则。
- 让模型遵循指定输出格式。
- 学习特定领域知识和任务模式。
- 提高任务准确率。

### SFT 的问题

- 对数据质量非常敏感。
- 容易受到错误标签影响。
- 可能出现过拟合。
- 只模仿示范答案，不一定能主动优化复杂推理过程。

## 3.3 LoRA 原理

LoRA（Low-Rank Adaptation）是一种参数高效微调方法。

原始权重：

W = W0 + ΔW

LoRA 将增量矩阵分解：

ΔW = BA

其中：

A ∈ R^(r×k)

B ∈ R^(d×r)

r 远小于原始维度。

训练时冻结 W0，只更新 A 和 B。

优点：

- 可训练参数少。
- 显存需求低。
- 训练速度快。
- 多个 LoRA Adapter 可以切换。

与全量微调相比，LoRA 不需要更新全部模型参数。

### LoRA 和 QLoRA

LoRA：冻结原始模型权重，训练低秩矩阵。

QLoRA：进一步将基础模型量化，通常使用 4-bit NF4，降低显存开销，再训练 LoRA 参数。

## 3.4 CoT（Chain of Thought）

CoT 是思维链，让模型通过中间推理步骤完成复杂任务。

例如：

输入：商品 A 和商品 B 是否同款？

直接回答：

是。

CoT 回答：

1. 检查商品名称。
2. 比较规格参数。
3. 比较型号。
4. 检查关键属性。
5. 得出是否同款的结论。

CoT 的价值：

- 将复杂问题拆分为多个推理步骤。
- 提供中间判断依据。
- 提高部分复杂任务的推理表现。
- 便于分析错误发生在哪一步。

### CoT 数据蒸馏

使用能力更强的 Teacher Model 生成推理链，再通过清洗、筛选、验证构建训练数据，训练 Student Model。

常见问题：

- Teacher 产生幻觉。
- 推理链冗长。
- 中间推理与最终答案不一致。
- 答案泄露。
- 训练数据存在逻辑错误。

解决：

- 规则校验。
- LLM Judge。
- 正则清洗。
- 答案一致性校验。
- 多模型交叉验证。
- 困难样本筛选。

## 3.5 PPO 原理

PPO（Proximal Policy Optimization）是一种策略梯度强化学习算法。

典型 RLHF 流程：

1. Actor 生成回答。
2. Reward Model 评估回答。
3. Critic 估计状态价值。
4. 计算 Advantage。
5. 使用 PPO Clip Loss 更新 Actor。
6. 使用 Value Loss 更新 Critic。

PPO Clip Objective：

L = min(r_t A_t, clip(r_t,1-ε,1+ε)A_t)

其中：

- r_t：新旧策略概率比。
- A_t：优势函数。
- ε：裁剪范围。

PPO 为什么需要 Critic？

Critic 估计 Value，辅助计算 Advantage，降低策略梯度估计方差。

PPO 的缺点：

- 需要 Actor、Critic、Reward Model、Reference Model 等多个模型。
- 显存开销大。
- 训练流程复杂。
- 超参数较多。

**1.PPO 最核心公式**

面试最常见的是 **PPO-Clip**：
$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta)\hat A_t,\; \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t \right) \right]
$$
其中最关键的概率比：
$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)} {\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$
各部分可以这样理解：

- $$
  \pi_{\theta_{\text{old}}}(a_t|s_t)
  $$

  ：旧策略采取动作 a_t 的概率

- $$
  \pi_\theta(a_t|s_t)
  $$

  ：新策略采取动作 a_t 的概率

- $$
  r_t(\theta)
  $$

  ：**新旧策略概率比**

- A^t：优势函数（Advantage），表示这个动作相比平均水平到底好不好

- ϵ\epsilon：裁剪范围，常见如 0.10.1 或 0.20.2

- `clip`：把 r_t 限制在 [1−ϵ,1+ϵ][1-\epsilon,1+\epsilon] 范围内

例如：ϵ=0.2

那么：rt∈[0.8,1.2]

PPO 的核心思想就是：**可以更新策略，但不要一次更新得太猛。**

------

**2. 为什么要有 Advantage？**

优势函数通常写成：
$$
A_t = Q(s_t,a_t)-V(s_t)
$$
直观理解：

At=采取这个动作有多好−这个状态下平均有多好

所以：At>0  说明这个动作比平均水平好，应该**提高这个动作的概率**。

反过来：At<0  说明这个动作不好，应该**降低这个动作的概率**。

实际 PPO 中通常使用 GAE（Generalized Advantage Estimation，广义优势估计）计算 A^t：
$$
\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
   \hat A_t = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} +\cdots
$$

面试如果没深入问，**不需要主动展开 GAE 公式**。

------

**3.Clip 到底在干什么？**

假设：
$$
\epsilon=0.2
$$
旧模型生成某个 token 的概率：
$$
\pi_{\text{old}}(a|s)=0.5
$$
新模型变成：
$$
\pi_\theta(a|s)=0.8
$$
那么：
$$
r=\frac{0.8}{0.5}=1.6
$$
说明策略变化太大。

PPO 会通过：
$$
\operatorname{clip}(1.6,0.8,1.2)=1.2
$$
限制继续从这种过大的策略变化中获得收益。

所以 PPO 并不是简单粗暴地说“参数不能变化”，而是：**限制新旧策略在动作概率上的变化**

**4.PPO 完整 Loss**

实际训练时通常不只有 Policy Loss，还会加入 Value Loss 和 Entropy：
$$
L = L_{\text{policy}} + c_1L_{\text{value}} - c_2H(\pi)
$$
其中：
$$
L_{\text{value}} = (V_\theta(s_t)-V_t^{target})^2
$$
而：
$$
H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)
$$
因此三个部分分别负责：

Policy Loss：优化策略\boxed{ \text{Policy Loss：优化策略} }Value Loss：让 Critic 更准确地估计价值\boxed{ \text{Value Loss：让 Critic 更准确地估计价值} }Entropy：鼓励探索，避免策略过早变得过于确定\boxed{ \text{Entropy：鼓励探索，避免策略过早变得过于确定} }

注意不同代码库对 Loss 的正负号定义可能不同，因为论文写的是最大化目标，而代码优化器通常做最小化。
$$
\min(r_t\hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)
$$

---

## 3.6 GRPO原理

GRPO（Group Relative Policy Optimization）是一种相对 PPO 简化的策略优化方法。

核心思想：

对于同一个 Prompt，采样一组回答，通过组内奖励相对比较计算 Advantage。

假设同一个 Prompt 生成 G 个回答：

o1,o2,...,oG

得到奖励：

r1,r2,...,rG

组内 Advantage：

A_i = (r_i - mean(r)) / std(r)

再使用 PPO 风格的策略优化目标更新模型。

### GRPO 相比 PPO 的优势

- 不需要单独训练 Critic。
- 使用组内奖励估计相对优势。
- 降低 Value Model 带来的显存开销。
- 更适合部分可验证奖励任务。

### GRPO 的局限

- 需要对同一 Prompt 采样多个回答。
- 生成成本较高。
- 奖励函数设计非常关键。
- 组内奖励差异不足时，训练信号可能较弱。

> GRPO 全称是 Group Relative Policy Optimization，也就是组相对策略优化。它可以理解为 PPO 在大模型强化学习场景下的一种改进。
>
> 它和 PPO 最大的区别在于 Advantage 的计算方式。PPO 通常需要额外训练一个 Critic 或 Value Model 来估计状态价值，而 GRPO 不需要单独的 Critic。
>
> GRPO 对同一个 Prompt 一次采样多个 Response，形成一个 Group，然后分别计算这些回答的 Reward，再利用组内 Reward 的均值和标准差进行标准化：
> $$
> A_i=\frac{R_i-\operatorname{mean}(R)}{\operatorname{std}(R)}
> $$
> 如果一个回答的 Reward 高于组内平均值，那么 Advantage 为正，就提高生成这个回答的概率；如果低于组内平均值，Advantage 为负，就降低它的概率。
>
> 在策略更新方面，GRPO 仍然使用类似 PPO 的 probability ratio 和 clipping mechanism，限制新旧策略变化过大；同时通常还会加入相对于 Reference Model 的 KL penalty，防止模型在强化学习过程中偏离原始模型太远。
>
> 所以我会把 GRPO 总结成一句话：**对同一个问题采样多个回答，通过组内相对奖励来估计 Advantage，替代 PPO 中的 Critic，再通过 PPO 式 Clip 和 KL 约束稳定地更新策略。**
>
> 它的主要优势是省掉了 Critic/Value Model，因此在大语言模型强化学习中可以降低额外的显存和计算开销。             

把 **GRPO（Group Relative Policy Optimization，组相对策略优化）**理解成：

> **对同一个问题一次采样多个回答，让这些回答在组内互相比较，用“相对好坏”作为 Advantage，从而不再需要单独训练 Critic/Value Model。**

------

**1. GRPO 最核心公式**

对于一个问题 qq，从旧策略中采样一组回答：
$$
\{o_1,o_2,\dots,o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot|q)
$$
然后 Reward Model 或规则奖励给每个回答打分：
$$
r_1,r_2,\dots,r_G
$$
GRPO 最关键的一步，是利用**组内奖励计算 Advantage**：
$$
A_i = \frac{r_i-\operatorname{mean}(r_1,\dots,r_G)} {\operatorname{std}(r_1,\dots,r_G)} 
$$
也就是：

Advantage=当前回答奖励−组内平均奖励组内奖励标准差

因此：

- Ai>0：这个回答比组内平均水平好 → **提高生成它的概率**
- Ai<0：这个回答比组内平均水平差 → **降低生成它的概率**
- Ai≈0：接近平均水平 → 更新较小

这就是 **Group Relative（组内相对）** 的含义。

------

**2. GRPO 的完整目标函数**

经典 GRPO 目标可以写成：
$$
\boxed{ J_{\text{GRPO}}(\theta) = \mathbb E \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_{i,t}(\theta)\hat A_i, \operatorname{clip} (r_{i,t}(\theta),1-\epsilon,1+\epsilon) \hat A_i \right) - \beta D_{\mathrm{KL}} \right) \right] }
$$
其中：
$$
r_{i,t}(\theta) = \frac{ \pi_\theta(o_{i,t}|q,o_{i,<t}) }{ \pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t}) }
$$
注意这里的 ri,t(θ) 是 **policy ratio（策略概率比）**，不要和前面的 reward ri混淆。

------

**3. 这个公式怎么理解？**

其实和 PPO 非常像，可以拆成三部分。

**第一部分：组内 Advantage**
$$
A_i = \frac{R_i-\mu_R}{\sigma_R}
$$
比如同一个数学题，模型生成 4 个答案：

| 回答     | Reward | Advantage 大致情况 |
| -------- | ------ | ------------------ |
| Answer 1 | 1.0    | 正                 |
| Answer 2 | 0.8    | 正                 |
| Answer 3 | 0.3    | 负                 |
| Answer 4 | 0.1    | 负                 |

那么 GRPO 就会认为：

> Answer 1、2 相对比较好，Answer 3、4 相对比较差。

因此不需要 Critic 告诉模型：V(s)=?

直接通过**同一个问题下多个回答之间的相对奖励**构造 baseline。

------

**第二部分：PPO 式 Clip**

GRPO 同样使用：
$$
\min \left( r_{i,t}\hat A_i, \operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i \right)
$$
它的作用和 PPO 基本一致：

防止新策略相对旧策略一次更新得太猛

所以可以记：
$$
\boxed{ GRPO = PPO\ Clip + Group\ Relative\ Advantage - Critic }
$$
这句话非常适合面试。

------

**4. KL 散度是干什么的？**

GRPO 通常还会加入参考模型 πref\pi_{\text{ref}} 的 KL 约束：
$$
D_{\mathrm{KL}} ( \pi_\theta \| \pi_{\text{ref}} )
$$
KL 常用无偏估计
$$
\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1
$$

---

目的就是：**防止强化学习后的模型偏离原来的参考模型太远。**

所以 β 控制 KL 惩罚强度：

β↑⇒对偏离 reference model 的惩罚越强

可以简单理解为：

Clip：限制相邻两次策略更新

KL：限制模型偏离ReferenceModel

------

**5. GRPO 和 PPO 最大区别**

PPO 通常需要 Critic / Value Model：At≈Rt−V(st)

也就是说，需要估计：V(st)

而 GRPO 不训练独立 Critic，而是对一个问题生成 G个回答：o1,o2,⋯ ,oG

通过组内奖励：
$$
A_i = \frac{R_i-\operatorname{mean}(R)} {\operatorname{std}(R)}
$$
估计回答的相对优势。

所以：PPO：Critic→Advantage

而：GRPO：Group Rewards→Advantage

这也是 GRPO 用在大模型 RL 中的一个重要优势：**省掉 Critic 带来的额外模型和显存/计算开销。** 

## 3.7 GRPO与PPO对比

| 维度       | PPO        | GRPO                 |
| ---------- | ---------- | -------------------- |
| Critic     | 需要       | 不需要独立 Critic    |
| Advantage  | GAE 等方法 | 组内相对奖励         |
| 采样       | 策略采样   | 同 Prompt 多回答采样 |
| 显存       | 较高       | 相对较低             |
| 训练复杂度 | 较高       | 相对简化             |

## 3.8 DPO原理

**DPO（Direct Preference Optimization，直接偏好优化）** 理解成：

> **不需要 Reward Model，也不需要 PPO/GRPO 那样在线做强化学习，而是直接利用“哪个回答更好”的偏好数据，让模型提高 preferred response 的概率、降低 rejected response 的概率。**
>
> DPO = Preference Data + Reference Model + 直接优化 Policy

> DPO 全称是 Direct Preference Optimization，也就是直接偏好优化。它主要用于大模型的 Preference Alignment。
>
> DPO 和传统 RLHF 最大的区别是，它不需要单独训练 Reward Model，也不需要再使用 PPO 做在线强化学习，而是直接利用 preference pair 来优化 Policy。
>
> DPO 的训练数据通常由 Prompt、Chosen Response 和 Rejected Response 构成，也就是 ywy_w 优于 yly_l。
>
> 它的核心 Loss 是：
> $$
> L_{\mathrm{DPO}} = -E\left[ \log\sigma \left( \beta \left[ \log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right] \right) \right]
> $$
> 其中 πθ\pi_\theta 是当前训练的 Policy Model，πref\pi_{\mathrm{ref}} 是冻结的 Reference Model。
>
> 这个公式直观上就是希望当前模型相对于 Reference Model，更加提高 Chosen Response 的相对概率，同时降低 Rejected Response 的相对概率。
>
> DPO 可以从带 KL 正则的 RLHF 目标推导出来，把 Reward 和最优 Policy 之间的关系代入 preference model，从而最终得到一个类似二分类或者 pairwise ranking 的优化目标。
>
> 所以 DPO 最大的优势就是：**把传统的 Reward Model + RL Policy Optimization 两阶段流程，简化成直接在偏好数据上优化 Policy，训练实现更加简单和稳定。

**1. DPO 的数据长什么样？**

DPO 的训练数据通常是一个三元组：(x,yw,yl)

其中：

- x：Prompt
- yw：chosen / preferred response，偏好的回答
- yl：rejected response，不偏好的回答

例如：

```
Prompt：解释一下什么是梯度下降。
Response A：正确、清晰、有逻辑。
Response B：存在明显错误。
人工偏好：A > B
```

DPO 的目标非常直接：P(yw∣x)↑P(yl∣x)↓

但这里有个关键问题：

> 不能无限制地提高 chosen、降低 rejected，否则模型可能偏离原来的 SFT 模型太远。

所以 DPO 引入了 **Reference Model（参考模型）**。

------

**2. DPO 最核心公式**
$$
\boxed{ L_{\mathrm{DPO}} = -\mathbb E_{(x,y_w,y_l)} \left[ \log\sigma \left( \beta \left[ \log\frac{\pi_\theta(y_w|x)} {\pi_{\mathrm{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)} {\pi_{\mathrm{ref}}(y_l|x)} \right] \right) \right] }
$$

------

**3. 第一层：Policy Model 和 Reference Model**

其中：πθ

是**当前正在训练的 Policy Model。**

而πref是：

> **冻结的 Reference Model。**

Reference Model 通常可以来自训练开始时的 SFT 模型副本。

所以：
$$
\log \frac{\pi_\theta(y_w|x)} {\pi_{\mathrm{ref}}(y_w|x)}
$$
表示：当前模型相对于 Reference Model，**对 chosen response 的偏好提高了多少**。

而：
$$
\log \frac{\pi_\theta(y_l|x)} {\pi_{\mathrm{ref}}(y_l|x)}
$$
表示：当前模型相对于 Reference Model，**对 rejected response 的偏好提高了多少**。

------

**4. 第二层：Chosen 和 Rejected 做差**

定义：
$$
z= \beta \left[ \log\frac{\pi_\theta(y_w|x)} {\pi_{\mathrm{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)} {\pi_{\mathrm{ref}}(y_l|x)} \right]
$$
DPO 希望：z→+∞

也就是：
$$
\boxed{ \log\frac{\pi_\theta(y_w|x)} {\pi_{\mathrm{ref}}(y_w|x)} > \log\frac{\pi_\theta(y_l|x)} {\pi_{\mathrm{ref}}(y_l|x)} }
$$
通俗理解就是：

> **相对于 Reference Model，我希望当前模型更加偏向 chosen，而不是 rejected。**

------

**5. 第三层：为什么有 sigmoid？**

DPO 使用：
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
然后：
$$
L=-\log\sigma(z)
$$
如果：z≫0

说明模型已经很好地区分：yw>yl

那么：σ(z)→1

所以：L→0L

反过来，如果模型更喜欢 rejected：z<0

那么 Loss 就会比较大。

因此 DPO 本质上可以理解成：

对 chosen / rejected 做一个 pairwise preference classification

------

**6. β是干什么的？**

β控制 Preference Optimization 与 Reference Policy 约束之间的权衡尺度。

它来自 DPO 对 KL-regularized RLHF 目标的推导。

> β\beta 是 DPO 中控制隐式 KL 正则/偏好优化尺度的重要超参数，它决定模型相对于 Reference Policy 进行偏好调整的强度。

------

**7. 为什么 DPO 不需要 Reward Model？**

传统 RLHF 一般是：SFT→Reward Model→PPO

首先收集：yw>yl

然后训练 Reward Model：rϕ(x,y)

再让 PPO：
$$
\max_\theta E[r_\phi(x,y)]
$$
也就是说：先把人类偏好学成 Reward Model，再通过 RL 优化 Policy。**

------

DPO 认为：既然最终目的就是让 Policy 学会人类偏好，那能不能直接从 preference pair 优化 Policy？

所以变成：Preference Data→DPO→Policy

省掉：Reward Model

同时也省掉：PPO Online RL\boxed{PPO\ Online\ RL}

所以：DPO最大优势=训练流程简单、稳定、成本较低

------

**8. DPO 和 SFT 有什么区别？**

SFT：
$$
L_{\mathrm{SFT}} = -\log\pi_\theta(y_w|x)
$$
核心：告诉模型：“这个答案是好的，你学习它。

但是它通常没有直接利用：y_l 的信息。

DPO：(x,yw,yl)

不仅告诉模型：A 是好的。还告诉模型：**A 比 B 更好。**

因此：SFT：学习正确答案\boxed{ SFT：学习正确答案 }

而：DPO：学习回答之间的相对偏好\boxed{ DPO：学习回答之间的相对偏好 }

------

**对比**

|                 | PPO                   | GRPO                   | DPO               |
| --------------- | --------------------- | ---------------------- | ----------------- |
| 数据            | Rollout               | Group Rollout          | Preference Pair   |
| Reward          | Reward Model / Reward | Reward / Verifier      | Preference 数据   |
| Critic          | 通常需要              | 不需要                 | 不需要            |
| 在线采样        | 需要                  | 需要                   | 通常不需要        |
| Reference Model | 通常有                | 通常有                 | 有                |
| 核心思想        | Advantage + Clip      | Group Advantage + Clip | Chosen > Rejected |

### DPO 优点

- 不需要显式训练 Reward Model。
- 不需要 PPO 采样和强化学习循环。
- 训练流程相对简单。
- 直接利用 chosen/rejected 数据。

### DPO 局限

- 对偏好数据质量敏感。
- 对 chosen/rejected 的相对排序进行优化，不直接衡量绝对质量。
- 仍需要参考模型。
- 可能受到长度偏置影响。

所以最核心区别：

PPO/GRPO：OnlineRL

DPO：OfflinePreferenceOptimization

更严谨地说，DPO 通常使用**预先收集好的 preference dataset** 直接训练，而不是像 PPO/GRPO 那样在每轮策略优化过程中进行 rollout 并依据 reward 更新。



## 3.9 SimPO原理

SimPO（Simple Preference Optimization）是一种 Reference-free 偏好优化方法。

与 DPO 最大的区别：

DPO 需要 Reference Model。

SimPO 不需要 Reference Model。

SimPO 使用策略模型自身的长度归一化序列对数概率作为隐式奖励：

r(x,y) = β/|y| × log πθ(y|x)

然后通过 Margin γ 要求 chosen 的隐式奖励比 rejected 高一定幅度。

核心目标：

r(x,y_w) - r(x,y_l) > γ

其中：

- β：奖励缩放系数。
- γ：偏好间隔 Margin。
- |y|：回答 Token 长度。

为什么使用长度归一化？

因为长回答包含更多 Token，累积 Log Probability 通常更负，直接比较序列概率可能引入长度偏差。

长度归一化可以让不同长度回答的奖励更具可比性。

**DPO与SimPO对比**

| 维度            | DPO                  | SimPO              |
| --------------- | -------------------- | ------------------ |
| Reference Model | 需要                 | 不需要             |
| 优化目标        | 相对参考模型的概率比 | 长度归一化隐式奖励 |
| Margin          | 通过偏好目标隐式约束 | 显式引入 γ         |
| 训练成本        | 较高                 | 相对较低           |
| 长度处理        | 原始序列概率形式     | 显式长度归一化     |

选择建议：

如果希望利用参考模型约束策略更新，可以选择 DPO。

如果希望降低模型资源开销，并尝试直接优化长度归一化偏好奖励，可以考虑 SimPO。

## 3.10 DAPO

> DAPO 全称是 Decoupled Clip and Dynamic Sampling Policy Optimization，我把它理解成 GRPO 针对大模型长 CoT 强化学习场景的一系列改进。
>
> 相比 GRPO，DAPO 主要有四个改进。
>
> **第一个是 Clip-Higher。** GRPO 类似 PPO，通常使用对称的 clipping，比如 [0.8,1.2][0.8,1.2]。DAPO 将上下 clipping bound 解耦，适当提高上界，比如变成 [0.8,1.28][0.8,1.28]。这样可以让低概率但是高 Advantage 的 Token 获得更大的概率提升空间，增强 exploration，同时缓解 entropy collapse。
>
> **第二个是 Dynamic Sampling。** GRPO 如果对一个 Prompt 采样出来的 Response 全对或者全错，那么组内 Reward 没有差异，Relative Advantage 就没有有效学习信号。<u>DAPO 会动态过滤这些 Prompt，</u>把训练资源更多用于既有正确回答又有错误回答的样本。
>
> **第三个是 Token-Level Policy Gradient Loss。** GRPO 的常见实现<u>会先在每条 Response 内对 Token Loss 求平均，再在样本之间平均</u>；DAPO <u>改成对所有有效 Token 统一聚合，使长 CoT 场景下的 loss weighting 更合理。</u>
>
> **第四个是 Overlong Reward Shaping。** <u>对于接近最大生成长度的回答，DAPO 不采用突然的硬惩罚，而是逐渐增加长度惩罚，从而降低因为截断造成的 reward noise。</u>
>
> 所以我会把 DAPO 总结成：**它保留了 GRPO 不需要 Critic 和组内相对 Advantage 的优点，然后通过 Clip-Higher、Dynamic Sampling、Token-Level Loss 和 Overlong Reward Shaping，提高长链推理强化学习的训练稳定性和效率。**                    

可以把 **DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）** 理解为：

> **DAPO = GRPO + 针对大模型长 CoT 强化学习的 4 个关键修正。**

它没有推翻 GRPO。**组内采样、组内相对 Advantage、PPO-style policy ratio** 这些核心思想仍然保留，但针对 GRPO 实际训练 reasoning model 时出现的几个问题，加入了 **Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping** 四项技术。DAPO 原论文报告，在 Qwen2.5-32B Base 上达到 AIME 2024 50 分，并以约一半训练步数超过其对比的 DeepSeek-R1-Zero-Qwen-32B。

| GRPO 的问题                                    | DAPO 改进                   |
| ---------------------------------------------- | --------------------------- |
| Clip 对探索限制太强，可能出现 entropy collapse | **Clip-Higher**             |
| 一个 Prompt 的回答全对/全错，没有有效梯度      | **Dynamic Sampling**        |
| Sequence-level loss 对不同长度回答的处理不理想 | **Token-Level PG Loss**     |
| 超长回答直接惩罚会产生 reward noise            | **Overlong Reward Shaping** |

1、GRPO 使用对称 Clip，而 DAPO 使用非对称 Clip，适当放宽 probability ratio 的上界，使低概率但高收益的 Token 有更大的提升空间，从而增强探索并缓解 entropy collapse。

DAPO 把上下界**解耦**：
$$
\boxed{ \operatorname{clip} (r,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}}) }
$$
DAPO 就把上界放宽：1.2→1.28

2、Dynamic Sampling 会过滤组内全对或者全错的 Prompt，因为这些样本的 Reward 方差为 0，无法提供有效的 relative advantage，从而减少无效训练样本

3、DAPO 改成 **Token-Level Policy Gradient Loss**：
$$
\boxed{ L_{\text{DAPO}} = \frac{1} {\sum_i |o_i|} \sum_i \sum_t L_{i,t} }
$$
也就是：

> **把整个 batch/group 里的有效 Token 放在一起做归一化。**

4、快接近最大长度的时候逐渐增加惩罚，而不是超过长度后突然一刀切。

## 3.11 Reward Function 奖励函数设计

奖励函数决定模型倾向于优化什么目标。

常见奖励：

**正确性奖励**

判断最终答案是否正确。

例如：

正确：+1

错误：0

**格式奖励**

检查输出是否符合规定格式，例如 JSON、XML 或指定标签。

**长度奖励**

控制模型输出长度，避免冗长推理。

需要注意：不能简单鼓励所有回答越短越好，否则可能导致模型跳过必要推理。

**过程奖励**

对中间推理步骤进行评价。

### Reward Hacking

模型可能找到奖励函数漏洞，获得高奖励，但实际任务表现没有提高。

解决：

- 设计多维奖励。
- 使用独立 Benchmark。
- 增加规则校验。
- 关注奖励与真实任务指标的一致性。

## 3.12 DeepSpeed

DeepSpeed 是微软推出的深度学习训练优化框架。

核心技术是 ZeRO。

**ZeRO Stage 1**

切分 Optimizer States。

**ZeRO Stage 2**

进一步切分 Gradients。

**ZeRO Stage 3**

进一步切分 Model Parameters。

| 阶段      | 优化器状态 | 梯度 | 参数 |
| ------- | ----- | -- | -- |
| Stage 1 | 切分    | 复制 | 复制 |
| Stage 2 | 切分    | 切分 | 复制 |
| Stage 3 | 切分    | 切分 | 切分 |

适用场景：

大模型训练时显存不足，需要通过参数、梯度和优化器状态分片降低单卡显存占用。

---

# 第四部分：多模态大模型与推理优化

## 4.1 多模态模型是什么？

多模态模型可以同时理解或处理多种类型的信息。

常见模态：

- 文本。
- 图片。
- 视频。
- 音频。
- 结构化数据。

典型架构：

Image → Vision Encoder → Projector → LLM

其中：

1. Vision Encoder 提取图像特征。
2. Projector 将视觉特征映射到 LLM 的隐藏空间。
3. LLM 接收视觉 Token 和文本 Token。
4. 通过语言模型完成理解和生成。

## 4.2 ViT 原理

ViT（Vision Transformer）将图片切分为多个 Patch。

每个 Patch 展平并映射成向量，再加入位置编码，输入 Transformer Encoder。

基本流程：

Image → Patch Embedding → Position Embedding → Transformer → Visual Features

与 CNN 相比：

CNN 使用局部卷积和空间归纳偏置。

ViT 使用 Self-Attention 建模全局关系。

## 4.3 为什么冻结 ViT，只微调 LLM？

冻结 ViT 的原因：

1. 视觉 Encoder 已经具有一定的通用视觉表征能力。
2. SKU 同款任务的核心优化可能更多集中在语义理解和跨模态推理。
3. 冻结 ViT 可以减少训练参数和显存开销。
4. 降低小规模业务数据下视觉表征过拟合的风险。
5. 加快训练速度，提高实验迭代效率。

但它并不是永远最优。

替代方案：

**方案一：全量微调 ViT + LLM**

优点：视觉和语言模块可以共同适应业务数据。

缺点：显存开销大，训练成本高，也更容易过拟合。

**方案二：仅微调 ViT**

适合视觉特征需要明显适配、语言模型已经足够稳定的情况。

**方案三：冻结 ViT + 微调 Projector 和 LLM**

在冻结视觉主干的同时，让视觉特征映射层适应任务。

**方案四：LoRA 微调视觉和语言模块**

通过低秩参数更新降低训练成本。

面试回答：

> 我们最初采用冻结 ViT、仅微调 LLM 的方案，主要是考虑到视觉编码器已经具备较强的通用特征提取能力，而任务优化重点在于商品属性理解、跨模态信息融合和判别推理。冻结视觉主干可以减少可训练参数、降低显存开销，并控制小规模业务数据下的过拟合风险。当然，如果实验发现视觉特征存在明显领域偏移，也可以进一步尝试解冻高层 ViT、微调 Projector 或使用视觉 LoRA，通过消融实验比较效果和成本。
>

## 4.4 VLM 与 LLM 的区别

LLM 主要处理文本。

VLM（Vision Language Model）同时处理视觉和语言信息。

例如：

- Qwen-VL。
- InternVL。
- LLaVA。

典型应用：

- 图片理解。
- OCR。
- 图文匹配。
- 多模态问答。
- 商品识别。

## 4.5 vLLM 原理

vLLM 是大模型高性能推理和服务框架。

核心技术之一：PagedAttention。

传统 KV Cache：

通常需要为每个请求分配连续的显存空间，容易出现显存碎片和空间浪费。

PagedAttention：

将 KV Cache 划分成固定大小的 Block，通过 Block Table 管理不同请求的 KV Cache。

优点：

- 降低显存碎片。
- 提高 KV Cache 利用率。
- 支持高并发请求。
- 提高吞吐量。

### Continuous Batching

传统静态 Batch：

等待整个 Batch 中所有请求结束，才能处理下一批。

Continuous Batching：

某个请求结束后，立即加入新的请求。

优势：

- 减少 GPU 空闲。
- 提高并发处理能力。
- 提高吞吐量。

### Prefill 与 Decode

Prefill：

处理输入 Prompt，计算输入 Token 对应的 KV Cache。

特点：计算密集。

Decode：

逐 Token 自回归生成。

特点：通常更受内存带宽和 KV Cache 访问影响。

---

# 第五部分：RAG 与 Agent

## 5.1 RAG 是什么？

RAG（Retrieval-Augmented Generation）检索增强生成。

核心流程：

1. 文档加载。
2. 文本清洗。
3. Chunk 切分。
4. Embedding 向量化。
5. 向量数据库存储。
6. 用户提出 Query。
7. 检索相关文档。
8. 将检索结果和 Prompt 组合。
9. LLM 生成答案。

RAG 的主要价值：

让模型能够使用外部知识，降低知识过时和部分幻觉问题。

## 5.2 为什么需要 Chunk？

文档过长时，直接整体 Embedding 会造成：

- 语义粒度过粗。
- 检索定位不准确。
- 超出模型上下文限制。
- 召回结果包含大量无关信息。

因此需要切片。

常见策略：

- 固定长度切片。
- 固定长度 + Overlap。
- 按段落切片。
- 按语义边界切片。
- 父子 Chunk。

Chunk 太小：上下文不完整。

Chunk 太大：检索噪声增多。

## 5.3 Embedding

Embedding 将文本映射到向量空间。

语义相近的文本通常具有更高的向量相似度。

常见相似度：

Cosine Similarity：

cos(A,B) = A·B/(||A|| ||B||)

常用于向量检索。

## 5.4 BM25

BM25 是基于词项匹配的经典稀疏检索算法。

核心考虑：

- TF：词项在文档中的出现频率。
- IDF：词项在整个语料中的区分能力。
- 文档长度归一化。

优点：

- 不需要训练 Embedding 模型。
- 对关键词、专有名词、精确术语检索有效。
- 可解释性较好。

缺点：

- 不擅长语义相似但关键词不同的表达。
- 对同义词和上下文理解有限。

## 5.5 BM25 与向量检索对比

| 维度    | BM25 | 向量检索  |
| ----- | ---- | ----- |
| 原理    | 词项匹配 | 语义相似度 |
| 表达差异  | 较弱   | 较强    |
| 精确关键词 | 较好   | 不一定   |
| 同义词   | 较弱   | 较好    |
| 计算    | 倒排索引 | 向量索引  |

为什么使用混合检索？

BM25 擅长精确关键词，向量检索擅长语义匹配，两者具有互补性。

可以使用 RRF 融合：

RRF(d) = Σ 1/(k + rank_i(d))

其中 rank_i(d) 是文档在第 i 个检索结果中的排名。

## 5.6 Reranker

Reranker 是二阶段排序模型。

第一阶段：召回大量候选文档。

第二阶段：使用更精细的模型对 Query 和 Document 进行相关性评分。

常见方式：

- Cross-Encoder。
- BGE Reranker。
- Cohere Rerank。

优点：可以提升最终排序质量。

缺点：增加推理耗时。

## 5.7 Agent 是什么？

Agent 不只是调用 LLM，而是让模型具备：

- 感知任务。
- 任务规划。
- 工具选择。
- 执行操作。
- 获取反馈。
- 根据结果继续决策。

常见架构：

用户输入 → LLM 判断 → Tool Calling → 工具执行 → 返回结果 → LLM 总结。

## 5.8 Function Calling / Tool Calling

模型根据工具描述生成结构化调用参数。

例如：

用户：查询某个商品库存。

模型决定调用：

get_inventory(product_id="123")

程序执行函数，将结果返回模型。

需要注意：真正执行工具的是外部程序或 Agent Runtime，而不是模型本身直接执行代码。

## 5.9 MCP

MCP（Model Context Protocol）是连接 AI 应用与外部工具、资源的标准化协议。

主要角色：

- Host：AI 应用。
- Client：Host 内部负责连接 Server 的客户端。
- Server：暴露工具、资源和提示模板的服务端。

MCP 主要解决不同 AI 应用与工具之间的集成问题。

与 Tool Calling 的区别：

Tool Calling 是模型决定调用什么工具及传递什么参数。

MCP 是标准化工具和资源如何向 AI 应用暴露。

两者不是互斥关系，MCP Server 提供的工具可以被 Agent 通过 Tool Calling 使用。

## 5.10 LangChain 与 LangGraph

LangChain：

提供 LLM 应用开发组件，例如 Prompt、Model、Retriever、Tool、Chain。

LangGraph：

面向有状态、多步骤、循环式 Agent 工作流。

适合：

- 多步骤任务。
- 条件分支。
- 循环执行。
- 状态持久化。
- Human-in-the-loop。

---

# 第六部分：拼多多 Temu 多模态 SKU 同款匹配项目

这是本次面试最重要的项目，必须能够完整讲清楚。

## 6.1 项目整体链路

任务目标：

判断两个商品是否属于同款商品。

输入：

- 商品标题。
- 商品图片：主图+概览图
- 商品规格。
- 商品属性：类别、平台...
- 其他业务信息。

输出：

是否同款，以及必要的判断依据。

完整流程：

数据构建 → GT 清洗 → CoT 蒸馏 → 数据质量优化 → 多模态 SFT → GRPO → 推理优化 → Benchmark 评测。

## 6.2 为什么选择大模型，而不是传统匹配模型？

替代方案：

**方案一：规则匹配**

根据品牌、型号、规格、属性等字段进行规则判断。

优点：简单、可解释、成本低。

缺点：难以处理字段缺失、表达不一致和复杂语义。

**方案二：传统机器学习**

提取标题相似度、属性相似度、图片相似度等特征，使用 XGBoost、LightGBM 进行分类。

优点：训练成本低，适合结构化特征。

缺点：依赖人工特征工程，对复杂多模态关系建模能力有限。

**方案三：双塔模型**

分别编码商品 A、B，得到 Embedding，通过相似度判断是否同款。

优点：可以预先计算商品向量，适合大规模召回。

缺点：交互信息不足，对细粒度属性差异的建模可能不够充分。

**方案四：Cross-Encoder**

将两个商品一起输入模型进行匹配判断。

优点：能够充分建模两个商品之间的交互关系。

缺点：每对商品都需要重新计算，推理成本高。

**方案五：多模态大模型**

联合理解文本、图片和属性信息，通过指令和推理链完成复杂判断。

优点：具备较强的跨模态理解、语义推理和复杂规则处理能力。

缺点：训练成本、推理成本较高，对数据质量和输出稳定性要求较高。

面试回答：

我们的任务不仅涉及标题语义相似度，还涉及图片、规格和细粒度属性的一致性判断。规则和传统模型可以处理部分简单匹配，但对于字段缺失、表达差异以及复杂商品属性关系，存在一定局限。双塔模型适合大规模召回，Cross-Encoder 更适合精细匹配，而多模态大模型能够统一处理图文信息，并通过指令学习业务规则。因此项目选择多模态大模型作为核心判别模型，同时需要关注推理成本和数据质量。

## 6.3 七维辅助标注体系

项目中设计了七维辅助标注体系，基于 Kimi K2 完成多轮自动化标注，覆盖率达到 99.4%。

面试可能追问：

为什么不直接让模型输出最终答案？

因为仅有最终二分类标签，无法充分反映商品匹配的判断依据。

辅助标注可以将复杂任务拆分为多个可分析的维度，帮助：

- 数据质量检查。
- 下游模型的错误归因。
- 业务规则分析。
- 训练样本筛选。

## 6.4 为什么要构建独立 Benchmark？

项目中构建了 5000 条独立 Benchmark。

原因：

训练集效果不能代表模型的真实泛化能力。

独立 Benchmark 可以用于：

- 比较不同模型。
- 比较不同数据策略。
- 比较不同训练方案。
- 分析困难样本。
- 进行消融实验。

为什么要分层采样？

避免随机抽样导致某些类别、困难样本或特殊场景占比不足。

为什么要清洗 GT？

错误标签会让模型学习错误监督信号，降低训练效果和评测可信度。

## 6.5 为什么需要 CoT 数据蒸馏？

项目中：

- 主导 149 万 SKU 对冷启动数据构造。
- 教师模型蒸馏生成 46.3 万条正负平衡推理链。
- 设计 LLM 改写与正则清洗流水线。
- 答案泄露残留率从 87.6% 降到 0。

为什么不用原始数据直接 SFT？

原始数据通常只有最终标签，缺少中间判断过程。

通过教师模型生成推理链，可以让学生模型学习如何分析商品属性、比较差异、得出结论。

为什么选择 Teacher-Student 蒸馏？

教师模型能力较强，但直接部署成本可能较高。

通过教师生成高质量数据，再训练较小的学生模型，可以在一定程度上兼顾能力和推理成本。

### CoT 数据还有哪些构造方式？

**方案一：人工标注**

质量可控，但成本高、效率低。

**方案二：直接使用 Teacher 生成**

速度快，但可能产生幻觉、冗余推理和错误答案。

**方案三：Self-Consistency**

同一个问题采样多条推理路径，通过投票或一致性筛选提高可靠性。

缺点：生成成本增加，且多数一致不代表一定正确。

**方案四：规则约束生成**

结合结构化输出、字段校验和业务规则限制生成内容。

优点：可控性强。

**方案五：Teacher 生成 + LLM 改写 + 规则清洗 + 独立验证**

这是比较完整的数据质量优化流程。

## 6.6 什么是答案泄露？

答案泄露（Answer Leakage）指推理链或训练输入中提前包含了最终答案，导致模型不需要真正推理就能预测结果。

例如：

输入：判断商品 A 和 B 是否同款。

错误的推理数据：

第一步：判断结果为同款。

第二步：比较商品规格。

第三步：输出同款。

这样模型可能直接学习答案与输出之间的关联，而不是学习有效推理。

解决：

1. 识别推理链中提前暴露结论的表达。
2. 使用正则规则清洗固定模式。
3. 使用 LLM 改写不合格推理链。
4. 检查最终答案与原始标签是否一致。
5. 通过抽样人工审核和自动化检测评估残留率。

为什么不能只用正则？

因为答案泄露的表达形式不固定，单纯规则很难覆盖所有语义变体。

为什么不能只用 LLM？

LLM 也可能漏检，甚至引入新的错误。

因此结合规则与 LLM 更稳妥。

## 6.7 为什么选择 Qwen3.5-2B？

项目中使用 Qwen3.5-2B 完成多模态 SFT。

模型选择通常需要考虑：

- 模型的多模态能力。
- 参数规模。
- GPU 显存限制。
- 训练速度。
- 推理成本。
- 业务任务适配能力。

为什么不直接使用更大的模型？

更大的模型可能具备更强的通用能力，但训练和推理成本更高。

如果小模型经过高质量数据训练后已经能够满足业务指标，就没有必要盲目追求参数规模。

注意：具体模型选择依据、显存预算和实际对比模型，以你的真实实验记录为准。

## 6.8 为什么 SFT 后 F1 从 27% 提升到 85%？

可以从三个方面分析：

1. 数据质量提升：清洗错误标签，优化正负样本比例。
2. 推理链增强：让模型学习更结构化的商品比较过程。
3. 任务适配：通过 SFT 将通用模型调整为 SKU 同款匹配模型。

但不能简单将提升全部归因于 CoT 或某一个训练因素。

应通过消融实验分别验证：

- 原始数据 SFT。
- 清洗数据 SFT。
- CoT 数据 SFT。
- 不同数据比例。
- 不同模型训练策略。

## 6.9 为什么 GRPO 不直接使用 SFT？

SFT 的优化目标是模仿示范答案。

GRPO 则根据模型采样回答的奖励信号优化策略。

SFT 的局限：

模型可能生成正确答案，但推理过程冗长、存在冗余内容。

GRPO 可以通过任务正确性和输出长度奖励，让模型在保持任务效果的同时优化推理行为。

### 项目中的奖励函数

根据项目经历，使用了：

1. 任务正确性奖励。
2. 输出长度奖励。

目标：

提高最终任务正确性，同时减少不必要的推理 Token。

### 为什么不直接用 PPO？

PPO 需要额外的 Critic/Value Model，并涉及更复杂的强化学习训练流程。

GRPO 通过同一 Prompt 下的多条采样回答进行组内相对奖励比较，不需要独立 Critic，因此可以减少一部分模型和显存开销。

### 为什么不用 DPO？

DPO 主要依赖离线 chosen/rejected 偏好数据。

GRPO 可以直接根据当前策略生成的回答和奖励信号进行优化，更适合希望在线探索不同推理轨迹、优化任务正确性和长度的场景。

两者并非互斥：

- 有高质量偏好数据时，可以考虑 DPO。
- 有可验证奖励或可设计的任务奖励时，可以考虑 GRPO。

## 6.10 为什么 CoT Token 降低 80%，F1 仍然不下降？

这是非常值得深入准备的问题。

可能的原因：

1. 原始推理链存在大量重复描述。
2. 部分推理步骤对最终判断没有贡献。
3. 业务规则可以通过更简洁的方式表达。
4. GRPO 的奖励函数鼓励模型减少冗余输出。
5. 二阶段 SFT 进一步让模型内化业务规则。

关键是：压缩无效推理，而不是删除必要的判断依据。

评价时不能只看 Token 数量，还要同时观察：

- F1。
- Precision。
- Recall。
- 推理正确率。
- 输出格式合规率。
- 平均推理长度。
- 推理延迟。

## 6.11 为什么采用二阶段 SFT？

项目经历中，先通过 GRPO 优化推理轨迹，再进行采样轨迹筛选和短 Prompt 构建二阶段 SFT 数据。

可以这样理解：

第一阶段 SFT：让模型学会完成任务。

GRPO：优化模型生成的推理轨迹。

第二阶段 SFT：将经过筛选的高质量轨迹重新蒸馏到模型中，让模型在不依赖复杂 Prompt 的情况下学习业务规则。

优势：

- 减少对长 Prompt 的依赖。
- 内化业务规则。
- 压缩推理长度。
- 降低推理成本。

注意：第二阶段 SFT 不能简单理解为 GRPO 的替代品，它是将强化学习探索得到的优质行为重新转化为监督数据。可以理解成我的项目中，GRPO压缩输出token，SFT压缩输入的token。

## 6.12 项目中可能被追问的指标

**Precision**

预测为正样本的结果中，真正为正的比例。

Precision = TP/(TP+FP)

**Recall**

所有真实正样本中，被模型找出来的比例。

Recall = TP/(TP+FN)

**F1**

Precision 和 Recall 的调和平均：

F1 = 2PR/(P+R)

为什么不用 Accuracy？

当正负样本不平衡时，模型全部预测为多数类也可能取得较高 Accuracy，但实际识别能力很差。

F1 可以综合考察 Precision 和 Recall。

---

# 第七部分：科研项目——LLM + Vine Copula 因果数据增强

## 7.1 项目解决什么问题？

研究问题：

小样本随机对照试验中，存在选择偏差和目标域协变量分布变化时，如何利用数据增强提高因果效应估计的稳定性？

传统问题：

源域拥有完整的 X、T、Y。

目标域可能只有 X，没有处理变量 T 和结果变量 Y。

需要利用源域的因果信息和目标域协变量分布，估计目标人群中的 ATE/CATE。

## 7.2 ATE 和 CATE

ATE（Average Treatment Effect）：

总体平均处理效应。

ATE = E[Y(1)-Y(0)]

CATE（Conditional Average Treatment Effect）：

给定协变量 X 时的条件平均处理效应。

CATE(x) = E[Y(1)-Y(0)|X=x]

ATE 关注总体平均效果。

CATE 关注不同人群的异质性处理效果。

## 7.3 选择偏差与 Covariate Shift

选择偏差：

样本进入研究或被观测的机制与协变量、处理分配或结果有关，导致观测样本不能代表目标人群。

Covariate Shift：

源域和目标域的协变量分布不同：

P_S(X) ≠ P_T(X)

但通常假设：

P_S(Y|X) = P_T(Y|X)

需要注意：Covariate Shift 是一种特定分布偏移，不等同于所有形式的选择偏差。

## 7.4 Vine Copula 是什么？

Copula 用于描述多个随机变量之间的依赖结构。

根据 Sklar 定理：

F(x1,...,xd) = C(F1(x1),...,Fd(xd))

其中：

- F：联合分布。
- Fi：边缘分布。
- C：Copula 函数，描述变量间依赖结构。

Vine Copula 通过一系列 Pair-Copula 分解高维联合依赖关系。

优点：

- 能够描述非线性依赖。
- 可以选择不同 Copula Family。
- 对复杂高维依赖结构具有更高灵活性。
- 能够处理不同变量对之间不同的依赖形式。

## 7.5 为什么不用多元高斯分布？

多元高斯分布具有较强的分布假设。

它通常难以充分描述复杂的非线性和非对称依赖关系。

Vine Copula 可以将高维联合依赖拆解成多个二维 Pair-Copula，更灵活地描述复杂依赖结构。

## 7.6 为什么不用普通 Gaussian Copula？

Gaussian Copula 使用高斯依赖结构。

它可以描述一定程度的非线性相关，但对尾部依赖等复杂结构的刻画有限。

Vine Copula 可以为不同变量对选择不同的 Copula Family，因此具有更强的结构灵活性。

代价是模型结构更复杂，参数估计和模型选择成本更高。

## 7.7 LLM 在项目中起什么作用？

不是让 LLM 直接预测 ATE。

而是利用 LLM 的领域知识和语义理解能力辅助：

- 识别变量语义。
- 判断变量之间可能的依赖关系。
- 辅助 Copula Family 选择。
- 提供高维依赖结构的先验信息。

最终的统计生成和因果效应估计仍由统计模型和实验机制完成。

## 7.8 什么是 CI-based 迭代接受机制？

CI（Confidence Interval，置信区间）。

通过迭代生成候选样本，并根据因果效应一致性和置信区间等条件进行筛选。

目标：

- 保持 ATE 估计稳定。
- 控制生成数据带来的偏差。
- 提高估计精度。
- 避免仅仅追求协变量分布相似，而破坏因果关系。

注意：置信区间变窄本身并不能证明估计更准确，必须结合偏差、覆盖率和其他因果指标一起评估。

## 7.9 为什么需要对比其他方法？

可能的基线：

**IPW（Inverse Probability Weighting）**

通过处理分配概率的倒数加权，构造伪总体。

优点：估计思想明确。

缺点：极端权重可能造成方差增大。

**Matching**

为处理组和对照组寻找协变量相似的样本。

优点：直观。

缺点：可能损失样本，匹配质量依赖距离度量。

**Outcome Regression**

直接建模结果变量与协变量的关系。

优点：实现简单。

缺点：依赖结果模型设定正确。

**Doubly Robust**

结合倾向评分模型与结果回归模型。

只要两个模型中至少一个正确，在相应识别假设下即可保持一致性。

**GAN / VAE**

用于生成合成数据。

优点：能够学习复杂数据分布。

缺点：生成保真度、训练稳定性和因果结构保持需要额外验证。

**普通 Gaussian Copula**

建模变量联合依赖。

优点：相对简单。

缺点：依赖结构灵活性有限。

**Vine Copula**

通过 Pair-Copula 结构灵活建模高维依赖。

项目选择它，是为了更细致地刻画小样本数据中复杂的变量依赖关系。

## 7.10 项目指标如何解释？

根据简历：

- 协变量分布 SMD 从 0.914 降至 0.075。
- 置信区间宽度降低 87.6%。
- 对比 11 种因果运输及生成基线。

SMD：

Standardized Mean Difference，标准化均值差。

用于衡量两组协变量分布的均值差异。

通常：

|SMD| 越小，表示均值差异越小。

但 SMD 主要反映均值差异，不能完整证明两个高维分布一致。

还需要考虑：

- KS Test。
- MMD。
- Wasserstein Distance。
- 分布重叠情况。

置信区间宽度：

Width = Upper - Lower

宽度降低意味着估计不确定性区间缩小。

但是要进一步检查：

- 真实 ATE 是否落在区间内。
- Coverage 是否合理。
- 估计偏差是否降低。
- 是否存在过度自信。

---

# 第八部分：个人项目——LongMem-RAG 长程记忆 Agent

## 8.1 为什么需要长程记忆？

普通 LLM 存在上下文窗口限制。

长对话中：

- 早期信息可能被截断。
- 重要用户偏好可能遗忘。
- 重复输入历史信息增加 Token 成本。
- 普通 RAG 不一定能够区分短期上下文与长期偏好。

因此设计三层记忆架构。

## 8.2 三层记忆架构

**工作记忆（Working Memory）**

存储当前任务相关的对话信息。

特点：

- 短期。
- 实时更新。
- 与当前任务高度相关。

**短期摘要记忆（Short-term Memory）**

对近期对话进行摘要。

特点：

- 保留阶段性目标。
- 降低历史上下文长度。
- 便于连续对话。

**长期向量记忆（Long-term Memory）**

将重要信息持久化存储，通过 Embedding 和向量检索召回。

特点：

- 长期保存。
- 支持语义检索。
- 可以跨会话使用。

## 8.3 为什么不直接使用向量数据库？

向量数据库只能解决向量存储和相似度检索问题。

它本身不具备：

- 记忆重要性判断。
- 生命周期管理。
- 信息冲突处理。
- 摘要压缩。
- 记忆更新策略。

因此需要在向量数据库上构建记忆管理逻辑。

## 8.4 为什么使用 BM25 + Vector Hybrid Search？

BM25 对关键词匹配较好。

Vector Search 对语义相似性较好。

两者互补。

例如：

用户问：之前讨论的那个 Agent 项目使用了什么数据库？

BM25 可能根据项目名称、ChromaDB 等精确词语召回。

Vector Search 可以根据语义召回历史描述。

然后使用 RRF 或其他融合方式合并结果。

## 8.5 为什么不用纯向量检索？

可能出现：

- 专有名词召回不稳定。
- 数字、代码、文件名匹配不准确。
- 相近语义内容干扰。

因此加入 BM25 提高精确关键词匹配能力。

## 8.6 重要性评分

项目使用规则与语义特征对记忆进行重要性评分。

考虑：

- 显式记忆指令。
- 用户偏好。
- 任务目标。
- 时间戳。
- 内容特征。

目的：

不是所有历史对话都需要永久保存。

需要优先保留长期有价值的信息。

## 8.7 MD5 增量更新

MD5 用于生成内容摘要。

通过比较文档 Hash 判断内容是否变化。

如果内容未变化：

跳过重复 Embedding。

如果内容变化：

重新处理对应文档。

优点：

- 减少重复计算。
- 降低 Embedding API 调用成本。
- 支持增量入库。

注意：MD5 适合内容一致性检测，不应该作为安全认证机制。

## 8.8 为什么使用 Redis？

Redis 是内存型 Key-Value 数据库。

适合缓存：

- 热门检索结果。
- 高频上下文。
- 会话状态。
- 临时数据。

优点：

- 访问速度快。
- 支持多种数据结构。
- 适合高频读写。

但缓存需要考虑过期策略、更新一致性和缓存穿透等问题。

## 8.9 RAG 如何评估？

Recall\@K：

前 K 个检索结果中是否包含正确答案对应的文档。

MRR：

第一个相关文档排名的倒数的平均值。

MRR = (1/N)Σ1/rank_i

Recall\@K 越高，说明正确内容越容易被召回。

MRR 越高，说明正确内容通常排名越靠前。

---

# 第九部分：偏好优化项目——DPO / SimPO

## 9.1 项目核心问题

使用 Qwen3-0.6B、LoRA 和 UltraFeedback，比较 Reference-relative 与 Reference-free 偏好优化。

研究重点：

偏好数据中是否存在长度偏置，以及不同数据处理策略对偏好排序的影响。

## 9.2 为什么偏好数据存在长度偏置？

人类标注者可能倾向于认为更详细的回答质量更高。

模型也可能因为更长的回答包含更多信息，而获得更高的偏好分数。

但是：

长不一定代表好。

短也不一定代表差。

因此需要控制长度因素。

## 9.3 Length Matching

通过匹配 chosen 和 rejected 的回答长度，降低长度差异带来的干扰。

## 9.4 IPW 加权 Pairwise Loss

IPW（Inverse Probability Weighting）逆概率加权。

核心思想：

根据样本进入某种分布或被观测的概率，对样本赋予不同权重。

在项目中用于分析和校正偏好数据中的长度偏置。

注意：IPW 的正确使用需要明确倾向概率模型、权重构造和识别假设，不能只说使用了 IPW 就代表偏差一定消除。

## 9.5 为什么要做消融实验？

需要分别验证：

- 原始偏好数据。
- Length Matching。
- IPW 加权。
- 不同训练策略。
- 不同长度分布。

通过控制变量，分析性能变化到底来自哪个因素。

---

# 第十部分：算法与计算机基础

## 10.1 时间复杂度

常见复杂度：

| 复杂度      | 典型算法          |
| -------- | ------------- |
| O(1)     | 数组随机访问        |
| O(logN)  | 二分查找          |
| O(N)     | 单次遍历          |
| O(NlogN) | 快速排序平均情况、归并排序 |
| O(N²)    | 双重循环、冒泡排序     |
| O(2^N)   | 子集枚举          |
| O(N!)    | 全排列           |

面试手撕代码时要能够说明：

- 时间复杂度。
- 空间复杂度。
- 边界条件。
- 是否存在更优解。

## 10.2 哈希表

核心思想：

通过 Hash Function 将 Key 映射到存储位置。

平均查找复杂度 O(1)。

常见用途：

- 两数之和。
- 频次统计。
- 去重。
- 前缀和计数。

哈希冲突解决：

- 链地址法。
- 开放寻址法。

## 10.3 二分查找

适用于有序区间。

每次将搜索范围缩小一半。

时间复杂度：

O(logN)

重点：

- 左闭右闭还是左闭右开。
- 循环条件。
- mid 更新。
- 边界处理。

## 10.4 双指针

常见场景：

- 有序数组。
- 滑动窗口。
- 链表。
- 快慢指针。

时间复杂度通常可以从 O(N²) 优化到 O(N)。

## 10.5 DFS 与 BFS

DFS：

深度优先搜索。

使用递归或栈。

适合：

- 回溯。
- 连通分量。
- 路径搜索。
- 树遍历。

BFS：

广度优先搜索。

使用队列。

适合：

- 无权图最短路径。
- 层序遍历。
- 最少操作次数。

## 10.6 动态规划

核心：

将原问题拆解为具有重叠子问题和最优子结构的小问题。

步骤：

1. 定义状态。
2. 确定状态转移方程。
3. 确定初始化。
4. 确定遍历顺序。
5. 检查边界条件。

常见问题：

- 背包问题。
- 最长递增子序列。
- 最长公共子序列。
- 编辑距离。
- 股票买卖。

## 10.7 排序算法

快速排序：

- 平均 O(NlogN)。
- 最坏 O(N²)。
- 原地排序。
- 不稳定。

归并排序：

- O(NlogN)。
- 稳定。
- 需要额外空间。

堆排序：

- O(NlogN)。
- 通常不稳定。
- 可以用于 TopK。

---

# 第十一部分：项目技术选型速查表

面试官问：为什么使用这个技术，而不是其他技术？

建议使用以下回答框架：

1. 业务场景是什么？
2. 当前方案解决什么问题？
3. 有哪些可替代方案？
4. 替代方案有什么优缺点？
5. 为什么最终选择当前方案？
6. 通过什么实验验证选择有效？

| 当前方案                    | 其他方案                          | 主要选择依据              |
| ----------------------- | ----------------------------- | ------------------- |
| 多模态大模型                  | 规则、XGBoost、双塔、Cross-Encoder   | 复杂图文语义和属性推理         |
| CoT 蒸馏                  | 人工标注、自生成、Self-Consistency     | 数据规模、推理质量、成本        |
| SFT                     | Prompt Engineering、Few-shot   | 任务行为稳定性             |
| GRPO                    | PPO、DPO、SimPO                 | 奖励机制、训练资源、优化目标      |
| 冻结 ViT                  | 全量微调、视觉 LoRA                  | 数据量、显存、视觉领域偏移       |
| vLLM                    | Transformers、TGI、TensorRT-LLM | 吞吐量、部署成本、兼容性        |
| BM25 + Vector           | 纯 BM25、纯 Vector               | 精确匹配和语义召回互补         |
| Vine Copula             | Gaussian Copula、GAN、VAE       | 高维依赖结构和生成保真         |
| Long-term Vector Memory | 全量历史上下文、纯摘要                   | 长期存储、语义召回和 Token 成本 |
| Redis                   | MySQL、文件存储                    | 高频缓存和低延迟访问          |

---

# 第十二部分：前沿知识与面试加分项

## 12.1 FlashAttention

通过 IO-aware 算法减少 GPU 高带宽显存与片上 SRAM 之间的数据读写。

核心：

- 分块计算 Attention。
- 减少中间矩阵写入 HBM。
- 降低显存占用。
- 提高计算效率。

与普通 Attention 的区别：

数学结果保持等价，主要优化计算实现和内存访问方式。

## 12.2 GQA

Grouped Query Attention。

将多个 Query Head 共享一个 Key/Value Head。

优点：

- 降低 KV Cache 占用。
- 提高推理效率。
- 在一定程度上保留多头注意力能力。

## 12.3 MLA

Multi-head Latent Attention。

通过低维潜在表示压缩 KV Cache，降低推理时的显存开销。

## 12.4 Speculative Decoding

投机解码。

使用小模型先生成候选 Token，再由大模型并行验证。

如果候选 Token 被接受，就可以加速生成。

优点：

- 减少大模型逐 Token 解码的串行开销。
- 在一定条件下提高推理速度。

## 12.5 Agentic Workflow

从单轮 LLM 调用发展到具有规划、工具执行、状态管理和反馈机制的智能体工作流。

重点技术：

- Tool Calling。
- Planning。
- Memory。
- Reflection。
- Multi-Agent。
- MCP。

## 12.6 长上下文与 RAG

长上下文并不意味着不需要 RAG。

长上下文适合：

- 一次性输入完整文档。
- 跨文档复杂推理。
- 需要保留完整上下文的任务。

RAG 适合：

- 大规模知识库。
- 高频增量更新。
- 精准定位知识。
- 控制输入 Token 成本。

实际可以结合使用。

---

# 第十三部分：复习清单

## P0：必须熟练

- [ ] Transformer、Self-Attention、Multi-Head Attention
- [ ] RoPE、LayerNorm、RMSNorm
- [ ] Cross Entropy、Focal Loss、Triplet Loss
- [ ] SGD、Adam、AdamW
- [ ] 过拟合、欠拟合、Bias-Variance
- [ ] SFT、LoRA、QLoRA
- [ ] CoT、数据蒸馏、答案泄露
- [ ] DPO、SimPO、PPO、GRPO
- [ ] Reward Function、Reward Hacking
- [ ] DeepSpeed ZeRO
- [ ] ViT、多模态模型架构
- [ ] vLLM、PagedAttention、KV Cache
- [ ] Precision、Recall、F1

## P0：拼多多项目

- [ ] 7 维辅助标注体系具体是什么？
- [ ] 为什么需要 5000 条独立 Benchmark？
- [ ] 149 万 SKU 对与 46.3 万条 CoT 数据是什么关系？
- [ ] 为什么答案泄露率一开始高达 87.6%？
- [ ] 如何做到答案泄露残留率为 0？
- [ ] 为什么冻结 ViT？
- [ ] 为什么选择 Qwen3.5-2B？
- [ ] F1 从 27% 到 85% 的关键因素是什么？
- [ ] 为什么 SFT 后还需要 GRPO？
- [ ] GRPO 奖励函数如何设计？
- [ ] 为什么不用 PPO、DPO？
- [ ] CoT Token 降低 80% 的原因是什么？
- [ ] 为什么还需要第二阶段 SFT？
- [ ] 如何证明模型效果没有下降？

## P1：科研项目

- [ ] ATE 和 CATE 的区别。
- [ ] 选择偏差与 Covariate Shift。
- [ ] Gaussian Copula 与 Vine Copula。
- [ ] LLM 如何辅助 Copula 结构选择？
- [ ] 什么是 CI-based 迭代接受机制？
- [ ] 为什么要关注置信区间覆盖率？
- [ ] IPW、Matching、Doubly Robust。
- [ ] 为什么需要 11 种基线比较？

## P1：其他项目

- [ ] 三层记忆架构。
- [ ] BM25 与向量检索。
- [ ] RRF。
- [ ] MD5 增量更新。
- [ ] DPO 与 SimPO。
- [ ] Length Matching 和 IPW。
- [ ] LangGraph、MCP、Tool Calling。

## P2：算法手撕

建议至少复习：

- [ ] 两数之和。
- [ ] 三数之和。
- [ ] 无重复字符的最长子串。
- [ ] 最长回文子串。
- [ ] 合并区间。
- [ ] 二分查找。
- [ ] 搜索旋转排序数组。
- [ ] 链表反转。
- [ ] 环形链表。
- [ ] 二叉树层序遍历。
- [ ] 二叉树最大深度。
- [ ] 岛屿数量。
- [ ] 最长递增子序列。
- [ ] 零钱兑换。
- [ ] LRU 缓存。

---

# 最后：面试回答的核心原则

不要把回答停留在技术名词上。

例如，面试官问：

**为什么使用 GRPO？**


面试回答：

>我们在 SFT 阶段已经让模型学会了商品同款判断，但发现模型虽然能够输出正确答案，推理链仍然比较冗长。因此我们进一步引入 GRPO，希望通过可验证的任务正确性奖励和输出长度奖励优化生成行为。相比 PPO，GRPO 不需要独立训练 Critic，训练资源开销相对更低。相比 DPO，它不完全依赖预先构造的离线偏好对，可以根据当前模型采样的回答进行组内相对优化。最终我们实现了 CoT Token 数量降低 80%，同时 F1 没有下降。



