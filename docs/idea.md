# TabPFN Predictive Relations → Representation
## 论文完整实现路线（Markdown 公式规范版）
From Prediction to Representation: Learning Predictive Relations from Tabular Foundation Models

> 日期：2026-09-10  
> 范围：TabPFN + 分类 + 样本级表征  
> 原则：**先解剖 Readout，再决定模型；不预设 Readout 一定值得蒸馏。**

---

# 1. 核心研究问题

给定 Support Set

$$
D_S=\{(x_i,y_i)\}_{i=1}^N
$$

和 Query $x_q$，TabPFN Decoder Readout 给出：

$$
\alpha_q(C)=[\alpha_{q1},...,\alpha_{qN}]
$$

其中 $C$ 表示当前 support context。

论文真正要回答：

1. **Beyond Labels**：Readout 是否包含超越类别标签的实例级关系？
2. **Directionality**：这种关系是否明显有向？
3. **Context Dependence**：固定 Query 和 Support pair 后，改变其他 Context，关系是否稳定？
4. 如果前三项成立，**什么样的 representation 才能忠实压缩这种关系？**

最终目标不是“再造一个分类器”，而是研究：

$$
\boxed{\text{Prediction Behavior}\rightarrow\text{Relational Structure}\rightarrow\text{Reusable Representation}}
$$

---

# 2. 三个雷区与对应生死实验

## 雷区 1：是否只是昂贵的 SupCon？

TabPFN 分类 Readout 最终参与 label voting，因此天然带类别信息。

对类别 $c$ 定义：

$$
M_{qc}=\sum_{i:y_i=c}\alpha_{qi}
$$

再定义类内关系：

$$
\beta_{qi}=
\frac{\alpha_{qi}}
{M_{q,y_i}+\epsilon}
$$

所以：

$$
\alpha_{qi}=M_{q,y_i}\beta_{qi}
$$

解释：

- $M$：类别级信号；
- $\beta$：同类别内部“具体参考谁”。

### 必做 Null Model

如果模型只知道类别，不知道实例：

$$
\tilde\alpha_{qi}=\frac{M_{q,y_i}}{N_{y_i}}
$$

比较：

- JS/KL$(\alpha,\tilde\alpha)$
- normalized within-class entropy
- Top-$\beta$ concentration

### 必做行为实验

同类别内分别删除：

- Top-$\beta$
- Random-same-class
- Bottom-$\beta$

比较预测变化：

$$
\Delta p,\quad TV,\quad flip\ rate
$$

### 必做 SupCon baseline

同 backbone、同训练预算比较：

- SupCon
- Readout distillation
- SupCon + Readout

### Stop 条件

若多数数据集：

- $\beta$ 近似均匀；
- Top-$\beta$ removal ≈ random-same-class；
- Readout Student ≈ SupCon；

则停止“新 representation 方法”主路线。

---

## 雷区 2：有向关系 vs 对称空间

TabPFN relation 可能满足：

$$
R(i,j)\neq R(j,i)
$$

而普通 cosine：

$$
\cos(z_i,z_j)=\cos(z_j,z_i)
$$

无法表达强非对称关系。

### 必测指标

$$
A_F=
\frac{\|R-R^\top\|_F}
{\|R\|_F+\epsilon}
$$

以及：

- Reciprocity@K
- pairwise rank reversal
- $R-R^\top$ heatmap

### 模型分支

**弱非对称：**

$$
z=f(x),\qquad s_{ij}=\cos(z_i,z_j)
$$

**强非对称：**

$$
h=f(x),\quad q=W_Qh,\quad k=W_Kh
$$

$$
s_{ij}=q_i^\top k_j
$$

最终表示可以保存：

$$
[h,q,k]
$$

### Pivot 条件

若 asymmetric student 明显优于 symmetric student，则主模型必须使用 Query/Key 双空间。

---

## 雷区 3：Context 依赖

真正关系应写为：

$$
R(q,i\mid C)
$$

而不是 $R(q,i)$。

### 必须区分两种变化

**Softmax competition：**

pre-softmax score 不变，但加入其他 support 后 denominator 改变，导致 $\alpha$ 变化。

**真正的 context shift：**

$$
\ell_{qi}(C_1)\neq \ell_{qi}(C_2)
$$

说明 post-ICL representation 本身改变。

### Context 实验

固定：

- Query $q$
- Anchor supports $a,b,...$

只替换其余 context，采样 $M=10\sim20$ 组。

记录：

- normalized readout $\alpha$
- pre-softmax score $\ell$
- pairwise difference $\ell_{qa}-\ell_{qb}$
- ranking

指标：

- raw-score variance
- Kendall / Spearman
- Top-K Jaccard
- preference flip rate

### 模型分支

**Weak Context：**

$$
x\rightarrow z
$$

**Moderate：**

$$
z=f(x),\quad c=g(C),\quad z'=Adapter(z,c)
$$

**Strong：**

$$
(x,C)\rightarrow(q_C,k_C)
$$

Context encoder 第一版使用 DeepSets，不先上复杂 Transformer。

---

# 3. 第一阶段：Readout Anatomy

前 7 周的目标不是训练新模型，而是搞清楚：

$$
\boxed{\text{Readout 到底是什么}}
$$

最终必须得到下面的属性表：

| 属性 | 结果 |
|---|---|
| Beyond-label information | Strong / Medium / Weak |
| Beyond-SupCon information | Strong / Medium / Weak |
| Directionality | Strong / Medium / Weak |
| Context dependence | Strong / Medium / Weak |
| Raw-score stability | Strong / Medium / Weak |

然后只选择一条主路线。

---

# 4. Teacher 数据接口

每个 Query 保存：

```text
dataset_id
split_id
seed
query_id
query_true_label        # 只用于评价，绝不输入 context
support_ids
support_labels
tabpfn_prediction
readout_alpha
decoder_raw_scores      # 尽可能保存 softmax 前 score
hidden_query
hidden_support
```

## 防止泄漏

1. Query label 永远不能进入 Support Context。
2. 构造 $R_{ij}$ 时必须 leave-one-out。
3. Test query 生成 Teacher relation 时只允许 Train Support。
4. Student 训练不能间接看到 Test Label。
5. 所有 sample id 必须随 permutation 正确追踪。

---

# 5. Synthetic 测试集

真实数据很难解释原因，所以先做 4 类合成任务。

### Synthetic A：Class-Only

同类样本 exchangeable。

预期：

$$
\beta\approx uniform
$$

用于验证“无实例结构”时指标不会误报。

### Synthetic B：Prototype

每类内部多个 cluster。

预期：Query 更偏向同类中的正确 prototype。

### Synthetic C：Boundary

制造明显决策边界与边界附近样本。

研究 Readout 更偏：

- prototype
- boundary point
- counter-example

### Synthetic D：Context Competition

固定 Query/anchors，逐步加入竞争 support。

专门区分：

- softmax competition
- true context shift

---

# 6. Representation 模型候选

只有 Anatomy 完成后才选。

## R0：Static Symmetric

$$
z=f_\theta(x)
$$

$$
s_{ij}=\cos(z_i,z_j)/\tau
$$

仅适合：弱非对称 + 弱 Context。

## R1：Static Directed

$$
h=f_\theta(x)
$$

$$
q=W_Qh,\qquad k=W_Kh
$$

$$
s_{ij}=q_i^\top k_j/\tau
$$

推荐作为默认强 baseline。

## R2：Context-Conditioned Directed

$$
c=g(C)
$$

$$
q_i=Q(h_i,c),\qquad k_i=K(h_i,c)
$$

$$
s_{ij}(C)=q_i(C)^\top k_j(C)
$$

只在 Week 6 证明强 Context 后采用。

---

# 7. Teacher Target 候选

必须比较四种监督信号。

### T1：完整 Readout

$$
\alpha_{qi}
$$

风险：类别信息和 softmax competition 混在一起。

### T2：Class-Residual Relation

$$
\beta_{qi}
$$

这是最关键 target，直接回应“是不是 SupCon”。

### T3：Pre-softmax score

$$
\ell_{qi}
$$

避免 softmax normalization 的竞争效应。

### T4：Centered score

$$
\ell'_{qi}=
\ell_{qi}-\frac1N\sum_j\ell_{qj}
$$

提高不同 query 间数值稳定性。

---

# 8. Loss

优先测试：

1. KL on $\alpha$
2. KL on $\beta$
3. ranking loss on $\ell$
4. KL + ranking

第一版不要堆很多 loss。

最关键的 ablation：

$$
\boxed{\alpha\ vs\ \beta}
$$

如果 $\beta$ 更好，说明真正有价值的是类别内部实例关系，而不是类别质量本身。

---

# 9. 必须有的 Baselines

1. Raw Features
2. PCA
3. Supervised MLP
4. SupCon
5. TabPFN Hidden Embedding
6. Decoder Readout Profile
7. Static Symmetric Student
8. Static Directed Student
9. Context-Conditioned Student（仅当需要）

对 SupCon 与 Readout Student 必须严格统一：

- backbone
- optimizer
- epochs
- batch size
- embedding dimension
- split
- seed budget

---

# 10. Representation 到底用来干什么？

TabPFN 已经能预测，因此论文不能只比较分类 accuracy。

重点任务：

### Retrieval

- Recall@K
- Precision@K
- NDCG
- Teacher Top-K Recall

### Linear Probe

冻结 encoder，训练线性分类器。

### kNN

直接使用 embedding 进行 kNN。

### Behavioral Faithfulness

Teacher / Student / SupCon 各自 Top-K removal：

$$
\Delta_{Student}\approx\Delta_{Teacher}\gg\Delta_{Random}
$$

### Efficient Reuse

预计算 support representation，之后 ANN 检索。

比较：

- latency
- throughput
- memory
- Recall@K

这才回答：

> “为什么不直接每次跑 TabPFN？”

---

# 11. 正式 Benchmark

## 快速阶段

- 4 synthetic
- 5–8 real datasets

## 正式阶段

20–30 个分类数据集，覆盖：

- binary / multiclass
- numerical / mixed categorical
- imbalance
- 不同 $N$
- 不同 $d$

正式实验开始前冻结 dataset manifest，禁止根据结果删数据集。

---

# 12. 16 周逐周计划

## Week 1：环境冻结 + Teacher Extraction

目标：可靠提取 $\alpha$、hidden embedding，并 hook $\ell$。

任务：

- 固定 TabPFN 版本、checkpoint、commit
- 实现 `TabPFNTeacher`
- 实现 sample-id tracking
- 实现 readout / hidden / raw-score extraction
- 写 unit tests

必须测试：

- $\alpha_i\ge0$
- $\sum_i\alpha_i\approx1$
- batch vs single query 一致
- permutation 后 sample id 不错位
- query 不泄漏到 support

交付：

```text
src/teacher/
tests/test_teacher.py
reports/week01_teacher_validation.md
```

容错：

- 官方 API 变化 → pin commit
- raw score 暂时拿不到 → 不阻塞 Week 1，但 Week 3 前必须完成

---

## Week 2：Synthetic Anatomy

目标：先验证分析方法正确。

完成：

- Class-Only
- Prototype
- Boundary
- Context Competition

实现：

- $M,\beta$
- class-only null
- normalized entropy
- JS/KL
- context sampling

Gate：

如果 synthetic 都不能按理论预期表现，先修实验，不进入真实数据。

交付：

`reports/week02_synthetic_anatomy.md`

---

## Week 3：雷区 1A —— Beyond Labels

真实数据 5–8 个。

计算：

- JS($\alpha$, null)
- within-class entropy
- Top-$\beta$ concentration
- class-residual profiles

Gate：

若多数数据集：

$$
\beta\approx uniform
$$

则 STOP 主方法路线。

可转为“TabPFN Decoder Retrieval Anatomy”分析工作。

---

## Week 4：雷区 1B —— SupCon 与 Same-Class Faithfulness

实验：

- Top-$\beta$ removal
- Random-same-class removal
- Bottom-$\beta$ removal
- SupCon baseline

关键判断：

$$
\Delta_{Top-\beta}>\Delta_{RandomSameClass}
$$

并检查 Readout 是否提供 SupCon 没有的信息。

Gate：

如果 removal 没差异且 SupCon ≈ Readout，停止蒸馏路线。

交付：

`reports/week04_beyond_supcon.md`

---

## Week 5：雷区 2 —— Directionality

构造双向关系。

计算：

- $A_F$
- Reciprocity@K
- rank reversal
- $R-R^T$

同时训练一个最小 symmetric vs directed relation reconstructor。

Gate：

- 弱非对称 → Static Symmetric 可保留
- 强非对称 → 主模型强制 Dual-space

交付：

`reports/week05_directionality.md`

---

## Week 6：雷区 3 —— Context Sensitivity

固定 Query + anchors，采样 10–20 个 contexts。

记录：

- $\alpha$
- $\ell$
- pairwise score difference
- ranking

计算：

- variance
- Kendall
- Spearman
- Top-K Jaccard
- ranking flip rate

Gate：

- Weak → static
- Moderate → context adapter
- Strong → context-conditioned model

交付：

`reports/week06_context.md`

---

## Week 7：Readout Anatomy 总结与路线冻结

本周不开发复杂模型。

输出：

`READOUT_ANATOMY_REPORT.md`

明确填：

```text
Beyond Label:
Beyond SupCon:
Directionality:
Context Dependence:
Recommended Model:
Recommended Teacher Target:
Go / Pivot / Stop:
```

只选择一个主路线：

- Route A：Static Symmetric
- Route B：Static Directed
- Route C：Context-Conditioned Directed

---

## Week 8：主模型 v1

Backbone 第一版保持简单：

- numerical → standardized MLP
- categorical → embedding
- hidden dim 128
- output dim 64

根据 Week 7 实现唯一主模型。

优先 target：

- $\beta$-KL
- raw-score ranking

交付：

```text
src/models/
src/losses/
scripts/train_student.py
```

---

## Week 9：Target / Loss / Dimension 消融

比较：

- $\alpha$
- $\beta$
- raw $\ell$
- centered $\ell$

Loss：

- KL
- MSE
- ranking
- KL + ranking

维度：

$$
32,64,128
$$

关键结论：

> 哪种 Teacher information 真正适合 representation？

---

## Week 10：正式 Benchmark

扩展到 20–30 datasets。

统一跑：

- Raw
- PCA
- Supervised MLP
- SupCon
- TabPFN Hidden
- Readout Profile
- Readout2Rep

主要指标：

- Teacher Top-K Recall
- NDCG
- kNN
- linear probe
- mean rank
- win/tie/loss

形成论文 Table 1 / 2。

---

## Week 11：Behavioral Faithfulness + Failure Cases

比较删除：

- Teacher Top-K
- Student Top-K
- SupCon Top-K
- Hidden Top-K
- Random

分析失败数据集：

- imbalance
- rare cluster
- boundary samples
- noise
- multiclass confusion

不删除失败结果。

交付：

论文核心 removal curve。

---

## Week 12：Robustness / Context Generalization

若 Static：

- 不同 support subsets
- 不同 context size
- 不同 class balance

若 Context Model：

- unseen context
- support distribution shift
- context subsampling

同时测试：

- feature noise
- missing values
- seeds

形成 robustness table。

---

## Week 13：效率与“为什么不用 TabPFN”

建立 ANN retrieval。

比较：

- full TabPFN readout retrieval
- TabPFN hidden retrieval
- SupCon
- Readout2Rep

指标：

- embedding extraction cost
- query latency
- throughput
- memory
- index size
- Recall@K

形成 speed-quality curve。

---

## Week 14：完整消融 + 统计

必须完成：

1. $\alpha$ vs $\beta$
2. symmetric vs directed
3. static vs context（若 relevant）
4. SupCon vs Readout
5. hidden teacher vs decoder teacher
6. embedding dimension
7. loss
8. sparse Top-K teacher
9. context size

统计：

- mean/std
- mean rank
- win/tie/loss
- bootstrap CI
- paired Wilcoxon

冻结全部 Table / Figure。

---

## Week 15：论文完整初稿

论文结构：

### 1. Introduction
提出：

> What relational structure does a tabular foundation model use during in-context prediction?

### 2. Related Work
- tabular representation
- TFM
- example-based interpretability
- metric / relational distillation

### 3. Anatomy of TabPFN Predictive Relations
- beyond labels
- directionality
- context dependence

### 4. Readout2Rep
只介绍 Week 7 选出的最终模型。

### 5. Experiments
- anatomy
- representation quality
- faithfulness
- robustness
- efficiency

### 6. Limitations

### 7. Conclusion

本周结束必须有完整 v1，不留“待补实验”占位。

---

## Week 16：审稿人攻击测试 + 投稿准备

模拟六个最强问题：

### “这不就是 SupCon？”
用：
- $\beta$
- class-only null
- same-class removal
- SupCon baseline

### “Attention 不是 explanation。”
不宣称因果解释；使用：
- predictive relation
- decoder retrieval weight
- behavioral faithfulness

### “有向关系为什么用 cosine？”
若强非对称，主模型已是 Q/K 双空间。

### “ICL 是 context-dependent，为什么静态？”
用 Week 6 结果决定；强 Context 时必须 context-conditioned。

### “为什么不直接 TabPFN？”
用 retrieval reuse、ANN、latency、memory 回答。

### “为什么不直接用 hidden embedding？”
完整比较 Hidden vs Readout vs Student。

最终输出：

```text
paper/
code/
configs/
results/
figures/
tables/
README.md
REPRODUCIBILITY.md
```

---

# 13. 三种最终 Outcome 与容错

## Outcome A：最理想

发现：

- beyond-label 强
- 非对称明显
- moderate context dependence
- relation 可压缩

论文定位：

> TabPFN induces structured, directed and context-sensitive predictive relations that can be distilled into reusable representations.

## Outcome B：静态空间足够

发现：

- beyond-label 强
- 非对称弱
- context 弱

主模型保持简单 static representation。

论文定位：

> TabPFN predictive relations define a useful task-aware representation geometry.

## Outcome C：主要是标签

发现：

- $\beta$ 近似 uniform
- SupCon ≈ Readout
- same-class removal 无差异

结论：

$$
\boxed{\text{停止 Readout2Rep 方法论文}}
$$

可转成机制分析：

> How much instance-level information is actually present in TabPFN decoder retrieval?

不要为了“做完方法”而硬加网络。

---

# 14. 推荐代码结构

```text
readout2rep/
├── configs/
├── src/
│   ├── data/
│   ├── teacher/
│   ├── anatomy/
│   ├── models/
│   ├── losses/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── tests/
├── experiments/
├── results/
├── figures/
├── reports/
└── paper/
```

其中：

```text
src/teacher/
  tabpfn_teacher.py
  readout_hook.py
  score_hook.py
  hidden_embedding.py

src/anatomy/
  label_decomposition.py
  null_models.py
  directionality.py
  context_sensitivity.py

src/models/
  symmetric.py
  directed.py
  contextual.py
```

---

# 15. 实验日志要求

每次运行保存：

```yaml
dataset_id:
dataset_version:
split_seed:
experiment_seed:
tabpfn_version:
tabpfn_commit:
checkpoint:
python_version:
torch_version:
cuda_version:
gpu:
teacher_target:
context_size:
query_count:
student_model:
embedding_dim:
loss:
optimizer:
learning_rate:
batch_size:
epochs:
git_commit:
timestamp:
```

原则：

- 结果由脚本自动汇总
- Figure/Table 自动生成
- 不手改 CSV
- 不隐藏失败结果
- benchmark manifest 在正式实验前冻结

---

# 16. 当前最近必须做的 6 个实验

暂时不要想 Week 8 以后的模型。

先只做：

1. 稳定提取 $\alpha$ 和 pre-softmax $\ell$
2. 分解 $\alpha=M\beta$
3. $\alpha$ vs class-only null
4. same-class Top-$\beta$ removal
5. $R$ vs $R^\top$
6. 固定 Query/Anchor、改变 Context，测 raw-score rank flip

这六个实验完成后，才决定最终 representation 模型。

---

# 17. 当前相关工作边界

需要持续对照以下工作：

- **TabPFN Extensions**：官方 interpretability / internal embeddings  
  https://github.com/PriorLabs/tabpfn-extensions
- **TabPFN v3 architecture**：Decoder / Query-Key 实现  
  https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v3.py
- **Towards Localization via Data Embedding for TabPFN**，NeurIPS TRL 2024  
  https://neurips.cc/virtual/2024/103159
- **In-Context Learning of Soft Nearest Neighbor Classifiers for Intelligible Tabular Machine Learning**，TRL @ ACL 2025  
  https://aclanthology.org/2025.trl-1.15/
- **KernelICL: Interpretable Tabular Foundation Models via In-Context Kernel Regression**，2026  
  https://arxiv.org/abs/2602.02162

---

# 18. 最终执行原则

整篇论文不能写成：

> “我们把 TabPFN Attention 拿来训练了一个 embedding。”

应该写成：

$$
\boxed{
\text{Before distilling prediction behavior,
we first determine what relational structure that behavior actually contains.}
}
$$

顺序必须是：

$$
\text{Readout}
\rightarrow
\text{Beyond Labels?}
\rightarrow
\text{Directional?}
\rightarrow
\text{Contextual?}
\rightarrow
\text{Choose Correct Representation}
\rightarrow
\text{Distill}
\rightarrow
\text{Evaluate Reuse}
$$

**Week 1–7 的任务不是“把模型做出来”，而是判断模型究竟应该长什么样。**
