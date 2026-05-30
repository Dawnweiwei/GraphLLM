# 面向光网络拓扑问答的 GraphTranslator 适配实验汇报

日期：2026-05-30

## 1. 研究目标

本阶段工作的目标是：基于给定的单条光网络拓扑、设备参数、业务路由性能和监控数据，构建一套面向“光网络拓扑理解与决策协同”的 Graph-LLM 方法原型。

我们希望模型不是简单依赖文本证据复读答案，而是能够利用图编码后的拓扑表示回答问题。当前重点覆盖文档中 5.1 和 5.2 的两类任务：

- 5.1 基础事实理解：OMS 源宿节点、设备参数、业务波长/路径/Q_margin 查询。
- 5.2 拓扑语义理解：全局有向链路列表、业务共同经过的 OMS、OMS 上下游关系。

核心技术路线是参考 GraphTranslator，将光网络图编码成连续向量，再通过 Translator/Q-Former 对齐到冻结 LLM 的输入空间，使 LLM 能基于 graph token 回答自然语言问题。

## 2. 相关方法依据

### 2.1 GraphTranslator

GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks，WWW 2024 / arXiv:2402.07197。

论文提出用一个轻量 Translator 模块连接预训练 Graph Model 和冻结 LLM。其关键思想是：

- 图模型负责产生节点表示；
- Producer 生成图-文本对齐数据；
- Stage 1 训练 Translator 做 Graph Model-text alignment；
- Stage 2 将 Translator 输出投影到 LLM 词嵌入空间，作为 soft prompt 与用户指令拼接，训练 Graph Model-LLM alignment。

官方论文页面和代码：

- GraphTranslator 论文：[arXiv:2402.07197](https://arxiv.org/abs/2402.07197)
- GraphTranslator 代码：[alibaba/GraphTranslator](https://github.com/alibaba/GraphTranslator)

### 2.2 GraphToken

GraphToken: Let Your Graph Do the Talking: Encoding Structured Data for LLMs，arXiv:2402.05862。

GraphToken 的思想是训练一个图编码函数，把结构化图信息转成可注入 LLM prompt 的连续 token。它强调：相比直接将图序列化成文本，学习得到的 graph token 可以更紧凑地携带结构信息。

论文页面：

- GraphToken 论文：[arXiv:2402.05862](https://arxiv.org/abs/2402.05862)

### 2.3 对我们的启发

这两篇工作共同说明：如果希望 LLM 理解拓扑结构，不能只把图表转成文本塞进 prompt。更合理的方式是：

1. 用图编码器先提取拓扑和属性信息；
2. 用 Translator 把图表示对齐到语言空间；
3. 让 LLM 在自然语言问题下利用 graph token 回答。

因此，我们目前没有采用“测试时给 producer_text 证据”的开卷设置，而是采用：

```text
graph token + 用户自然语言问题 -> 答案
```

Producer text 只用于 Stage 1 对齐训练，不作为主测试阶段的输入证据。

## 3. 光网络数据结构理解

原始样本包含以下主要部分：

- sites：站点节点，例如 A、B、C、D、E、F；
- oms_links：OMS 有向链路，例如 E 到 F 的 OMS7；
- oms_segments：每条 OMS 内部的 site_stage / span / device / fiber 信息；
- services：业务信息，包括 service_id、lambda_id、path_oms_ids、transponder、Q_margin；
- performance：设备或链路监控指标。

该数据的关键特点是：

1. 节点本身通常只有 ID；
2. 大量有效信息集中在边或边的内部 segment 上；
3. service 不是简单节点属性，而是和 OMS 路径、波长、性能指标共同组成一组关系。

因此我们采用 edge-as-node 思路，把 OMS、device、service、lambda 都显式建成图节点，避免把复杂边属性压扁成单一文本字段。

## 4. 数据构造方法

### 4.1 QA 数据

QA 数据由 `data/optical/build_optical_qa.py` 生成。

生成后的每条样本包含：

```text
id
task_type
subtask
input
output
answer_type
evidence_span
difficulty
sample_id
focus_type
focus_id
producer_text
```

其中 `input` 是自然语言问题，`output` 是标准答案，`focus_type/focus_id` 用于决定该问题应该使用哪个图实体的 embedding。

当前任务包括：

| 类别 | subtask | 示例 |
|---|---|---|
| topology_qa | fact_extraction | OMS7 对应的链路是哪个源节点到哪个目的节点？ |
| device_qa | parameter_lookup | OMS7 在 Site1 的 EDFA 类型和增益是多少？ |
| service_qa | service_lookup | 业务12对应的波长、路径OMS和Q_margin分别是什么？ |
| topology_reasoning | adjacency_understanding | 请给出该网络中所有已存在的有向链路。 |
| topology_reasoning | path_membership | 所有业务是否都经过同一条 OMS 链路？ |
| topology_reasoning | reverse_relation | OMS7 的上游节点和下游节点分别是什么？ |

### 4.2 图结构构造

图编码由 `data/optical/build_optical_gnn_embeddings.py` 完成。

我们构造了一个异构 edge-as-node 图：

```text
site -> oms -> site
oms <-> device
service <-> lambda
service <-> oms
```

具体节点类型包括：

- site：站点节点；
- oms：有向 OMS 链路节点；
- device：OMS 中的 EDFA / VOA / fiber 相关设备节点；
- service：业务节点；
- lambda：波长节点。

边类型包括：

- site_src_to_oms；
- oms_to_site_dst；
- oms_to_device / device_to_oms；
- service_to_lambda / lambda_to_service；
- service_to_oms / oms_to_service。

这样做的原因是光网络中大量语义位于“边”上，例如 OMS7 不是普通属性，而是 E 到 F 的有向连接，并且它下面还有 segment 和设备参数。

### 4.3 GraphSAGE 图编码器

我们实现了 GraphSAGE 风格的图编码器：

- 输入维度：64；
- 隐藏维度：256；
- 输出维度：768；
- 输出维度与后续 Translator 输入维度保持一致。

GraphSAGE 使用 relation-specific transforms，对不同边类型使用不同的聚合变换。

自监督训练目标包括：

1. link reconstruction：重构图中存在的边；
2. node type classification：预测节点类型；
3. OMS src/dst 预测：保持 OMS 有向拓扑关系；
4. device type / gain 预测：保持设备属性；
5. service lambda / Q_margin 预测：保持业务属性。

已训练好的 GraphSAGE 统计如下：

```text
link_accuracy: 0.7756
type_accuracy: 1.0
oms_src_accuracy: 1.0
oms_dst_accuracy: 1.0
device_type_accuracy: 1.0
device_gain_mae: 1.0914
service_lambda_accuracy: 0.9721
service_q_margin_mae: 1.6396
```

这说明图编码器本身已经能较好捕捉 OMS 源宿、节点类型、设备类型和 service lambda；但 Q_margin 这种连续数值仍有一定误差。

### 4.4 Focus Embedding 选择

不同问题使用不同 focus embedding：

| 问题类型 | focus embedding |
|---|---|
 OMS 源宿查询 | OMS node embedding |
 OMS 上下游查询 | OMS node embedding |
 设备参数查询 | device node embedding |
 业务查询 | service node embedding |
 全局链路列表 | graph-level pooled embedding |
 共同 OMS 判断 | graph-level pooled embedding |

实际导出的 embedding 是：

```text
0.7 * focus_node_embedding + 0.3 * graph_embedding
```

若问题是全局问题，则直接使用 graph-level embedding。

## 5. Stage 1 设计

### 5.1 原始 Stage 1

最初版本的 Stage 1 使用 QA 行中的 `producer_text` 作为对齐文本。该文本较偏向概览式摘要，例如：

```text
样本包含若干站点、若干 OMS 链路和若干业务。
已存在的有向链路包括...
示例业务...
```

该设计能帮助模型学习局部 OMS 拓扑，但对 service 和 graph-level 任务不够针对。

### 5.2 Focused Stage 1 v1

为了更贴近 GraphTranslator 的 Producer 思想，我们设计了 focused Stage1 数据，由 `data/optical/sample_stage1_rows.py` 抽样。

Stage1 行按照实体类型生成专门 producer text：

#### OMS-level summary

```text
OMS7 是从 E 到 F 的有向 OMS 链路，
上游节点是 E，下游节点是 F。
共有 N 条业务经过 OMS7。
设备参数：Site1 包含 EDFA 类型 21S，增益 19 dB...
```

#### Device-level summary

```text
OMS7 在 Site1 的设备参数：EDFA 类型为 21S，增益为 19 dB。
```

#### Service-level summary

```text
业务12使用波长λ33，路径OMS为7，Q_margin为9.65 dB，transponder为8019_400g。
```

#### Network-level summary

```text
样本 sample_0001 包含 8 条有向 OMS 链路：
A到B（OMS2），B到C（OMS3）...
所有业务共同经过 OMS7。
```

v1 的 20k Stage1 采样比例为：

```text
oms_summary: 6000
device_summary: 5400
service_summary: 7000
network_summary: 1600
```

### 5.3 Focused Stage 1 v2

v1 提升了全局邻接表，但降低了 path_membership。因此 v2 将 network summary 拆成两类：

```text
network_adjacency_summary
network_common_oms_summary
```

v2 的 20k Stage1 采样比例为：

```text
oms_summary: 5200
device_summary: 5000
service_summary: 6600
network_adjacency_summary: 1600
network_common_oms_summary: 1600
```

实验结果表明，v2 能修复一部分 path_membership，但会明显损伤邻接表和局部拓扑能力，因此当前不作为主版本。

## 6. Stage 2 训练设计

Stage 2 参考 GraphTranslator 的 GM-LLM alignment：

```text
GraphSAGE embedding
  -> Q-Former / Translator
  -> linear projection to ChatGLM hidden size
  -> prepend as soft graph tokens
  -> frozen ChatGLM2-6B generates answer
```

重要设置：

- LLM：ChatGLM2-6B；
- LLM 参数冻结；
- GraphSAGE embedding 冻结；
- 主要训练 Translator/Q-Former 和投影层；
- Stage2 输入不提供 producer_text 证据；
- Prompt 只包含系统说明和自然语言问题：

```text
Answer the optical-network question using the graph tokens.
Output only the final answer.
Question: {question}
Answer:
```

这不是开卷测试，因为没有提供答案事实，只告诉 LLM 前面的 soft token 是图信息。

Stage2 训练数据：

```text
optical_train_translator_rows_20k.tsv
```

评测数据：

```text
optical_test_translator_rows_1k.tsv
```

## 7. 评测指标

最初 exact match 过于严格，不适合真实问答。例如：

```text
标准答案：OMS7对应链路为E到F。
模型答案：E 是源节点，F 是目的节点，也就是 E -> F。
```

这对用户是正确的，但 exact match 会判错。

因此我们新增任务感知评测脚本：

```text
data/optical/evaluate_optical_qa.py
```

指标包括：

| 任务 | 指标 |
|---|---|
 fact_extraction / reverse_relation | src_accuracy, dst_accuracy, direction_accuracy |
 parameter_lookup | edfa_type_accuracy, gain_accuracy |
 service_lookup | lambda_accuracy, path_oms_f1, q_margin_accuracy |
 adjacency_understanding | edge_set_precision, edge_set_recall, edge_set_f1 |
 path_membership | yes_no_accuracy, common_oms_accuracy |

注意：评测脚本后来修复了 `路径OMS为8-1` 的解析问题。修复前 service path 被低估，修复后发现 service 路径识别并不差，真正弱项是 lambda 和 Q_margin 数值。

## 8. 实验结果

### 8.1 三版模型对比

| 指标 | baseline | focused v1 | focused v2 |
|---|---:|---:|---:|
 overall exact_match | 36.30% | 37.00% | 32.20% |
 overall slot_accuracy | 58.14% | 58.61% | 56.19% |
 fact_extraction slot | 96.71% | 98.50% | 96.11% |
 reverse_relation slot | 95.18% | 96.08% | 90.06% |
 device_qa slot | 59.88% | 62.57% | 57.19% |
 service_qa slot | 29.24% | 27.07% | 26.70% |
 adjacency edge_set_f1 | 6.21% | 29.30% | 11.45% |
 path_membership slot | 61.75% | 38.25% | 55.72% |

### 8.2 最佳版本

当前最佳版本是 focused Stage1 v1。

它带来的主要提升：

```text
fact_extraction slot: 96.71% -> 98.50%
reverse_relation slot: 95.18% -> 96.08%
device_qa slot: 59.88% -> 62.57%
adjacency edge_set_f1: 6.21% -> 29.30%
```

说明 focused Stage1 确实让 Translator 更好地抽取了：

- OMS 源宿关系；
- OMS 上下游方向；
- EDFA 类型和部分增益信息；
- graph-level 邻接结构。

但 v1 的代价是：

```text
path_membership slot: 61.75% -> 38.25%
```

说明把全局邻接表和共同 OMS 混在同一类 network summary 中，会让 graph-level embedding 更偏向“列边”，反而损害“业务路径交集”判断。

### 8.3 v2 为什么没有采用

v2 将 network-level 文本拆成 adjacency 和 common_oms 两类。结果：

```text
path_membership slot: 38.25% -> 55.72%
adjacency edge_set_f1: 29.30% -> 11.45%
reverse_relation slot: 96.08% -> 90.06%
```

这说明 v2 虽然修复了 path_membership 的一部分能力，但整体损害更大。当前不建议作为主模型。

## 9. 结果分析

### 9.1 已经有效的部分

#### 局部 OMS 拓扑理解

模型对单条 OMS 的源宿关系、上下游方向已经比较稳定：

```text
fact_extraction slot: 98.50%
reverse_relation slot: 96.08%
```

这说明 edge-as-node + OMS focus embedding 是合理的。

#### 设备类型识别

focused v1 后：

```text
edfa_type_accuracy: 91.02%
```

模型已经较好学到 EDFA 类型，但增益仍只有：

```text
gain_accuracy: 34.13%
```

说明类别型设备属性较容易编码，连续数值属性仍较难。

#### 全局邻接表

focused v1 明显提升：

```text
edge_set_f1: 6.21% -> 29.30%
```

这说明 graph-level producer text 对全局拓扑确实有效。

### 9.2 仍然薄弱的部分

#### Service 查询

修正评测后，service 查询的路径其实不算差：

```text
path_oms_f1: baseline 65.11%, focused v1 63.35%
```

但 lambda 和 Q_margin 仍很弱：

```text
lambda_accuracy: 约 12%
q_margin_accuracy: 约 6-10%
```

这说明 service node embedding 可以学到路径模式，但对 service_id 到 lambda / q_margin 的精确绑定还不够好。

#### Q_margin 和 gain 等连续数值

GraphSAGE 阶段的 service_q_margin_mae 约为 1.64，说明图编码器本身对 Q_margin 的精确回归还不够。进入 Translator/LLM 后，数值误差会进一步放大。

#### path_membership

path_membership 依赖所有业务路径的集合交集。当前 graph-level embedding 是所有节点平均池化，可能不足以精确表达“所有 service path 的交集”这种集合运算。

## 10. 当前文件与实验产物

主要代码：

```text
data/optical/build_optical_qa.py
data/optical/build_optical_gnn_embeddings.py
data/optical/sample_stage1_rows.py
data/optical/evaluate_optical_qa.py
third_party/GraphTranslator/Translator/models/translator_models/translator_qformer_optical.py
third_party/GraphTranslator/Translator/models/translator_models/translator_chatglm_optical.py
```

主要配置：

```text
pretrain_optical_stage1_focused.yaml
pretrain_optical_stage2_qa_20k_focused.yaml
generate_optical_qa_1k_focused.yaml

pretrain_optical_stage1_focused_v2.yaml
pretrain_optical_stage2_qa_20k_focused_v2.yaml
generate_optical_qa_1k_focused_v2.yaml
```

服务器实验结果：

```text
/root/autodl-tmp/GraphLLM_project/data/optical/sage/eval_optical_qa_1k_taskaware_v2.json
/root/autodl-tmp/GraphLLM_project/data/optical/sage/eval_optical_qa_1k_focused_taskaware_v2.json
/root/autodl-tmp/GraphLLM_project/data/optical/sage/eval_optical_qa_1k_focused_v2_taskaware.json
```

当前最佳 checkpoint：

```text
/root/autodl-tmp/GraphLLM_project/third_party/GraphTranslator/Translator/model_output/pretrain_optical_stage2_qa_20k_focused/checkpoint_0.pth
```

## 11. 下一步建议

### 11.1 优先优化 service 数值属性

当前 service path 已经能学到一部分，真正需要提升的是：

```text
service_id -> lambda_id
service_id -> q_margin
```

建议：

1. 对 service node 增加更强的 ID 编码；
2. 将 lambda 和 Q_margin 做离散 bucket 或专用数值 token；
3. 在 GraphSAGE 自监督中提高 service_lambda 和 service_q_margin 的 loss 权重；
4. 单独生成 service-focused Stage1 数据，不与 graph-level 数据混在一起训练或加权训练。

### 11.2 为 graph-level 任务设计专用 pooling

当前 graph-level embedding 是 mean pooling。对于：

```text
所有业务是否经过同一条 OMS？
```

mean pooling 可能不足以表达集合交集。

建议尝试：

- 增加 GRAPH special node；
- GRAPH 节点显式连接所有 service 和 OMS；
- 使用 attention pooling 替代 mean pooling；
- 单独训练 graph-level objective。

### 11.3 Stage1 增加 contrastive / matching loss

目前 Stage1 主要还是生成式对齐，尚未完整复现 GraphTranslator 的三目标：

- contrastive；
- generative；
- matching。

后续可以进一步加入：

```text
Service12 embedding <-> Service12 summary 为正样本
Service12 embedding <-> Service13 summary 为负样本
```

这可能尤其有助于区分大量格式相似的 service 样本。

## 12. 阶段结论

本阶段验证了三点：

1. GraphTranslator 的两阶段训练框架可以迁移到光网络拓扑问答；
2. edge-as-node 图构造适合光网络这种“边上属性丰富”的数据；
3. 面向任务定制的 Stage1 producer text 会显著影响模型能力。

当前最佳版本 focused v1 证明：只要 Stage1 对齐目标设计得更贴近光网络语义，模型不仅不会丢失原来的 OMS 局部拓扑能力，还能显著提升全局邻接表理解能力。

但 service 数值查询和业务路径交集仍需进一步优化。下一阶段应重点从 service 表示、graph-level pooling 和 contrastive/matching Stage1 目标入手。
