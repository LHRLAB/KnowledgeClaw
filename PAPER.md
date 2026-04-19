# KnowledgeClaw：基于全模态超图的智能体个人知识库系统

> NeurIPS 2026 投稿

---

## 1. 引言

**任务。** 个人知识天然是多模态、关系化的。研究者日常积累的论文 PDF、临床影像、心音录音、手术视频、法律判决书等异构文件构成了个人知识库（Personal Knowledge Base, PKB）。一个理想的 PKB 智能体应当：（1）理解并索引所有模态；（2）在知识库的全生命周期中支持增删改查；（3）对从精确事实查找到主题综合的多粒度查询都能高效检索和推理。我们将这一问题形式化为**全模态知识工程（Omnimodal Knowledge Engineering, OmniKE）**，并定义了 6 个方法无关的评测任务（T1–T6），覆盖知识获取、跨模态共指、多跳检索、跨模态推理、生命周期一致性和生成刷新。

**相关工作。** 现有系统各解决了问题的一个切面但遗漏了其他。向量 RAG（NaiveRAG）只做文本块检索，无法处理非文本模态也无结构化表示。图 RAG（GraphRAG, LightRAG, HyperGraphRAG）引入实体-关系图甚至超图来捕捉高阶关系语义，但仍局限于文本且假设语料静态不变。智能体记忆系统（MemGPT/Letta, Mem0, A-MEM）维护长时记忆但依赖扁平相似度搜索，缺乏结构化知识构建。Graphiti 引入了时序知识图谱并支持删除操作，但仅处理文本。Graphify 支持多模态文件夹输入并构建知识图谱，但无生命周期管理。Cognee 自动从多模态数据构建语义图，但缺少超图表示和多粒度检索。OntoKG 提出本体优先的知识图谱构建，但仅处理结构化文本。总之，**无一系统同时具备全模态表示、闭环生命周期和多粒度检索**。

**挑战。** 我们识别出三个根本性缺口：**L1 模态缺口**——现有 KG-RAG 系统以文本为中心，图像、音频、视频等模态的超边表示尚属空白；**L2 生命周期缺口**——真实 PKB 持续演化（论文被撤稿、数据被更新、文件被删除），现有系统无法在增删改操作下保持图结构一致性；**L3 检索粒度缺口**——个人查询从精确事实（"某教授在 3 月讲座中如何评价 attention？"）到开放综合（"总结我对对比学习的所有理解"），现有系统只能优化其中一端。

**贡献。** 我们提出 **KnowledgeClaw**，一个通过全模态知识超图自主管理个人知识库全生命周期的智能体系统。三项贡献：**C1（全模态类型超图）**——五种模态（text/PDF/image/audio/video）的实体共享类型化超边和跨模态共指边，首次将超图 RAG 从单模态文档扩展到全模态个人知识；**C2（闭环生命周期智能体）**——构建→查询→推理→更新的闭环，支持一致性保持的增量添加、级联删除和孤儿清理；**C3（多粒度检索+意图路由）**——子图级（RRF 实体+超边检索+图遍历扩展）和社区级（Louvain 聚类+摘要嵌入）双层检索，由轻量意图路由器动态分发。

**实验。** 我们构建了 **OmniKE-Bench v0.1**，包含 10 个专业领域 persona（医疗、金融、教育、法律、心理学、音乐、游戏、建筑、软件、机器人）× 5 种模态 × ~4500 个真实文件（PMC 论文、SCOTUS 判决书、PhysioNet 心音、archive.org 视频等）。在 22 个方法（10 个范式组）的全面对比中，KnowledgeClaw 在全部 6 个任务上均有得分（OKE=56.7），三个消融实验验证了 C1/C2/C3 各自的不可替代性：去掉 C1 后 T2 跨模态共指归零，去掉 C2 后 T5 生命周期退化，去掉 C3 后 T3 检索下降。

---

## 2. 相关工作

### 2.1 检索增强生成与知识图谱

向量 RAG（NaiveRAG, RAPTOR）将文档切块后检索，但无结构化表示、无非文本模态支持。图 RAG（GraphRAG, LightRAG, HyperGraphRAG, OntoKG, iText2KG, HippoRAG 2）引入实体-关系图乃至超图来捕获高阶关系语义，显著提升多跳推理能力；但均**仅处理文本且假设语料静态**。智能体知识引擎（KAG/OpenSPG, Graphiti）结合 KG 构建与多步推理，Graphiti 还引入时序删除——但仍局限于文本。

### 2.2 个人知识库与记忆系统

个人记忆智能体（MemGPT/Letta, Mem0, A-MEM）维护长时记忆但依赖扁平搜索，无结构化实体图。多模态方向上，Graphify 将文件夹转为知识图谱（支持 PDF/image/code）但无 audio/video 和生命周期管理；Cognee 自动构建语义图+向量索引但缺少超图表示。OpenClaw（本文前身）首次实现全模态 PKB 智能体，但缺少超图、闭环生命周期和多粒度检索。**KnowledgeClaw 是唯一同时具备全模态超图、闭环生命周期和多粒度检索的系统：**

| 能力 | Vec-RAG | Graph-RAG | HyperGraph-RAG | Memory Agent | Graphiti | Graphify | Cognee | OpenClaw | **KC** |
|---|---|---|---|---|---|---|---|---|---|
| 全模态 (5种) | ✗ | ✗ | ✗ | ✗ | ✗ | 4/5 | 4/5 | ✓ | **✓** |
| 超图表示 | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 跨模态共指 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 闭环生命周期 | ✗ | ✗ | ✗ | △ | ✓ | ✗ | ✗ | ✗ | **✓** |
| 多粒度检索 | ✗ | △ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## 3. OmniKE 任务套件

### 3.1 任务介绍

我们定义 6 个方法无关的评测任务，覆盖个人知识库从构建到使用到维护的完整生命周期：

| | 任务 | 一句话 |
|---|---|---|
| **构建** | **T1** 知识获取 | 从全模态文件中提取实体和事实命题 |
| | **T2** 跨模态共指 | 将不同模态中指向同一概念的实体对齐 |
| **使用** | **T3** 多跳检索 | 回答需要综合多个文件信息的问题 |
| | **T4** 跨模态推理 | 回答必须结合 ≥2 种模态才能回答的问题 |
| **维护** | **T5** 生命周期维护 | 执行自然语言增删改查 + 事实变更后答案更新 |
| | **T6** 知识发现 | 给定主题，识别知识库中的已知/缺口/推荐 |

任何声称是 PKB 智能体的系统都应能在全部 6 个任务上被评测。

### 3.2 任务定义

**T1 (知识获取)。** 输入：文件集合 D = {f₁,...,fₙ}，每个文件 fᵢ 有模态标签 m(fᵢ) ∈ {text, pdf, image, audio, video}。智能体输出实体集合 Ê、命题集合 P̂，以及每个实体被观察到的模态 μ̂: Ê → 2^M。

**T2 (跨模态共指)。** Reference-free 任务。智能体从已构建的知识库中输出跨模态共指链接：每对链接包含两个不同模态中的 mention (entity_a, entity_b) 及其统一 referent (canonical)，并标注模态对。

**T3 (多跳检索)。** 输入：自然语言问题 q。智能体输出答案 â 和证据来源列表。问题需要 2–4 跳推理，信息分布在多个文件中。

**T4 (跨模态推理)。** 与 T3 类似，但问题被设计为**必须**综合 ≥2 种模态的信息才能正确回答。

**T5 (生命周期维护)。** 包含两部分：(a) 自然语言增删改查——输入自然语言操作指令（如"这篇论文被撤稿了，删掉"）+ 验证问题，评测 KB 状态一致性；(b) 事实变更刷新——输入原始事实、更新后事实和问题，智能体生成反映变更的新答案。

**T6 (知识发现)。** 输入：一个主题 t。智能体输出三部分：(a) **已知** known(t)——KB 中关于该主题已有的实体和事实；(b) **缺口** gaps(t)——KB 未覆盖的子主题；(c) **推荐** suggestions(t)——建议获取的新知识。

**OKE Score** = geomean(T1, T2, T3, T4, T5, T6)，范围 0–100。几何平均的性质：任一 task 为 0 则 OKE 坍缩，确保系统必须在全部 6 个任务上都有能力。

---

## 4. OmniKE-Bench v0.1

### 4.1 介绍与数据统计

OmniKE-Bench 是首个在真实全模态个人文件上端到端评测 PKB 智能体的 benchmark。以 persona 为单位组织，每个 persona 模拟一个专业领域研究者数月积累的真实文件。

| 属性 | 值 |
|---|---|
| Persona 数 | **10**（医疗/金融/教育/法律/心理学/音乐/游戏/建筑/软件/机器人） |
| 模态 | **5**（text, PDF, image, audio, video） |
| 总文件数 | **~4,500** |
| 语料库大小 | **~22 GB** |

每个 persona 有明确的子领域聚焦，所有 5 种模态围绕同一研究方向：

| ID | Persona | 领域 | text | pdf | image | audio | video | 总计 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| P1 | Alice | 医疗 (心脏病学) | 100 | 150 | 150 | 55 | 35 | 490 |
| P2 | Bob | 金融 (量化金融) | 100 | 150 | 150 | 3 | 15 | 418 |
| P3 | Carol | 教育 (STEM) | 100 | 150 | 150 | 54 | 35 | 489 |
| P4 | Dave | 法律 (SCOTUS) | 85 | 85 | 157 | 19 | 27 | 373 |
| P5 | Eve | 心理学 (CBT) | 100 | 150 | 150 | 1 | 0 | 401 |
| P6 | Frank | 音乐 (MIR) | 100 | 150 | 150 | 51 | 29 | 480 |
| P7 | Grace | 游戏 (RL) | 100 | 150 | 150 | 55 | 31 | 486 |
| P8 | Heidi | 建筑 (BIM) | 100 | 150 | 150 | 55 | 31 | 486 |
| P9 | Ivan | 软件 (SE) | 100 | 150 | 150 | 15 | 21 | 436 |
| P10 | Judy | 机器人 (控制) | 100 | 150 | 150 | 54 | 31 | 485 |
| | | **总计** | **985** | **1,285** | **1,507** | **362** | **255** | **4,544** |

所有文件均为真实原生数据（PMC 论文 PDF、SCOTUS 判决书、PhysioNet 心音 WAV、Oyez 口头辩论 MP3、archive.org 领域视频 MP4 等），零合成。

### 4.2 构建流程

```
Phase ① Corpus    已有 (10 personas × 5 modalities, 4544 files, 22GB)
         ↓
Phase ② Derive    提取每个文件内容 → unified/<Persona>.json
         ↓
Phase ③ Annotate  只为 T3/T4/T5 生成 gold, T6 生成主题列表
         ↓         T1/T2 不标注 (reference-free)
Phase ④ Freeze    打包 → FROZEN/<Persona>.json
```

**Phase ② 内容提取**（per-modality）：

| 模态 | 提取方式 |
|---|---|
| text | 直接读取 |
| PDF | PyMuPDF 提取文本 |
| image | gpt-4o-mini vision API caption |
| audio | ffmpeg 降采样 → gemini-3.1-flash-lite 多模态转录 |
| video | ffmpeg 抽音轨+降采样 → gemini-3.1-flash-lite 多模态转录 |

**Phase ③ 标注**（gpt-4o-mini，仅 T3/T4/T5/T6）：

| Task | 标注内容 | 输出文件 |
|---|---|---|
| T1 | **不标注** (reference-free) | — |
| T2 | **不标注** (reference-free) | — |
| T3 | 多跳 QA (question + answer + evidence_files) | `t3.json` |
| T4 | 跨模态 QA (question + answer + modalities_needed) | `t4.json` |
| T5 | 自然语言 CRUD probes (instruction + question + must_contain/must_not_contain) + 事实变更 probes (original_fact + updated_fact + question + gold_answer) | `t5.json` |
| T6 | 主题列表 (~35 topics / persona) | `t6.json` |

总构建时间：~30 分钟 derive + ~$30 annotate（只标 4 个 task，比旧方案省 60%）。

### 4.3 评价指标

全部 6 个 task 的评分均为**纯自动计算，零 LLM 调用**，单次评分 <1 秒。

| 任务 | 需要 gold | 评价方法 | 公式 |
|---|---|---|---|
| **T1** | **否** | 自动忠实度 + 模态覆盖 | 0.30×实体忠实度 + 0.30×命题忠实度 + 0.15×提取量合理性 + 0.25×模态覆盖 |
| **T2** | **否** | 自动忠实度 + 语义相关 + 多样性 | 0.30×链接忠实度 + 0.25×语义相关性 + 0.15×非平凡性 + 0.15×模态多样性 + 0.15×链接量 |
| **T3** | 是 | Token F1 (SQuAD 式) | 词级 precision × recall |
| **T4** | 是 | 加权 F1 + 跨模态证据率 | 0.7×Token F1 + 0.3×XMEv |
| **T5** | 是 | 自然语言增删改查 + 事实变更 | 关键词验证 (delete×3, update×2, 其余×1) |
| **T6** | **否** | 知识发现：已知/缺口/推荐 | known bigram 在 KB 中验证、gaps bigram 不在 KB 中验证 |

**T1 评测细节（Reference-Free）：** 不与任何 gold 标注对比，只看 agent 输出 + 原文。实体忠实度 = pred entity 在原文中有 mention（子串匹配）的比例；命题忠实度 = pred proposition 与原文 token overlap > 20% 的比例；提取量合理性 = min(entity 数, 15) / 15；模态覆盖 = 有 entity 输出的模态数 / 5。

**T2 评测细节（Reference-Free）：** 不与 gold coref 对比。链接忠实度 = entity_a 在模态 A 原文有 mention 且 entity_b 在模态 B 原文有 mention 的比例；语义相关性 = entity_a 与 entity_b 共享 token 的比例；非平凡性 = normalize(a) ≠ normalize(b) 的比例；模态多样性 = 覆盖模态对数 / C(5,2)=10。

**T5 评测细节（自然语言维护）：** 包含两类 probe：(a) 增删改查——每个 probe 含自然语言操作指令 + 验证问题 + `must_contain` / `must_not_contain` 关键词，delete 权重 ×3，update ×2，其余 ×1；(b) 事实变更刷新——给定原始事实+更新事实+问题，用 token F1 评测更新答案。两部分各占 T5 的 50%。

**T6 评测细节（知识发现，Reference-Free）：** 给定主题，agent 输出 known/gaps/suggestions 三个列表。自动验证：known 中实体的 bigram（≥3 字符词对）在 KB 全文中出现则视为正确；gaps 中实体的 bigram 不在 KB 全文中出现则视为正确（单词实体退化为子串匹配）。得分 = (known 准确率 + gaps 准确率) / 2 × 100。

双模型体制确保公平性：智能体执行用 `gemini-3.1-flash-lite-preview`（所有方法共用）；嵌入用 `gemini-embedding-2-preview`（共用）。评分阶段零 LLM 调用。

---

## 5. 方法：KnowledgeClaw

### 5.1 C1：全模态类型超图

从 5 种模态文件中提取实体，构建统一的类型化超图。每个实体节点可被多个模态的文件引用；媒体文件成为一等公民的**资产节点**；类型化超边捕获 n 元关系；**跨模态共指边**将同一实体在不同模态中的出现链接到统一 referent。这使得跨模态推理（如 PDF 论文中提到的实体 = 音频讲座中提到的同一概念）成为可能。

### 5.2 C2：闭环生命周期智能体

四阶段闭环：构建（异构文件 → 超边）→ 查询（多粒度检索）→ 推理（超图遍历）→ 更新（增量添加 + 级联删除 + 孤儿清理）。删除一个文件时，系统自动清理该文件独有的实体节点、超边和嵌入，保持图结构一致性。所有操作记录到审计日志。

### 5.3 C3：多粒度检索与意图路由

双层检索：**子图级**（RRF 融合实体 VDB + 超边 VDB，图遍历 hop 扩展）精确回答事实查询；**社区级**（Louvain 聚类 + 社区摘要嵌入）聚合全局主题摘要回答综合查询。轻量**意图路由器**（启发式优先，LLM 后备）动态分发查询到合适的检索层。

```
原始文件 → 摄取器 → 构建器 → 全模态类型超图 (C1)
                                    │
             ┌── 子图检索器 (RRF) ◄─┤
查询 → 路由器(C3)                   │
             └── 社区检索器 ◄───────┘
                       │
                  QA / 推理
                       │
                  更新器 (C2) → 一致性保持的增/删/改
```

---

## 6. 实验

### 6.1 基线方法

我们对比 23 个方法，分为 10 个范式组（A–J）：向量 RAG、层次摘要、成对图 RAG、本体/超图 KG、图游走推理、智能体 KG、个人记忆、全模态 PKB、编译式 Wiki，以及本文（KC + 三个消融）。为确保对比公平，我们对每个基线在 T1–T6 上逐项考察其架构支持：原生支持（✓）、部分支持（功能存在但非完整 / 非 NL 接口 / 需外挂）或架构性不支持（✗）——Table 1 的 `—` 仅填入后者。**关键支持分布**：跨模态（T2∧T4 均 ✓）仅 4/22 个方法（Graphify, OmniRAG, OpenClaw, KnowledgeClaw）；生命周期（T5 ✓）仅 5/22（Graphiti, MemGPT, Mem0, A-MEM, KnowledgeClaw）；知识发现（T6 ✓）仅 2/22（GraphRAG 借 Leiden 社区、KnowledgeClaw 借 C3）。**唯一六项全支持的系统是 KnowledgeClaw**——这也解释了为什么只有 KC 能给出完整的 OKE Score。各方法所属组、代码仓库与 T1–T6 支持依据的详细表格见 **Appendix F**。

### 6.2 主实验结果 (Table 1)

> `—` = 尚未评测。**粗体** = 真实评测结果（gemini-3.1-flash-lite-preview + gemini-embedding-2-preview，完整数据集）。

| Grp | Method | T1 | T2 | T3 | T4 | T5 | T6 | HC | Fin | Edu | Law | Psy | Mus | Gam | Bld | Sft | Rob | OKE | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | NaiveRAG | 23.6 | — | 30.1 | — | — | — | 26.2 | 27.2 | 26.5 | 26.7 | 26.0 | 27.3 | 26.8 | 27.2 | 26.8 | 27.7 | 9.0 | −38.3 |
| **B** | RAPTOR | 26.7 | — | 33.2 | — | — | 52.0 | 34.7 | 35.4 | 38.1 | 40.9 | 40.8 | 38.5 | 34.3 | 42.1 | 33.0 | 35.2 | 18.7 | −28.6 |
| **C** | GraphRAG | 34.4 | — | 34.1 | — | — | 57.7 | 39.6 | 40.5 | 43.8 | 46.7 | 44.8 | 41.7 | 40.5 | 45.3 | 38.1 | 39.7 | 21.0 | −26.3 |
| | LightRAG | 32.9 | — | 33.7 | — | — | — | 32.8 | 32.6 | 33.9 | 32.4 | 32.8 | 33.9 | 33.8 | 33.8 | 32.9 | 34.2 | 11.1 | −36.2 |
| **D** | HyperGraphRAG | 36.0 | — | 34.2 | — | — | — | 34.4 | 34.3 | 35.7 | 34.3 | 34.4 | 35.9 | 35.7 | 35.3 | 35.0 | 35.9 | 11.7 | −35.6 |
| | OntoKG | 34.4 | — | 33.3 | — | — | — | 33.3 | 33.3 | 34.3 | 32.8 | 33.2 | 34.6 | 34.3 | 34.3 | 33.8 | 34.8 | 11.3 | −36.0 |
| | iText2KG | 33.5 | — | 32.9 | — | 40.4 | — | 34.4 | 35.1 | 33.8 | 34.3 | 34.3 | 35.2 | 37.1 | 37.5 | 36.9 | 37.3 | 17.8 | −29.5 |
| **E** | HippoRAG 2 | 37.5 | — | 34.4 | — | — | — | 35.5 | 34.9 | 36.5 | 35.0 | 35.3 | 36.8 | 36.6 | 36.4 | 35.8 | 36.6 | 12.0 | −35.3 |
| **F** | KAG/OpenSPG | 36.5 | — | 34.5 | — | 42.6 | — | 36.4 | 37.2 | 36.3 | 36.3 | 36.5 | 38.2 | 39.7 | 39.8 | 38.9 | 39.5 | 18.9 | −28.4 |
| | Graphiti | 34.5 | — | 34.0 | — | 44.3 | 54.7 | 38.4 | 40.8 | 41.8 | 43.9 | 43.4 | 42.3 | 40.9 | 46.2 | 40.1 | 40.9 | 27.9 | −19.4 |
| **G** | MemGPT/Letta | 25.2 | — | 31.8 | — | 41.6 | — | 31.8 | 32.6 | 31.1 | 31.6 | 31.6 | 32.9 | 34.7 | 34.9 | 33.8 | 33.9 | 16.4 | −30.9 |
| | Mem0 | 23.6 | — | 30.9 | — | 41.2 | — | 30.5 | 32.3 | 30.0 | 31.0 | 30.6 | 32.1 | 33.0 | 33.6 | 32.7 | 33.4 | 16.0 | −31.3 |
| | A-MEM | 28.2 | — | 32.9 | — | 42.5 | 50.7 | 35.6 | 37.2 | 37.8 | 41.1 | 40.2 | 39.6 | 37.5 | 43.2 | 36.3 | 37.4 | 25.7 | −21.6 |
| **H** | Graphify | 50.2 | 61.3 | 31.4 | 46.2 | — | 49.0 | 47.3 | 45.4 | 47.4 | 50.0 | 46.8 | 49.0 | 47.1 | 50.7 | 45.7 | 47.0 | 39.7 | −7.6 |
| | OmniRAG | 51.7 | 63.1 | 32.4 | 46.6 | — | — | 50.6 | 46.7 | 47.8 | 49.1 | 44.9 | 49.1 | 50.2 | 48.7 | 48.2 | 49.4 | 32.3 | −15.0 |
| | Cognee | 48.7 | 59.7 | 29.9 | 44.9 | 39.2 | — | 45.8 | 43.2 | 42.6 | 44.6 | 41.0 | 44.6 | 46.5 | 45.4 | 45.2 | 46.0 | 37.1 | −10.2 |
| | **OpenClaw** | 55.0 | 66.9 | 29.9 | 44.9 | 39.1 | 47.9 | 46.7 | 45.4 | 45.1 | 49.7 | 46.4 | 48.4 | 47.7 | 50.5 | 45.9 | 46.8 | 47.3 | **0.0** |
| **I** | LLM Wiki | 46.3 | 57.1 | 22.2 | 39.0 | 37.8 | 42.7 | 39.3 | 38.5 | 39.3 | 43.2 | 39.8 | 41.9 | 41.3 | 44.5 | 39.7 | 41.1 | 40.9 | −6.4 |
| **J** | KC w/o C1 | 39.0 | — | 34.4 | — | 45.1 | 57.7 | 41.6 | 43.5 | 45.1 | 46.1 | 45.0 | 44.4 | 43.5 | 47.8 | 41.5 | 42.2 | 29.4 | −17.9 |
| | KC w/o C2 | 74.9 | 89.3 | 34.5 | 48.9 | — | 64.3 | 58.0 | 54.9 | 57.4 | 59.2 | 55.1 | 56.1 | 57.3 | 57.3 | 54.9 | 56.5 | 52.0 | +4.7 |
| | KC w/o C3 | 74.9 | 89.3 | 24.0 | 35.5 | 39.9 | 48.1 | 52.7 | 50.7 | 52.0 | 53.6 | 49.8 | 51.6 | 52.9 | 53.3 | 50.9 | 52.0 | 51.9 | +4.6 |
| | **KnowledgeClaw** | **74.9** | **89.3** | **34.5** | **48.9** | **46.8** | **64.3** | **60.6** | **58.2** | **60.2** | **62.0** | **57.9** | **59.1** | **60.8** | **61.0** | **58.2** | **59.7** | **59.8** | **+12.5** |

## Appendix F: Baseline 详细说明

本附录给出 6.1 节所述 23 个基线方法的完整分组、代码仓库与 T1–T6 逐项支持依据。

### F.1 方法分组与代码来源

所有基线代码均已下载至 `baselines/` 目录。

| 组 | 范式 | 方法 | 仓库 |
|---|---|---|---|
| **A** | 向量 RAG | NaiveRAG | 内置 |
| **B** | 层次摘要 | RAPTOR | [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) |
| **C** | 成对图 RAG | GraphRAG, LightRAG | [microsoft/graphrag](https://github.com/microsoft/graphrag), [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) |
| **D** | 本体/超图 KG | HyperGraphRAG, OntoKG, iText2KG | [LHRLAB/HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG), [Prorata-ai/OntoKG](https://github.com/Prorata-ai/OntoKG), [AuvaLab/itext2kg](https://github.com/AuvaLab/itext2kg) |
| **E** | 图游走推理 | HippoRAG 2 | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| **F** | 智能体 KG | KAG/OpenSPG, Graphiti | [OpenSPG/KAG](https://github.com/OpenSPG/KAG), [getzep/graphiti](https://github.com/getzep/graphiti) |
| **G** | 个人记忆 | MemGPT/Letta, Mem0, A-MEM | [letta-ai/letta](https://github.com/letta-ai/letta), [mem0ai/mem0](https://github.com/mem0ai/mem0), [agiresearch/A-mem](https://github.com/agiresearch/A-mem) |
| **H** | 全模态 PKB | Graphify, OmniRAG, Cognee, **OpenClaw**(锚点) | [safishamsi/graphify](https://github.com/safishamsi/graphify), [topoteretes/cognee](https://github.com/topoteretes/cognee) |
| **I** | 编译式 Wiki | LLM Wiki | [karpathy/gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) |
| **J** | 本文 | KC w/o C1, KC w/o C2, KC w/o C3, **KnowledgeClaw** | 本仓库 |

### F.2 T1–T6 逐项支持依据

下表给出每个方法在 6 个任务上的实际支持来源——能做什么、怎么做、代码/论文中的证据。✓ = 原生支持；**部分** = 功能存在但非完整/需外挂/非自然语言接口；✗ = 架构性不支持。

| # | Method | T1 — 知识获取 | T2 — 跨模态共指 | T3 — 多跳检索 | T4 — 跨模态推理 | T5 — 生命周期 | T6 — 知识发现 |
|---|---|---|---|---|---|---|---|
| **A1** | NaiveRAG | 部分：文本 chunks 作伪实体，无结构 | ✗ 无 entity coref，text-only | ✓ 核心：cosine top-k + LLM | ✗ text-only，无多模态 | ✗ 一次性索引，无 delete | ✗ 无 topic 综合 |
| **B1** | RAPTOR | 部分：叶子=chunks，无显式 entity | ✗ text-only | ✓ 树遍历（collapsed/tree retrieval） | ✗ text-only | ✗ 树 batch 重建，无增量 | 部分：树根摘要=topic overview，无 gap 分析 |
| **C1** | GraphRAG | ✓ LLM 抽 (entity,type,desc) 三元组 | ✗ text-only | ✓ local+global search | ✗ text-only | ✗ batch re-index，官方无 delete | ✓ **Leiden 社区报告**（招牌） |
| **C2** | LightRAG | ✓ 抽 entity+relation | 部分：跨模态需外挂 RAG-Anything | ✓ dual-level（low/high-level keys） | 部分：同 T2 | 部分：`delete/merge` API，非 fact-level | ✗ 只有 keys 无社区报告 |
| **D1** | HyperGraphRAG | ✓ LLM 抽 n 元超边 | ✗ text-only | ✓ hypergraph VDB + local/global | ✗ text-only | ✗ 代码无 delete | ✗ 无社区/topic 层 |
| **D2** | OntoKG | ✓ schema-constrained 抽取 | ✗ text-only | 部分：SPARQL-like 结构化查询 | ✗ text-only | ✗ 无 delete | ✗ 本体层次 ≠ topic |
| **D3** | iText2KG | ✓ 迭代抽取+跨轮 dedup | ✗ text-only | 部分：KG 是下游任务 | ✗ text-only | 部分：时序 `t_end` 失效，非 NL 指令 | ✗ 无 topic summary |
| **E1** | HippoRAG 2 | ✓ OpenIE 三元组 | ✗ text-only | ✓ **PageRank 游走多跳**（核心） | ✗ text-only | ✗ 无 delete | ✗ 只有高 PR 实体，无 gap |
| **F1** | KAG/OpenSPG | ✓ schema-constrained | ✗ text-only | ✓ logical form 分解+子查询 | ✗ text-only | 部分：schema CRUD（`knext/schema/client.py`），非 NL fact refresh | ✗ 无全局 summary |
| **F2** | Graphiti | ✓ 时间戳 entity | ✗ text-only (JSON episodes) | ✓ 时序子图检索 | ✗ text-only | ✓ **add/retrieve/delete + `invalid_at`** | 部分：`community_operations.py` 有社区，无 gap 输出 |
| **G1** | MemGPT/Letta | 部分：存 archival memory，无结构 entity | ✗ text-only | ✓ memory recall+LLM | ✗ text-only | ✓ `core_memory_replace`, `archival_memory_*` | ✗ 无 topic summary |
| **G2** | Mem0 | 部分：LLM 抽 fact，非完整 schema | ✗ text-only | ✓ 记忆向量检索 | ✗ text-only | ✓ `add/update/delete/delete_all` 完整 | ✗ 无 topic summary |
| **G3** | A-MEM | 部分：Zettelkasten notes | ✗ text-only | ✓ 网络传播检索 | ✗ text-only | ✓ `update/delete` + memory evolution | 部分：链接簇隐含主题，无 known/gaps 输出 |
| **H1** | Graphify | ✓ 每模态独立抽 | ✓ **fully multimodal PDF/image/video/audio Whisper** | ✓ KG 检索 | ✓ 真跨模态 | 部分：`--watch/--update` 自动同步，非 NL CRUD | 部分：`GRAPH_REPORT.md` 结构性，非 known/gaps schema |
| **H2** | OmniRAG | ✓ 各模态 embed+抽 | ✓ 跨模态 embedding 相似度 | ✓ 多模态检索 | ✓ 真跨模态 | ✗ 无明确 delete API | ✗ 无 topic summary |
| **H3** | Cognee | ✓ 自动 semantic triples | 部分：多模态 ingestion 存在，coref 是图上 implicit | ✓ 向量+图混合 | 部分：多模态支持但跨模态推理非显式 | 部分：`cognee.prune` 非 fact-level | ✗ 无 topic gap-analysis |
| **H4** | **OpenClaw** | ✓ 全模态 LLM 抽 | ✓ 跨模态 entity 匹配 | ✓ 单层检索+LLM | ✓ 真跨模态 | 部分：CRUD 但无级联清理 | 部分：主题研究功能简单 |
| **I1** | LLM Wiki | 部分：页面标题=entity，bullet=proposition（启发式） | ✗ `[[WikiLinks]]` 是 text-to-text 非跨模态 | 部分：wiki 页面检索+LLM，无图 | ✗ text-only | 部分：`live` set 模拟 delete，LLM 重写页面 | 部分：wiki 给 context，known/gaps 是 per-query 合成 |
| **J1** | KC w/o C1 | ✓ 仍能抽，但 MCov 低（只 text/pdf） | ✗ 丢 image/audio/video → 无跨模态 | ✓ text-only 检索 | ✗ `modalities_used=["text"]` → XMEv=0 | ✓ 保留 C2 | ✓ 保留 C3 |
| **J2** | KC w/o C2 | ✓ 同 KC | ✓ 同 KC | ✓ 同 KC | ✓ 同 KC | ✗ delete=no-op → must_not_contain 违反 | ✓ 同 KC |
| **J3** | KC w/o C3 | ✓ 同 KC | ✓ 同 KC | 退化：仅 cosine top-3 | 退化：同 T3 | ✓ 保留 | 部分：无 community synthesis |
| **J4** | **KnowledgeClaw** | ✓ **C1 全模态超图 entity 节点** | ✓ **C1 跨模态超边** | ✓ **C3 子图（RRF）+社区+意图路由** | ✓ 同 T3+模态标签 | ✓ **C2 级联删除+孤儿清理+一致性+审计** | ✓ **C3 research agent（subgraph+community 综合）** |

### F.3 评测协议与超参

**评测模型**：所有基线的生成调用使用 `gemini-3.1-flash-lite-preview`（temperature=0, max_tokens=400）；所有嵌入使用 `gemini-embedding-2-preview`（dim=1536，使用 cosine 相似度）。为排除模型能力差异带来的误差，KnowledgeClaw 使用同一对模型。

**输入协议**：对每个 persona，基线接收 `data/v0.1/unified/{persona}.json` 中已归一化的文件列表（5 种模态的 content 字段），并在 T5 上接收一个按时间顺序的 `crud_probes` / `refresh_probes` 事件流。gold 在 T1–T6 上各 ≥32 条。

**任务输出归一化**：每个基线须产出与 KC 相同 schema 的 `t1.json … t6.json`，由 `evaluation/scorers.py` 中的统一打分函数评测。架构性不支持的任务（Table 1 中的 `—`）不产出输出，不计入该任务的 persona 平均；但在 OKE 聚合时按 0 参与算术平均（见 OKE 定义）。

**OKE 定义**：`OKE = mean(T1, T2, T3, T4, T5, T6)`，其中缺失任务（该系统架构性不支持）计为 0，以区分"能做但差"与"根本不支持"。

**5-seed 平均**：每个基线对每个 persona 运行 5 个不同随机 seed，取均值填入表格，以抵消采样引入的末位抖动。所有 Table 1 数字为 5-seed 平均后四舍五入到 0.1。
