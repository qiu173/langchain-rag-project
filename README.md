# LangChain 文档智能问答系统

用 RAG（检索增强生成）技术构建针对 LangChain 官方文档的问答系统，
并对不同分块（chunking）策略、检索策略进行量化对比实验。

> **"用 RAG 做 RAG 工具的文档问答"** —— 这个项目的核心叙事：
> LangChain 文档内容分散在多个模块（核心概念 / 集成 / API Reference），
> 用户的真实问题经常需要跨文档片段综合回答，这正好暴露了 naive RAG
> 的局限性（分块割裂上下文、单一检索策略召回不全），本项目针对这些
> 具体问题做了对比实验和改进。

## 数据版本锁定

本项目基于 LangChain 文档构建，版本信息：

- 仓库：https://github.com/langchain-ai/docs
（LangChain 官方文档已于 2026 年迁移到独立仓库，Mintlify 构建，源文件路径为 `src/oss/`）
- 分支：`main`（该仓库持续部署，没有 release tag，因此改用锁定 commit hash）
- Commit：`cd2543874db5b72b8b6e88270dcac8d5830b4dab`
- 拉取日期：2026-07-20
- 文档数量：70 篇（仅保留 `langchain` 子目录下的核心概念/Agent 框架文档，
  过滤掉了 langgraph/deepagents/langsmith 等相关但非本项目核心的文档）

> 锁定 commit 是为了保证实验结果可复现——文档更新很快，
> 如果不锁版本，几周后重新跑实验会得到不一样的结果和数据。

## 项目结构

```
langchain-rag-project/
├── scripts/
│   └── fetch_docs.py        # 拉取并清洗文档
├── src/
│   ├── chunking.py          # 三种分块策略实现
│   ├── ingest.py            # 向量化 + 存入 Chroma
│   ├── retriever.py         # 检索模块（含 rerank）
│   ├── router.py            # 查询路由（判断是否需要检索/调用工具）
│   └── rag_pipeline.py      # 端到端 pipeline
├── eval/
│   ├── build_testset.py     # 辅助构建测试集
│   ├── testset.json         # 人工标注的测试集（49条）
│   └── evaluate.py          # 评估脚本，输出量化对比表
├── data/                     # 存放拉取下来的文档（不进 git）
├── .env.example
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY 或 DEEPSEEK_API_KEY

# 3. 拉取文档（锁定到 main 分支的某个 commit，脚本会打印实际 commit hash）
python scripts/fetch_docs.py --ref main

# 4. 分块 + 建索引（三种策略分别建独立 collection）
python src/ingest.py --strategy fixed --collection langchain_fixed
python src/ingest.py --strategy recursive --collection langchain_recursive
python src/ingest.py --strategy semantic --collection langchain_semantic

# 5. 运行评估，对比不同策略效果（--rerank 会同时测试加/不加 rerank 的版本）
python eval/evaluate.py --collections langchain_fixed langchain_recursive langchain_semantic --rerank
```

## 实验设计（重点部分）

### 变量 1：分块策略对比
- **固定长度分块**：baseline，简单粗暴，容易切断语义
- **递归字符分块**：按标题层级 -> 段落 -> 句子递归切分，LangChain 文档本身是 markdown，天然适合
- **语义分块**：基于 embedding 相似度动态确定切分点

### 变量 2：是否加 rerank
对比加 / 不加 rerank 模型（如 bge-reranker）对 Top-K 命中率的影响

### 变量 3：查询路由（Agent 能力）
不是所有问题都要走检索——比如"你好"、"LangChain 是谁开发的"这类问题，
系统应该判断出不需要检索或只需要简单检索，而"如何用 LCEL 实现一个带
记忆的多轮对话链，并且要支持流式输出"这种问题需要多路检索 + 综合。

### 评估指标
- 检索层面：Recall@5, MRR
- 测试集：49 条人工构造/审核的问答对（见下方"测试集构建"）

## 实验结果

| 分块策略 | Recall@5（无 rerank） | Recall@5（+rerank） | MRR（无 rerank） | MRR（+rerank） |
| 固定长度分块 | 56.8% | 68.2% | 0.441 | 0.588 |
| 递归结构化分块 | 55.3% | 64.8% | 0.438 | 0.540 |
| 语义分块 | 54.5% | **87.9%** | 0.362 | **0.811** |

*（基于 44 条需要检索的问题评估，测试集共 49 条，其余 5 条为路由测试问题不参与检索评估）*

### 关键发现

**语义分块单独使用时并非最优解，但与 rerank 结合后效果显著反超。**
不加 rerank 时，三种分块策略的 Recall@5 相差不大（54.5%~56.8%），语义
分块反而略低；但加入 rerank 之后，语义分块的 Recall@5 跳升至 87.9%，
比固定分块和递归分块分别高出 **19.7 个百分点**和 **23.1 个百分点**，
MRR 提升幅度也远超另外两种策略。

对此的解释是：语义分块生成的 chunk 语义边界更完整（按内容主题切分，
而非机械按字符数切分），这类完整的 chunk 更有利于 cross-encoder
rerank 模型做细粒度的语义匹配打分；而固定/递归分块产生的 chunk 有时
会切断一段完整的语义（例如把一个代码示例从中间截断），即使 rerank
重排，chunk 本身信息不完整也难以获得准确的相关性判断。

**结论**：分块策略和排序策略需要匹配评估，不能孤立地说某种分块策略
"更好"——语义分块 + rerank 的组合是本项目测试集上的最优配置，已设为
`rag_pipeline.py` 的默认参数。

### Chunk 数量对比（验证上述假设的数据支撑）

| 分块策略 | Chunk 数量 |
| 固定长度分块 | 3600 |
| 递归结构化分块 | 3412 |
| 语义分块 | **323** |

语义分块产生的 chunk 数量仅为另外两种策略的约 **1/11**，直接印证了
"语义分块合并出更大、更完整的语义单元"这一假设——固定分块和递归分块
机械地按字符数切分，产生大量细碎 chunk；语义分块则按内容语义边界
动态合并，单个 chunk 包含更完整的上下文。这正是它在 rerank 环节表现
更好的根本原因：完整的语义单元让 cross-encoder 有更充分的信息做
细粒度相关性判断，而细碎的 chunk 即使重排也难以补全缺失的上下文。

## 测试集构建思路

`eval/testset.json` 实际包含 49 条问答对，构建方式：

1. **LLM 辅助生成候选问题**（40 条，`single_doc` 类型）：用
   `eval/build_testset.py` 从实际拉取到的文档中随机抽样，让 LLM 针对
   每篇文档生成一个真实用户可能会问的问题，再人工审核筛选、补充标准
   答案摘要
2. **跨文档综合问题**（4 条，`multi_hop` 类型）：手动构造，答案需要
   综合 2-3 篇文档才能完整回答，用于测试系统的多跳检索能力
3. **路由测试问题**（5 条，`direct` 类型）：闲聊、常识性问题，用于
   验证 `router.py` 能正确判断"不需要检索"，避免所有问题都无脑走 RAG


## 已知局限与改进方向

**语义分块在处理内容分散于大 chunk 中的细节性问题时，召回率会明显下降。**

实际问答测试中发现：针对"`@tool` 装饰器中 docstring 的作用"这类问题，
系统未能检索到正确答案；但结构近似的问题（如"两个 tool_calls 只提供一个
ToolMessage 会报什么错"）却能准确命中并给出高质量回答。

推测原因：语义分块会把内容合并成更大、更完整的语义单元（见上文
"Chunk 数量对比"，平均每个 chunk 体积是另外两种策略的 10 倍以上）。当某个
细节性知识点被合并进一个整体讨论其他主题的大 chunk 时，该 chunk 的向量
表示会被主导内容"稀释"，导致语义相关性分数不够高，无法进入 Top-K 候选、
连 rerank 都没有机会补救。这与 87.9% 的 Recall@5（而非 100%）是一致的——
约 12% 的问题会落入这一失败模式。

**补充发现**：进一步测试同一问题在递归分块策略下的表现，发现即使
chunk 切分更细，该问题依然未能命中正确文档。这说明失败原因不完全
是"大 chunk 稀释"单一因素，也可能与 embedding 模型（bge-small-zh-v1.5）
处理"中文提问 + 英文专业术语"这类跨语言查询时的语义对齐能力有关——
这是当前 embedding 模型选型带来的固有局限，而非分块策略可以单独解决
的问题。

**可能的改进方向**（本项目范围外，留作后续迭代）：
- 语义分块增加最大 chunk 长度上限，超过阈值强制二次切分，避免单个 chunk
  过度膨胀
- 检索阶段引入混合检索（向量检索 + 关键词检索 BM25），弥补纯向量检索
  对细节术语不敏感的问题
- 扩大 rerank 阶段的候选召回数量（当前 top_k×4），增加细节内容被召回
  进入候选集的概率
 - 更换或对比支持更强跨语言对齐能力的 embedding 模型（如 bge-m3），
  验证中文提问检索英文文档场景下的召回率是否有实质提升