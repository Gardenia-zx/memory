# 记忆系统：分关实现指南

目标：自己实现一个能够保存、提取、更新、检索和遗忘信息的 Agent 记忆系统，并用 LongMemEval、LoCoMo 和工程用例验收。

本文前半保留逐关契约，便于追溯设计原因；当前实现已经覆盖第 1—11 关的工程开发。

## 初始实现选择

- Python。当前检查到的解释器为 3.13.9。
- SQLite 保存原始消息和后续结构化记忆；消息序号与后台任务支持多个独立连接并发写入。
- 单元测试使用 unittest；模型和向量服务均可注入，离线功能测试不会访问网络。
- 实体、事实和关系先按图模型设计，用关系表表达；专用图数据库根据评测收益再决定。
- 每一关保存稳定的来源 ID 和作用域，使后续可以追溯错误、重放历史和计算证据覆盖率。

SQLite 可由 Python 的 sqlite3 模块访问，无须另起数据库服务。参考：[Python sqlite3 文档](https://docs.python.org/3/library/sqlite3.html)。

## 实现路线

| 关卡 | 实现内容 | 可观察的验收成果 |
|---|---|---|
| 1 | 原始消息记录与重放 | 重启后原文仍在；按用户、会话隔离；重复写入不产生新消息 |
| 2 | 近期队列与基础上下文组装 | 消息和工具交互顺序正确；预算不足时明确触发压缩需求；不静默丢消息 |
| 3 | 原始历史检索与最小评测 | 关键词检索能找回来源；无相关信息返回空；保存不依赖事实提取的首份基线 |
| 4 | 结构化事实存储与读取 | 手工录入事实，保存作用域、来源、有效时间、状态和版本；新会话能查回旧事实 |
| 5 | 关系判断与记忆更新 | 重复、补充、状态变化、纠错、多值事实和争议各有明确处理；保留历史版本 |
| 6 | 模型接入、候选提取与函数执行器 | 从消息产生带证据候选；结构化校验后更新；明确记住操作提交成功后才确认 |
| 7a | 历史摘要与队列压缩 | 摘要有来源与覆盖边界；摘要提交后才淘汰对应消息 |
| 7b | 任务状态与恢复 | 目标、约束、待办和执行状态独立保存；恢复时不重复已完成操作 |
| 7c | 事件与经历整理 | 经历保留来源；区分计划、执行和已验证结果 |
| 8 | 外部资料和混合检索 | 文档按版本分块并保留引用；全文与向量召回可对照；资料与用户偏好保持来源边界 |
| 9 | 后台任务、一致性与遗忘 | 提取任务可重试；旧任务不覆盖新事实；删除向所有已有派生内容传播且不复活 |
| 10 | 图关系检索 | 实体消歧、事实与来源关联、有限范围关系扩展可用；能和无图方案比较 |
| 11 | 正式基准验收 | 完成 LongMemEval、LoCoMo 官方评分、分类报告、成本报告与工程回归 |

## 当前完成状态（2026-10-26）

开发实现已经完成。`python -X utf8 -m unittest discover -s tests -v` 当前运行 368 项功能测试，全部通过。覆盖范围如下：

| 阶段 | 当前实现 |
|---|---|
| 1—5 | 原始消息、上下文、检索评测、版本化事实、关系判断、纠错/撤回/迁移、争议及人工解决 |
| 6 | OpenAI-compatible Responses/Chat Completions HTTP 适配器、严格候选提取、完整事实快照、固定计划执行、崩溃后幂等恢复 |
| 7 | 摘要生成/存储/压缩，独立任务状态，带来源且区分计划/执行/验证的事件 |
| 8 | 外部文档版本与稳定分块，BM25 风格全文检索、可注入向量检索及混合 RRF |
| 9 | 原子消息序号、租约任务、任务代际保护、过期恢复、全派生表物理遗忘和防复活墓碑 |
| 10 | Unicode 实体别名消歧、合并重定向、带真实来源的关系、有界/时态图扩展和配对对照 |
| 11 | 官方 JSON 形状适配、标签隔离只读快照、lexical/session-first 基线、逐题 checkpoint、预测导出、官方评分命令/LoCoMo 桥接、分类/证据/成本报告 |

已使用硅基流动真实 API 完成 LoCoMo 的 1、20 和 199 题开发烟测；尚未执行完整 LongMemEval/LoCoMo 官方评分、最终准确率或性能压测。仓库没有写入 API key，下面记录的是开发基线，不是正式基准成绩。

### 第 6 关：模型、候选和执行流水线

- `src/model_adapter.py` 定义 `ModelAdapter`、无配置保护和 `OpenAIResponsesAdapter`。通过 `api_mode` 支持 `POST /responses` 和 OpenAI-compatible `POST /chat/completions`；两种模式均要求严格 JSON，并统一记录输入/输出 token。
- `src/candidate_extractor.py` 只接受严格 JSON。模型不能指定 user_id、message/session 来源或事实 ID；这些字段由可信调用上下文和稳定 ID 工厂绑定。
- `src/memory_pipeline.py` 先持久化候选，再逐条保存固定关系计划，最后调用 `MemoryUpdater`。崩溃发生在事实提交后、流水线进度提交前时，重试复用同一 action/version，不再次生成更高版本。
- `MemoryPipeline.handle_job` 可直接注册为 `BackgroundWorker` 的 `extract_message` 处理器；完成记录不会在重试时再次调用模型。

API key 有意留空。接入示例：

```python
from src.model_adapter import ModelConfig, OpenAIResponsesAdapter

config = ModelConfig(
    model="",       # 部署时填写
    api_key="",     # 部署时填写；不要提交到仓库
    base_url="https://api.openai.com/v1",
)
adapter = OpenAIResponsesAdapter(config)
```

空 model/key 会在网络请求前抛出 `ModelNotConfiguredError`。也可以注入任何实现 `complete_json(task=..., payload=...)` 的本地或其他供应商适配器。Responses 请求与 JSON mode 约定依据[官方 OpenAI 文档](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)。

### 第 7 关：摘要、任务和事件

- `SummaryGenerator` 从真实连续消息区间生成摘要；相同 summary_id 重试不再次调用模型。`SummaryStore` 校验覆盖边界，不能拆开 assistant tool call 和后续 tool 结果。
- `ContextCompactor` 只使用已提交摘要替换队列前缀，保留最近原文、原始库和 seq 水位；没有合适摘要时明确抛出预算异常。
- `TaskStore` 独立保存目标正文、结构化约束/待办 payload 和 pending/done/cancelled 状态，可通过 pending 列表恢复。
- `EventStore` 保存 occurred_at、phase（planned/executed/verified）、参与者、结构化 payload 和真实消息来源。

### 第 8 关：外部资料和混合检索

- `ExternalDocumentStore` 使用 `(user_id, document_id, version)` 保存版本，旧 active 版变为 superseded；分块包含稳定 chunk_id、字符起止偏移和 metadata 引用。
- 文档支持 user/session 作用域、左闭右开有效时间和逻辑撤回，始终与聊天消息/用户事实保持不同的 typed source。
- `HybridRetriever` 同时检索消息和文档分块。`lexical` 可离线运行；`vector`/`hybrid` 必须显式注入 embedding。`HashEmbedding` 只用于确定性的功能测试，不冒充语义模型。
- `SessionFirstRetriever` 先按每个会话中最高的消息 BM25 分选择会话，再返回全部消息或每会话 Top-N 消息；始终保留用户隔离、`as_of` 时间过滤和真实来源。

### 第 9 关：后台一致性与遗忘

- `MessageStore` 用独立序列表和 SQLite 写锁原子分配 seq，多个连接并发写入仍连续且身份重试不占新序号。
- `JobStore` 提供 enqueue/claim/complete/fail/recover，任务带租约、最大尝试次数和 subject generation；新一代会取消旧 pending/running，旧租约或过期租约不能提交。
- `MemoryForgetter` 根据 SQLite 外键图计算子表优先删除顺序，在一个事务内清除当前 schema 中所有带 user_id 的表，并保存不可变墓碑收据。任何遗漏、IGNORE 或触发器失败都会整体回滚；旧消息和旧任务均不能复活用户数据。

### 第 10 关：实体图

- `EntityGraphStore` 保存实体、NFKC/casefold 别名、合并重定向、关系有效期和真实消息来源。
- 图扩展限制 max_depth/max_nodes，处理环、合并链、用户隔离和时间过滤；达到节点上限时不会返回缺失端点的边。
- `GraphHistoryRetriever.compare` 对相同 query/top_k 返回 without_graph/with_graph，供正式评测做配对比较。实现仍使用 SQLite 关系表；是否引入专用图数据库由用户的性能/准确率结果决定。

### 第 11 关：正式基准工具链

- `src/eval/benchmark_datasets.py` 读取本地官方 JSON：LongMemEval 的 question/haystack 结构，以及 LoCoMo 的 conversation/session/qa 结构。答案、证据、人工 summary/observation 不进入建库消息。LoCoMo 发布文件中的复合证据字符串会拆成真实 dialog ID，可无歧义修正的冒号/前导零写法会规范化；仍不存在的标注保留在 `unresolved_evidence` 供诊断，不伪造消息来源。
- `run_benchmark` 每份历史只建一次库，每题使用新的只读 SQLite 连接；问答和评分反馈不能写回固定快照。报告包含分类回答分、来源/会话证据覆盖、建库和查询 token/cost（调用方提供 usage 时）及快照哈希；`session_first` 返回量不是固定 Top-K，必须结合实际来源数解释。
- `export_predictions_jsonl` 生成 LongMemEval 的 question_id/hypothesis JSONL；`export_locomo_predictions` 生成带 `*_prediction` 和 evidence context 的 LoCoMo 输入。
- `official_score_command` 只生成 argv、工作目录和官方脚本哈希，不会隐式运行可能付费的裁判。`eval/score_locomo_official.py` 调用官方 `eval_question_answering` 并输出总体/分类回答分和检索召回。
- `eval/benchmark_config.example.json` 保留通用默认值；正式运行使用已被 Git 忽略的 `eval/benchmark_config.local.json`，不要提交密钥。

适配器按 [LongMemEval 官方仓库](https://github.com/xiaowu0162/LongMemEval)和 [LoCoMo 官方仓库](https://github.com/snap-research/locomo)的数据/评分接口实现。夹具只验证功能和隔离，不代表正式分数。

`external/` 包含本地克隆的官方仓库和原始数据，已加入 `.gitignore`，不随本项目提交。下面的默认路径假定分别准备在 `external/LongMemEval` 和 `external/locomo`；测试所需的小型合成夹具保留在 `tests/fixtures/`。

#### 运行真实基准入口

`eval/run_benchmark.py` 提供可执行基线：完整历史写入 MessageStore，可选择 `HybridRetriever` 的 `lexical` 模式或 `SessionFirstRetriever` 的 `session_first` 模式，再调用 OpenAI-compatible Responses 或 Chat Completions API 生成严格 JSON 答案。它明确标记为 `raw_history_baseline`，不会逐消息调用 MemoryPipeline，也不代表事实、向量或图检索成绩。

回答提示版本 `raw-lexical-json-v2` 使用每条检索消息的 `occurred_at` 解析相对时间，保留日/月/周及持续时长的原始粒度，并要求自然语言日期而非 ISO 日期；没有证据时仍返回 `No information available`。

`--retriever lexical` 返回全历史 BM25 排名前 `top_k` 的消息。`--retriever session_first --session-count N` 先按会话内最高消息分选择 N 个会话；默认返回这些会话的全部消息，`--turns-per-session M` 可限制每个会话的消息数。相邻消息扩展通过私有配置的 `run_config.neighbors=true` 开启，目前没有单独的 CLI 开关。

先复制私有配置；该固定文件名已加入 `.gitignore`：

```powershell
Copy-Item eval\benchmark_config.example.json eval\benchmark_config.local.json
```

配置以及 `--report`/`--predictions` 中的相对路径都以配置文件所在目录为基准。LongMemEval 可填写：

```json
{
  "benchmark": "longmemeval",
  "dataset_path": "../external/LongMemEval/data/longmemeval_s_cleaned.json",
  "model": "填写实际回答模型",
  "official_repository": "../external/LongMemEval"
}
```

LoCoMo 对应改为 `"benchmark":"locomo"`、`"dataset_path":"../external/locomo/data/locomo10.json"`、`"official_repository":"../external/locomo"`；入口会把报告中的 dataset_variant 固定为 `locomo10`。保留示例中的其他字段；密钥优先从 `api_key_env` 指定的环境变量读取，不要写入配置。`max_context_characters` 是实际执行的检索上下文字符上限，不是近似 token 数。

当前本机的私有配置已设为 LoCoMo + 硅基流动，关键字段如下。硅基流动使用 `chat_completions`；通用示例仍默认 `responses`，两种协议互不混用。

```json
{
  "api_key_env": "SILICONFLOW_API_KEY",
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "base_url": "https://api.siliconflow.cn/v1",
  "api_mode": "chat_completions",
  "max_output_tokens": 2048
}
```

适配器会向 `/chat/completions` 发送 `response_format: {"type":"json_object"}`，并把硅基流动的 `prompt_tokens`/`completion_tokens` 统一记录为输入/输出 token。非正常结束或截断的回答会明确失败，不会把残缺 JSON 计入结果。

先只进行一次真实模型调用：

```powershell
$secureKey = Read-Host "SiliconFlow API key" -AsSecureString
$env:SILICONFLOW_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
& "D:\User\envs\memory\python.exe" -X utf8 -m eval.run_benchmark `
  --config eval\benchmark_config.local.json `
  --limit-histories 1 --limit-questions 1
```

烟测只生成 `data/benchmark_runs/<benchmark>-smoke-report.json`，故意不生成部分预测或官方评分命令；烟测时显式传入 `--predictions` 会报错。

##### LoCoMo 199 题开发基线（2026-10-26）

以下两轮使用同一份 LoCoMo 数据、第一份 history 的同一批 199 题、`deepseek-ai/DeepSeek-V4-Flash`、`raw-lexical-json-v2`、`top_k=5` 和 `max_context_characters=60000`。数据集 SHA-256 为 `553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365`；两轮均为新调用，`reused_answers=0`。

```powershell
# 全历史消息 BM25 Top-5 基线
& "D:\User\envs\memory\python.exe" -X utf8 -m eval.run_benchmark `
  --config eval\benchmark_config.local.json `
  --limit-histories 1 --limit-questions 199 `
  --retriever lexical `
  --report ..\data\benchmark_runs\lexical-199-report.json

# 先选 6 个会话，再返回这些会话的全部消息
& "D:\User\envs\memory\python.exe" -X utf8 -m eval.run_benchmark `
  --config eval\benchmark_config.local.json `
  --limit-histories 1 --limit-questions 199 `
  --retriever session_first --session-count 6 `
  --report ..\data\benchmark_runs\session-first-6s-199-report.json
```

`data/` 是本地运行目录，不随 Git 提交；可复现的结果摘要固定如下：

| 指标 | lexical | session_first（6 sessions） |
|---|---:|---:|
| 题数 | 199 | 199 |
| source evidence recall | 0.4467005 | 0.8578680 |
| session recall | 0.8104061 | 0.8577834 |
| 平均返回来源数/题 | 5 | 140.98 |
| 输入 tokens | 181,956 | 3,838,217 |
| 输出 tokens | 41,644 | 79,004 |
| 正式 answer score | 未执行 | 未执行 |

`session_first` 的 source evidence recall 绝对提高 `0.4112`、相对提高约 `92.05%`，输入 tokens 同时增加 `21.09` 倍。这个结果没有答案或 evidence 标签泄漏，并继续过滤问题 `as_of` 之后的消息。

两组不是固定消息数或固定 token 预算下的算法对比：lexical 是实际 Recall@5；`session_first` 默认选出 6 个会话后返回其中全部消息，本轮每题统计 111—172 条来源，平均 140.98 条。因此 `0.8578680` 的正确表述是“6-session、最多 60,000 字符上下文下的 evidence recall”，不能称为 Recall@5。当前将其保留为高召回开发基线；后续测试 `turns_per_session=1/3/5/10`、相邻 turn 扩展及其他 history 的留出表现，再决定最终默认策略。这 199 题不是正式全量成绩，且未运行官方答案评分。

确认开发烟测后，使用显式 `--full` 运行完整数据：

```powershell
& "D:\User\envs\memory\python.exe" -X utf8 -m eval.run_benchmark `
  --config eval\benchmark_config.local.json --full
```

完整运行生成 report、LongMemEval JSONL/LoCoMo JSON 预测，以及在配置了官方仓库时生成 `*-official-command.json`；命令只保存不执行，API key 不写入任何产物。若要比较多组配置，请为 report、predictions、official command 设置不同文件名，默认同基准的下一次运行会原子替换上一次产物。

入口按问题写入 `<report>.checkpoint.jsonl`。相同数据、检索配置和报告路径的重跑会复用已完成答案；数据、prompt、retriever 或预算改变时 fingerprint 不匹配，不会混用旧 checkpoint。完整 LoCoMo 会进行 1,986 次回答调用，LongMemEval cleaned S 会进行 500 次；仍应先估算费用。正式评分需显式执行 command artifact 中的 argv，例如：

```powershell
$command = Get-Content data\benchmark_runs\locomo-official-command.json -Raw | ConvertFrom-Json
Push-Location $command.cwd
try { & $command.argv[0] $command.argv[1..($command.argv.Count - 1)] } finally { Pop-Location }
```

LongMemEval 官方裁判会再次使用 API key 并产生调用费用。LoCoMo 回答 F1 不需要 API key，但官方脚本通常还需安装 `regex`、`numpy`、`nltk` 和 `bert-score`（后者通常连带 PyTorch）；评分环境与回答环境可以分开准备。

第 4 关形成第一个可检查的“手工记忆写入 → 检索 → 组装上下文”闭环。
第 6 关再检查真实模型的提取和回答能力。前面的固定候选只证明程序数据流正确。

评测贯穿全部关卡：每关先跑合成开发案例；第 3 关保存检索轨迹并建立基线，同时用两个基准的少量开发样本验证说话者、时间和来源 ID 的适配。第 6 关接通真实模型的端到端评分；第 11 关冻结配置，运行正式验收。

## 第一关：原始消息记录

本关的目标是回答三个问题：谁在什么会话里说了什么、按什么顺序进入系统、重启后能否完整读出。

### 准备文件

核心代码统一放在 src/，测试统一放在 tests/。测试通过 from src.message_store import ... 导入实现；运行命令时位于项目根目录。第一关自行创建：

- `src/message_store.py`：消息记录与读取实现。
- `tests/test_message_store.py`：本关验收测试。
- `tests/test_message_store_regressions.py`：幂等、参数边界和原始消息保留的回归测试。
- `demo_messages.py`：写入少量合成消息、关闭数据库、重新打开并打印结果。

数据库路径作为参数传入。演示数据库可以放在 `data/messages.sqlite`；测试使用临时文件，每个测试隔离。

### 记录契约

| 字段 | 含义 | 约束 |
|---|---|---|
| message_id | 上游为该消息生成的稳定 ID | 上游保证新消息 ID 唯一；重试沿用同一 ID；查询仍限定用户和会话 |
| user_id | 数据所属用户 | 非空字符串，来自调用方身份上下文 |
| session_id | 所属会话 | 非空字符串；查询同时限定用户与会话 |
| seq | 该用户会话内的写入顺序 | 存储层分配，从 1 开始递增；重试不分配新序号 |
| occurred_at | 原消息发生时间 | 调用者传入带时区时间；统一存储为 UTC，保留原时间语义 |
| recorded_at | 首次写入系统的时间 | 存储层生成的 UTC 时间；重试不修改 |
| payload | 原始消息结构 | JSON 对象，role 为 user、assistant 或 tool；允许 content、tool_calls、tool_call_id 等内容 |

来源引用由 user_id、session_id、message_id 一起定位；不能仅用会话内的 message_id 查询其他用户的数据。

payload 直接保留交互内容。工具参数即使是坏 JSON 字符串，也应保留该字符串；工具错误也属于原始记录。这里验证消息外层和可序列化性，不判断工具是否成功、参数是否正确或内容是否值得记住。系统指令作为配置管理，不在本关的交互消息中重复追加。

两个时间字段用途不同：occurred_at 表示消息发生时间；recorded_at 服务于导入、重试与审计。消息里说“上个月搬家”时，搬家事实的有效时间由后续提取器另行处理，不能直接用消息时间替代。seq 表达写入顺序，不能靠 occurred_at 排序代替它。

### 本关三个接口

1. append_message：接收 user_id、session_id、message_id、occurred_at、payload；返回含 seq、recorded_at 的完整已保存记录。
2. list_messages：接收 user_id、session_id、after_seq（默认 0）；按 seq 升序返回 seq 大于 after_seq 的记录列表。
3. get_message：接收 user_id、session_id、message_id；存在则返回记录，不存在则返回 None。

本关可以用一个小类管理数据库连接，或用普通函数；不需要先设计通用 Repository 基类或数据库插件体系。

### 写入规则

1. 正常消息写入并提交后才返回成功。
2. 同一作用域、同一 message_id 再次写入时，直接返回第一次保存的记录，不新增、不覆盖，也不比较内容或时间。
3. 上游保证一个 ID 绑定一条不可变消息；新消息使用新 ID，重试复用原 ID。
4. 新消息的发生时间规范化为 UTC；重试返回原记录，保留首次写入的 recorded_at 和 seq。
5. 不修改调用者传入的 payload；读取后修改返回对象，也不能改变已持久化内容。
6. 分配 seq 和插入消息放在同一事务中；数据库约束同时保护消息 ID 和序号的唯一性。
7. 第一关按单写入者实现；多会话并发写入、事务冲突重试在第 9 关扩展验收。

### 输入校验

- append/get/list 对其接收的用户、会话和消息 ID 检查非空字符串；纯空白不合法，不悄悄修改 ID。
- append 先检查 ID，再查已有记录；命中时直接返回。仅对尚未保存的新消息检查 payload 和 occurred_at。
- 新 payload 必须是字典，role 为 user、assistant 或 tool；保持 JSON 序列化检查和 allow_nan=False，不使用 default=str 隐式改写内容。
- occurred_at 必须是包含时区的 ISO 格式字符串；允许 +08:00 等偏移，然后统一到 UTC。工具参数中的坏 JSON 字符串仍应原样保留。
- after_seq 必须是大于等于 0 的整数，bool 不作为整数接受。
- 参数不合法应在插入前报错，不能留下记录或占用序号。ID、role、after_seq、时间格式违反约定时使用 ValueError；JSON 序列化失败可以沿用 TypeError/ValueError。

### 推荐实现顺序

1. 学会打开和关闭一个 SQLite 文件，建立消息表和两个唯一约束。
2. 先实现一次正常写入和读取，验证中文 JSON 往返。
3. 实现按用户、会话过滤和按 seq 排序。
4. 实现按 ID 去重；同一作用域的 ID 已存在时直接返回首次记录。
5. 实现 after_seq 查询。
6. 关闭数据库再打开，确认数据仍然存在。
7. 补齐错误输入和数据隔离测试。

SQL 使用参数绑定；消息写入时不要拼接 SQL 字符串。先明确事务边界，再写序号分配。

### 必须通过的验收案例

| 案例 | 预期 |
|---|---|
| 同一会话连续写入三条消息 | seq 为 1、2、3，读取顺序相同 |
| 两位用户使用相同 session_id | 彼此读取结果仍完全隔离 |
| 同一用户有两个会话 | 分别读取；各自独立编号 |
| 相同消息重试两次 | 只保存一条，返回相同 seq 和 recorded_at |
| 同 ID 但正文不同 | 按 ID 返回首次记录，不比较或覆盖新内容 |
| 中文、工具调用、坏参数字符串和工具错误 | JSON 往返后内容完整保留 |
| 仅查询 after_seq=2 | 只返回 seq 大于 2 的消息 |
| 关闭连接后重新打开文件 | 内容、ID、时间和序号仍一致 |
| 不存在的消息或会话 | get 返回 None；list 返回空列表 |
| 非法外层消息或不带时区的时间 | 明确拒绝，不留下半条记录 |
| 修改输入或已读返回对象 | 数据库已保存内容不受影响 |
| 消息发生时间晚到或乱序 | 按写入顺序返回；occurred_at 原语义仍保留 |

完成测试文件后，在本目录运行 `python -X utf8 -m unittest discover -s tests -v`。
在项目根目录运行这条命令，会一起执行基础测试与回归测试。

### 完成后应能解释

1. 为什么原始消息和提取后的事实需要分开保存？
2. 为什么不能只保存 role 和 content，而丢掉工具调用字段？
3. 为什么 seq、occurred_at 和 recorded_at 是三个不同概念？
4. 为什么重复请求不能直接插入，也不能随意覆盖？

## 第二关：上下文队列

第一小步实现 ContextQueue，绑定一个用户会话，从 MessageStore 增量读取消息。
历史库保存完整记录；队列保存当前会话的内存工作副本。新增消息仍先写入 MessageStore，再显式刷新队列。

核心文件为 `src/context_queue.py`，验收测试放在 `tests/test_context_queue.py`。

### 第一小步接口

- 构造：ContextQueue(store, user_id=..., session_id=...)。构造后队列为空，不自动读取数据库。
- 内部保存完整记录列表 `_records`，初始 `last_seq=0`。完整记录用于后续来源追溯和裁剪。
- refresh()：调用 list_messages，传入绑定的 user_id、session_id 和 after_seq=last_seq；追加新增记录，返回新增数量。没有新增记录时返回 0，不改变状态。
- 刷新成功后，last_seq 取最后一条新增记录的 seq；不能使用队列长度或新增数量代替水位。
- to_messages()：按队列顺序导出各记录的 payload，并返回深拷贝。导出结果不包含 user_id、session_id、seq 等存储元数据。
- 队列不执行数据库删除；以后从上下文移除消息时，也不删除完整历史。不在导出方法中隐式刷新。

### 第一小步验收

1. 空库刷新返回 0，导出空列表，last_seq 保持 0。
2. 当前会话两条消息按序加载；其他用户、其他会话的消息不混入；再次刷新返回 0，结果不重复。
3. 历史库追加第三条消息后，刷新前队列保持原样；刷新返回 1，随后导出三条消息，last_seq 更新。
4. 修改导出列表或其中的嵌套 payload，不影响下次导出和数据库内容。

本小步先完成增量同步；下一小步增加 token 预算与明确的预算不足处理。摘要压缩在第 7a 关接入。

## 第三关：原始历史检索与评测

`src/history_retriever.py` 提供当前用户跨会话的关键词检索。查询与字符串正文使用 casefold 归一化，查询按空白拆词并去重；按命中的不同关键词数降序排序，同分保留原读取顺序。

`src/eval/retrieval_evaluator.py` 评测固定历史快照中的来源召回：

```python
from src.eval.retrieval_evaluator import evaluate_retrieval

report = evaluate_retrieval(retriever, cases, top_k=5)
```

每个 case 包含 case_id、user_id、query、relevant_sources；每个正确来源包含 user_id、session_id、message_id。正确来源应由数据加载阶段确认存在于历史中，标签留在评测层。无相关来源的案例使用空列表。

- top_k 必须是正整数，拒绝 bool。来源按完整三元组比较，先截取前 k 条，再去重评分。
- 有答案案例分别计算 Recall@k、Hit@k，再按案例求平均；无答案案例单独计算返回空的比例。某类案例缺失时，对应汇总为 None，写入 JSON 后为 null。
- report.details 保留逐案例的来源标签、实际返回来源、检索分数和指标。检索异常直接抛出，跨用户标签或返回结果报错。
- 本地合成数据用于保存开发基线；正式基准的适配与评分按下方约定逐步接入。

评测器的测试位于 `tests/test_retrieval_evaluator.py`。在本目录运行 `python -X utf8 -m unittest tests.test_retrieval_evaluator -v`。

固定开发数据和运行入口位于项目根目录的 `eval/`，核心指标计算继续使用上面的评测器：

```python
from eval.run_retrieval_eval import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORT_PATH,
    run_evaluation,
    save_report,
)

report = run_evaluation(DEFAULT_DATASET_PATH, top_k=2)
save_report(report, DEFAULT_REPORT_PATH)
```

`eval/data/retrieval_dev.json` 保存版本号、原始历史和人工来源标签。运行函数每次创建新的内存数据库，仅导入 history，校验案例 ID 唯一、来源存在且属于当前用户；重复的历史来源报错。返回值包含原有指标和明细，以及 dataset_id、retriever 版本信息。

在本目录运行 `python -X utf8 -m eval.run_retrieval_eval` 会执行内置开发集，打印数据集、检索器版本、案例数量和三个汇总指标，并生成 `eval/reports/retrieval_baseline.json`。重复运行会覆盖同一路径的报告。

run_evaluation 仅返回结果字典；save_report 创建所需目录，以 UTF-8、中文原文和缩进保存完整 JSON。未评测的类别指标保存为 null，控制台显示为 N/A。报告保留逐案例来源、分数和指标；序列化失败时不覆盖旧报告。

数据加载与运行的测试位于 `tests/test_run_retrieval_eval.py`。运行 `python -X utf8 -m unittest tests.test_run_retrieval_eval -v`。

报告保存与入口的测试位于 `tests/test_retrieval_report_output.py`。运行 `python -X utf8 -m unittest tests.test_retrieval_report_output -v`。

## 第四关：结构化事实存储与读取

`src/fact_store.py` 中的 FactStore 接收已有 MessageStore，复用其连接；连接由 MessageStore 统一关闭。facts 保存事实版本，fact_sources 保存完整消息来源。新增默认使用 version=1、status=active，也支持显式传入版本和状态。

- get_fact(user_id=..., fact_id=..., version=1) 按用户、事实 ID 和指定版本查询。不存在返回 None；存在返回完整字典，将 object_json 解码为 object，并附带按来源会话和消息 ID 排序的 sources。
- list_facts(user_id=..., subject_id=None, predicate=None, session_id=None, as_of=None) 列出当前 active 的事实。subject_id、predicate 提供时精确匹配；不传会话只返回用户级事实，传会话则返回用户级与该会话的事实。as_of 默认为当前 UTC 时间，提供时必须带时区；有效期采用左闭右开区间，未知边界不作排除条件。返回值按 fact_id、version 升序排列，每条包含与 get_fact 相同的完整字段。
- add_fact 接收 user_id、fact_id、subject_id、predicate、object、sources，以及可选版本、状态、作用域和有效期。相同用户、事实 ID 和目标版本已存在时，直接返回首次记录，忽略其他新入参；相同内容使用不同 fact_id，仍创建不同事实。版本迁移使用 update_fact，让新版本写入和旧版本失效一起提交。
- object 使用 `{"type": "literal", "value": "咖啡"}` 结构，value 必须能严格 JSON 序列化。sources 为非空列表，每项包含 user_id、session_id、message_id；来源必须属于当前用户且已存在，重复来源按完整三元组去重。
- scope 为 user 时 session_id 必须为 None；scope 为 session 时必须提供非空 session_id。事实适用的会话与来源消息的会话分别保存。
- valid_from、valid_to 可为 None；提供时必须包含时区，统一存为 UTC，两端存在时要求终点晚于起点。recorded_at 为事实首次入库的 UTC 时间。
- 事实和所有来源在同一事务中提交；任何一条插入失败均回滚。共享连接在开启事务前启用 SQLite 外键约束。

测试位于 `tests/test_fact_store.py`。在本目录运行 `python -X utf8 -m unittest tests.test_fact_store -v`。

列表查询的测试单独放在 `tests/test_fact_store_listing.py`，运行 `python -X utf8 -m unittest tests.test_fact_store_listing -v`。

## 第五关：事实版本更新

- update_fact 接收 user_id、fact_id、new_version、object、sources。new_version 必须是至少为 2 的严格整数；同一次更新重试沿用相同目标版本，目标已存在就返回原记录。
- 新写入要求事实存在、最新版本为 active，且目标版本等于最新版本加 1。不满足条件时抛出 ValueError；成功和幂等重试都返回完整事实字典。
- 新版本继承主体、谓词、作用域、会话和有效期，替换对象和来源；旧版本及其来源保留，状态改为 superseded。
- add_fact 和 update_fact 各自管理完整事务，共用不提交事务的内部 _insert_fact 方法。更新旧版、插入新版和插入来源任一步失败，都整体回滚；调用方在方法外捕获异常。
- retract_fact 接收 user_id、fact_id、expected_version。预期版本必须是至少为 1 的严格整数，并与最新版本匹配；事实不存在、版本不匹配或状态不允许时抛出 ValueError。最新版本已撤回时直接返回，支持幂等重试。
- 撤回在一个事务内将最新 active 版本改为 retracted，不新增版本，事实内容、有效期、记录时间和全部来源保留。get_latest_fact 仍返回撤回版本，list_facts 排除它；数据库错误向外传播并整体回滚。

更新回归测试位于 `tests/test_fact_store_updates.py`，运行 `python -X utf8 -m unittest tests.test_fact_store_updates -v`。

撤回回归测试位于 `tests/test_fact_store_retractions.py`，运行 `python -X utf8 -m unittest tests.test_fact_store_retractions -v`。

`src/memory_updater.py` 提供统一入口 `MemoryUpdater(fact_store).apply(user_id=..., decision=...)`，每次执行一条确定的决策。事实关系判断由独立的 judge_fact_relation 负责。

| action | decision 中的业务字段 | 行为 |
|---|---|---|
| ADD | fact_id、subject_id、predicate、object、sources；可选 scope、session_id、valid_from、valid_to | 新增 version=1、status=active 的事实；相同目标版本返回原记录 |
| UPDATE | fact_id、new_version、object、sources | 调用 update_fact 保存同一事实的新版本，可用于纠错或来源合并 |
| DELETE | fact_id、expected_version | 调用 retract_fact 撤回最新版本 |
| TRANSITION | fact_id、new_version、new_fact_id、effective_at、object、sources；可选 valid_to | 关闭旧区间并新增事实，整体事务提交 |
| NOOP | 无额外字段 | 不执行存储操作 |

- decision 必须包含 action。用户归属由独立 user_id 参数提供，决策中不能覆盖 user_id，也不能自行指定 ADD 的 version 或 status；未知字段和未知动作抛出 ValueError。
- 返回结构统一为 `{"action": "UPDATE", "fact": 已保存的事实字典}`；NOOP 的 fact 为 None，仅表示无需执行存储操作，不证明某条事实存在或写入成功。
- TRANSITION 另返回 previous_fact：关闭旧区间后的版本；fact 为新事实的第 1 版，完整返回是 `{"action":"TRANSITION","previous_fact":旧值关闭版,"fact":新值第1版}`。
- 写入和版本校验复用 FactStore，异常向外传播。已有目标版本的重试保留按 ID 返回原记录的规则；新请求缺失必要字段时由存储层明确拒绝。MemoryUpdater 不额外包裹事务。

统一入口的测试位于 `tests/test_memory_updater.py`，运行 `python -X utf8 -m unittest tests.test_memory_updater -v`。

### 事实关系判断器

`src/fact_relation_judge.py` 提供纯函数，先判断、再由调用方选择执行。它不查询或写入数据库，也不调用模型或从文本识别意图。

```python
from src.fact_relation_judge import judge_fact_relation

plan = judge_fact_relation(
    user_id="u1",
    candidate={
        "fact_id": "f-tea",  # 上游分配的稳定新 ID
        "subject_id": "user:u1",
        "predicate": "prefers_drink",
        "object": {"type": "literal", "value": "茶"},
        "sources": [{"user_id": "u1", "session_id": "s1", "message_id": "m2"}],
    },
    existing_facts=existing_facts,
    cardinality="multi",
)
if plan["decision"] is not None:
    result = updater.apply(user_id="u1", decision=plan["decision"])
```

`existing_facts` 必须覆盖相关主体、谓词和精确作用域的所有有效区间，以及每个事实 ID 的最新版本（包括 superseded/retracted）；也可以同时包含旧版本。独立调用判断器时可传入 FactStore 读取的完整记录；`MemoryPipeline.load_fact_snapshot` 已提供自动完整快照。判断器按每个 fact_id 的最高版本判断，不会在最新版本失效后改用旧 active 版本。**不能把单次 `list_facts(as_of=...)` 的结果当作完整快照**。列表内的所有事实都须属于独立参数指定的 user_id，否则报错。

候选公共字段为 subject_id、predicate、sources，以及可选的 scope（默认 user）、session_id、valid_from、valid_to。来源必须为非空的完整同用户引用，按三元组去重；消息是否真实存在由执行时的 FactStore 校验。事实作用域与来源会话分别处理，不从来源推断作用域。

| intent | 额外候选字段 | 含义 |
|---|---|---|
| assert（默认） | fact_id、object | 提出一条新事实；fact_id 必须由上游稳定分配 |
| correct | object、target | 纠正指定的同一事实；target 为 `{"fact_id":"f1","version":1}` |
| retract | target | 撤回指定版本，不要求提供 object |
| change | object；可选 fact_id、target，valid_from 为生效时间 | 完整提供新 ID、目标和生效时间时生成 TRANSITION，否则待处理 |

correct/retract 缺少 target 时返回待处理；target 结构非法时报错。纠错及撤回只能作用于同主体、谓词、精确 scope/session_id 的最新 active 版本，不能按列表第一项或相似值自动选择目标。候选不能传入 user_id、version、status、action 等未知字段；cardinality 是独立的谓词规则参数，只接受 single、multi、unknown（默认）。

返回字段为 relation、target、reason、decision。target 为已有事实的 fact_id/version，不能唯一指定时为 None；decision 仅含 MemoryUpdater 允许的动作字段，待处理时为 None。

| relation | 处理 |
|---|---|
| SUPPLEMENT | 无相关 active 事实，生成 ADD |
| DUPLICATE | 精确作用域、有效区间与结构化值相同；默认 NOOP，显式开启来源合并时可生成 UPDATE |
| MULTIVALUE | 同区间不同值，明确 cardinality=multi 时生成 ADD，保留旧值 |
| CORRECTION | 明确纠错且目标、区间匹配，生成 UPDATE，固定 new_version=目标版本+1 |
| RETRACTION | 明确撤回，生成 DELETE，固定 expected_version=目标版本 |
| TEMPORAL_CHANGE | 完整明确的 change 生成 TRANSITION；普通区间差异或信息不足时 decision=None |
| UNRESOLVED | 缺少依据、目标失效或不匹配、不同值但未明确允许多值，decision=None |

日期统一 UTC 后比较；结构化值按严格 JSON 的存储形态比较，忽略字典键顺序，区分布尔值与数字，不做同义词或大小写推断。默认的重复 NOOP 不合并新增来源，显式开启方式见下节。已有同值和异值并存时，只有 multi 允许直接判定重复，否则保留争议。assert/correct 发现区间不同时仍保守返回 TEMPORAL_CHANGE、decision=None，不从区间差异推断 change 意图。

这是新候选的规划入口。assert 的 fact_id 若已在快照出现，返回 UNRESOLVED，防止把新提案误交给已有身份。**执行重试直接复用原 decision，不重新判断、不重新生成 ID，也不重新取最新版加一**；MemoryUpdater/FactStore 原有的按 ID/目标版本返回首次记录、不比较重试正文的规则保持不变。判断结果不代表已经保存，存储失败仍向外抛出。当前按单写入者使用，调用方应在处理下一条候选前执行当前计划；本步不提供并发快照或批量计划事务。

测试位于 `tests/test_fact_relation_judge.py`。在本目录运行 `python -X utf8 -m unittest tests.test_fact_relation_judge -v`。

### 重复事实来源合并

`judge_fact_relation(..., merge_duplicate_sources=True)` 开启来源合并；参数只接受严格布尔值，默认 False 保持已验收的重复 NOOP 行为。开启后，唯一重复目标的快照必须含该版本完整的 sources，可直接使用 FactStore 读出的记录；只对选中的重复目标校验来源，其他或历史记录不增加来源要求。

当候选有新来源时，relation 仍为 DUPLICATE，decision 使用现有 UPDATE，固定 fact_id 和 new_version=目标版本+1，对象取目标的规范 JSON 值，sources 为目标全部来源与候选来源按完整三元组去重的并集。原版本的来源保留，新版本包含合并后的来源；主体、谓词、作用域和有效期沿用目标。比如 v1 有 m1，再次出现同一事实且来源是 m2，执行后 v1 保留 m1，v2 包含 m1、m2。来源都已记录时仍返回 NOOP，不创建新版本。

```python
plan = judge_fact_relation(
    user_id="u1", candidate=candidate, existing_facts=existing_facts,
    merge_duplicate_sources=True,
)
if plan["decision"] is not None:
    result = updater.apply(user_id="u1", decision=plan["decision"])
```

多个同值事实 ID 无法唯一选择合并目标时，开启合并会返回 UNRESOLVED、decision=None。显式 correct 指定的同值目标也支持来源合并；混合不同值、时间区间差异和失效目标的原有保守规则保持。真实来源消息的存在性仍由 FactStore 校验，任何写入失败整体回滚。

当前须单写入者逐条规划、执行，再读取新快照处理下一候选。**不能从同一旧快照分别规划两次来源合并**：两条 UPDATE 会使用相同目标版本，后者按版本身份返回第一次记录，其新来源不会自动追加。执行重试只复用原 decision，即便该版本后来已更新或撤回，也不重新创建版本。

新增测试为 `tests/test_duplicate_sources.py`，共 20 个方法（13 个规则测试、7 个 SQLite 集成测试）。本轮仅完成静态审查和 AST 语法检查，由用户运行测试，助手不代跑：

```powershell
& "D:\User\envs\memory\python.exe" -X utf8 -m unittest tests.test_duplicate_sources -v
```

### 有效期迁移

`FactStore.transition_fact(user_id=..., fact_id=..., new_version=..., new_fact_id=..., effective_at=..., object=..., sources=..., valid_to=None)` 将明确时间变化作为一个操作提交。new_version 是旧事实关闭区间后的目标版本，必须等于旧最新 active 版本加 1；new_fact_id 必须是该用户尚未使用的新 ID。

例如旧事实 f-beijing 第 1 版保存“2020 年起住在北京”，明确从 2026 年 10 月 1 日搬到杭州：

```python
plan = judge_fact_relation(
    user_id="u1",
    candidate={
        "intent": "change",
        "fact_id": "f-hangzhou",
        "target": {"fact_id": "f-beijing", "version": 1},
        "subject_id": "user:u1",
        "predicate": "lives_in",
        "object": {"type": "literal", "value": "杭州"},
        "valid_from": "2026-10-01T00:00:00+08:00",
        "sources": [{"user_id": "u1", "session_id": "s1", "message_id": "m-move"}],
    },
    existing_facts=existing_facts,
    cardinality="single",
)
if plan["decision"] is not None:
    result = updater.apply(user_id="u1", decision=plan["decision"])
```

成功后 f-beijing v1 仅状态改为 superseded，原对象、有效区间、时间及来源保留；v2 为 active，仍保存北京，但 valid_to 关闭在搬家时刻。f-hangzhou v1 为 active，从搬家时刻开始。关闭版合并原来源和变化来源；新事实只使用变化来源。按 as_of 查询时，边界之前返回北京，边界起返回杭州。active 表示未撤回或取代，过去/未来是否适用由有效期决定。

effective_at 必须带时区且严格位于旧区间内部：若旧起点或终点已知，不能等于或越过该边界。新 valid_to 若有必须晚于 effective_at；它由新候选单独提供，默认 None，不继承旧值的终点，也不凭旧值终点推断新值的结束时间。单值/unknown 规则下，判断器会检查新区间是否与其他同主体、谓词、精确作用域的最新 active 事实重叠，重叠返回 UNRESOLVED；区间恰好接壤允许。multi 只迁移显式目标，其他值保留。即使对象相同，明确的 change 仍按指定时间切分。

新增 fact_transitions 表，以 `(user_id, new_fact_id)` 唯一记录本次迁移的两个结果版本，并用外键引用它们。已有数据库初始化时只增加该表，不改写原事实。已完成迁移的重试先按这个身份读取固定版本，忽略新传入的旧目标、版本、正文和时间；后来纠错或撤回也不会让重试重新迁移或恢复旧值。仅有普通 ADD 占用了 new_fact_id 时不能当作迁移成功，会拒绝新迁移。

新增事实、旧版失效、关闭版及来源、迁移记录在唯一的 FactStore 事务内完成，任一步失败全部回滚；内部只调用不提交的 _insert_fact，MemoryUpdater 不套外层事务。返回 `{"previous_fact":关闭版,"fact":新事实第1版}`，数据库异常继续抛出。底层仍逐条执行固定计划；完整快照加载和模型候选提取已由 `MemoryPipeline` 接入，争议记录见下节。

本小步新增测试为 `tests/test_fact_store_transitions.py` 和 `tests/test_fact_temporal_plans.py`。在本目录运行：

```powershell
& "D:\User\envs\memory\python.exe" -X utf8 -m unittest tests.test_fact_store_transitions tests.test_fact_temporal_plans -v
```

### 争议记录

`src/conflict_store.py` 提供 `ConflictStore(fact_store)`，复用已有 FactStore 的连接，连接仍由 MessageStore 管理。record/get/list 入口保存和读取争议，不修改事实或消息；人工解决入口见下一节。`MemoryPipeline` 会把有相关事实且关系为 UNRESOLVED 的模型候选自动记为争议；人工解决结果仍由调用方明确选择，不能由模型暗中替代。

```python
from src.conflict_store import ConflictStore

conflicts = ConflictStore(fact_store)
saved = conflicts.record_conflict(
    user_id="u1",
    conflict_id="c1",  # 上游分配的稳定 ID
    candidate={
        "subject_id": "user:u1",
        "predicate": "prefers_drink",
        "object": {"type": "literal", "value": "茶"},
        "sources": [{"user_id": "u1", "session_id": "s1", "message_id": "m2"}],
    },
    related_facts=[{"fact_id": "f1", "version": 1}],
    reason="新旧值不同，缺少纠错或多值共存依据",
)
retried = conflicts.record_conflict(user_id="u1", conflict_id="c1")
one = conflicts.get_conflict(user_id="u1", conflict_id="c1")
pending = conflicts.list_conflicts(user_id="u1")
```

示例中的来源消息及事实版本须事先存在。record_conflict 的 candidate、related_facts、reason 默认 None 是为了支持已有 ID 的无正文重试；新增时三者仍必须有效。身份为 `(user_id, conflict_id)`，先校验这两个 ID 并查已有记录，命中直接返回首次记录，不比较新正文，也不校验其他新传入字段；同内容的新 ID 仍创建独立记录。

- candidate 必须为字典，不能包含顶层 user_id。其业务字段按严格 JSON 保存，使用 allow_nan=False，不猜测或重新校验意图。candidate.sources 为非空列表，沿用完整 user_id/session_id/message_id 三元组；要求同用户且消息存在，按完整三元组去重，只保留三元组字段。
- related_facts 为非空列表，每项只含 fact_id/version；版本是严格正整数，bool 不接受。必须在当前用户下精确存在，可引用 superseded/retracted 的历史版本，不替换为最新版本。重复 fact_id/version 去重。
- reason 为非空字符串，保留原文；新增时 status 固定由存储层写 open，recorded_at 为新记录的 UTC 时间。解决成功后 status 为 resolved，原候选、关联、原因和 recorded_at 保留。
- 返回字段为 user_id、conflict_id、candidate、related_facts、reason、status、recorded_at。来源按会话/消息 ID 排序，事实引用按 fact_id/version 排序；get_conflict 缺失返回 None，list_conflicts 只返回当前用户全部记录，按 recorded_at/conflict_id 升序。

数据保存在 conflicts、conflict_facts、conflict_sources 三表。candidate_json 剥离 sources，读取时从来源表重建 candidate.sources，避免双份来源不同步。两个关联表均使用含 user_id 的复合外键，连接争议及真实事实版本/消息。已有数据库初始化时只增加表，不重写既有数据。

record_conflict 在唯一事务中写入三表；校验失败、数据库错误或插入被忽略时全部回滚，错误向外抛出。读取不提交写事务。输入及已返回字典的修改不会改变已保存记录。当前仍按单写入者使用。

本步只接收有现存事实参与的争议；缺目标、缺时间等其他待处理候选不应无条件接入。对应测试为 `tests/test_conflict_store.py`，由用户运行：

```powershell
& "D:\User\envs\memory\python.exe" -X utf8 -m unittest tests.test_conflict_store -v
```

### 人工解决争议

`resolve_conflict(*, user_id, conflict_id, outcome=None, target=None, new_fact_id=None, reason=None)` 原子执行一次人工决定，并关闭对应争议；`get_resolution(*, user_id, conflict_id)` 读取处理记录，不存在返回 None。

| outcome | 需要的附加参数 | 行为 |
|---|---|---|
| KEEP_EXISTING | 无 | 保留当前事实，保存 NOOP 决定 |
| ADD_CANDIDATE | new_fact_id | 人工明确允许候选共存，使用新 ID 新增 v1 active，保留已有事实 |
| CORRECT | target：fact_id/version | 使用已存候选纠正指定相关事实，生成 target.version+1 |
| RETRACT | target：fact_id/version | 逻辑撤回指定相关事实的当前版本，保留内容、来源及历史 |
| TRANSITION | target、new_fact_id | 关闭目标旧区间，按候选时间新增事实，与争议关闭一起提交 |

新解决必须提供非空 reason。target 只适用于 CORRECT/RETRACT/TRANSITION；new_fact_id 只适用于 ADD_CANDIDATE/TRANSITION；对应结果要求的参数必须填写，其他参数不能混入。不会自动选择目标、生成新 ID 或判断关系。采用的主体、谓词、对象、作用域、有效期和来源均来自已存 candidate，不能在解决时替换候选正文。新增事实使用显式 new_fact_id，候选中原有 fact_id 不作为结果 ID；任何版本已占用该新 ID 时拒绝，也不能将别处的普通 ADD 或已完成时间迁移当作本次解决成功。

三个有 target 的结果均要求它出现在争议的 related_facts 中，并仍是当前用户下最新 active 版本；已更新或撤回的旧目标不允许继续执行。候选与目标的主体、谓词、精确 scope/session_id 须一致。CORRECT 还要求 UTC 有效区间相同；RETRACT 不使用候选对象和时间，只撤回指定版本。只改变选定事实，不自动批量修改其他关联事实。ADD_CANDIDATE 是人工明确共存决定，不从不同值自动推断谓词允许多值。

TRANSITION 使用已存候选的 valid_from 作为 effective_at，并取候选的 object、sources、valid_to；切点必须严格位于目标原区间内部，新终点须晚于切点，默认开放且不继承旧终点。旧关闭版为 target.version+1，新事实为新 ID 的第 1 版；关闭版保留旧对象并合并原来源和变化来源。决定保留候选原时间字符串，实际事实统一存 UTC。手工迁移仅处理选定目标，不推断全局单值或修改其他事实；高层 judge 的自动规则规划仍依赖完整快照与 cardinality 约束。

```python
resolved = conflicts.resolve_conflict(
    user_id="u1", conflict_id="c1", outcome="CORRECT",
    target={"fact_id": "f1", "version": 1},
    reason="已确认候选用于纠正此前记录",
)
retried = conflicts.resolve_conflict(user_id="u1", conflict_id="c1")
audit = conflicts.get_resolution(user_id="u1", conflict_id="c1")
```

返回字段为 user_id、conflict_id、outcome、decision、result、reason、resolved_at。decision 使用既有 ADD/UPDATE/DELETE/TRANSITION/NOOP 格式，result 为 `{"action":动作,"fact":事实或None}`；TRANSITION 另含 previous_fact，表示旧值的关闭版。它们是首次处理时保存的 JSON 快照，**不能据此判断事实当前仍然 active**。后来事实纠错或撤回不会改变该快照，现状仍通过 FactStore 查询。

新增 conflict_resolutions 表，以 `(user_id, conflict_id)` 为唯一处理身份，并通过复合外键关联原争议。重试先按该身份查记录，命中直接返回，忽略新的 outcome/target/new_fact_id/reason，不比较正文、不重新执行，也不重新计算版本。每个争议本步只解决一次，不提供重开。

旧版处理表的 CHECK 只允许前三种结果。初始化会在同一建表事务内升级该已知旧约束：创建 conflict_resolutions_v2，逐列复制全部旧记录，替换旧表。原 decision_json、result_json、原因和时间原样保留；失败整体回滚。已支持五种结果的表跳过升级；临时表名已占用时拒绝迁移，不删除未知对象。

事实及来源写入、迁移记录、处理快照、open→resolved 在唯一事务内提交，失败全部回滚。FactStore 的公开 update_fact/retract_fact/transition_fact 仍各自管理普通写入事务，复用不提交的内部方法；组合解决仅调用 _insert_fact/_update_fact/_retract_fact/_transition_fact，避免调用公开写方法造成提前提交。事实和来源插入均检查实际写入数量，防止插入被忽略后误报完成。get_conflict/record_conflict 的返回字段不变，解决后会读到 resolved 状态；list_conflicts 继续返回当前用户全部记录。

三种基础人工决定的 `tests/test_conflict_resolutions.py`、撤回/迁移与旧约束升级的 `tests/test_conflict_resolution_actions.py` 都已纳入完整功能回归。五种人工结果仍由调用方明确指定；模型候选、规则规划和争议记录已接入 `MemoryPipeline`，但模型不能自行冒充人工解决。按需运行：

```powershell
& "D:\User\envs\memory\python.exe" -X utf8 -m unittest tests.test_conflict_resolution_actions -v
```

## 正式基准的固定约定

- LongMemEval 使用对话版官方 cleaned S/M 数据；Oracle 为单独的证据诊断实验。[官方仓库](https://github.com/xiaowu0162/LongMemEval)
- LoCoMo 第一阶段使用 locomo10.json 的 QA 任务，保留说话者、时间和来源 ID。[官方仓库](https://github.com/snap-research/locomo)
- 历史通过真实记忆链路回放；标准答案、证据标签和人工摘要不进入建库过程。
- 同一历史的多道问题使用固定的只读记忆快照，问答和裁判反馈不写回。
- 等待后台任务与索引提交后开始离线主测；即时可见性另测。
- 固定回答模型、提示词、工具权限、上下文预算、数据和评分版本；开发子集与正式保留集分开。
- 分别报告官方总分、分类分、证据检索表现、建库成本和查询成本。图检索与无图检索做配对对比。
- 用户隔离、删除不复活、幂等和并发一致性使用独立工程用例验收。
