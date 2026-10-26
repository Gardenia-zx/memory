# 记忆系统开发交接（当前，2026-10-26）

> 本节是当前唯一有效状态。后面的“历史阶段快照”仅用于追溯第 5 关当时的契约，其中的“尚未实现/待运行”不再代表当前进度。

## 当前结论

第 1—11 关的工程开发已经完成。最后一次完整功能回归：

```powershell
cd D:\cmu_agent\memory
python -X utf8 -m unittest discover -s tests -v
```

结果为 368 tests、全部通过。LongMemEval cleaned S 与 LoCoMo locomo10 已完成整份适配加载验证；硅基流动真实 API 已完成 LoCoMo 1、20 和第一份 history 的 199 题开发烟测。尚未执行完整基准官方评分、最终准确率或性能压测；示例及私有配置中的 API key 均为空，开发烟测不能写成正式成绩。

## 当前模块地图

- 原始与上下文：`message_store.py`、`context_queue.py`、`context_builder.py`、`history_retriever.py`。
- 结构化记忆：`fact_store.py`、`fact_relation_judge.py`、`memory_updater.py`、`conflict_store.py`。
- 模型流水线：`model_adapter.py`、`candidate_extractor.py`、`memory_pipeline.py`。
- 长期结构：`summary_store.py`、`summary_generator.py`、`context_compactor.py`、`task_store.py`、`event_store.py`。
- 外部资料与检索：`external_document_store.py`、`hybrid_retriever.py`、`session_retriever.py`。
- 可靠性与图：`job_store.py`、`forgetter.py`、`entity_graph.py`。
- 正式评测工具：`src/eval/benchmark_datasets.py`、`src/eval/benchmark_runner.py`、`eval/run_benchmark.py`、`eval/score_locomo_official.py`、`eval/diagnostics/`、`eval/benchmark_config.example.json`。

模型流水线的固定顺序为：原消息提交 → 候选提交 → 关系计划提交 → 事实动作提交 → 进度收据提交 → 返回完成。模型不能覆盖 user_id、来源、session 或事实 ID；事实已写而进度保存失败时，重试复用原 action/version。

后台任务具有 generation、租约、最大尝试次数和过期恢复。用户遗忘动态发现所有带 user_id 的表，按外键子表优先在一个事务删除并写墓碑；旧消息或旧任务不能复活数据。

基准 runner 每份历史只建一次，问题使用新的 SQLite 只读连接；答案、证据、人工摘要和评分反馈不进入建库。报告支持总体/分类回答分、来源/会话证据覆盖、build/query usage、快照哈希和逐题 checkpoint。官方 scorer 只生成命令或由显式桥接调用，不会隐式产生付费请求。

## 当前不可破坏的契约

1. user_id 来自可信调用参数，不能从模型或正文覆盖。
2. 消息来源使用完整 `(user_id, session_id, message_id)`；文档来源含版本和 chunk ID。
3. 首次写入身份不可变；重试返回第一次提交内容。
4. 固定事实计划不能在重试时重新计算版本。
5. 所有外部时间必须带时区，内部 UTC；有效期左闭右开。
6. 摘要、事件、事实和图关系必须保留真实来源。
7. 新任务 generation 使旧任务失效；过期租约不能在 recovery 前抢先提交。
8. 遗忘任何一步失败或被 IGNORE 都整体回滚。
9. 基准标签永不进入问答可见快照。

## 当前测试分布

- 第 6 关：`test_memory_pipeline.py`、`test_model_adapter.py`。
- 第 7 关：`test_stage7_memory.py`、`test_summary_generator.py`。
- 第 8 关：`test_external_documents_hybrid.py`。
- 第 9 关：`test_message_store_concurrency.py`、`test_job_store.py`、`test_forgetter.py`。
- 第 10 关：`test_entity_graph.py`。
- 第 11 关：`test_benchmark_adapters.py`、`test_real_benchmark_cli.py`、`test_benchmark_recovery.py`、`test_retrieval_diagnostics.py`、`test_session_retriever.py`。
- 跨阶段闭环：`test_full_memory_lifecycle.py`。

在限制系统临时目录的环境中先设置：

```powershell
$env:TEMP = 'D:\cmu_agent\memory\.test-tmp'
$env:TMP = 'D:\cmu_agent\memory\.test-tmp'
python -X utf8 -m unittest discover -s tests -v
```

`.test-tmp/` 已加入 `.gitignore`。

## 用户接手的外部验收

1. 已创建被 Git 忽略的 `eval/benchmark_config.local.json`：LoCoMo、硅基流动 `chat_completions`、`deepseek-ai/DeepSeek-V4-Flash`；API key 使用 `SILICONFLOW_API_KEY` 环境变量，未写入文件。
2. 已完成同一 199 题的 lexical Top-5 与 session-first 6-session 开发对照；命令、数据 SHA、结果和预算口径固定在 README。session-first 是高召回/高上下文基线，不是同预算 Recall@5。
3. 下一步换其他 history 做留出验证，再测试 `turns_per_session`/邻居扩展的召回—上下文曲线。
4. 全量必须显式传 `--full`；导出预测后显式运行官方 scorer。
5. 正式验收记录总体/分类准确性、证据覆盖、成本、吞吐和延迟，并固定数据 SHA-256、官方仓库 revision、prompt 和预算。

有意保留的边界：没有内置真实密钥或默认付费模型；没有让模型冒充人工争议裁决；没有专用图数据库；没有伪造正式基准成绩。

---

# 历史阶段快照（已归档，不再代表当前进度）

交接快照：2026-10-21。用于从旧会话迁移到新会话；代码现状以工作区文件为准。先读本文，再按需读 README 和相关实现，不要从头重做项目。

## 1. 目标与当前接续点

用户在学习 Agent 的 memory 章节后，自行设计并逐步实现记忆系统。目标是保存完整交互历史，管理近期上下文，提取和维护有来源、作用域、有效期及版本的事实，检索后组装上下文，并最终通过 LongMemEval、LoCoMo 和工程回归验收。

当前处于第 5 关：事实关系判断与记忆更新。

- 第 1～4 关的已实现内容，用户已反馈相关测试通过。
- FactStore 的版本更新、逻辑撤回已实现，用户已反馈测试通过。
- MemoryUpdater 的四种动作统一入口及 9 个测试，用户已在新会话明确反馈通过。
- 只读规则函数 judge_fact_relation 的 20 个测试，用户已明确反馈通过，不重跑。
- 有效期迁移：FactStore.transition_fact、MemoryUpdater 的 TRANSITION 动作，以及完整 change 候选的执行计划，用户已审查并反馈 32 个测试通过，不重跑；候选、意图、谓词多值规则和完整事实快照仍由调用方提供。
- ConflictStore 的争议记录与读取接口，用户已审查并反馈 18 个测试全部通过，不重跑。
- 人工争议解决的保留旧事实、候选共存、指定版本纠错三种结果，用户已审查并反馈 19 个测试全部通过，不重跑。
- RETRACT/TRANSITION 的争议解决及旧处理表约束升级，用户已审查并反馈 22 个测试通过，不重跑。
- 最新补充重复事实的来源合并：判断器显式开启后生成现有 UPDATE，新增 20 个测试方法待用户审查与运行。助手仅做静态审查与 AST 语法检查，未运行测试。当前没有自动候选提取、快照加载、自然语言意图识别或真实模型接入。
- 迁移时不重跑测试、不重写现有模块。测试结果由用户提供，不把静态检查或待确认结果写成已经通过。

## 2. 工作区与协作要求

- 项目所在目录：`D:\cmu_agent\memory`；桌面项目根目录和新会话默认 cwd：`D:\cmu_agent`。
- 已有 Conda 解释器：`D:\User\envs\memory\python.exe`，Python 3.13；用户使用 PyCharm 和 PowerShell。
- `D:\cmu_agent` 当前不是 Git 仓库。直接使用现有本地目录，无需创建工作树或迁移到其他项目。
- 核心代码放 `memory/src/`，测试放 `memory/tests/`；开发评测入口、数据和报告放 `memory/eval/`。
- 用中文交流，按小步开发，解释接口契约、状态含义与实现理由；不要一次跳到完整复杂框架。
- 用户最新要求继续下一小步，沿用助手直接实现及补测试、用户审查并运行的分工。用户曾要求自己写核心、助手指导并后补测试，后续分工以用户最新要求为准，不对已授权实现重复确认。
- 已由助手写好的有效期迁移实现及 32 个测试，用户已审查并反馈通过。后续沿当前第 5 关路线逐步推进，不因讨论语义判断缺口而提前接模型或重建项目。
- **除非用户明确要求代跑，否则不要运行测试。** 用户多次说“我来测试”“不需要再重复测试”。用户报告通过后，直接进入下一小步，不重跑旧测试。
- 可以对新代码做静态/语法检查，但不能把它说成运行测试通过。不要导入生产模块执行逻辑来变相重跑测试。
- 不需要对已授权的常规可逆修改反复确认。保持改动小，不做无关格式整理，不新增无必要依赖；当前使用标准库 sqlite3、json、datetime、unittest。
- 给测试命令时，先确保对应测试文件真实存在。命令工作目录应为 `D:\cmu_agent\memory`。
- 旧消息中的异常模块路径已过时。当前统一使用 `from src.error import ContextBudgetExceeded`，实现文件是 `src/error/context_budget_exceeded.py`。

## 3. 最初的架构意图

主要上下文包括：基本不变的系统指令、变化很少的用户偏好、队列消息。队列头部将保存历史摘要，近期交互追加到队尾。

外部上下文包括：完整历史消息，以及外部资料、整理后的消息、未来可能有用信息的归档。

组件包括：队列管理器、模型记忆操作的函数执行器、候选事实提取器、事实关系判断和记忆更新器。实体/事实/来源先用关系表表达图结构；专用图数据库在后期通过评测决定是否引入。

以上是整体目标。历史摘要、独立用户偏好管理、归档、模型提取和专用图检索尚未实现，不要与现有能力混淆。

## 4. 开发路线和进度

| 关卡 | 内容 | 当前状态 |
|---|---|---|
| 1 | 原始消息记录、隔离、重放与幂等 | 已实现，用户反馈测试通过 |
| 2 | 近期队列、增量刷新、上下文预算 | 已实现基础能力，用户反馈通过；摘要压缩未做 |
| 3 | 历史关键词检索、最小评测与报告 | 本地开发基线已实现并验收；正式数据适配未做 |
| 4 | 结构化事实存储、查询及上下文组装 | 已实现，用户反馈通过 |
| 5 | 关系判断、纠错、撤回、多值、状态变化、冲突 | 更新、撤回、统一入口、规则判断器、有效期迁移、争议记录及五种解决结果已验收；重复事实来源合并已实现待验收 |
| 6 | 模型适配、候选提取、结构化函数执行 | 未开始；不要提前调用真实模型 |
| 7a | 摘要、覆盖边界与队列压缩 | 未开始 |
| 7b | 任务状态、待办与恢复 | 未开始 |
| 7c | 事件和经历整理 | 未开始 |
| 8 | 外部资料、分块和混合检索 | 未开始 |
| 9 | 后台任务、并发、一致性与遗忘传播 | 未开始；当前按单写入者设计 |
| 10 | 实体消歧、图关系扩展及无图对照 | 未开始 |
| 11 | LongMemEval、LoCoMo 正式验收与成本报告 | 未开始 |

第 3 关的路线原本包括正式数据开发样本适配，但目前落地的是合成开发集，不能将数据适配写成已完成。两个正式基准是用户明确要求的最终验收内容。

## 5. 主要文件

```text
memory/
  README.md
  DEVELOPMENT_HANDOFF.md
  src/
    message_store.py
    context_queue.py
    context_builder.py
    history_retriever.py
    fact_store.py
    memory_updater.py
    fact_relation_judge.py
    conflict_store.py
    error/context_budget_exceeded.py
    error/__init__.py
    eval/retrieval_evaluator.py
  eval/
    run_retrieval_eval.py
    data/retrieval_dev.json
    reports/retrieval_baseline.json
  tests/
    test_message_store.py
    test_message_store_regressions.py
    test_context_queue.py
    test_history_retriever.py
    test_retrieval_evaluator.py
    test_run_retrieval_eval.py
    test_retrieval_report_output.py
    test_fact_store.py
    test_fact_store_listing.py
    test_context_builder.py
    test_fact_store_updates.py
    test_fact_store_retractions.py
    test_memory_updater.py
    test_fact_relation_judge.py
    test_fact_store_transitions.py
    test_fact_temporal_plans.py
    test_conflict_store.py
    test_conflict_resolutions.py
    test_conflict_resolution_actions.py
    test_duplicate_sources.py
```

`eval/data/retrieval_dev.json` 已核实存在，虽然普通 `rg --files` 的结果可能没有显示它。工作区还有更早的实验内容，当前开发目标始终是 `memory/`。

## 6. 必须保留的存储和上下文契约

### ID 幂等规则：用户明确指定

上游用分布式算法保证新 ID 不重复；重试沿用原 ID。**同一标识返回第一次写入的内容，不比较正文，也不因新的正文不同而报冲突。** 新 ID 即使内容相同，也属于新记录。

- 消息标识是 `(user_id, session_id, message_id)`。
- 事实版本标识是 `(user_id, fact_id, version)`。
- 先校验定位所需 ID/版本，已存在则返回；其他新传入业务字段即使变化、缺失或非法，也应维持已有目标的幂等行为。
- 不要在新包装层里重新引入“同 ID 正文必须一致”的检查。

### MessageStore

- 连接由 MessageStore 拥有和关闭，row_factory 为 sqlite3.Row。先在 autocommit=True 时启用外键，再切换 autocommit=False；不要在已有事务中才执行 PRAGMA foreign_keys=ON。
- messages 保存 user_id、session_id、message_id、会话内递增 seq、occurred_at、recorded_at、payload_json。
- validate_ids 是公共静态方法，拒绝非字符串、空或全空白，保留 ID 原值。
- append_message 新消息接受 user/assistant/tool 的原始 JSON payload；保留 tool_calls、坏的工具参数字符串、content=None 等原始交互结构。时间要求带时区并统一 UTC，JSON 严格编码。
- get_message 返回解码 payload 的完整记录或 None；list_messages 按 seq 升序、严格大于 after_seq；list_user_messages 限定用户，按 session_id、seq 排序。

### ContextQueue 与 ContextBuilder

- 队列绑定用户和会话，构造时为空；只有显式 refresh 才增量加载，last_seq 是实际消息序号。
- to_messages 深拷贝导出 payload，不返回数据库元字段，也不写入或删除历史。
- ContextBuilder 输出：system 提示词；可选的 user 角色 JSON 数据块 `{"type":"memory_context","facts":[...]}`；随后平铺近期原始消息。没有自定义 memory 角色，也不把整个近期消息列表嵌成单条 user 消息。
- 只使用绑定用户/会话/as_of 对应的事实；不隐式 refresh，不把合成 memory_context 存回原始消息。
- 对完整最终消息列表计数一次；max_tokens 和计数结果必须是严格非负整数，等于预算允许，大于预算抛 ContextBudgetExceeded；当前不自动丢消息或压缩。

### FactStore 数据结构

- 复用 MessageStore 的同一个连接，不自行关闭连接。
- facts 主键 `(user_id, fact_id, version)`，含 subject_id、predicate、object_json、scope、session_id、valid_from、valid_to、status、recorded_at。
- version 为正整数；status 为 active/superseded/retracted。scope=user 时 session_id 必须为 None；scope=session 时必须提供非空会话 ID。
- fact_sources 外键连接事实版本和真实 messages 来源，保存 source_session_id/source_message_id，始终限定相同 user_id。
- object 目前只支持 `{"type":"literal","value":...}`，value 必须能严格 JSON 编码，不允许 NaN/Infinity。
- sources 是非空完整来源列表，每项包含 user_id/session_id/message_id，要求同一用户且消息存在，按完整三元组去重。
- 时间带时区、统一 UTC，两端存在时 valid_to>valid_from。

### FactStore 各方法

- `get_fact(user_id=..., fact_id=..., version=1)`：精确版本读取，不存在返回 None；object 解码，sources 按会话和消息 ID 排序。默认版本 1 不是“最新版本”。
- `get_latest_fact(user_id=..., fact_id=...)`：按 version DESC 读取，不过滤 status，撤回版仍应返回。
- `list_facts(user_id=..., subject_id=None, predicate=None, session_id=None, as_of=None)`：仅 active，按有效期 `[valid_from, valid_to)` 过滤，未知边界开放。未传会话仅返回用户级事实；传会话返回用户级加该会话事实。as_of 是有效时间，不是历史数据库快照时间。
- `add_fact(...)`：用户要求公开参数中保留 version=1、status=None（None 归 active）。查重和读回使用实际传入版本，不能再固定取版本 1。这个底层入口支持显式版本/状态，常规版本迁移走 update_fact。
- `_insert_fact(...)`：复用校验、幂等查询、事实及来源插入、精确读回，**不管理事务，不执行 commit/rollback**。
- `_insert_fact` 检查事实 INSERT 的 rowcount==1、来源 executemany 的 rowcount==去重来源数，插入被忽略时报错；仍不提交或回滚。
- `add_fact`、`update_fact` 各自用唯一的 `with self.conn:` 调用内部方法。不要从 update_fact 调用会自行提交事务的公开 add_fact；同一连接嵌套 with 不是 savepoint，内层会提前提交。
- `update_fact(user_id=..., fact_id=..., new_version=..., object=..., sources=...) -> dict`：new_version 严格整数且>=2；目标版本已存在则直接返回。否则要求 latest 存在、active、目标等于 latest.version+1。同一事务中条件更新旧版为 superseded、插入新版 active 和新来源。继承主体、谓词、作用域、会话和有效期，新来源只属于新版，旧来源继续保留。
- 公开 update_fact/retract_fact/transition_fact 仍各有唯一 with conn，算法分别在不提交事务的 `_update_fact/_retract_fact/_transition_fact` 中。争议解决由 ConflictStore.resolve_conflict 管理组合事务，内部仅调私有写方法，不能调用公开写方法形成嵌套提交；MemoryUpdater 不改事务管理方式。
- `retract_fact(user_id=..., fact_id=..., expected_version=...) -> dict`：预期版本严格整数且>=1。事务内先确认最新版本号匹配；同版本已 retracted 直接返回；active 才能改为 retracted。条件 UPDATE 检查 rowcount==1，事务内重新读取返回。不新增版本，不改内容/有效期/recorded_at/来源。
- `transition_fact(user_id=..., fact_id=..., new_version=..., new_fact_id=..., effective_at=..., object=..., sources=..., valid_to=None)`：详见第 9 节的迁移契约；同一事务关闭旧区间并创建独立新事实，返回 previous_fact/fact。
- 不符合写入条件时抛 ValueError；数据库异常原样向外传播并回滚。写入失败不返回 None，也不转成 NOOP。
- 撤回后同一 fact_id 没有 active 版本，这是用户已理解并接受的语义。不会自动激活旧版；“恢复旧值/撤销一次更新”是另一种尚未实现的操作。逻辑撤回保留数据，不等于第 9 关的物理遗忘传播。
- SQL UPDATE 改变数据库，不会自动改变之前返回的 Python 字典；检查新状态需要重新读取。

## 7. 最新 MemoryUpdater 的精确契约

入口：`MemoryUpdater(fact_store).apply(*, user_id: str, decision: dict) -> dict`。

这是决策执行入口，尚不提取候选、不检索对比、不自动选择动作、不调用模型，也不是批量事务执行器。

| action | 允许的决策字段（均含 action） | 行为 |
|---|---|---|
| ADD | fact_id/subject_id/predicate/object/sources，可选 scope/session_id/valid_from/valid_to | 调用 add_fact，业务入口明确 version=1、status=active |
| UPDATE | fact_id/new_version/object/sources | 调用 update_fact |
| DELETE | fact_id/expected_version | 调用 retract_fact |
| TRANSITION | fact_id/new_version/new_fact_id/effective_at/object/sources，可选 valid_to | 调用 transition_fact，返回 fact 加 previous_fact |
| NOOP | 只有 action | 不查询或写入事实，返回 fact=None |

- 独立 user_id 来自可信调用上下文，decision 不能覆盖它；未知动作、未知字段、非字典决策都抛 ValueError。
- ADD 不能由 decision 自行指定 version/status；这与底层 add_fact 支持参数化并不冲突，是业务入口的生命周期约束。
- 返回 `{"action": action, "fact": 已保存记录}`；NOOP 返回 `{"action":"NOOP","fact":None}`。NOOP 只表示无需执行存储操作，不证明某事实存在或写入成功。
- TRANSITION 另含 previous_fact，表示关闭旧区间后的版本；fact 表示新事实第 1 版。原四种动作的返回结构不变。
- 通过 `.get()` 将业务字段交给 FactStore 校验，保留已存在目标版本优先返回的幂等规则。不要为了“必填校验”破坏重试缺正文仍返回已有版本的行为。
- 不在这里额外包裹 `with conn:`，不吞存储异常。当前输入字段白名单严格，后续需要 reason/target 等元信息时应明确扩展契约，不能默默透传未知字段。
- `tests/test_memory_updater.py` 有 9 个真实 SQLite 内存库测试，覆盖动作链、幂等（含撤回后重试旧 UPDATE）、用户隔离、结构校验、缺字段和数据库故障回滚。用户已反馈通过，不重跑。

## 8. 评测现状与最终验收要求

- HistoryRetriever 是大小写归一化、空白拆词、按不同关键词子串命中数评分的简单检索，不是中文分词或语义检索。稳定按得分降序，同分沿用消息顺序；只查当前用户，top_k=0 和空查询允许返回空。
- 核心评测函数在 src/eval/retrieval_evaluator.py；项目根 eval/run_retrieval_eval.py 负责载入开发集、建干净内存库、验证标签和保存报告。
- 评分基于完整来源三元组；先截断前 k 再去重，不补位。有答案算宏平均 Recall@k、Hit@k，无答案单独算返回空比例；缺少对应类别时为 None。
- 当前合成集有 8 条历史、8 个案例（6 有答案、2 无答案）。已保存 k=2 基线：Recall=7/9、Hit=5/6、无答案空返回率=1。这不是 LongMemEval/LoCoMo 的正式成绩。
- 最终必须接入 LongMemEval 与 LoCoMo：原始历史走真实记忆链路，答案/人工摘要/证据标签不能进入建库；同一历史的多个问题使用固定只读快照，问答和裁判反馈不写回。
- 正式接入时核对官方数据与评分脚本，固定模型、提示词、配置、预算、数据和评分版本；报告官方总分/分类分、证据检索、建库/查询成本和工程回归。图检索与无图做配对比较。
- 现有 README 记录的拟用数据是 LongMemEval cleaned S/M 对话版，Oracle 单列；LoCoMo locomo10.json QA。当前尚未下载和实现正式适配，不要暗示已完成。

## 9. 事实关系判断器与后续小步

`src/fact_relation_judge.py` 提供 `judge_fact_relation(*, user_id, candidate, existing_facts, cardinality="unknown", merge_duplicate_sources=False) -> dict`，返回 relation/target/reason/decision。纯函数，不读写数据库，不调用模型；decision 可交给 MemoryUpdater.apply，待处理为 None，与 NOOP 分开。详细输入和示例见 README 的“事实关系判断器”。

- candidate 的 intent 为 assert（默认）/correct/retract/change，公共字段为 subject_id、predicate、sources，可选 scope/session_id/valid_from/valid_to。assert 要求稳定的新 fact_id 和 object；correct 要求 object，可给 target；retract 不需要 object，可给 target；change 要求 object，可选 fact_id/target。target 为 fact_id/version，缺目标返回待处理、非法结构报错。未知候选字段不允许。
- existing_facts 是调用方提供的同用户完整快照，须覆盖相关事实所有区间及非 active 的最新版本；可含旧版本，按每个 fact_id 的最大 version 选择，不恢复旧 active。单次 list_facts(as_of) 不保证完整，当前没有自动快照加载。不同用户输入报错，不跨精确 scope/session_id 选目标。
- SUPPLEMENT→ADD；DUPLICATE 默认 NOOP，显式开启来源合并时可 UPDATE；MULTIVALUE→ADD（仅明确 multi）；CORRECTION→UPDATE；RETRACTION→DELETE。完整 change 可生成 TRANSITION；其他 TEMPORAL_CHANGE 与 UNRESOLVED 的 decision=None。不同对象不自动覆盖，同值与异值并存时只有 multi 可直接判断重复。
- 时间归一 UTC；对象按严格 JSON 存储形态比较，忽略键顺序，区分 true 与 1。来源只校验结构和同用户，真实消息存在性由写入层校验。默认重复 NOOP 不合并新增来源。
- correct/retract 目标必须最新 active 且主体/谓词/精确作用域匹配；纠错还必须有效区间相同。目标失效不换其他目标。UPDATE 固定 new_version=target.version+1，DELETE 固定 expected_version。
- 判断器只规划新候选；assert 的 fact_id 已出现在快照时返回 UNRESOLVED。执行重试直接复用原 decision，不重新判断或计算最新版+1，原存储幂等规则不变。调用方逐条判断、执行，当前无并发/批量快照事务。
- `tests/test_fact_relation_judge.py` 有 20 个测试方法（18 个规则/边界测试、2 个 SQLite 集成测试），用户已反馈通过，不重跑。

重复事实来源合并契约（最新实现，待验收）：

- `merge_duplicate_sources` 严格 bool，默认 False 保留旧行为，包括旧快照不带 sources 仍可判断重复。True 只对实际选中的唯一重复目标校验 sources：该版本的完整、非空、同用户三元组来源；无关或历史记录不新增来源校验。
- 存在新增来源时，relation=DUPLICATE，decision 复用 UPDATE，固定目标 fact_id/new_version=target.version+1，object 为目标规范值，sources 是旧来源在前、新来源在后的去重并集；旧版及旧来源保留，新版同值且包含合并来源。来源全已知仍 NOOP，无新版本。不新增存储表、写入 API 或 MemoryUpdater 动作。
- 显式 correct 同值且精确目标/区间匹配也可合并；多个同值 fact_id 的 assert 不能唯一选目标，开启合并时 UNRESOLVED/decision=None。single/unknown 同值与异值混合、区间差异、复用候选 ID、目标失效等旧保守分支保持。
- 单写入者逐条规划、执行、读取新快照再处理下一候选。两条合并计划若来自同一旧快照，会共用目标版本，后者只按版本身份返回首条记录而不追加其来源；当前不支持这种批量或并发累积。重试只复用原 decision，后续更新/撤回也不重新合并。
- `tests/test_duplicate_sources.py` 有 20 个独立测试方法（13 个规则/兼容测试、7 个真实 SQLite 集成测试），不导入旧 TestCase。覆盖默认兼容、唯一目标/来源校验、UTC/JSON、歧义、深拷贝、更新及撤回后的固定版本重试、读新快照连续累积来源、消息存在性/用户归属、三写点 FAIL/IGNORE 整体回滚和磁盘重开。本轮两份 Python 文件已通过 AST 语法检查，测试待用户审查与运行；助手未执行测试或导入生产模块。

有效期迁移契约（用户已验收）：

- 对明确 change，须同时提供稳定新 fact_id、显式 target 和带时区 valid_from（映射为 effective_at）；不完整继续待处理。目标须最新 active、精确主体/谓词/作用域匹配。切点须严格在旧区间内部，新 valid_to 若有须晚于切点；新终点独立提供、默认 None，不继承旧终点。
- 计划固定旧 fact_id/new_version=target.version+1 与 new_fact_id。同值显式 change 也按时间切分，不推断 NOOP。single/unknown 下新区间若与其他同上下文最新 active 事实重叠，返回 UNRESOLVED；左闭右开接壤不算重叠，multi 仅迁移选中的目标。
- 存储在唯一事务内插入新事实 v1、把原版改为 superseded、插入旧 fact_id 的关闭区间版 active、插入迁移记录。原版正文、完整有效期、记录时间及来源保留；关闭版 valid_to=切点且来源为原来源+变化来源，新事实 valid_from=切点、仅变化来源。历史 active 区间仍由 list_facts(as_of) 过滤。
- 新增 fact_transitions 表，主键 `(user_id,new_fact_id)`，外键引用新 v1 和旧关闭版。初始化旧数据库只增加表。重试只先校验 user_id/new_fact_id，查到迁移记录即返回固定两个版本，忽略其他新业务字段，绝不比较正文；后来纠错、撤回后也不重新迁移。普通 ADD 使用过的新 ID 没有迁移记录时必须拒绝，不能伪装成功。
- 方法返回 previous_fact/fact；MemoryUpdater 的 TRANSITION 返回 action/previous_fact/fact，不嵌套事务。_insert_fact 不提交，任何一步失败整体回滚；来源存在性、严格 JSON、时间与用户归属继续由写入层校验。
- 测试：tests/test_fact_store_transitions.py 有 18 个方法，tests/test_fact_temporal_plans.py 有 14 个方法，用户已反馈通过，助手未代跑。覆盖半开区间、来源、重试、五处写入失败及忽略迁移记录的回滚、已有数据库增表和磁盘重开。详细候选示例见 README 的“有效期迁移”。

争议记录契约（用户已验收）：

- `src/conflict_store.py` 提供 `ConflictStore(fact_store)`，复用已有连接和 MessageStore，不拥有或关闭连接。
- `record_conflict(*, user_id, conflict_id, candidate=None, related_facts=None, reason=None)`、`get_conflict(*, user_id, conflict_id)`、`list_conflicts(*, user_id)`。record 业务字段默认 None 只为已有 ID 重试；先验证 user_id/conflict_id 并查已有记录，命中即返回，不检查新正文。新增记录时业务字段仍必须合法。
- 三表：conflicts 保存独立用户/争议 ID、剥离 sources 后的 candidate_json、reason、status=open、UTC recorded_at；conflict_facts 保存非空的 fact_id/version 引用；conflict_sources 保存候选的非空真实消息来源。关联表均以含 user_id 的复合外键连接主记录及真实事实版本/消息，引用按完整标识去重。
- related_facts 只含 fact_id/version，须在当前用户下存在，允许 superseded/retracted 历史版本；本步只记录有现存事实参与的争议，不把所有 UNRESOLVED 无条件转换成争议。candidate 必须为严格 JSON 字典，顶层不能提供 user_id，sources 须同用户且真实存在；不调用判断器、不执行候选动作。
- 来源只存关系表，读取时重建 candidate.sources；输出 user_id/conflict_id/candidate/related_facts/reason/status/recorded_at。来源按会话/消息 ID 排序，事实引用按 fact_id/version 排序；list 按 recorded_at/conflict_id 升序返回该用户全部记录，get 缺失返回 None。
- record 在唯一事务中写三表，失败整体回滚；不修改任何事实或消息。新增 status 为 open，解决成功后为 resolved；记录/读取不自动集成或调用模型。
- 三个 INSERT 阶段均检查 rowcount，插入被忽略时同样报错回滚；SQLite 错误原样抛出。tests/test_conflict_store.py 有 18 个测试方法，用户已审查并反馈全部通过，助手未代跑，不重跑。

人工争议解决契约（五种结果均已验收）：

- `resolve_conflict(*, user_id, conflict_id, outcome=None, target=None, new_fact_id=None, reason=None)` 与 `get_resolution(*, user_id, conflict_id)`。单次人工决定，支持 KEEP_EXISTING/ADD_CANDIDATE/CORRECT/RETRACT/TRANSITION。新解决要求争议 open 和非空 reason，拒绝不适用参数；不推断意图，不允许替换已存候选正文。
- KEEP_EXISTING 保存 NOOP、不改事实；ADD_CANDIDATE 用显式新 new_fact_id（任意版本占用则拒绝）及已存候选业务字段新增 v1，保留旧值，表示人工明确共存；CORRECT 必须指定 related_facts 内仍 latest active 的精确 target，要求主体、谓词、精确作用域和 UTC 有效区间相同，使用候选 object/sources 更新到 target.version+1。只作用选定事实，不批量修改其他引用。
- RETRACT 需要 target，不接受 new_fact_id，decision 为 DELETE/expected_version；候选主体、谓词、精确作用域须匹配，忽略对象/时间，保留历史仅逻辑撤回。TRANSITION 需要 target 和未用 new_fact_id，从已存候选取切点 valid_from、object/sources/valid_to，旧关闭版固定 target.version+1，新事实 v1，result 另含 previous_fact。两者目标均必须是 related_facts 内仍 latest active 的版本；新解决不能借用外部迁移收据冒充成功。手工迁移只处理指定目标，不暗中推断全局单值/重叠规则，高层 judge 仍用完整快照/cardinality。
- 新表 conflict_resolutions：主键 user_id/conflict_id、复合外键→conflicts，保存 outcome、decision_json、result_json、reason、resolved_at。返回 user_id/conflict_id/outcome/decision/result/reason/resolved_at；get 缺失 None。decision/result 是首次处理时的不可变 JSON 快照，后续事实变化不改变它；当前状态须查 FactStore。第 9 关遗忘传播须覆盖这份派生快照。
- 先校验 user_id/conflict_id 并查处理记录；已有处理直接返回，完全忽略其他重试字段，不重执行、不比较正文。新解决不能把被其他操作占用的事实 ID 或版本当成本次成功。原争议内容、来源、关联与 recorded_at 保留，仅状态 open→resolved；get/list/record 返回字段不变，list 仍列用户全部争议。当前不支持重开。
- 事实/来源写入、迁移记录、处理记录、争议状态在同一个 resolve_conflict 事务中提交；只调用不提交的内部写入方法。任何失败/IGNORE 整体回滚，数据库错误原样抛出。tests/test_conflict_resolutions.py 的 19 个方法用户已反馈通过，不重跑。tests/test_conflict_resolution_actions.py 的 22 个方法覆盖撤回、时间边界、版本/作用域/用户隔离、身份重试、快照、写入失败和忽略、旧库升级回滚及磁盘重开，用户也已审查并反馈通过，不重跑。
- 旧 conflict_resolutions 有三值 CHECK，初始化时在原 with conn 内检查 sqlite_master.sql：已含 RETRACT/TRANSITION 跳过，否则创建固定名 conflict_resolutions_v2（五值 CHECK），显式列 INSERT SELECT 复制全部旧记录，DROP 旧表并 RENAME。旧 JSON 文本、reason、resolved_at 不改，失败全部回滚。临时名占用则拒绝，不删除未知表；仅升级当前已知旧结构，不引入通用迁移框架。

第 5 关最新的重复事实来源合并待用户验收；之后按路线进入第 6 关的模型适配与候选提取，不能把当前人工规则/决定称作自动语义关系判定。

## 10. 可提供给用户的命令

最新重复事实来源合并测试命令（待用户运行）：

```powershell
cd D:\cmu_agent\memory
& "D:\User\envs\memory\python.exe" -X utf8 -m unittest tests.test_duplicate_sources -v
```

按需由用户运行的开发评测：

```powershell
cd D:\cmu_agent\memory
& "D:\User\envs\memory\python.exe" -X utf8 -m eval.run_retrieval_eval
```

不要仅因为进入了新会话就自动运行这些命令；不要重建环境或另建一套目录。
