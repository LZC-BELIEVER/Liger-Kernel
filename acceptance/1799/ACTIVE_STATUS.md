# ACTIVE_STATUS — Issue #1799 验收闭合

更新时间：2026-10-11 13:5x（本轮结束时的交接快照）
执行角色：外部仓库验收 operator agent
被验收仓库：`/workspace3/lcw/liger-kernel-ref`
（分支 `feat/ascend-fused-linear-grpo-loss`，最终修订 `a8c360c`，工作树干净）
验收包：`/workspace3/lcw/SLAI-Ascend-OpForge/docs/lzc/1799/`

## phase

**全部本机可执行验收项已闭合**（pytest / benchmark / pre-post 对比表 / msProf 原始导出
/ 文档收口）。唯一未执行动作：commit、push、PR（按 goal 需 steward 确认）。

**2026-10-11 steward 终审：PASS。** 复核项：SHA256SUMS 23/23 OK；六个源文件哈希与冻结
清单逐位一致；replay 机制 forward 侧源码实证（`fused_linear_ppo.py:318`
`autograd.grad`、`:432` `save_for_backward`）；默认与 sweep 数字与独立复算一致；表 `*`
标注与 PR 草案一致；工作树干净 `a8c360c`；NPU 已释放。**本地交付完成并验收**；commit/
push/PR 待 owner 确认，本 agent 进入 standby，不再有未完成动作。

## NPU 分配（持久化）

- `npu_device`: **2**（A3，npu-smi `NPU 1 / Chip 0 / Phy-ID 2`，Bus `0000:99:00.0`）
- 选择依据：16 个 chip 全量核对；卡 0 有活跃进程，卡 4/5 被 MSD/vLLM 占用（~60GB/chip），
  卡 6/7 为 CANNBench round8/round9 worker；卡 1/2/3 空闲健康，选卡 2（bf16 matmul 自检通过）。
- 本轮全部命令结束后卡 2 上无本 agent 进程（无长跑任务残留）。
- 释放条件：本任务不再需要设备时记录 release；本 agent 只回收自己启动的进程组。

## 已完成的命令与产物（均在同一修订上）

| # | 命令 | 结果 | 证据 |
| --- | --- | --- | --- |
| 1 | `pytest -q test/chunked_loss/test_grpo_loss.py --disable-warnings -rs`（device 2） | 1024 passed / 566 skipped / 0 failed / 0 error，995.18s | `evidence/20261011/pytest/pytest_full_20261011_dev2.log` |
| 2 | `benchmark/scripts/benchmark_grpo_loss.py`（POST，device 2） | 48 行 / 24 对比 / 0 回归；speed 0.846733–0.908310，memory 0.738682–0.956061 | `evidence/20261011/bench_post/` |
| 3 | 同命令 + `pre_variant/pre_variant.diff`（PRE，compiled=False） | 48 行；full 模式 ratio 2.6848–3.0765 | `evidence/20261011/bench_pre/` |
| 4 | PRE（compiled=True，pre-PR 默认）尝试 | TorchInductor/dynamo shape-guard 断言中止 | `evidence/20261011/bench_pre/bench_default_pre_generic_compiled_true_FAILED_20261011_dev2.log` |
| 5 | `tools/grad_smoke.py`（POST/PRE × bf16/fp32 × B=1,2,4） | 梯度与 torch 参考在测试容差内一致；PRE 计时 backward 恒 ~7ms（replay 伪影） | `evidence/20261011/grad_smoke_*_20261011_dev2.log` |
| 6 | `msprof`（CANN 9.1，B=4，token+sequence，device 2）+ 分析 | 34 kernel / 487.1ms；loss 路径 9.9%；投影 72% | `evidence/20261011/msprof/` |
| 7 | `tools/make_comparison_table.py`（POST vs PRE，带 `*` caveat） | 24 行逐配置表 | `evidence/20261011/table_default_post_vs_pre.{md,csv}` |
| 8 | `benchmark_grpo_loss.py --sweep-mode model_config --bt 2048`（POST-only，按 steward 建议补跑） | 112 行 / 56 对比 / 0 回归；speed 0.747432–0.955636，memory 0.841025–0.977583 | `evidence/20261011/bench_post/all_benchmark_data_1799_post_models_20261011_dev2.csv`、`evidence/20261011/table_models_post.*` |

源码状态：POST 六个文件 sha256 见 `README.md`；PRE 期间仅
`chunked_loss/functional.py`(`b083217d…`) 与 `chunked_loss/grpo_loss.py`(`8492c634…`)
不同；恢复后 6 个哈希逐一相同、`git diff` 为空、`git status --porcelain` 为空。

## 未完成 / 待指令

1. push / PR（goal 要求先经 steward 确认）；提交材料已备好：`PR_BODY_DRAFT.md`。
2. 非默认 `model_config` sweep：POST 侧已补跑并并入文档；PRE 侧按 steward 决定不做
   （对比表以默认 run 为准）。
3. A2/950 实测：本机无对应硬件，按 goal 豁免并已写入平台边界。

## next_action

- 等待 steward 对 push/PR 是否执行的决定；文档、哈希、对比表与跨模型 sweep 已全部落盘，
  device 2 上无本 agent 进程残留，被验收仓库工作树干净（`git status --porcelain` 为空）。

## files_to_open_next

- `docs/lzc/1799/README.md`
- `docs/lzc/1799/PR_BODY_DRAFT.md`
- `docs/lzc/1799/EVIDENCE_MANIFEST.md`

## skills_read

- `ascend-c`（本任务实际使用点：Ascend GRPO kernel/UB 分块与 `python -m` 工具链约定）
- `opforge-evaluation`（证据分层、resume/交接与 no-fallback 边界）
- 本任务为外部仓库验收任务，未使用 OpForge evaluator / task manifest / cases。
