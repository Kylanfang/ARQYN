<!-- paths generalized for distribution -->
# acceptance_v2.md — arqyn-v2 工程锁定验收表（2026-10-07）

> 口径：passed=有机械证据；failed=实测未达；not_verified=本轮未测（额度压缩，用户 10/7 指令）。
> 置换门 failed 系用户明示接受缓解后锁定（"接受部署固定候选序合同……后续使用中再精进"，10/6-10/7 指令）。

| §6.4 门槛行 | 状态 | 证据 |
|---|---|---|
| 1 数据与合同（交叉 0/无泄漏/三路编译一致） | **passed** | dataset_audit_v2d.json 五判据；M1a contract-crosscheck.json（input IDs 全等） |
| 2 运行身份（v2 声明必有适配器/身份随行） | **passed** | IDENTITY.json+runner identity 块；model_mode=v2 无适配器双拒绝测试 |
| 3 主任务（OOD 条件类 macro≥0.85） | **passed** | val 面 macro 1.0（best_val_metrics.json step400）；处女 test top1 0.775（v2/lock/v2_test200_metrics.json）——test 面按条件类 macro 未分解（not_verified 细项：分解脚本在 v2/eval 可复算） |
| 4 困难条件（不可执行≤5%/状态对照≥90%） | **passed** | 不可执行 0.0（四代连续）；状态对照 9/9=1.0（best_val_metrics + metrics_v2 m5 复算同值） |
| 5 候选置换（一致率≥95%/≥100 诊断根） | **failed → accepted-with-mitigation** | J-0015 实测 0.475（受教两序 0.875/0.925 vs 未教随机 0.60-0.675）；缓解=部署固定冻结召回序（same_order 40/40）+条件守卫；用户 10/6 裁决；改进路线在 post-training-backlog 第 1 条 |
| 6 常规回归（常规任务保住/相对 v1 降幅≤2pp） | **passed** | 常规束 0.857=v1=基座（J-0015 常规监控零回退面）；J-0016 正式版 val 全绿 1.0 |
| 7 校准（字段绑定/服务概率与重放一致/NLL-Brier-ECE） | **passed（clarify 除外）** | calibration.json：T=1.995 仅 calib 拟合，ECE 0.082→0.035，top1 不变；概率链修复实证（M1a：服务=独立重放 max diff 0.0）；clarify=uncalibrated（calib 无 yes 根，如实） |
| 8 训练真实性（日志/更新数/权重变化/重载） | **passed** | J-0016 stdout 600 步+train_summary（early_stop_val 如实）；权重变化数值在案；fresh evaluate=val40 top1 1.0（部署同构装载复评一致） |

**本轮压缩未测（not_verified，全部挂积压清单）**：①test 面五线完整对照（v1 线、增强规则线未跑；基座/规则/v2 三线已跑）；②时延/资源 P50·P95（沿用 M1a worker 实测：冷加载 25.4s、热 P50 536ms/P95 624ms，未对 J-0016 适配器重测）；③K=5/8/12 候选数诊断束；④batch_test.py 包装器已知问题（底层 evaluate CLI 六次验证可用）；⑤第二种子稳定性。

**最终判定：locked（工程试用版）** —— 8 行中 7 passed、1 failed-accepted-with-mitigation（用户裁决）。对外能力引用口径=处女 test top1 0.775 / none 召回 1.0 / 常规束 0.857。
