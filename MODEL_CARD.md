<!-- paths generalized for distribution -->
# arqyn-v2 模型卡（工程试用版）

锁定时间：2026-10-07 02:1x（用户指令：接受部署固定候选序合同，压缩收尾，后续真实使用驱动后训练）

## 一、身份（全部可复核）

- 适配器：`<adapter_dir>`（IDENTITY.json 含逐文件 sha256；adapter_model.safetensors sha256 头 d3c460d9811f51f7…，93,442,816 字节级同 v1 结构）
- 来源：J-0016（600 步验证早停，best=step400，选型规则 val_only_predeclared_condition_consistency_and_macro_metrics，3.5 小时/峰值显存 3859MiB）
- 基座：internlm/Intern-Decision-2B @ revision 8797836c…（全新 LoRA，resume_from 空）
- 训练数据：v2d 全量 1176 根/2352 行/82 家族（sha 9f5964de…；oracle 1.0/泄漏 0/交叉 0/覆盖六轴不缩窄——audit_v2d 五判据全过）
- 配方：masked QLoRA，NF4+doubleQuant+BF16 compute+SDPA，lr3e-5/r16/α32/dropout0.10/batch1×accum4/max_len2048/warmup50/seed 沿合同，K=3 确定性序变体（order_variant_policy=deterministic_storage_reverse_seeded，提交 5d94b20+5cb0b10）
- 输入合同：arqyn.decision.input_contract/v2@c80c465c…（训练/评测/部署三路同源，GPU 实测 input IDs 全等）
- 校准：T=1.995449（仅 calib 200 根拟合；ECE 0.082→0.035，NLL 0.445→0.353，top1 不变）；clarify 未校准（calib 无 yes 根，如实 uncalibrated）
- 部署装载：load_base_model(NF4)+PeftModel.from_pretrained().eval()（与服务/worker 同构）

## 二、实测成绩（同协议、部署同构装载、三口径可复算）

| 评估面 | n | top1 | top2 | none 召回 | 备注 |
|---|---|---|---|---|---|
| val（冻结 40 根，选型用） | 40 | **1.000** | 1.000 | 1.000 | 已饱和，仅选型用途 |
| calib（冻结 200 根） | 200 | 0.895 | 0.955 | 1.000 | 校准拟合面 |
| **test（处女 200 根，家族隔离 OOD，唯一一次消费）** | 200 | **0.775** | 0.820 | **1.000** | **本卡 headline：真实泛化数字** |

test 同批对照：未微调基座 0.535（none 召回 0.662）→ **微调增益 +24.0pp**；规则 0.435（none 召回 0.0）→ **+34.0pp 且拒答面碾压**。v1 线未跑（额度压缩，见验收表 not_verified）。

五代轨迹（val-40 正序）：J-0010 0.475 → J-0013 0.65 → J-0014 0.70 → J-0015(pilot) 0.875 → **J-0016(正式) 1.0**；状态对照 1/9 → 8/9 → 9/9 → 9/9；常规回归束 0.286 → … → 0.857（追平基座/v1，零回退面）。

## 三、已知局限（使用时必须知道）

1. **顺序敏感（最主要局限）**：受教序（存储/逆序）top1 0.875-1.0，未受教随机序降至 0.60-0.675，翻序错向 21/21——模型利用槽位先验而非序不变语义。**缓解在位：生产候选序=确定性冻结召回（same_order 40/40 实证），运营口径即固定序成绩；任何改动候选排序的调用方须重新评估**。
2. 合成工作台情境训练与评估——真实用户分布未测；test 0.775 是"合成 OOD 家族"口径。
3. clarify 字段未校准（T=1）；none 高置信场景概率未经温度时不作风险阈值用。
4. val-40 已被迭代消费（选型合法），其 1.0 有虚高；对外引用一律用 test 0.775。
5. 不做飞行控制/遥测判断（数据零飞行内容）；不自主执行操作（条件守卫后的建议层）。

## 四、三态定位（不得混用）

- **训练完成**：✅（台账 J-0016/R-0072 级、真实日志、权重变化、checkpoint 齐备）
- **合成测试通过**：✅ 部分——主任务/困难条件/数据合同/运行身份过；候选置换门 failed（已裁决接受缓解，见 acceptance_v2.md）；校准过（clarify 除外）
- **真实任务通过**：❌ 未验证（后训练阶段解决；采集通道=uav/problems.py）

## 五、启用方式

1. 服务装载：产品仓 uav/decision 以 model_mode=v2 + adapter 路径 `<adapter_dir>`（声明 v2 无适配器会明确失败）；校准产物 `v2/lock/calibration.json` 绑定本适配器身份。
2. 离线验证：`v2/lock/batch_test.py`+`cases.jsonl`（用户先写 user_expected 再跑；**包装器有已知问题见积压清单第 7 条，临时用 evaluate CLI 直跑分区格式文件**）。
3. 误判记录：SKEIN `POST /api/uav/problems`（自动附对象引用/版本/环境）——聚成后训练数据集的唯一入口。

> v1（研究原型/冻结基线）保留于 ../arqyn-v1 只读，仅作对照，禁引用其旧 val 0.9667 为能力宣传。
