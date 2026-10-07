# ARQYN v2 —— 工作台 Agent 下一步操作决策适配器（工程试用版）

**ARQYN 是一个 LoRA 适配器：把 `internlm/Intern-Decision-2B` 微调成工作台 Agent 的"下一步操作决策"模型——输入用户请求、可校验的工作台状态和候选操作列表，排序出下一步最该执行的操作；没有任何受支持的操作可用时显式拒答 `none`，而不是硬猜。**

![License](https://img.shields.io/badge/License-Apache__2.0-blue)
![Base](https://img.shields.io/badge/base-Intern--Decision--2B-success)
![Format](https://img.shields.io/badge/format-LoRA%20%2F%20PEFT%20%2F%20NF4-informational)
![VRAM](https://img.shields.io/badge/NF4_inference-~4GB-yellow)
![Status](https://img.shields.io/badge/version-v2_engineering_preview-orange)

| | |
|---|---|
| 形态 | LoRA 适配器（非完整模型；**不含基座权重**，需另行下载） |
| 基座 | `internlm/Intern-Decision-2B` @ revision `8797836c65fc91a2435b1fb6850b5f0aabd75cc3`（Apache-2.0） |
| 版本 | v2 · 工程试用版（2026-10-07 锁定，验收判定见 `ACCEPTANCE.md`） |
| 许可 | Apache-2.0（`LICENSE` + `NOTICE`） |
| 部署 | NF4 4bit 推理约 4GB 显存，可完全离线（8GB 消费级 GPU 实测可跑） |

## 为什么需要它

工作台类 Agent 的每一步都面对同一个问题：候选操作有一排（刷新目录、补配置、查任务……），这一步该执行哪个？三条现成路线都不够好：

- **让通用 LLM 自由生成操作**：会幻觉出候选里不存在的操作，或该说"没有合适操作"时硬猜——未微调基座在 held-out 测试上首位命中率只有 0.535，且该拒答的场景里 1/3 在乱猜（none 召回 0.66）；
- **规则基线**：可解释但上限低（0.435），而且完全不会拒答（none 召回 0.00，每条拒答样例都硬猜）；
- **人工兜底**：不可扩展。

ARQYN 把这件事收敛成受约束的任务：模型只能在候选列表内做单选排序，或回答 `none`；输出概率经温度校准，置信度可以直接用于阈值判断。在 200 个从未见过的场景家族（家族隔离 OOD）上首位命中率 **0.775**，拒答召回 **1.00**。

## 核心能力

| 能力 | 说明 | 对应文件 |
|---|---|---|
| 下一步操作排序 | 在候选操作列表内单选：输出最优操作；无合适项输出 `decision.none` | `adapter_model.safetensors` + `adapter_config.json` |
| 拒答（none） | 前置条件不满足时拒答而非臆造；held-out none 召回 1.00 | `MODEL_CARD.md`、`examples/cases.jsonl`（case-003 即拒答示例） |
| 置信度校准 | 温度 T=1.995449（仅 calib 200 根、operation 字段拟合）：ECE 0.082→0.035，NLL 0.445→0.353，top1 不变 | `calibration.json` |
| 澄清判断（clarify） | 附带"是否需要向用户澄清"的 yes/no 判断（**未校准**，见已知边界） | `MODEL_CARD.md`、`calibration.json`（note 字段） |
| 离线批测 | 无副作用离线评测：示例用例 + 批测参考脚本（需部署侧评测运行环境） | `examples/cases.jsonl`、`examples/batch_test.py` |
| 完整性校验 | 逐文件 sha256 清单；适配器权重校验头 `d3c460d981…` | `checksums.sha256` |
| 合规署名 | 基座来源、revision、改动范围声明（Apache-2.0 §4(b)） | `NOTICE`、`LICENSE` |

## 实测成绩（诚实口径）

同协议、部署同构装载（NF4 + double quant，BF16 compute，SDPA）三线对照。**held-out test = 200 个从未见过的实例根、家族隔离 OOD，仅消费一次**：

| 评估线 | n | top1 | top2 | none 召回 |
|---|---|---|---|---|
| **本适配器** | 200 | **0.775** | 0.820 | **1.000** |
| 未微调基座 | 200 | 0.535 | — | 0.662 |
| 规则基线 | 200 | 0.435 | — | 0.000 |

- 相对基座 **+24.0pp**、相对规则基线 **+34.0pp**；也是三线中唯一拒答召回非零的一条。
- 另外两个评估面**勿直接对外引用**：calib 200 根 top1 0.895（校准拟合面）；val 40 根 1.000（已饱和，且被迭代选型消费，有虚高）。**对外能力引用一律用 test 0.775。**
- 验收状态：8 项门槛 7 项 passed，1 项（候选置换一致性门）failed→经裁决以"部署固定候选序合同"缓解后锁定，最终判定 **locked（工程试用版）**。逐项证据见 `ACCEPTANCE.md`。

## 快速开始

```bash
# 1) 依赖（建议独立虚拟环境）
pip install torch transformers peft bitsandbytes accelerate

# 2) 下载基座（约 4.2GB，公开可下；revision 必须与上面一致）
huggingface-cli download internlm/Intern-Decision-2B \
  --revision 8797836c65fc91a2435b1fb6850b5f0aabd75cc3

# 3) 校验本包完整性（12 个文件应全部 OK）
sha256sum -c checksums.sha256
```

加载（NF4 4bit，部署同构）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("internlm/Intern-Decision-2B", trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    "internlm/Intern-Decision-2B",
    revision="8797836c65fc91a2435b1fb6850b5f0aabd75cc3",
    quantization_config=bnb, device_map="cuda", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "<本仓库克隆目录>").eval()
```

自测用例：`examples/cases.jsonl` 每行一条 `{case_id, user_question, state(条件字段), user_expected(先写你自己的判断再跑)}`；`examples/batch_test.py` 是离线批测的参考实现——依赖部署侧的评测运行环境与基座，本仓不含该环境；其已知问题与替代路径记录在 `ACCEPTANCE.md`。

## 工作方式

```
用户请求 + 工作台状态(可校验条件字段) + 候选操作列表     ← 输入合同 v2：训练/评测/部署三路同源
        │
        ▼
Intern-Decision-2B（语言骨干，视觉塔冻结）＋ LoRA 适配器
        │   单选问题：operation(候选内 choice) + clarify(yes/no)
        ▼
top-1 操作 / decision.none ＋ 概率 → 温度 T=1.995449 校准 → 置信度
        │
        ▼
建议输出（只建议，不执行）
```

训练配方（masked QLoRA，均可在 `adapter_config.json` / `step.json` / `MODEL_CARD.md` 复核）：

- LoRA r=16 / α=32 / dropout 0.10，仅语言骨干线性层（q/k/v/o_proj、gate/up/down_proj、in_proj_* 等）；视觉塔与 projector 冻结未动（见 `NOTICE` 改动声明）；
- NF4 double-quant + BF16 compute + SDPA；lr 3e-5，batch 1×accum4，max_len 2048，warmup 50；
- 数据：合成工作台场景 1,176 根 / 2,352 行 / 82 家族（oracle 全对、无泄漏、家族切分）；每样本每 epoch K=3 个确定性候选序变体（`order_variant_policy=deterministic_storage_reverse_seeded`）；
- 600 步验证早停，best=step400（`step.json`）；训练峰值显存 3859MiB。

## 目录导航

| 路径 | 作用 |
|---|---|
| `adapter_model.safetensors` | ★ LoRA 权重本体（约 67MB，git-lfs；sha256 前缀 `d3c460d981`） |
| `adapter_config.json` | PEFT 配置：r/α/dropout/target_modules，peft 0.21.2，基座已解析为 HF hub 名 |
| `calibration.json` | 温度校准产物：T、拟合前后 ECE/NLL/Brier、可靠性分桶数据 |
| `step.json` | 训练步信息：best=step400、K=3 序变体策略 |
| `checksums.sha256` | 逐文件 sha256 完整性清单（`sha256sum -c` 验证） |
| `examples/` | 示例用例 `cases.jsonl` + 离线批测脚本 `batch_test.py` |
| `MODEL_CARD.md` | 完整模型卡：身份、成绩、局限、三态定位 |
| `ACCEPTANCE.md` | 验收表：8 门槛逐项状态与证据（含 not_verified 清单） |
| `给队友说明.md` | 中文速览：包内容、跑通步骤、成绩、边界 |
| `LICENSE` / `NOTICE` | Apache-2.0 许可 + 基座署名与改动声明 |
| `README.md` | 本文件 |

## 与相关项目的关系

- **基座**：[internlm/Intern-Decision-2B](https://huggingface.co/internlm/Intern-Decision-2B)（Apache-2.0），上游为 Qwen/Qwen3.5-2B——署名与改动声明见 `NOTICE`；
- **上游项目**：[InternLM/Intern-Decision](https://github.com/InternLM/Intern-Decision)；
- 本仓库只含 v2 适配器本体；训练代码、训练数据集与部署侧服务不在本仓分发；更早的研究原型版本不随本仓提供。

## 许可

Apache-2.0。分发时请一并保留 `LICENSE` 与 `NOTICE`；本包不含基座权重，请自行从 Hugging Face 下载。

## 已知边界（使用前必读）

1. **候选顺序敏感（最主要局限）**：以上成绩在固定候选排序合同下测得。受教序 top1 0.875–1.0，未受教的随机序降至 0.60–0.675，翻序错向 21/21——模型利用了槽位先验而非序不变语义。缓解=部署时固定冻结召回序；**任何改变候选排序的调用方必须重新评估**。
2. **合成数据训练**：训练与评测均为合成工作台场景（oracle 校验、家族隔离），真实用户分布上的表现未测。
3. **clarify 字段未校准**：温度仅对 operation 字段拟合（校准集无 yes 样本），其概率不可当风险阈值使用。
4. **只建议不执行**：输出是建议层，不会执行任何真实操作。
5. **评测规模小**：test 仅 200 根，分数当估计值看；val-40 已被选型消费（1.0 有虚高），勿引用。
6. **时延数据沿用上一代**：同构装载实测冷加载 25.4s、热请求 P50 536ms / P95 624ms，**未对 v2 权重重测**（`ACCEPTANCE.md` not_verified 项）。
