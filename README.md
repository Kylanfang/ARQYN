# ARQYN v2 -- Workbench Decision Adapter (engineering preview)

LoRA adapter that turns **internlm/Intern-Decision-2B** into a *next-step decision*
model for the workbench: given a user request, verifiable workbench state, and a
candidate operation list, it ranks the operation to run next (or answers `none`
when no supported operation fits).

## Facts

| | |
|---|---|
| Base model | `internlm/Intern-Decision-2B` @ `8797836c65fc91a2435b1fb6850b5f0aabd75cc3` |
| Adapter | LoRA r=16 / alpha=32 / dropout=0.10, language backbone only (vision tower frozen) |
| Training | masked QLoRA, NF4 double-quant + BF16 compute, multi candidate-order per epoch |
| Task | single-choice operation ranking + `clarify` (no/yes) |
| Trained on | synthetic workbench scenarios (1,176 roots / 2,352 rows, oracle-verified, family-split) |
| Eval precision | NF4 + double quant, BF16 compute, SDPA |
| Calibration | temperature 1.995 (fitted on a held-out calib split, operation field) |
| License | Apache-2.0 (see LICENSE / NOTICE) |

## Held-out test (200 unseen-instance roots, family-isolated)

| line | top1 | none-recall |
|---|---|---|
| **this adapter** | **0.775** | **1.00** |
| untuned base | 0.535 | 0.66 |
| rule baseline | 0.435 | 0.00 |

`none` = "no suitable operation". The adapter refuses correctly on all such cases;
the rule baseline guesses on every one.

## Load

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
model = PeftModel.from_pretrained(base, "./arqyn-v2").eval()
```

## Known limitations

1. **Candidate-order sensitivity** -- trained with a fixed candidate-ordering
   policy; scores are for that order. Re-evaluate if you change candidate ranking.
2. **Synthetic scenarios** -- not real-user logs; real-distribution performance
   is not yet measured.
3. `clarify` field is **uncalibrated** (temperature fitted for `operation` only).
4. Suggests operations only -- it does **not** execute them.
5. Small evaluation splits (200 roots); treat scores as estimates.

## Files

- `adapter_model.safetensors` -- LoRA weights
- `adapter_config.json` -- PEFT config (base id resolved to HF hub name)
- `calibration.json` -- temperature + binding info
- `MODEL_CARD.md`, `ACCEPTANCE.md` -- full card and acceptance table
- `examples/` -- sample cases + offline test script
- `checksums.sha256` -- integrity

## Attribution

Built on [Intern-Decision](https://github.com/InternLM/Intern-Decision) /
[internlm/Intern-Decision-2B](https://huggingface.co/internlm/Intern-Decision-2B)
(Apache-2.0), itself fine-tuned from Qwen3.5-2B. See `NOTICE`.
