# ⚖️ Legal Text Simplifier — Qwen2.5-3B + QLoRA

Fine-tuned **Qwen2.5-3B-Instruct** to translate dense, archaic legal language from the **Pakistan Penal Code (PPC)** and **Code of Criminal Procedure (CrPC)** into plain, accessible English — making the law understandable to people who aren't lawyers.

> Built end-to-end on a single free-tier Colab GPU using QLoRA, proving you don't need enterprise compute to fine-tune a useful domain-specific LLM.

---

## 🧩 Why this project

Legal text is intentionally precise — and unintentionally unreadable. Most citizens can't parse a PPC section without a lawyer's help. This project explores whether a small open-source LLM can be taught to **preserve legal meaning while rewriting it in plain English**, closing that comprehension gap.

It also sits at the intersection of my two academic tracks — Computer Science and Law — making it a direct, practical demonstration of applying NLP to legal accessibility (a space adjacent to the growing LegalTech / AI-governance field).

---

## ✨ Features

- 🔧 Fine-tuned on a **custom instruction dataset** built from real PPC/CrPC sections
- ⚡ **4-bit NF4 quantization (QLoRA)** — trains on a 16GB T4 GPU
- 🪶 **LoRA adapters** for lightweight, efficient fine-tuning (no full-model retraining)
- 🎛️ **Interactive Gradio demo** for real-time testing with adjustable generation settings
- 🤗 Built on the Hugging Face ecosystem — `transformers`, `trl`, and `peft`

---

## 🏗️ Model Details

| Component | Detail |
|---|---|
| **Base model** | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| **Fine-tuning method** | QLoRA (4-bit NF4 quantization + LoRA adapters) |
| **Trainer** | TRL `SFTTrainer` |
| **Hardware** | Google Colab, NVIDIA T4 (16GB) |

---

## 📊 Dataset

A custom instruction-style dataset built from PPC and CrPC sections:

- **686** training examples / **172** test examples
- Format: `instruction`, `input`, `output`
  - `instruction` — task prompt (e.g. *"Simplify the following legal text into plain English."*)
  - `input` — original legal section text
  - `output` — plain-English simplified version

**Example:**
```yaml
instruction: Simplify the following legal text into plain English.
input: >
  Section 1 of the Pakistan Penal Code (PPC): Title and extent of operation
  of the Code. This Act shall be called the Pakistan Penal Code, and shall
  take effect throughout Pakistan.
output: >
  This section just says the law is officially named the "Pakistan Penal
  Code" and applies everywhere in the country.
```

> ⚠️ **Note:** swap in one of your *actual* simplified outputs here — the placeholder above is illustrative only, since a great README example is often the first thing a recruiter screenshots.

Before training, the dataset is converted into conversational format (`messages` with `user`/`assistant` roles) so it's compatible with the tokenizer's chat template and `SFTTrainer`.

---

## ⚙️ Training Configuration

- **Quantization:** 4-bit NF4, double quantization, fp16 compute dtype (T4-compatible)
- **LoRA:** rank 16, alpha 32, dropout 0.05 — applied to attention + MLP projection layers
- **Optimizer:** `paged_adamw_8bit`
- **Epochs:** 3
- **Effective batch size:** 8 (batch size 2 × gradient accumulation 4)
- **Max sequence length:** tuned from the dataset's token-length distribution

---

## 📁 Project Structure

```
.
├── sft_legal_simplification.csv      # training dataset
├── pakistan_penal_code.csv           # source PPC data
├── crpc.csv                          # source CrPC data
├── train.ipynb                       # training notebook (QLoRA + SFTTrainer)
├── app.py / gradio_demo.ipynb        # Gradio inference demo
└── qwen2.5-3b-legal-simplification/  # saved LoRA adapter + tokenizer
```

---

## 🚀 Usage

### Load the fine-tuned model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "qwen2.5-3b-legal-simplification"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
```

### Run inference

```python
messages = [{
    "role": "user",
    "content": "Simplify the following legal text into plain English.\nSection 5 of the Pakistan Penal Code..."
}]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### Run the Gradio demo

```bash
python app.py
```

Or in a notebook:

```python
demo.launch(share=True)
```

This launches an interactive UI where you can paste any PPC/CrPC section and get a simplified version back, with adjustable generation length and temperature.

---

## 📈 Results & Limitations

> *Fill this in with your actual numbers — recruiters and engineers weight this section heavily.*

- Qualitative comparison of base Qwen2.5-3B vs. fine-tuned outputs on held-out test sections
- Optional: ROUGE/BLEU score against reference simplifications, or a small human-eval rubric (faithfulness vs. base model)
- Known limitations — e.g. occasional loss of legal nuance, performance on sections outside PPC/CrPC, hallucinated clauses on edge cases

---

## 🙏 Acknowledgements

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba Cloud
- [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft) by Hugging Face
- [Gradio](https://www.gradio.app/) for the demo interface

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
