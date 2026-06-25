Legal Text Simplifier — Qwen2.5-3B + QLoRA

Fine-tuned Qwen2.5-3B-Instruct to simplify dense legal language from the Pakistan Penal Code (PPC) and Code of Criminal Procedure (CrPC) into plain, accessible English — making legal text understandable to non-lawyers.

Overview

Legal documents like the PPC and CrPC are written in formal, archaic legal language that's hard for the average person to understand. This project fine-tunes a small open-source LLM to rewrite legal sections in plain English while preserving their original meaning, using parameter-efficient fine-tuning (QLoRA) so the whole pipeline runs on a single free-tier GPU (Colab T4).

Features


Fine-tuned on a custom instruction dataset built from PPC/CrPC sections
4-bit quantized base model (QLoRA) — trains on a 16GB GPU
LoRA adapters for efficient, lightweight fine-tuning
Interactive Gradio demo for real-time testing
Built on Hugging Face transformers, trl, and peft


Model

Base modelQwen/Qwen2.5-3B-InstructFine-tuning methodQLoRA (4-bit NF4 quantization + LoRA adapters)TrainerTRL SFTTrainerHardware usedGoogle Colab, NVIDIA T4 (16GB)

Dataset

A custom instruction-style dataset built from Pakistan Penal Code and Code of Criminal Procedure sections:


686 training examples / 172 test examples
Format: instruction, input, output

instruction: task prompt (e.g. "Simplify the following legal text into plain English.")
input: original legal section text
output: plain-English simplified version





Example:

instruction: Simplify the following legal text into plain English.
input: Section 1 of the Pakistan Penal Code (PPC): Title and extent of operation
       of the Code. This Act shall be called the Pakistan Penal Code, and shall
       take effect throughout Pakistan.
output: Section 1 of the Pakistan Penal Code (PPC): Title and extent of operation
        of the Code. This Act shall be called the Pakistan Penal Code, and shall
        take effect throughout Pakistan.

Before training, the dataset is converted into conversational format (messages with user/assistant roles) so it works with the tokenizer's built-in chat template and SFTTrainer.

Training Configuration


Quantization: 4-bit NF4, double quantization, fp16 compute dtype (T4-compatible)
LoRA: rank 16, alpha 32, dropout 0.05, applied to attention + MLP projection layers
Optimizer: paged_adamw_8bit
Epochs: 3
Effective batch size: 8 (batch size 2 × gradient accumulation 4)
Max sequence length: tuned based on dataset token-length distribution


Project Structure

.
├── sft_legal_simplification.csv     # training dataset
├── pakistan_penal_code.csv          # source PPC data
├── crpc.csv                         # source CrPC data
├── train.ipynb                      # training notebook (QLoRA + SFTTrainer)
├── app.py / gradio_demo.ipynb        # Gradio inference demo
└── qwen2.5-3b-legal-simplification/  # saved LoRA adapter + tokenizer

Usage

Load the fine-tuned model

pythonimport torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "qwen2.5-3b-legal-simplification"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

Run inference

pythonmessages = [{
    "role": "user",
    "content": "Simplify the following legal text into plain English.\nSection 5 of the Pakistan Penal Code..."
}]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))

Run the Gradio demo

bashpython app.py

Or in a notebook:

pythondemo.launch(share=True)

This launches an interactive UI where you can paste any PPC/CrPC section and get a simplified version back, with adjustable generation length and temperature.

Acknowledgements


Qwen2.5 by Alibaba Cloud
TRL and PEFT by Hugging Face
Gradio for the demo interface


License

Specify your license here (e.g. MIT).
