# MAQA-LLaMA — Arabic Medical Chatbot Fine-tuning

<p align="center">
  <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="100"/>
</p>

<p align="center">
  <a href="https://huggingface.co/AliAbdelrasheed/maqa_llama"><img src="https://img.shields.io/badge/🤗%20HuggingFace-maqa__llama-blue"/></a>
  <a href="https://huggingface.co/AliAbdelrasheed/maqa_llama_4bit"><img src="https://img.shields.io/badge/🤗%20HuggingFace-maqa__llama__4bit-blue"/></a>
  <a href="https://huggingface.co/AliAbdelrasheed/maqa_llama_4bit_GGUF"><img src="https://img.shields.io/badge/🤗%20HuggingFace-maqa__llama__4bit__GGUF-blue"/></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue"/>
</p>

> ⚠️ **Disclaimer:** This project is intended for research and educational purposes only.
> The models are **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## Overview

This repository contains the training notebook and documentation for fine-tuning
**Meta Llama 3 8B Instruct** on the **MAQA dataset** (Medical Arabic Questions & Answers)
to build an Arabic-language medical question-answering assistant.

The project was developed as a **graduation project at Nile University** (B.Sc. IT – Big Data, 2024),
motivated by the lack of accessible, reliable Arabic-language healthcare information for
Arabic-speaking communities.

### Published Models

| Model | Format | Size | Link |
|---|---|---|---|
| `maqa_llama` | BF16 (full precision) | ~16 GB | [🤗 HuggingFace](https://huggingface.co/AliAbdelrasheed/maqa_llama) |
| `maqa_llama_4bit` | 4-bit bitsandbytes | ~5 GB | [🤗 HuggingFace](https://huggingface.co/AliAbdelrasheed/maqa_llama_4bit) |
| `maqa_llama_4bit_GGUF` | GGUF q4_k_m | 4.92 GB | [🤗 HuggingFace](https://huggingface.co/AliAbdelrasheed/maqa_llama_4bit_GGUF) |

---

## Background & Motivation

Arabic-speaking communities face significant barriers to healthcare access:
- Language barriers when seeking medical information online
- Limited availability of Arabic-language medical resources
- Reliance on unverified sources leading to misinterpretation of symptoms

This project aimed to address this by building a chatbot grounded in **real doctor-patient
interactions** rather than synthetic data, using the largest available Arabic medical dataset.

---

## Dataset — MAQA

**MAQA** (Medical Arabic Questions & Answers) is the largest Arabic medical Q&A dataset
available for NLP research.

| Property | Value |
|---|---|
| Total records | **430,000** question-answer pairs |
| Columns | `question` (patient) · `answer` (doctor diagnosis + treatment notes) |
| Medical specialisations | **20** |
| Sources | altibbi.com · tbeeb.net · cura.healthcare |
| Language | Arabic (Modern Standard Arabic) |
| All questions | Unique and cleaned |

> Reference: *"Deep learning for Arabic healthcare: MedicalBot"* — Social Network Analysis
> and Mining, Springer (2023).
> Available at [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Y2JBEZ).

A **sampled subset** of the full dataset (`sampled_maqa.csv`) was used for fine-tuning,
split 70% training / 30% evaluation.

---

## Development Journey

The project went through multiple iterations before arriving at the final fine-tuned models:

### Phase 1 — RAG (Retrieval-Augmented Generation)
- Attempted to build a retrieval pipeline using **LangChain** and **FAISS** as a vector store
- Embedded MAQA data into a vector database for similarity-search-based retrieval
- **Outcome:** Functional for prototype testing but limited in conversational generalisation;
  approach was abandoned in favour of full fine-tuning

### Phase 2 — LoRA Fine-tuning with Unsloth
- Moved to **direct supervised fine-tuning** using QLoRA (Low-Rank Adaptation)
- Used the [Unsloth](https://github.com/unslothai/unsloth) framework for 2x faster training
  and 30% less VRAM usage on Google Colab Pro
- Published 3 model variants to HuggingFace covering full-precision, 4-bit GPU, and CPU-compatible formats

---

## Training Details

### Model & Framework

| Component | Value |
|---|---|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` (Meta Llama 3 8B Instruct) |
| Fine-tuning method | QLoRA |
| Framework | [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace TRL (SFTTrainer) |
| Training environment | Google Colab Pro |
| Chat template | Llama-3 (ShareGPT style) |

### LoRA Configuration

| Hyperparameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Gradient checkpointing | `unsloth` (30% less VRAM) |
| Random seed | 3407 |

### Training Arguments

| Hyperparameter | Value |
|---|---|
| Epochs | 1 |
| Batch size (per device) | 52 |
| Learning rate | 2e-4 |
| LR scheduler | Linear |
| Warmup steps | 200 |
| Optimiser | AdamW 8-bit |
| Max sequence length | 2048 |
| Evaluation | Every 500 steps |

### System Prompt

The model was fine-tuned with the following Arabic system prompt:

```
أنت طبيب محترف ولديك خبرة في كل مجالات الطب.
يجيب على أسئلة المرضى حول الأمراض، باستخدام لهجة رسمية وودية،
وإجابات موجزة ومفيدة يسهل على الجميع فهمها.
```

*Translation: "You are a professional doctor with expertise in all fields of medicine.
Answer patients' questions about diseases using a formal and friendly tone,
with concise and helpful answers that everyone can understand."*

Special tokens `<|question|>` and `<|answer|>` were added to the tokenizer
to structure patient input and doctor response during training.

---

## Repository Structure

```
maqa-llama-finetuning/
│
├── Llama_3_8b_Instruct_Unsloth_2x_faster_finetuning.ipynb   # Main training notebook
├── README.md                                                  # This file
└── docs/
    └── graduation_project_report.pdf                         # Full project documentation
```

---

## How to Reproduce

### Requirements

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
pip install pandas scikit-learn datasets
```

### Steps

1. Download the MAQA dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Y2JBEZ)
2. Sample or use the full dataset, save as `sampled_maqa.csv` with columns `question` and `answer`
3. Upload to Google Drive and update the path in Cell 8 of the notebook
4. Run all cells in `Llama_3_8b_Instruct_Unsloth_2x_faster_finetuning.ipynb`
5. Models will be pushed to your HuggingFace account via `push_to_hub_merged` and `push_to_hub_gguf`

> **Note:** A Google Colab Pro subscription is recommended for sufficient GPU memory
> (the notebook was trained with batch size 52 which requires significant VRAM).

---

## Inference Examples

### Example Question
**Input (Arabic):**
> أشعر بألم أسفل البطن بخاصرتي والألم يجي على فترات، ما الأسباب؟

*"I feel pain in my lower abdomen near my side, and the pain comes intermittently. What are the causes?"*

---

## Limitations

- Not a substitute for professional medical advice or clinical diagnosis
- Cannot prescribe or recommend specific medications
- Trained on a sampled subset of MAQA; coverage across all 20 specialisations may be uneven
- Optimised for Modern Standard Arabic; Egyptian/Levantine dialect performance may vary
- Web-scraped training data may contain noise or outdated medical information

---

## Future Work

- Fine-tuning on the **full 430,000-row** MAQA dataset
- Multi-turn conversational memory for sustained dialogue
- Dialect-specific variants (Egyptian Arabic, Gulf Arabic)
- Multimodal input (e.g. skin condition images)
- Integration with electronic medical records (EMR) systems
- Mobile application via Flutter

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — for making LLM fine-tuning accessible on limited hardware
- [HuggingFace TRL](https://github.com/huggingface/trl) — SFTTrainer
- MAQA dataset authors — *"Deep learning for Arabic healthcare: MedicalBot"*, Springer (2023)
- Nile University, School of Information Technology and Computer Science

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## Author

**Ali Abdelrasheed**
Nile University · B.Sc. Information Technology – Big Data · Class of 2024
🤗 [HuggingFace](https://huggingface.co/AliAbdelrasheed) · 💻 [GitHub](https://github.com/finalbro22)
