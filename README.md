# 🧠 Scaling Bottlenecks in Large Language Models  
### Comparative Study: Dense Transformers vs. Sparse Mixture-of-Experts (MoE)

This project investigates how **sparse Mixture-of-Experts (MoE)** layers compare to **dense transformer architectures** in terms of inference latency, memory usage, and qualitative output behavior.  
The experiments were conducted as part of the *Scientific Writing* course at the **University of Basel (HS 2025)**.

---

## ⚙️ Methodology Overview

The study explores whether performance and efficiency differences between **dense** and **sparse** transformer architectures can be observed in a **controlled small-scale setting**.

Two models were implemented from scratch using **PyTorch**:

1. **Dense baseline** – 3-layer encoder-only transformer with standard feed-forward blocks.  
2. **Sparse MoE variant** – identical to the baseline, except the second feed-forward block is replaced by a **Mixture-of-Experts layer** with four experts and **Top-2 routing**.

Both models include a next-token prediction head for basic language-modeling tasks.  
No training was performed — all comparisons are based on **inference-time behavior**.

---

## 📊 Dataset and Experiment Setup

- **Dataset:** 1,000 abstracts from the **arXiv Computer Science** category (via Hugging Face).  
- **Preprocessing:** Tokenized with *bert-base-uncased*, sequence length fixed to 128 tokens.  
- **Hardware:** HP Omen 14 (Intel Core Ultra 9, 32 GB RAM, RTX 4070 GPU).  
- **Software:** PyTorch, Ubuntu VM, IntelliJ IDEA.

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **Latency per token** | Average inference time per token over 10 runs |
| **Peak GPU memory** | Maximum allocated memory recorded with `torch.cuda.max_memory_allocated()` |
| **Parameter activation** | Number of active parameters per token (MoE: Top-2 experts) |

A small **qualitative output analysis** was also performed by comparing generated next-token predictions from both models for diverse input prompts.

---

## ⚠️ Limitations
- Experiments are **small-scale** and not representative of production-level models.  
- The MoE implementation omits advanced optimizations like load-balancing loss or expert dropout.  
- Qualitative analysis is **exploratory** and based on a small set of inputs.

---

## 🎯 Goal
To provide a reproducible, interpretable example of how **sparse expert routing** affects compute efficiency and model behavior — contributing to the discussion around the **scaling limits of LLMs** and the **trade-offs of sparse architectures**.

---

## 📚 Author
**Manuel Buser**  
Master’s Student in Computer Science (Machine Intelligence)  
University of Basel  
[LinkedIn](https://www.linkedin.com/in/manuel-buser) • [GitHub](https://github.com/manuel-buser)
