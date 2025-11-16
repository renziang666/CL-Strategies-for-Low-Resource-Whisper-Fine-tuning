# A Continual Learning Approach for Adapting Whisper to Low-Resource Languages

This repository provides the official implementation for two novel continual learning (CL) approaches, **UGP** (an advanced AGEM) and **LaBER** (an advanced ER), designed to adapt the Whisper model to low-resource Southeast Asian languages.

It also includes the code for several baseline methods (Full Fine-tuning, LoRA, EWC, ER, AGEM, and PAPP) for comprehensive comparison.

## Table of Contents

- [A Continual Learning Approach for Adapting Whisper to Low-Resource Languages](#a-continual-learning-approach-for-adapting-whisper-to-low-resource-languages)
  - [Table of Contents](#table-of-contents)
  - [🔧 Installation](#-installation)
  - [🚀 Quick Start](#-quick-start)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Training](#2-training)
      - [Our Proposed Methods](#our-proposed-methods)
      - [Baseline Methods](#baseline-methods)
    - [3. Evaluation](#3-evaluation)
  - [💻 Hardware Requirements](#-hardware-requirements)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [📜 Citation](#-citation)

## 🔧 Installation

1.  Clone the repository:
    ```bash
    git clone github.com/renziang666/CL-Strategies-for-Low-Resource-Whisper-Fine-tuning
    
    ```

2.  Create and activate the Conda environment using the provided file:
    ```bash
    conda env create -f environment.yml
    conda activate your_env_name
    ```

## 🚀 Quick Start

### 1. Data Preprocessing

Run the following script to prepare your dataset:

```bash
python prepare_data/preprocess_data.py
````

**Note:** When using whisper-large-v3 models, ensure your data is processed to 1280 dimensions.

### 2\. Training

You can run our proposed methods or the baseline methods for comparison.

#### Our Proposed Methods

  * **UGP (Advanced AGEM):**
    ```bash
    python method/AGAM_new/AGEM_NEW_DEBUG_v2.py
    ```
  * **LaBER (Advanced ER):**
    ```bash
    python method/ER_new/finetune_ER_new.py
    ```

#### Baseline Methods

  * **Full Fine-tuning:**
    ```bash
    python method/full_finetune/full_finetune.py
    ```
  * **LoRA:**
    ```bash
    python method/lora/finetune_lora.py
    ```
  * **EWC (Elastic Weight Consolidation):**
    ```bash
    python method/EWC/finetune_EWC.py
    ```
  * **ER (Experience Replay):**
    ```bash
    python method/ER_new/finetune_ER_new.py
    ```
    > **Note:** You may need to adjust the loss weight for the target language when using ER.
  * **AGEM (Averaged Gradient Episodic Memory):**
    ```bash
    python method/AGEM_old/AGEM_DEBUG_large.py
    ```
  * **PAPP:**
    ```bash
    python method/PPAP/your_papp_script.py
    ```

### 3\. Evaluation

Use the corresponding script to evaluate your trained models.

  * **For all languages (except Thai):**
    ```bash
    python eval_wer/evaluate_wer.py
    ```
  * **For Thai (TH):**
    ```bash
    python eval_wer/eval_th.py
    ```
  * **For LoRA (specific script):**
    ```bash
    python method/lora/eval_lora.py
    ```

## 💻 Hardware Requirements

All experiments were developed and tested on a **single NVIDIA A40** GPU.

## 🙏 Acknowledgements

We would like to express our gratitude to the **SATLab, Department of Electronic Engineering, Tsinghua University** for their invaluable support, resources, and discussions that contributed to this project.

## 📜 Citation

If you find this work useful in your research, please consider citing our paper:


