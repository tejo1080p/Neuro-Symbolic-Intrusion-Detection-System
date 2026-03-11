# Hybrid Neuro-Symbolic Intrusion Detection System

### Bridging deep learning accuracy with symbolic explainability in cybersecurity

![Status](https://img.shields.io/badge/Research-Active-0ea5e9?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-Cybersecurity-ef4444?style=for-the-badge)
![AI](https://img.shields.io/badge/AI-Neuro--Symbolic-8b5cf6?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-22c55e?style=for-the-badge)

---

## Project Overview

Traditional Intrusion Detection Systems (IDS) face a difficult trade-off:

- high predictive performance
- high interpretability

Deep learning detects complex attacks but is often a black box.
Rule-based systems are interpretable but struggle with complex traffic behavior.

This project introduces a **Hybrid Neuro-Symbolic IDS** that integrates:

- 🧠 **CNN Neural Layer** for deep traffic pattern recognition
- ⚙️ **XGBoost Layer** for robust tabular classification
- 📜 **Symbolic Rule Layer** for human-readable security reasoning

Result: an IDS that is both **accurate** and **explainable**.

---

## System Architecture

### Three-layer architecture

- **Neural Component** -> CNN traffic pattern detection
- **Machine Learning Component** -> XGBoost classifier
- **Symbolic Component** -> Rule-based interpretability engine

### Hybrid fusion decision

Final prediction is generated through weighted fusion of:

- CNN confidence
- XGBoost probability
- symbolic rule confidence

### Conceptual architecture diagram

```text
NSL-KDD Network Features
        |
 +------+------+ 
 |             |
CNN        XGBoost
 |             |
 +------+------+ 
        |
 Symbolic Rule Engine
        |
 Hybrid Neuro-Symbolic Fusion
        |
 Attack / Normal + Explanation
```

---

## Key Features

- 🧠 Hybrid neuro-symbolic AI architecture
- 🔍 Explainable intrusion detection decisions
- 📈 CNN-based deep traffic pattern learning
- ⚡ XGBoost tabular feature modeling
- 📜 Symbolic rule-based intrusion reasoning
- 🧩 SHAP global feature explainability
- 🧪 Explainability metrics (coverage, complexity, fidelity)
- 🔁 Reproducible notebook-based research pipeline

---

## Dataset

**NSL-KDD intrusion detection dataset**

### Dataset highlights

- protocol, service, and flag features
- connection statistics and host-level traffic attributes
- anomaly indicators such as `serror_rate`, `rerror_rate`, and related metrics

### Binary target

- `0` -> Normal traffic
- `1` -> Intrusion attack

### Preprocessing

The pipeline handles loading, cleaning, mixed-feature transformation, and reproducible train-test split.

---

## Project Pipeline

1. Data Analysis
2. Data Preprocessing
3. Baseline Model Training
4. CNN Deep Learning Training
5. Symbolic Rule Extraction
6. Hybrid Neuro-Symbolic Fusion
7. Explainability Evaluation
8. Model Comparison and Visualization

---

## Models Implemented

### Baseline models

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- SVM
- MLP Neural Network

### Deep models

- CNN Model 1 - Basic
- CNN Model 2 - Deep
- CNN Model 3 - Advanced

### Hybrid model

- Hybrid Neuro-Symbolic IDS (CNN + XGBoost + symbolic reasoning)

---

## Evaluation Metrics

### Predictive metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- False Positive Rate (FPR)
- Brier Score

### Explainability metrics

- Rule Coverage
- Rule Complexity
- Explanation Fidelity
- SHAP Feature Importance

---

## Results Snapshot

### 🚀 Performance badges (Hybrid)

![Hybrid Accuracy](https://img.shields.io/badge/Hybrid%20Accuracy-0.9949-22c55e?style=flat-square)
![Hybrid F1](https://img.shields.io/badge/Hybrid%20F1-0.9947-16a34a?style=flat-square)
![Hybrid ROC-AUC](https://img.shields.io/badge/Hybrid%20ROC--AUC-0.9999-15803d?style=flat-square)
![Hybrid PR-AUC](https://img.shields.io/badge/Hybrid%20PR--AUC-0.9999-166534?style=flat-square)

### 📊 Baseline and component results (from notebook outputs)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Brier Score |
|------|---------:|----------:|-------:|---:|--------:|-------:|----:|------------:|
| Best Baseline (Balanced Random Forest) | 0.9622 | 0.9945 | 0.9266 | 0.9594 | 0.9965 | 0.9965 | 0.0047 | 0.0265 |
| XGBoost | 0.9956 | 0.9970 | 0.9938 | 0.9954 | 0.9999 | 0.9999 | 0.0028 | 0.0031 |
| CNN (Deep) | 0.9896 | 0.9887 | 0.9896 | 0.9892 | 0.9995 | 0.9995 | 0.0104 | 0.0076 |

### 🧬 Hybrid model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Brier Score |
|------|---------:|----------:|-------:|---:|--------:|-------:|----:|------------:|
| XGBoost Layer | 0.9956 | 0.9970 | 0.9938 | 0.9954 | 0.9999 | 0.9999 | 0.0028 | 0.0031 |
| CNN Layer | 0.9896 | 0.9887 | 0.9896 | 0.9892 | 0.9995 | 0.9995 | 0.0104 | 0.0076 |
| **Hybrid Neuro-Symbolic IDS** | **0.9949** | **0.9982** | **0.9912** | **0.9947** | **0.9999** | **0.9999** | **0.0017** | **0.0188** |

---

## Explainability Framework

### Rule-based explainability

Symbolic rules provide human-readable reasoning for attack decisions.

```text
IF serror_rate > 0.7 AND srv_serror_rate > 0.7
THEN Possible SYN Flood Attack
```

### SHAP explainability

SHAP explains which features most influence XGBoost intrusion predictions.

Top discovered SHAP features include:

- `numeric__src_bytes`
- `categorical__service_http`
- `numeric__dst_bytes`
- `numeric__count`

### Explainability metrics (measured)

| Metric | Value |
|--------|------:|
| Rule Coverage | 46.36% |
| Average Rule Length | 3.27 |
| Max Rule Length | 4 |
| Explanation Fidelity | 0.9670 |

---

## Explainability Demonstration

✅ Example alert rationale:

Sample classified as **Attack** because:

- CNN detected anomalous traffic pattern
- XGBoost predicted high attack probability
- Symbolic rules fired for suspicious `serror_rate` and `srv_serror_rate`

This supports faster analyst triage and incident response.

---

## Repository Structure

```text
ShockAwareNeuroSymbolic/
|
|-- data/
|   |-- raw/
|   |   |-- archive(2)/
|   |   `-- nsl-kdd/
|   `-- processed/
|       `-- nsl_kdd_processed.csv
|
|-- notebooks/
|   |-- 01_data_analysis.ipynb
|   |-- 02_baseline_models.ipynb
|   `-- 03_neuro_symbolic_model.ipynb
|
|-- src/
|   |-- preprocessing.py
|   |-- baseline_models.py
|   |-- evaluation.py
|   |-- symbolic_rules.py
|   `-- nsai_model.py
|
|-- models/
|   |-- *.joblib
|   `-- cnn/
|
`-- README.md
```

---

## Installation

### 1) Clone repository

```bash
git clone https://github.com/<your-username>/ShockAwareNeuroSymbolic.git
cd ShockAwareNeuroSymbolic
```

### 2) Create virtual environment

```bash
python -m venv .venv
```

### 3) Activate environment

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 4) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

---

## Usage

Run notebooks in sequence:

1. `notebooks/01_data_analysis.ipynb`
2. `notebooks/02_baseline_models.ipynb`
3. `notebooks/03_neuro_symbolic_model.ipynb`

Launch Jupyter:

```bash
jupyter notebook
```

---

## Future Work

- ⚡ Real-time streaming intrusion detection
- 🌐 Larger modern datasets beyond NSL-KDD
- 🕸️ Graph neural networks for traffic topology learning
- 🤖 Automated symbolic rule induction and adaptation

---

## License

MIT License (placeholder)

---

## Acknowledgments

- NSL-KDD dataset
- scikit-learn
- XGBoost
- PyTorch / TensorFlow
- SHAP explainability framework

---

## Citation

If you use this repository in research, please cite this project and the NSL-KDD dataset.
