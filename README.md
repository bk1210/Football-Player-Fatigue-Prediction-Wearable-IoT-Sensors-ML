# 🏃 Football Player Fatigue Prediction — Wearable IoT Sensors & ML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow)
![Kaggle](https://img.shields.io/badge/Kaggle-P100_GPU-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A machine learning framework that predicts three-class fatigue states in football players from wearable IoT sensor data — with Karvonen heart rate labeling, SMOTE balancing, LOSO validation, and a real-time coach substitution-alert dashboard.**

[Features](#-features) • [How It Works](#-how-it-works) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack) • [Contact](#-contact)

</div>

---

## 📌 Overview

**Football Player Fatigue Prediction** repurposes the PAMAP2 physical activity dataset for sports-specific fatigue monitoring. Instead of binary high/low labels, this system introduces a **three-class Karvonen-based labeling scheme** (LOW / MEDIUM / HIGH) grounded in cardiac physiology — with zero data leakage into model training.

The project offers three core contributions:
- 🏷️ **Karvonen Auto-Labeler** — assigns fatigue states from heart rate reserve % and movement intensity, then discards intermediates before training to prevent leakage
- 🌲 **Personalized Random Forest** — per-subject model fine-tuning that handles large inter-individual physiological variation
- ⚽ **Coach Dashboard** — 90-minute match simulation with real-time fatigue phase tracking and substitution alerts

---

## ✨ Features

### 🏷️ Karvonen Fatigue Labeling
- Heart rate reserve % computed via the Karvonen formula (HRrest=60, HRmax=195)
- Movement intensity as Euclidean norm of chest accelerometer
- Three classes: LOW (light cardiac load + high movement), HIGH (cardiac drift), MEDIUM (otherwise)
- Intermediate features discarded after labeling — **fully leakage-free**

### 📊 Sliding Window Feature Extraction
- 512-sample windows (~5.12s at 100 Hz), 50% overlap
- 11 statistical descriptors per channel: mean, std, min, max, median, Q1/Q3, RMS, range, MAD, energy
- 8 channels × 11 features = **88 features per window**
- Majority vote label assignment per window

### ⚖️ SMOTE Class Balancing
- Original distribution heavily skewed toward MEDIUM (83.4%)
- SMOTE (k=3) applied exclusively on training set
- Balanced to 451 samples per class (1,353 total)

### 🌲 Four Models Evaluated
- **Random Forest** — 200 trees, max depth 20, balanced class weights
- **SVM (RBF kernel)** — C=10, gamma=scale, balanced class weights
- **LSTM** — 2-layer (64→32 units), BatchNorm + Dropout, early stopping
- **Personalized RF** — dedicated per-subject model with subject-level SMOTE

### 🔍 LOSO Cross-Validation
- LeaveOneGroupOut with subject IDs as groups
- Subject-independent generalization evaluation
- Exceeds original PAMAP2 benchmark (97.96% vs ~88%)

### ⚽ Coach Decision Support Dashboard
- 90-minute match simulation with phase-by-phase fatigue profiling
- LOW → MEDIUM transition through first half, HIGH accumulation in final 20 minutes
- Visual heart rate, movement intensity, and fatigue class timeline
- Substitution alert triggered when HIGH fatigue detected

---

## 🖥️ Demo

### Fatigue Prediction
```
Window  → HR: 158bpm | Accel norm: 8.2 | Over 70 mins played
Output  → 🔴 HIGH Fatigue — Substitution Recommended
```

### Personalized vs Global (Subject 3)
```
Global RF (LOSO) → F1: 48.3%
Personalized RF  → F1: 100.0%   (+51.7 pp improvement)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- GPU recommended (Kaggle P100)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/bk1210/football-fatigue-prediction.git
cd football-fatigue-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the Dataset
Get the PAMAP2 dataset from UCI:
👉 [PAMAP2 Physical Activity Monitoring Dataset](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

Place the subject `.dat` files inside a `PAMAP2_Dataset/Protocol/` folder in the project root.

### Step 4 — Run the Notebook
```bash
jupyter notebook iot_fatigue.ipynb
```

Or upload directly to **Kaggle** and run with P100 GPU.

---

## 📖 Usage

### Running the Full Pipeline

Open `iot_fatigue.ipynb` and run all cells — the notebook handles:

1. Data loading from PAMAP2 `.dat` files (9 subjects, 100 Hz)
2. Preprocessing — remove transitional segments, interpolate missing values, clip HR to [30, 220]
3. Karvonen-based three-class fatigue labeling (leakage-free)
4. Sliding window feature extraction (88 features per window)
5. SMOTE balancing on training set
6. Training and evaluating all four models (RF, SVM, LSTM, Personalized RF)
7. LOSO cross-validation across all 9 subjects
8. Feature importance analysis
9. 90-minute football match simulation + coach dashboard

---

## 🏗️ Architecture

### End-to-End Pipeline

```
PAMAP2 Raw Data (9 subjects, 100 Hz, 54 channels)
    │
    ▼
Preprocessing
[Remove activityID=0 | Interpolate NaN | Clip HR to 30-220 bpm]
    │
    ▼
Fatigue Label Creation (Karvonen Formula) ← Core Contribution
    ├─► HRR% = (HR - HRrest) / (HRmax - HRrest)
    └─► Movement Intensity = √(ax² + ay² + az²)
         │
         ▼
    LOW / MEDIUM / HIGH  →  intermediates discarded (leakage-free)
    │
    ▼
Sliding Window Feature Extraction
[512 samples | 50% overlap | 8 channels × 11 stats = 88 features]
    │
    ▼
SMOTE Balancing (training set only, k=3)
    │
    ├──► Random Forest (200 trees, depth 20)
    ├──► SVM RBF (C=10, gamma=scale)
    ├──► LSTM (64→32 units, BatchNorm, Dropout)
    └──► Personalized RF (per-subject, 80/20 split)
    │
    ▼
Evaluation: 70/30 Test Split + LOSO Cross-Validation
    │
    ▼
Coach Dashboard — 90-min Match Simulation + Substitution Alerts
```

### Project Structure

```
football-fatigue-prediction/
│
├── iot_fatigue.ipynb                    # Full pipeline — labeling, training, LOSO, dashboard
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## 📊 Results

### Model Comparison — 70/30 Test Split

| Model | Accuracy | F1-Macro |
|---|---|---|
| Random Forest | 97.56% | 76.23% |
| SVM (RBF) | 94.15% | 63.96% |
| LSTM | 91.22% | 57.85% |
| **Personalized RF** | **97.87%** | **79.93%** |

### LOSO Cross-Validation vs Baseline

| Method | LOSO Accuracy | LOSO F1-Macro |
|---|---|---|
| Reiss & Stricker (PAMAP2 Benchmark) | ~88.0% | — |
| **This Work (RF)** | **97.96% ± 2.57%** | **87.55% ± 18.69%** |

### Comparison with Base Papers

| Criterion | Krishnaleela & Prakash (2025) | Liu et al. (2023) | **This Work** |
|---|---|---|---|
| Dataset | PAMAP2 | Own (proprietary) | PAMAP2 |
| Task | HAR (18 classes) | Fatigue (binary) | Fatigue (3-class) |
| SMOTE | No | No | ✅ Yes |
| LOSO | No | No | ✅ Yes |
| Personalized | No | No | ✅ Yes |
| Performance | 99.86% Acc | 94.15% Acc | **97.87% Acc / 79.93% F1** |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| scikit-learn | RF, SVM, SMOTE, LOSO, metrics |
| TensorFlow 2.19 | LSTM model training |
| imbalanced-learn | SMOTE oversampling |
| NumPy / Pandas | Feature engineering, sliding window |
| Matplotlib / Seaborn | EDA, training curves, dashboard |
| Kaggle (P100 GPU) | Training environment |

---

## 📦 Dependencies

```txt
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
tensorflow>=2.19.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- [ ] Collect a football-specific dataset with GPS + continuous HR from professional players
- [ ] Raw 100 Hz stream modeling with CNN-BiLSTM on full time-series
- [ ] On-device edge inference using optimized BiLSTM
- [ ] Multi-task learning — joint activity type + fatigue state prediction
- [ ] Prospective validation linking HIGH-fatigue episodes to injury occurrence
- [ ] Real-time MQTT streaming pipeline for live match deployment

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Bharath Kesav R**
- 📧 Email: bharathkesav1275@gmail.com
- 🐙 GitHub: [@bk1210](https://github.com/bk1210)
- 🎓 Institution: Amrita Vishwa Vidyapeetham, Coimbatore

---

## 🙏 Acknowledgements

- [Reiss & Stricker (2012)](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) — for the PAMAP2 dataset
- [Liu et al. (2023)](https://doi.org/10.1007/s00779-022-01688-w) — Base paper on wearable fatigue detection
- [Op de Beeck et al. (2018)](https://dl.acm.org/doi/10.1145/3219819.3219847) — Personalization motivation

---

<div align="center">

**⭐ If you found this project useful, please give it a star on GitHub! ⭐**

*Built with ❤️ for football/sports analystics and data analytics *

</div>
