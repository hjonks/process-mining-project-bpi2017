# Process Mining: Dutch Bank Loan Application Analysis
### BPI Challenge 2017 · Python · pm4py · scikit-learn · Power BI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi&logoColor=black)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

End-to-end process mining analysis of **1,202,267 real event logs** from a Dutch bank's loan application process. The project identifies operational bottlenecks, predicts SLA breaches using machine learning, and quantifies process deviation — producing an executive-ready Power BI dashboard.

> **[View Live Dashboard →](https://app.powerbi.com/YOUR_LINK_HERE)**

---

## Key Findings

| Finding | Result |
|---|---|
| Dataset | 1,202,267 events · 31,509 loan applications |
| Primary bottleneck | W_Call after offers — **30.1h** mean queue time |
| Cumulative delay | **238,000+ days** of total process time lost |
| Rework prevalence | **100%** of cases contain activity repetitions |
| Redundant events | **732,259** wasted activity occurrences |
| ML model (AUC) | **0.83** — Gradient Boosting classifier |
| Top SLA predictor | Process complexity (`n_unique_activities`) |
| Process conformance | **0%** — zero cases follow the ideal path |
| SLA breach rate | **64.1%** of non-conformant cases miss the 14-day deadline |

---

## Repository Structure

```
Process_Mining_Project/
│
├── data/
│   ├── raw/
│   │   └── BPI Challenge 2017.xes
│   └── processed/
│       ├── event_log_cleaned.csv
│       └── case_features.csv
│
├── results/
│   ├── figures/
│   │   ├── eda_overview.png
│   │   ├── process_variants.png
│   │   ├── bottleneck_analysis.png
│   │   └── ml_performance.png
│   ├── tables/
│   │   ├── bottleneck_summary.csv
│   │   ├── case_features_enriched.csv
│   │   └── feature_importance.csv
│   └── reports/
│       ├── process_mining_summary.txt
│       └── bottleneck_summary.txt
│
├── scripts/
│   ├── paths.py
│   ├── day1_eda_discovery.py
│   └── day2_bottleneck_ml.py
│
└── powerbi/
    └── Process_Mining_Dashboard.pbix
```

---

## 📊 Methodology

### EDA & Process Discovery
- Loaded and cleaned 1.2M+ event log entries from `.xes` format via `pm4py`
- Engineered case-level features: duration, SLA breach flag, outcome classification
- Process variant analysis — identified all unique activity sequences across 31,509 cases
- Established baseline KPIs: median duration, breach rate, outcome distribution

### Bottleneck, ML & Conformance

**Bottleneck Analysis**
Calculated inter-activity waiting time for every transition. Identified `W_Call after offers` as primary bottleneck: 195,081 occurrences × 30.1h average = **238,000+ days** cumulative delay.

**Rework Detection**
Found **732,259 rework events** across 100% of cases — every application is revisited at least once.

**Root Cause ML Model**
Best model: **Gradient Boosting (AUC: 0.83)**. `n_unique_activities` is the strongest SLA breach predictor.

**Conformance Checking**
0% of cases follow the normative process. Non-conformant cases face a 64.1% SLA breach rate.

---

## 📈 Visualisations

### EDA Overview
![EDA Overview](results/figures/eda_overview.png)

### Process Variant Analysis
![Variants](results/figures/process_variants.png)

### Bottleneck Analysis
![Bottlenecks](results/figures/bottleneck_analysis.png)

### ML Model Performance
![ML](results/figures/ml_performance.png)

---

## How to Run

```bash
# Install dependencies
python -m pip install pm4py pandas numpy matplotlib seaborn scikit-learn scipy openpyxl

# Place BPI Challenge 2017.xes in data/raw/
# Download from: https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884

cd scripts
python eda_discovery.py
python bottleneck_ml.py
```

---

## Tech Stack

`Python` · `pm4py` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn` · `Power BI`

---

## 💼 Business Recommendations

1. **Redesign W_Call after offers** — 238,000+ days of cumulative delay from one activity
2. **Implement straight-through processing** — 100% rework rate signals broken intake validation
3. **Deploy SLA predictor at intake** — AUC 0.83 model enables proactive case escalation

---

## 📚 Dataset

van Dongen, B. (2017). *BPI Challenge 2017*. 4TU.ResearchData.
https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b

---

## Author

**Dhruv Chaudhary** · MSc Business Analytics & Decision Science · University of Leeds

[LinkedIn](https://www.linkedin.com/in/dhruv-chaudhary-16a1ba227/) · [GitHub](https://github.com/hjonks)
