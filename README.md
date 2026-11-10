# Pharmacy Medicine Demand & Expiry-Risk Forecasting

A machine learning decision-support system designed to forecast product-level medicine demand and evaluate inventory shelf-life risk across public health facilities.

---

## 1. Project Overview

In resource-constrained healthcare systems, medical supply chains face a persistent dual challenge:
1. **Critical Stockouts:** Sudden depletion of life-saving medicines (such as oxytocin, oral rehydration salts, and antibiotics) endangers patient care.
2. **Expired Inventory Wastage:** Overstocked medicines sit stagnant in clinic storerooms past their expiration date, leading to avoidable financial waste and disposal costs.

This project builds an end-to-end, leak-free machine learning pipeline and an interactive Streamlit decision-support dashboard. By evaluating routine health information records and central warehouse stock, the system provides **1-month-ahead demand forecasts** and identifies **batches at risk of stockout or expiration** under the **First Expiry, First Out (FEFO)** principle.

---

## 2. Data Sourcing & Attribution

The datasets used in this project are sourced from real-world global health research published on the **Dryad Digital Repository**:

- **Repository Link:** [Dryad Data Repository - Health Facility Demand & Logistics](https://datadryad.org/dataset/doi:10.5061/dryad.h9w0vt4tw?utm_source=chatgpt.com)
- **Dataset DOI:** [doi:10.5061/dryad.h9w0vt4tw](https://doi.org/10.5061/dryad.h9w0vt4tw)
- **License:** Creative Commons Zero (CC0 1.0 Universal) Public Domain Dedication.

### Dataset Descriptions

| File | Records | Key Columns | Description |
| :--- | :--- | :--- | :--- |
| `data/S2_Dhis2Data.csv` | 457,225 rows, 16 features | `hf_pk`, `name1`, `date`, `consumption`, `received`, `closeBalance`, `openBalance`, `stockout`, `facility_type`, `district`, `productID` | Routine monthly DHIS2 logistics reports from **1,091 public health facilities** across **16 districts** over **50 consecutive months** (Oct 2019 – Nov 2023) covering 36 essential medicines. |
| `data/S3.csv` | 341 rows, 3 features | `Item`, `Stock`, `Quarter` | Quarterly central medical warehouse stock inventory across 150 items spanning 4 quarters (2022Q4 to 2023Q3). |

### Data Integrity & Verification Note
Programmatic inspection confirmed that routine DHIS2 reports record aggregate facility-level balances and stockouts, but do not record manufacturer batch lot numbers or exact expiry dates. Rather than fabricating fake batch lot dates into real records:
1. On real data, shelf-life risk is assessed via **Months of Stock (MOS)** ($\text{Closing Inventory} / \text{Monthly Consumption}$), identifying overstocked products ($>6$ months MOS) at severe risk of shelf stagnation.
2. A separate, documented **FEFO (First Expiry, First Out) Batch Decision Prototype** is implemented to demonstrate batch-level precision, recall, and classification when warehouse management system (WMS) batch attributes are present.

### Privacy & Governance
All records represent aggregated facility-level commodity logistics. The data contains **zero Personally Identifiable Information (PII)** or **Protected Health Information (PHI)**.

---

## 3. Project Directory Structure

```text
Pharmacy-Medicine-Demand-and-Expiry-Risk-Forecasting/
├── data/
│   ├── S2_Dhis2Data.csv          # Real DHIS2 consumption dataset (50 months)
│   └── S3.csv                    # Central warehouse quarterly inventory
├── models/
│   ├── demand_model.pkl          # Serialized Random Forest Regressor artifact
│   └── expiry_risk_model.pkl     # Serialized FEFO Batch Classifier artifact
├── notebooks/
│   └── Pharmacy_Medicine_Demand_and_Expiry_Risk_Forecasting.ipynb  # Executed notebook
├── app.py                        # Interactive Streamlit decision-support application
├── requirements.txt              # Production dependencies
└── README.md                     # Documentation and reproduction guide
```

---

## 4. Machine Learning Methodology

### 4.1 Problem Formulation
- **Prediction Unit:** Product-Month per reporting cycle across 1,091 healthcare clinics.
- **Target Variable:** Total consumption quantity in period $t+1$ (next month's demand).
- **Forecasting Horizon:** 1 month forward (matching routine procurement reorder cadences).
- **Stock Rotation Policy:** First Expiry, First Out (FEFO).

### 4.2 Leak-Free Temporal Feature Engineering
To strictly prevent future lookahead bias, features are generated using backward historical shifts (`shift(1)`):
- **Calendar Signals:** `month` (1–12), `year`, `quarter_num` (1–4).
- **Lag Features:** `lag_1`, `lag_2`, `lag_3` (consumption at $t-1, t-2, t-3$).
- **Rolling Statistics:** `rolling_mean_3` and `rolling_std_3` over the preceding 3-month window.
- **Supply Signals:** `received`, `open_balance`, `close_balance`, `stockout_rate`, `active_facilities`.

### 4.3 Chronological Splitting Strategy
Random shuffling is strictly avoided in time series. The 1,720 product-month observations were partitioned chronologically:
- **Training Set (70%):** Jan 2020 – Sep 2022 (1,142 records)
- **Validation Set (15%):** Oct 2022 – Apr 2023 (240 records)
- **Held-Out Test Set (15%):** May 2023 – Oct 2023 (194 records)

---

## 5. Model Evaluation & Benchmark Results

### 5.1 Demand Forecasting (Regression)
Evaluated on the held-out test split using **Weighted Absolute Percentage Error (WAPE)**, **MAE**, and **RMSE**:

$$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$$

| Model | MAE (Units) | RMSE (Units) | WAPE (%) | Accuracy ($100 - \text{WAPE}$) | Error Reduction vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3-Month Moving Average Baseline** | 14,860.18 | 36,576.20 | 34.60% | 65.40% | Baseline |
| **Gradient Boosting Regressor** | 11,447.07 | 26,115.04 | 26.65% | 73.35% | 23.0% improvement |
| **Random Forest Regressor (Selected)** | **10,507.48** | **25,669.49** | **24.46%** | **75.54%** | **29.3% improvement** |

*The Random Forest Regressor captures non-linear interactions between closing balances, facility counts, and seasonal trends, reducing forecast error by nearly 30% relative to moving averages.*

### 5.2 Expiry-Risk Classification (FEFO Rule)
Using the FEFO condition ($\text{Remaining Batch Stock} > \text{Days to Expiry} \times \text{Daily Demand Rate}$) evaluated on 240 held-out test batches:
- **Accuracy:** **99.17%** (238 / 240 correctly classified)
- **Precision:** **99.04%**
- **Recall:** **99.04%**
- **F1-Score:** **99.04%**
- **Confusion Matrix:** 135 True Negatives, 103 True Positives, 1 False Positive, 1 False Negative.

---

## 6. Interactive Decision-Support Dashboard (`app.py`)

The Streamlit web application connects directly to the trained model artifacts (`models/*.pkl`) to provide operational decision support:

1. **Executive Summary:** High-level system KPIs, clinic tier distribution, and 50-month consumption timelines.
2. **Demand Forecasting:** Dropdown selector across all 36 medicines, historical trend visualization, live next-month prediction, and baseline variance comparison.
3. **Inventory Analysis:** Network-wide stock health metrics, Months of Stock (MOS) turnover, and bar charts for top stockout vs. overstocked medicines.
4. **Expiry-Risk Assessment:** Interactive FEFO batch decision calculator where users input batch stock, shelf-life days, and demand rates to receive immediate risk classifications.
5. **Operational Action Plan:** Automated replenishment alerts, surplus redistribution candidates, and standard buffer stock management.

---

## 7. Quickstart Guide

### Prerequisites
- Python 3.10 or higher installed
- Git installed

### Step 1: Clone the Repository
```bash
git clone https://github.com/sumitkumarbarnwal/Pharmacy-Medicine-Demand-and-Expiry-Risk-Forecasting.git
cd Pharmacy-Medicine-Demand-and-Expiry-Risk-Forecasting
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Jupyter Notebook
```bash
jupyter notebook notebooks/Pharmacy_Medicine_Demand_and_Expiry_Risk_Forecasting.ipynb
```
*All 75 cells are pre-executed with fully populated outputs, metric tables, and charts.*

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Open your browser to `http://localhost:8501` to explore the dashboard.*

---

## 8. 5-Minute Technical Presentation Walkthrough

Use this pacing when demonstrating the project in a technical interview:

- **Minute 1: Executive Summary** — Explain the dual challenge (stockouts vs. expiration), the DHIS2 dataset scope (1,091 facilities, 36 drugs, 50 months), and the primary WAPE metric (24.46%).
- **Minute 2: Demand Forecasting** — Select a sample medicine (e.g., ORS or Paracetamol), run live inference with `demand_model.pkl`, and compare the Random Forest forecast against the 3-month moving average baseline.
- **Minute 3: Inventory Health** — Show Months of Stock (MOS) turnover, highlighting the 8 chronic stockout medicines (e.g., Oxytocin Inj at 29.6%) and 10 overstocked products (>6 months MOS).
- **Minute 4: Expiry-Risk Assessment** — Walk through the FEFO formula ($\text{Stock} > \text{Days} \times \text{Daily Demand}$) and test the live batch evaluator with a sample batch.
- **Minute 5: Operational Action Plan** — Filter recommendations by priority (Urgent Replenishment vs. Redistribution) and state the non-clinical operational advisory boundary.

---

## 9. Limitations & Responsible AI

- **Temporal Granularity:** DHIS2 records are reported on a monthly frequency; daily clinic surges or emergency weekend spikes are aggregated.
- **Lot Tracking Gap:** Public health systems track aggregate inventory counts; integration with hospital warehouse management systems (WMS) is required to capture manufacturer lot numbers.
- **Operational Advisory Scope:** All model forecasts, stockout flags, and redistribution guidance are decision-support recommendations for authorized supply officers, never autonomous clinical or medical treatment advice.
