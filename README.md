# 🏭 Predidctive Maintenance Using AI 

Predict machine failures **before they happen** — using real industrial sensor data, a comparison of three ML models, SHAP-based explainability, and a generative-AI layer that turns risk scores into plain-English maintenance reports.

Instead of fixing machines after they break (reactive) or on a fixed calendar schedule (preventive), this project predicts failure risk from live sensor readings so maintenance can happen exactly when it's needed — reducing downtime, cost, and safety risk.

**Live demo:** _add your deployed Streamlit URL here_

---

## 📌 What this project does

- Takes real sensor readings — **air temperature, process temperature, rotational speed, torque, tool wear, machine type** — as input
- Predicts **failure probability** using a trained XGBoost classifier
- Flags machines as **Low / Medium / High risk** against a user-adjustable threshold
- Explains *why* a machine is at risk using **SHAP** (which sensor readings drove the score, and by how much)
- Generates a **plain-English maintenance report** using Google Gemini, grounded in the actual SHAP drivers (not hallucinated)
- Lets you ask **follow-up questions** about a specific machine's prediction
- Logs every prediction to a local **SQLite** database, browsable in a History tab
- Ships with a **synthetic demo fleet generator** so the Upload & Predict tab can be demoed without a real CSV on hand

---

## 🧰 Tools & Technologies used

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| ML models | Scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Generative AI | Google Gemini (`gemini-2.5-flash`) |
| Web app | Streamlit |
| Charts | Plotly |
| Database | SQLite |
| Model persistence | Joblib |
| Dataset access | ucimlrepo |

---

## 🤖 Machine Learning models compared

Three models were trained and evaluated on a stratified, held-out test split (20%), with class imbalance explicitly handled (`class_weight='balanced'` for Logistic Regression / Random Forest, `scale_pos_weight` for XGBoost) since only ~3% of machines in the real dataset actually fail.

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.178 | 0.868 | 0.295 | 0.934 |
| Random Forest | 0.941 | 0.706 | 0.807 | 0.972 |
| **XGBoost ✅ (selected)** | **0.821** | **0.809** | **0.815** | **0.979** |

**Why XGBoost was chosen:** the best model is selected automatically based on F1-score on the minority (failure) class — not raw accuracy, which is meaningless on a 97%/3% imbalanced dataset. Logistic Regression catches a lot of failures but drowns maintenance teams in false alarms (17.8% precision). Random Forest is very precise but misses ~29% of real failures. XGBoost gives the best balance of catching real failures **and** keeping false alarms manageable — the tradeoff that matters most in a maintenance context, where a missed failure is expensive but "inspect everything" isn't practical either.

*(Full confusion matrices and metrics are saved to `models/metrics.json` and viewable in the app's Model Metrics tab.)*

---

## 🔍 Explainability — not just a number

A risk score alone isn't actionable. For every prediction, SHAP values show **which specific sensor readings** pushed the risk up or down, e.g.:

> *82% failure risk — primarily driven by elevated torque (+0.32) and tool wear (+0.21), partially offset by normal rotational speed (−0.08).*

This is then handed to Gemini, which turns it into a maintenance-team-readable recommendation instead of a spreadsheet of SHAP values.

---

## 📁 Project Structure

```
predictive_maintenance/
│
├── app.py                      → Streamlit web app (6-tab dashboard)
├── download_data.py             → fetches the real AI4I 2020 dataset from UCI
├── generate_demo_data.py        → creates a synthetic demo fleet for the Upload & Predict tab
├── requirements.txt
├── .env.example                 → template for GEMINI_API_KEY
│
├── src/
│   ├── preprocessing.py         → cleaning, feature engineering, train/test split, scaling
│   ├── train_models.py          → trains + compares 3 models, saves the best one
│   ├── explain.py               → SHAP explanation for a single prediction
│   ├── llm_report.py            → Gemini-powered report + follow-up Q&A
│   └── database.py              → SQLite prediction history
│
├── models/
│   ├── model.pkl                → best trained model (XGBoost)
│   ├── scaler.pkl                → fitted StandardScaler
│   ├── feature_names.json
│   └── metrics.json              → evaluation results for all 3 models
│
└── data/
    ├── ai4i2020.csv               → real UCI dataset (training only)
    └── demo_fleet_sample.csv      → synthetic demo data (never used for training)
```

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**, UCI Machine Learning Repository — 10,000 real industrial sensor records with a genuinely imbalanced ~3.3% failure rate.

**Citation:** Stephan, S. (2020). *AI4I 2020 Predictive Maintenance Dataset.* UCI Machine Learning Repository.

**Features used for prediction:**

| Feature | Meaning |
|---|---|
| Type | Machine quality variant (Low / Medium / High) |
| Air temperature | Ambient temperature (K) |
| Process temperature | Internal process temperature (K) |
| Rotational speed | Spindle speed (rpm) |
| Torque | Applied torque (Nm) |
| Tool wear | Cumulative tool wear (min) |
| Power *(engineered)* | Torque × Rotational speed |
| Temp_Diff *(engineered)* | Process temperature − Air temperature |

---

## 🖥️ App Pages

- **Upload & Predict** — upload a fleet CSV, get risk scores for every machine
- **Manual Entry** — enter one machine's readings, get an instant prediction + SHAP + AI report
- **AI Report** — plain-English maintenance recommendation, with follow-up Q&A
- **Trends** — fleet-wide charts of risk distribution and sensor patterns
- **History** — every past prediction, logged to SQLite
- **Model Metrics** — precision/recall/F1/ROC-AUC and confusion matrices for all 3 models, transparently

---

## 🚀 How to run this project

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/predictive-maintenance-ai.git
cd predictive-maintenance-ai
```

**2. Create a virtual environment** *(optional but recommended)*
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**
```bash
python download_data.py
```

**5. Set up your Gemini API key**
```bash
cp .env.example .env
```
Add your key to `.env` (free, no credit card required — get one at https://aistudio.google.com/apikey).

**6. Train the models**
```bash
python src/train_models.py
```

**7. (Optional) Generate demo fleet data**
```bash
python generate_demo_data.py
```

**8. Run the app**
```bash
streamlit run app.py
```

---

## 📈 Model Evaluation

All three models were evaluated on the same held-out test set using Precision, Recall, F1-Score, ROC-AUC, and confusion matrix — see the table above and the **Model Metrics** tab in the app for full transparency. XGBoost was selected and is the model shipped in `models/model.pkl`.

---

## 🙋 What I learned building this

- How to spot and avoid **target leakage** in a real-world imbalanced dataset
- How to train, evaluate, and fairly compare multiple ML models on the minority-class metrics that actually matter
- How to make a "black box" model's predictions explainable using SHAP
- How to combine a traditional ML pipeline with an LLM layer — grounding the LLM in real SHAP output instead of letting it hallucinate
- How to build and deploy a multi-tab Streamlit app backed by a SQLite history log
- How to structure a project so the pipeline (training) and the product (the app) are cleanly separated

---

## 🔮 Future improvements

- Optimize the risk threshold based on the real-world cost of a missed failure vs. an unnecessary inspection, rather than a fixed 0.7 default
- Add model monitoring / drift detection for when the app is fed live sensor streams
- Add unit tests for the preprocessing and prediction pipeline
- Support multiple simultaneous LLM providers (Gemini / OpenAI / local) via a config flag

---

## 🧑‍💻 Author

Made by **Latika**
 | LinkedIn: https://www.linkedin.com/in/latika-muteja-a0168b277/

