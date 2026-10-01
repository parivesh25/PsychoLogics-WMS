import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# PSYCHOLOGICS — WELFARE MONITORING SYSTEM (POC)
# Prototype uses synthetic/experimental data for demonstration.
# ==========================================================

# 1. DATA INGESTION
df = pd.read_csv("employee_records.csv")
df_sample = df.head(5000).copy()

# Required personnel columns
required_columns = ["Age", "Salary"]
missing_columns = [c for c in required_columns if c not in df_sample.columns]

if missing_columns:
    raise ValueError(
        f"Missing required columns in employee_records.csv: {missing_columns}"
    )

# 2. FEATURE ENGINEERING
# Synthetic wellness features for the POC.
# In production these would come from voluntary validated
# wellness assessments and approved organizational systems.

np.random.seed(42)

df_sample["Anxiety_Score_GAD7"] = np.random.randint(
    0, 21, size=len(df_sample)
)  # 0–20

df_sample["Depression_Score_PHQ9"] = np.random.randint(
    0, 28, size=len(df_sample)
)  # 0–27

df_sample["Work_Hours_Per_Week"] = np.random.randint(
    40, 90, size=len(df_sample)
)

# Additional organizational indicators
df_sample["Leave_Days_Last_90"] = np.random.randint(
    0, 16, size=len(df_sample)
)

df_sample["Deployment_Days_Last_90"] = np.random.randint(
    0, 91, size=len(df_sample)
)

# 3. WELFARE RISK SCORE
# Higher value = higher potential welfare concern.
# This is a prototype scoring formula, NOT a clinical diagnosis.

anxiety_component = df_sample["Anxiety_Score_GAD7"] / 20.0
depression_component = df_sample["Depression_Score_PHQ9"] / 27.0
workload_component = (
    (df_sample["Work_Hours_Per_Week"] - 40) / 50.0
).clip(0, 1)

deployment_component = df_sample["Deployment_Days_Last_90"] / 90.0
leave_component = 1 - (df_sample["Leave_Days_Last_90"] / 15.0)

df_sample["Composite_Welfare_Risk_Index"] = (
    anxiety_component * 0.30
    + depression_component * 0.30
    + workload_component * 0.20
    + deployment_component * 0.10
    + leave_component * 0.10
).clip(0, 1)

# 4. CREATE PROTOTYPE LABEL
# Top 15% of the composite score is treated as high-risk
# for model demonstration only.
risk_cutoff = df_sample["Composite_Welfare_Risk_Index"].quantile(0.85)

df_sample["High_Welfare_Risk_Flag"] = (
    df_sample["Composite_Welfare_Risk_Index"] >= risk_cutoff
).astype(int)



# 5. RANDOM FOREST AI ENGINE
feature_columns = [
    "Age",
    "Salary",
    "Anxiety_Score_GAD7",
    "Depression_Score_PHQ9",
    "Work_Hours_Per_Week",
    "Leave_Days_Last_90",
    "Deployment_Days_Last_90",
]

X = df_sample[feature_columns]
y = df_sample["High_Welfare_Risk_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
)

rf_model.fit(X_train, y_train)

# 6. MODEL EVALUATION
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 7. RISK PROBABILITY / EARLY WARNING
df_sample["Risk_Probability"] = rf_model.predict_proba(X)[:, 1]

alerts_output = (
    df_sample[df_sample["Risk_Probability"] >= 0.50]
    .copy()
    .sort_values(by="Risk_Probability", ascending=False)
)

# 8. CONSOLE DEMO OUTPUT
print("=" * 62)
print("       PSYCHOLOGICS: WELFARE MONITORING AI ENGINE")
print("=" * 62)

print(f"[*] Dataset Used: employee_records.csv ({len(df_sample)} records)")
print("[*] AI Model: Random Forest")
print("[*] Model Status: Trained Successfully")
print(f"[*] Prototype Test Accuracy: {accuracy * 100:.2f}%")
print(
    f"[*] Personnel Monitored: {len(df_sample)} | "
    f"Early-Warning Alerts: {len(alerts_output)}"
)

print("-" * 62)
print("LIVE EARLY-WARNING ALERTS")
print("-" * 62)

display_columns = [
    c
    for c in ["Employee_Name", "Department"]
    if c in df_sample.columns
]

display_columns += [
    "Anxiety_Score_GAD7",
    "Depression_Score_PHQ9",
    "Work_Hours_Per_Week",
    "Risk_Probability",
]

print(
    alerts_output[display_columns]
    .head(15)
    .to_string(index=False)
)

print("-" * 62)
