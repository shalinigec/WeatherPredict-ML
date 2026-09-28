"""
Weather (Rain) Prediction
Models: Decision Tree, Random Forest, Extreme Gradient Boosting (XGBoost)

Rain is a categorical target ("rain" / "no rain"), so the main pipeline uses
classifiers. Regressor versions are included at the end (predict a 0-1 rain
score, then threshold at 0.5) in case the assignment requires them.

Run in Google Colab, where the file is at /content/weather_forecast_data.csv
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

try:
    display  # available in Jupyter / Colab
except NameError:
    display = print

RANDOM_STATE = 42
DATA_PATH = "/content/weather_forecast_data.csv"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Rain"] = df["Rain"].astype(str).str.strip().str.lower()

print("Shape:", df.shape)
print("\nFirst rows:")
display(df.head())
print("\nInfo:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:")
display(df.describe().round(3))
print("\nTarget distribution:\n", df["Rain"].value_counts())

# ---------------------------------------------------------------------------
# 2. Exploratory data analysis
# ---------------------------------------------------------------------------
features = [c for c in df.columns if c != "Rain"]

plt.figure(figsize=(5, 4))
sns.countplot(x="Rain", data=df, palette="Set2")
plt.title("Rain vs No Rain")
plt.show()

df[features].hist(bins=30, figsize=(12, 7), color="steelblue")
plt.suptitle("Feature distributions")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))
for ax, col in zip(axes, features):
    sns.boxplot(x="Rain", y=col, data=df, ax=ax, palette="Set2")
    ax.set_title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature correlation")
plt.show()

# ---------------------------------------------------------------------------
# 3. Encode target and split
# ---------------------------------------------------------------------------
df["Rain"] = df["Rain"].map({"no rain": 0, "rain": 1})
assert df["Rain"].notna().all(), "Unexpected values in Rain column"

X = df[features]
y = df["Rain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 4. Classification models
#    (tree-based models do not need feature scaling)
# ---------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6,
                                            random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200,
                                            random_state=RANDOM_STATE,
                                            n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1,
                             max_depth=4, eval_metric="logloss",
                             random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []

plt.figure(figsize=(6, 5))
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["no rain", "rain"]))

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d",
                cmap="Blues", xticklabels=["no rain", "rain"],
                yticklabels=["no rain", "rain"])
    plt.title(f"Confusion matrix: {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.show()

    rows.append({
        "Model": name,
        "Train Acc": accuracy_score(y_train, model.predict(X_train)),
        "Test Acc": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV F1 (mean)": cv_f1.mean(),
        "CV F1 (std)": cv_f1.std(),
        "Train time (s)": train_time,
    })

# ---------------------------------------------------------------------------
# 5. Compare results
# ---------------------------------------------------------------------------
# ROC curves
plt.figure(figsize=(6, 5))
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curves"); plt.legend()
plt.show()

comparison = pd.DataFrame(rows).set_index("Model")
comparison["Overfit gap"] = comparison["Train Acc"] - comparison["Test Acc"]
print("\n=== MODEL COMPARISON ===")
display(comparison.round(4).sort_values("F1", ascending=False))

comparison[["Test Acc", "Precision", "Recall", "F1", "ROC-AUC"]].plot(
    kind="bar", figsize=(9, 5), ylim=(0, 1.05), rot=0,
    title="Model comparison")
plt.ylabel("Score"); plt.legend(loc="lower right")
plt.show()

# Feature importance side by side
imp = pd.DataFrame({n: m.feature_importances_ for n, m in models.items()},
                   index=X.columns)
imp.plot(kind="bar", figsize=(9, 5), rot=30,
         title="Feature importance by model")
plt.ylabel("Importance")
plt.show()

best = comparison["F1"].idxmax()
print(f"\nBest model by F1: {best}")

# ---------------------------------------------------------------------------
# 6. Save models + metadata for the FastAPI backend
# ---------------------------------------------------------------------------
import json
import os
import joblib
import sklearn
import xgboost

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

files = {}
for name, model in models.items():
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    files[name] = fname

metadata = {
    "features": list(X.columns),               # column order the models expect
    "labels": {"0": "no rain", "1": "rain"},
    "best_model": best,                        # chosen by F1 above
    "files": files,
    "metrics": comparison[["Test Acc", "Precision", "Recall", "F1",
                           "ROC-AUC"]].round(4).to_dict(orient="index"),
    "versions": {"scikit-learn": sklearn.__version__,
                 "xgboost": xgboost.__version__},
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved {len(files)} models + metadata.json to ./{MODEL_DIR}/")
print("Default model for the API:", best)

# ---------------------------------------------------------------------------
# 7. Optional: regressor versions (rain score 0-1, threshold at 0.5)
# ---------------------------------------------------------------------------
reg_models = {
    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=6, random_state=RANDOM_STATE),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost Regressor": XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        random_state=RANDOM_STATE),
}

reg_rows = []
for name, m in reg_models.items():
    m.fit(X_train, y_train)
    score = m.predict(X_test)
    label = (score >= 0.5).astype(int)
    reg_rows.append({
        "Model": name,
        "R2 (train)": r2_score(y_train, m.predict(X_train)),
        "R2 (test)": r2_score(y_test, score),
        "RMSE": np.sqrt(mean_squared_error(y_test, score)),
        "Accuracy@0.5": accuracy_score(y_test, label),
        "F1@0.5": f1_score(y_test, label),
    })

reg_comparison = pd.DataFrame(reg_rows).set_index("Model")
print("\n=== REGRESSOR COMPARISON ===")
display(reg_comparison.round(4).sort_values("R2 (test)", ascending=False))

reg_comparison[["R2 (train)", "R2 (test)"]].plot(
    kind="bar", figsize=(8, 5), rot=0, title="R-squared: train vs test")
plt.axhline(0, color="k", lw=0.8)
plt.ylabel("R-squared")
plt.show()

print(f"Best regressor by test R2: {reg_comparison['R2 (test)'].idxmax()}")

# R-squared of the classifiers' predicted rain probability (for a like-for-like
# comparison with the regressors above)
clf_r2 = {n: r2_score(y_test, m.predict_proba(X_test)[:, 1])
          for n, m in models.items()}
print("\nR2 (test) of classifier probabilities:")
display(pd.Series(clf_r2, name="R2 (test)").round(4).to_frame())