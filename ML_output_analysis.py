import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# KONFIGURACJA
# ===============================
INPUT_DIR = "input_from_Jezusek2"
OUTPUT_DIR = "output_dziecko3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_FILE = f"{INPUT_DIR}/predictions.csv"
FEAT_FILE = f"{INPUT_DIR}/feature_importance.csv"

# ===============================
# WCZYTANIE DANYCH
# ===============================
print("Wczytywanie danych od Jezuska 2...")
df = pd.read_csv(PRED_FILE)

print(f"Liczba zdarzeń: {len(df)}")
print(df.head())

# ===============================
# PODSTAWOWE STATYSTYKI
# ===============================
accuracy = (df["true_label"] == df["pred_label"]).mean()

report = classification_report(
    df["true_label"],
    df["pred_label"],
    target_names=["Kaskada", "Tor"],
    output_dict=True
)

report_df = pd.DataFrame(report).T
report_df.to_csv(f"{OUTPUT_DIR}/classification_report.csv")

print("\nDokładność:", accuracy)

# ===============================
# MACIERZ POMYŁEK
# ===============================
cm = confusion_matrix(df["true_label"], df["pred_label"])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Kaskada", "Tor"],
    yticklabels=["Kaskada", "Tor"]
)
plt.xlabel("Predykcja")
plt.ylabel("Prawda")
plt.title("Macierz pomyłek")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.pdf")
plt.close()

# ===============================
# ROC CURVE
# ===============================
fpr, tpr, _ = roc_curve(df["true_label"], df["pred_proba"])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Krzywa ROC")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.pdf")
plt.close()

# ===============================
# PRECISION–RECALL
# ===============================
precision, recall, _ = precision_recall_curve(
    df["true_label"], df["pred_proba"]
)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precision_recall.pdf")
plt.close()

# ===============================
# HISTOGRAM ENERGII (tot)
# ===============================
plt.figure(figsize=(7, 5))
plt.hist(df[df.true_label == 0]["tot"], bins=50, alpha=0.6, label="Kaskady")
plt.hist(df[df.true_label == 1]["tot"], bins=50, alpha=0.6, label="Tory")
plt.xlabel("tot (energia / ładunek)")
plt.ylabel("Liczba zdarzeń")
plt.legend()
plt.title("Rozkład energii")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/energy_distribution.pdf")
plt.close()

# ===============================
# 3D – NAJCIEKAWSZE PRZYPADKI
# ===============================
print("Tworzenie wizualizacji 3D...")

# 1️⃣ BARDZO PEWNE TORY
high_conf_tracks = df[
    (df.pred_label == 1) & (df.pred_proba > 0.95)
].sample(n=min(3000, len(df)), random_state=42)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    high_conf_tracks["pos_x"],
    high_conf_tracks["pos_y"],
    high_conf_tracks["pos_z"],
    c=high_conf_tracks["pred_proba"],
    s=2,
    cmap="viridis"
)
ax.set_title("3D: Bardzo pewne tory")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3d_high_conf_tracks.pdf")
plt.close()

# 2️⃣ NAJGORSZE BŁĘDY MODELU
mistakes = df[df.true_label != df.pred_label]

if len(mistakes) > 0:
    mistakes = mistakes.sample(
        n=min(3000, len(mistakes)),
        random_state=42
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        mistakes["pos_x"],
        mistakes["pos_y"],
        mistakes["pos_z"],
        c=mistakes["pred_proba"],
        s=2,
        cmap="coolwarm"
    )
    ax.set_title("3D: Błędne klasyfikacje")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3d_misclassified.pdf")
    plt.close()

# ===============================
# WAŻNOŚĆ CECH (JEŚLI JEST)
# ===============================
if os.path.exists(FEAT_FILE):
    feat = pd.read_csv(FEAT_FILE)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=feat,
        y="feature",
        x="importance",
        color="steelblue"
    )
    plt.title("Ważność cech (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.pdf")
    plt.close()

# ===============================
# PODSUMOWANIE DO RAPORTU
# ===============================
summary = {
    "Accuracy": accuracy,
    "AUC_ROC": roc_auc,
    "N_events": len(df),
    "N_errors": int((df.true_label != df.pred_label).sum())
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)

print("\n✅ DZIECKO 3 ZAKOŃCZYŁO PRACĘ")
print(summary_df)
