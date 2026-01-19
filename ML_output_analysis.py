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
# GŁÓWNA FUNKCJA ANALIZY
# ===============================

def run_analysis(
    pred_file,
    output_dir,
    feat_file=None,
    class_names=("Kaskada", "Tor"),
    high_conf_threshold=0.95,
    max_points_3d=3000
):
    """
    Główna funkcja analizy wyników ML

    Parameters
    ----------
    pred_file : str
        CSV z predykcjami
    output_dir : str
        Folder wyjściowy na wykresy i raporty
    feat_file : str or None
        CSV z ważnością cech (opcjonalne)
    class_names : tuple
        Nazwy klas (default: ("Kaskada", "Tor"))
    high_conf_threshold : float
        Próg pewności dla 3D wizualizacji
    max_points_3d : int
        Maksymalna liczba punktów w 3D
    """

    os.makedirs(output_dir, exist_ok=True)

    print("[ML_ANALYSIS] Wczytywanie danych...")
    df = pd.read_csv(pred_file)

    print(f"[ML_ANALYSIS] Liczba zdarzeń: {len(df)}")

    # ===============================
    # METRYKI
    # ===============================
    accuracy = (df["true_label"] == df["pred_label"]).mean()

    report = classification_report(
        df["true_label"],
        df["pred_label"],
        target_names=list(class_names),
        output_dict=True
    )

    report_df = pd.DataFrame(report).T
    report_df.to_csv(f"{output_dir}/classification_report.csv")

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
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predykcja")
    plt.ylabel("Prawda")
    plt.title("Macierz pomyłek")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.pdf")
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
    plt.savefig(f"{output_dir}/roc_curve.pdf")
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
    plt.savefig(f"{output_dir}/precision_recall.pdf")
    plt.close()

    # ===============================
    # HISTOGRAM ENERGII
    # ===============================
    plt.figure(figsize=(7, 5))
    plt.hist(df[df.true_label == 0]["tot"], bins=50, alpha=0.6, label=class_names[0])
    plt.hist(df[df.true_label == 1]["tot"], bins=50, alpha=0.6, label=class_names[1])
    plt.xlabel("tot (energia / ładunek)")
    plt.ylabel("Liczba zdarzeń")
    plt.legend()
    plt.title("Rozkład energii")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_distribution.pdf")
    plt.close()

    # ===============================
    # 3D – WYSOKA PEWNOŚĆ
    # ===============================
    print("[ML_ANALYSIS] Tworzenie wizualizacji 3D...")

    high_conf_tracks = df[
        (df.pred_label == 1) & (df.pred_proba > high_conf_threshold)
    ]

    if len(high_conf_tracks) > 0:
        high_conf_tracks = high_conf_tracks.sample(
            n=min(max_points_3d, len(high_conf_tracks)),
            random_state=42
        )

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
        plt.savefig(f"{output_dir}/3d_high_conf_tracks.pdf")
        plt.close()

    # ===============================
    # 3D – BŁĘDY
    # ===============================
    mistakes = df[df.true_label != df.pred_label]

    if len(mistakes) > 0:
        mistakes = mistakes.sample(
            n=min(max_points_3d, len(mistakes)),
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
        plt.savefig(f"{output_dir}/3d_misclassified.pdf")
        plt.close()

    # ===============================
    # WAŻNOŚĆ CECH
    # ===============================
    if feat_file is not None and os.path.exists(feat_file):
        feat = pd.read_csv(feat_file)

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=feat,
            y="feature",
            x="importance",
            color="steelblue"
        )
        plt.title("Ważność cech (Random Forest)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.pdf")
        plt.close()

    # ===============================
    # PODSUMOWANIE
    # ===============================
    summary = {
        "Accuracy": float(accuracy),
        "AUC_ROC": float(roc_auc),
        "N_events": int(len(df)),
        "N_errors": int((df.true_label != df.pred_label).sum())
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)

    print("[ML_ANALYSIS] Zakończono analizę")
    print(summary_df)

    return summary

