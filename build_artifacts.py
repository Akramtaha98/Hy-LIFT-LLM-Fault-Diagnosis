"""
build_artifacts.py — generate publishable figures & tables for your project.

This script:
1) Loads a labeled dataset CSV with columns: message,label
2) Uses your rule-based classifier from classify.py
3) Computes accuracy, precision, recall, F1 (macro and per-class)
4) Exports CSV tables and PNG figures

Usage:
  python build_artifacts.py --data sample_logs.csv --outdir artifacts

No pandas/numpy required; we only depend on matplotlib for figures.
"""

import argparse, csv, os
from typing import List, Dict, Tuple

# Import your simple classifier (must be in the same folder as this script)
from classify import predict_fault, parse_log_line

# We import matplotlib only when needed to avoid import errors in minimal setups
import matplotlib.pyplot as plt

LABELS = ["congestion", "backhaul", "hardware", "interference", "core"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Expect CSV with at least columns: message,label
    Returns: messages, labels
    """
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "message" not in r.fieldnames or "label" not in r.fieldnames:
            raise ValueError("Input CSV must have columns: 'message' and 'label'")
        for row in r:
            X.append(row["message"])
            y.append(row["label"])
    return X, y

def predict_messages(messages: List[str]) -> List[str]:
    """
    Run your classifier for each message and return predicted labels.
    """
    preds = []
    for m in messages:
        parsed = parse_log_line(m)
        out = predict_fault(parsed["message_lower"])
        preds.append(out["label"])
    return preds

def build_confusion_matrix(labels: List[str], y_true: List[str], y_pred: List[str]) -> List[List[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm

def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    correct = sum(1 for a,b in zip(y_true, y_pred) if a == b)
    return correct / max(1, len(y_true))

def precision_recall_f1(labels: List[str], cm: List[List[int]]):
    """
    Compute per-class precision, recall, f1 from confusion matrix.
    cm rows=true, cols=pred.
    """
    n = len(labels)
    per_class = {}
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        prec = tp / max(1, (tp + fp))
        rec  = tp / max(1, (tp + fn))
        f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(cm[i])}
    return per_class

def macro_avg(per_class):
    ks = list(per_class.keys())
    P = sum(per_class[k]["precision"] for k in ks) / max(1, len(ks))
    R = sum(per_class[k]["recall"] for k in ks) / max(1, len(ks))
    F = sum(per_class[k]["f1"] for k in ks) / max(1, len(ks))
    return {"precision": P, "recall": R, "f1": F}

def save_tables(outdir, labels, cm, per_class, acc, y_true, y_pred):
    tables_dir = os.path.join(outdir, "tables")
    data_dir   = os.path.join(outdir, "data")
    ensure_dir(tables_dir)
    ensure_dir(data_dir)

    # Confusion matrix table
    with open(os.path.join(tables_dir, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + ["pred_"+lab for lab in labels])
        for i, lab in enumerate(labels):
            w.writerow(["true_"+lab] + cm[i])

    # Per-class metrics
    with open(os.path.join(tables_dir, "metrics_by_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","precision","recall","f1","support"])
        for lab in labels:
            m = per_class[lab]
            w.writerow([lab, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m["support"]])

    # Overall metrics
    M = macro_avg(per_class)
    with open(os.path.join(tables_dir, "metrics_overall.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["accuracy", f"{acc:.4f}"])
        w.writerow(["macro_precision", f"{M['precision']:.4f}"])
        w.writerow(["macro_recall", f"{M['recall']:.4f}"])
        w.writerow(["macro_f1", f"{M['f1']:.4f}"])

    # Save prediction pairs for audit
    with open(os.path.join(data_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["message","true_label","pred_label"])
        for m,t,p in zip(y_true, y_true, y_pred):
            w.writerow([m,t,p])

def save_figures(outdir, labels, cm, per_class):
    figs_dir = os.path.join(outdir, "figures")
    ensure_dir(figs_dir)

    # 1) Confusion matrix heatmap
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(cm)  # default colormap; no explicit colors
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), va="center", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # 2) Class support bar chart
    supports = [per_class[lab]["support"] for lab in labels]
    fig2 = plt.figure(figsize=(6, 3.8))
    ax2 = fig2.add_subplot(111)
    ax2.bar(range(len(labels)), supports)  # default colors
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Count")
    ax2.set_title("Class Support")
    fig2.tight_layout()
    fig2.savefig(os.path.join(figs_dir, "class_support.png"), dpi=200)
    plt.close(fig2)

def main(data_path: str, outdir: str):
    ensure_dir(outdir)

    X, y_true = load_dataset(data_path)
    y_pred = predict_messages(X)

    cm = build_confusion_matrix(LABELS, y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    per_class = precision_recall_f1(LABELS, cm)

    save_tables(outdir, LABELS, cm, per_class, acc, y_true, y_pred)
    save_figures(outdir, LABELS, cm, per_class)

    print(f"Artifacts written to: {outdir}")
    print("- tables/: metrics_overall.csv, metrics_by_class.csv, confusion_matrix.csv")
    print("- figures/: confusion_matrix.png, class_support.png")
    print("- data/: predictions.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_logs.csv", help="CSV with message,label columns (e.g., from data_gen.py)")
    ap.add_argument("--outdir", default="artifacts", help="Output folder for figures & tables")
    args = ap.parse_args()
    main(args.data, args.outdir)
