"""
Métricas puras (sin dependencias externas) para clasificación y regresión.
Se incluye opcionalmente visualización con matplotlib SOLO para gráficos.
"""

from typing import Iterable, Any, List, Tuple, Dict, Optional
import math

# =====================
# CLASIFICACIÓN (PURO)
# =====================

def confusion_matrix_binary(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    aciertos = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return aciertos / n


def precision_recall_f1_binary(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[float, float, float]:
    cm = confusion_matrix_binary(y_true, y_pred)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def confusion_matrix_multiclass(y_true: Iterable[Any], y_pred: Iterable[Any]) -> Tuple[List[Any], List[List[int]]]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        mat[idx[yt]][idx[yp]] += 1
    return labels, mat


def f1_macro_multiclass(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    labels, _ = confusion_matrix_multiclass(y_true, y_pred)
    f1s: List[float] = []
    for lab in labels:
        yt_bin = [1 if yt == lab else 0 for yt in y_true]
        yp_bin = [1 if yp == lab else 0 for yp in y_pred]
        _, _, f1 = precision_recall_f1_binary(yt_bin, yp_bin)
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# =================
# REGRESIÓN (PURO)
# =================

def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def mse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    y_bar = sum(y_true) / n
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - y_bar) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


# ======================
# GRÁFICOS (matplotlib)
# ======================

try:
    import matplotlib.pyplot as plt  # opcional, para gráficos
except Exception:  # pragma: no cover - en caso de no estar disponible
    plt = None


def plot_confusion_matrix(
    labels: List[Any],
    mat: List[List[int]],
    title: str = "Matriz de Confusión",
    save_path: Optional[str] = None,
    show: bool = False,
    block: bool = True,
    close: bool = True,
) -> None:
    if plt is None:
        print("matplotlib no disponible para graficar.")
        return
    import numpy as np
    arr = np.array(mat)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Verdadero")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Figura guardada en: {save_path}")
    if show:
        plt.show(block=block)
    if close:
        plt.close(fig)


def plot_regression_diagnostics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    save_prefix: Optional[str] = None,
    show: bool = False,
    block: bool = True,
    close: bool = True,
) -> None:
    if plt is None:
        print("matplotlib no disponible para graficar.")
        return
    y_true = list(y_true)
    y_pred = list(y_pred)
    # Pred vs Real
    fig1 = plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('Real')
    plt.ylabel('Predicho')
    plt.title('Predicción vs Real (métricas puras)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_prefix:
        path1 = f"{save_prefix}_pred_vs_real.png"
        fig1.savefig(path1)
        print(f"Figura guardada en: {path1}")
    if show:
        plt.show(block=block)
    if close:
        plt.close(fig1)
    # Residuos vs Predicho
    residuos = [yt - yp for yt, yp in zip(y_true, y_pred)]
    fig2 = plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuos, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicho')
    plt.ylabel('Residuo (Real - Pred)')
    plt.title('Residuos vs Predicción (métricas puras)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_prefix:
        path2 = f"{save_prefix}_residuos.png"
        fig2.savefig(path2)
        print(f"Figura guardada en: {path2}")
    if show:
        plt.show(block=block)
    if close:
        plt.close(fig2)
