import numpy as np


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def aami_metrics(cm: np.ndarray, classes: list[str]) -> dict:
    """Per-class sensitivity (recall) and positive predictivity (precision),
    plus overall accuracy. Rows = true, columns = predicted."""
    total = cm.sum()
    per_class = {}
    for i, name in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class[name] = {
            "sensitivity": float(sens),
            "ppv": float(ppv),
            "support": int(cm[i, :].sum()),
        }
    return {
        "overall_accuracy": float(np.trace(cm) / total) if total else 0.0,
        "per_class": per_class,
    }
