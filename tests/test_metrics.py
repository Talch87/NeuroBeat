import numpy as np

from neurocardio.eval.metrics import aami_metrics, confusion


def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    cm = confusion(y_true, y_pred, n_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1 and cm[2, 0] == 1
    assert cm.sum() == 5


def test_aami_metrics_known_values():
    cm = np.array([[10, 0, 0], [0, 5, 0], [0, 0, 2]])
    m = aami_metrics(cm, classes=["N", "SVEB", "VEB"])
    assert abs(m["overall_accuracy"] - 1.0) < 1e-9
    assert abs(m["per_class"]["VEB"]["sensitivity"] - 1.0) < 1e-9
    assert abs(m["per_class"]["VEB"]["ppv"] - 1.0) < 1e-9


def test_aami_metrics_partial():
    cm = np.array(
        [
            [90, 0, 1],
            [0, 10, 0],
            [2, 0, 6],
        ]
    )
    m = aami_metrics(cm, classes=["N", "SVEB", "VEB"])
    veb = m["per_class"]["VEB"]
    assert abs(veb["sensitivity"] - 6 / 8) < 1e-9
    assert abs(veb["ppv"] - 6 / 7) < 1e-9
