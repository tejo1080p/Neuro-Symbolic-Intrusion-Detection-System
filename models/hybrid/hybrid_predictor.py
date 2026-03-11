
import numpy as np


def hybrid_predict_proba(xgb_proba, cnn_proba, rule_proba):
    xgb_proba = np.asarray(xgb_proba, dtype=float)
    cnn_proba = np.asarray(cnn_proba, dtype=float)
    rule_proba = np.asarray(rule_proba, dtype=float)

    hybrid_score = (0.45 * xgb_proba) + (0.35 * cnn_proba) + (0.20 * rule_proba)

    high_rule_conf_mask = rule_proba > 0.7
    hybrid_score[high_rule_conf_mask] += 0.10

    strong_disagreement_mask = np.abs(cnn_proba - xgb_proba) > 0.5
    hybrid_score[strong_disagreement_mask] += 0.10 * rule_proba[strong_disagreement_mask]

    return np.clip(hybrid_score, 0.0, 1.0)


def hybrid_predict_label(xgb_proba, cnn_proba, rule_proba, threshold=0.5):
    return (hybrid_predict_proba(xgb_proba, cnn_proba, rule_proba) >= threshold).astype(int)
