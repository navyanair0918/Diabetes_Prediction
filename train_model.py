"""Select, evaluate, and save the final BRFSS diabetes-status classifier."""
import json
import pickle

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml_pipeline import FEATURES, LABELS, MODEL_PATH, REPORT_PATH, TARGET, load_dataset, make_candidates


def best_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1_values))
    return float(thresholds[best]), float(f1_values[best])


def main():
    X, y = load_dataset()

    # Keep the test split untouched during model and threshold selection.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    )

    candidates = make_candidates()
    candidate_results = []
    validation_thresholds = {}
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        scores = candidate.predict_proba(X_val)[:, 1]
        threshold, validation_f1 = best_f1_threshold(y_val, scores)
        validation_thresholds[name] = threshold
        candidate_results.append({
            "name": name,
            "validation_pr_auc": round(float(average_precision_score(y_val, scores)), 4),
            "validation_f1_at_tuned_threshold": round(validation_f1, 4),
            "validation_threshold": round(threshold, 4),
        })

    # PR-AUC is the preselected model-selection metric because the positive class is a minority.
    candidate_results.sort(key=lambda item: item["validation_pr_auc"], reverse=True)
    selected_name = candidate_results[0]["name"]
    threshold = validation_thresholds[selected_name]
    selected_model = candidates[selected_name]
    selected_model.fit(X_train_val, y_train_val)

    test_scores = selected_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)
    report_by_class = classification_report(
        y_test, test_predictions, labels=[0, 1],
        target_names=[LABELS[0], LABELS[1]], output_dict=True, zero_division=0,
    )
    confusion = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    true_negative, false_positive = int(confusion[0, 0]), int(confusion[0, 1])
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, test_predictions)), 4),
        "specificity": round(true_negative / (true_negative + false_positive), 4),
        "precision_positive": round(float(precision_score(y_test, test_predictions, zero_division=0)), 4),
        "recall_positive": round(float(recall_score(y_test, test_predictions, zero_division=0)), 4),
        "f1_positive": round(float(f1_score(y_test, test_predictions, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, test_predictions, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_scores)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_scores)), 4),
        "confusion_matrix": confusion.tolist(),
        "by_class": report_by_class,
    }

    # Permutation importance is computed on a fixed 5,000-row test sample for display.
    importance_n = min(5_000, len(X_test))
    importance_sample = X_test.sample(n=importance_n, random_state=42)
    importance_labels = y_test.loc[importance_sample.index]
    importance = permutation_importance(
        selected_model, importance_sample, importance_labels,
        scoring="f1_macro", n_repeats=2, random_state=42, n_jobs=1,
    )
    importances = sorted(
        [{"feature": feature, "importance": round(float(value), 5)}
         for feature, value in zip(FEATURES, importance.importances_mean)],
        key=lambda item: item["importance"], reverse=True,
    )

    report = {
        "dataset_name": "Diabetes Health Indicators Dataset (BRFSS 2015)",
        "dataset_source": "Kaggle: alexteboul/diabetes-health-indicators-dataset",
        "rows": int(len(X)),
        "predictors": len(FEATURES),
        "target": TARGET,
        "class_counts": {LABELS[int(label)]: int(count) for label, count in y.value_counts().sort_index().items()},
        "positive_prevalence": round(float(y.mean()), 4),
        "selected_model": selected_name,
        "selection_metric": "Validation average precision (PR-AUC)",
        "threshold": round(threshold, 4),
        "threshold_method": "Threshold maximizing validation F1 for the positive class",
        "split": {
            "model_selection_train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "final_training_rows": int(len(X_train_val)),
            "test_rows": int(len(X_test)),
            "random_seed": 42,
        },
        "candidate_validation_results": candidate_results,
        "test_metrics": metrics,
        "test_feature_importance": importances,
        "importance_method": f"Permutation importance on {importance_n:,} held-out records using macro-F1",
        "interpretation_note": "Survey-label classification; not future-onset prediction, diagnosis, or calibrated clinical risk.",
    }

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(selected_model, model_file)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset: {len(X):,} records · {len(FEATURES)} predictors")
    print("Validation model selection (average precision / PR-AUC):")
    for result in candidate_results:
        print(f"  {result['name']}: {result['validation_pr_auc']:.4f}")
    print(f"Selected final model: {selected_name}")
    print(f"F1-tuned threshold from validation: {threshold:.4f}")
    print("Final held-out test metrics:")
    for key, value in metrics.items():
        if isinstance(value, (float, int)):
            print(f"  {key}: {value:.4f}")
    print(f"  Confusion matrix [TN, FP; FN, TP]: {metrics['confusion_matrix']}")
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved report: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
