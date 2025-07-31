from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
import os

def evaluate_baseline(y_test, model_name, model_preds, model_probs=None, info_gain_dict=None, output_path="visuals/baseline_results.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    majority_class = Counter(y_test).most_common(1)[0][0]
    baseline_preds = [majority_class] * len(y_test)

    model_acc = accuracy_score(y_test, model_preds)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    acc_improvement = model_acc - baseline_acc

    console_results = [
        f"{model_name} Accuracy: {model_acc:.3f}",
        f"Baseline Accuracy (Majority Class): {baseline_acc:.3f}",
        f"Improvement over Baseline: {acc_improvement:.3f}"
    ]

    text_results = console_results.copy()

    if model_probs is not None:
        auc_score = roc_auc_score(y_test, model_probs)
        console_results.append(f"{model_name} AUC: {auc_score:.3f}")
        text_results.append(f"{model_name} AUC: {auc_score:.3f}")

    if info_gain_dict:
        sorted_ig = sorted(info_gain_dict.items(), key=lambda x: x[1], reverse=True)
        text_results.append("\nTop Information Gain Features:")
        for feature, ig in sorted_ig[:5]:
            text_results.append(f"{feature}: {ig:.4f}")

    # Print to console
    print("\n" + "\n".join(console_results))

    # Write to file
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\nEvaluation Results: {model_name}\n")
        f.write("-" * 40 + "\n")
        for line in text_results:
            f.write(line + "\n")
