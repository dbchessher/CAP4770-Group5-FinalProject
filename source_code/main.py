# main.py
from mushroom_preprocess import download_mushroom_data
from mushroom_cleaning import load_and_clean_data
from decision_tree import run_decision_tree
from super_tree import run_supertree
from random_forest import run_random_forest
from sunburst import generate_sunburst
from baseline import evaluate_baseline
from calculate_information_gain import compute_information_gain

def main():
    print("Downloading dataset...")
    data_path = download_mushroom_data(data_dir="data/")

    print("Cleaning and encoding...")
    X, y = load_and_clean_data(data_path)

    print("Running Decision Tree...")
    dt_model, dt_X_test, dt_y_test = run_decision_tree(X, y, output_path="visuals/decision_tree.png")

    print("Running SuperTree...")
    st_model, st_X_test, st_y_test = run_supertree(X, y, output_path="visuals/super_tree.png")

    print("Running Random Forest...")
    rf_model, rf_X_test, rf_y_test = run_random_forest(X, y, output_path="visuals/random_forest_importance.png")

    print("Generating Sunburst Chart...")
    generate_sunburst(data_path, output_path="visuals/sunburst_chart.html")

    print("Calculating Information Gain...")
    info_gain = compute_information_gain(X, y)

    print("Evaluating Decision Tree...")
    evaluate_baseline(
        dt_y_test,
        model_name="Decision Tree",
        model_preds=dt_model.predict(dt_X_test),
        model_probs=dt_model.predict_proba(dt_X_test)[:, 1],
        # info_gain_dict=info_gain
    )

    print("Evaluating SuperTree...")
    evaluate_baseline(
        st_y_test,
        model_name="SuperTree",
        model_preds=st_model.predict(st_X_test),
        model_probs=st_model.predict_proba(st_X_test)[:, 1],
        # info_gain_dict=info_gain
    )

    print("Evaluating Random Forest...")
    evaluate_baseline(
        rf_y_test,
        model_name="Random Forest",
        model_preds=rf_model.predict(rf_X_test),
        model_probs=rf_model.predict_proba(rf_X_test)[:, 1],
        info_gain_dict=info_gain
    )

if __name__ == "__main__":
    main()
