import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import yaml
import joblib


REPO_ROOT = Path(__file__).resolve().parents[2]  # adjust depth if needed
CONFIG_PATH = REPO_ROOT / "config.yaml"

def load_config(config_path: str):
    """Load and parse YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_path(rel_path):
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(base_dir, rel_path))

    # Resolve all paths
    paths = {k: resolve_path(v) for k, v in config.get("paths", {}).items()}
    return paths, config

def pw10(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / y_true < 0.10) * 100

def get_grad_boosting_model(filename: str):
    df = pd.read_csv(filename)
    df = df.dropna(subset=['weight', 'estimated_weight_kg_aug', 'height_cm', 'sex'])

    # kg to lb
    df['estimated_weight_lb'] = df['estimated_weight_kg_aug'] * 2.20462

    features = ['estimated_weight_lb', 'height_cm', 'sex']
    target = 'weight'  # already in lb

    numeric_features = ['estimated_weight_lb', 'height_cm']
    categorical_features = ['sex']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features) # male 0, female 1
        ]
    )

    # K fold cross val
    # split data into sets, uses set as a test set once (187/7 ~ 26 points per set)
    kf = KFold(n_splits=7, shuffle=True, random_state=42)

    mae_list, rmse_list, r2_list, pw10_list = [], [], [], []
    y_true_all, y_pred_all, sex_all, est_all = [], [], [], []

    male_pw10_list = []
    female_pw10_list = []

    # train per fold
    for train_idx, test_idx in kf.split(df):
        X_train, X_test = df.iloc[train_idx][features], df.iloc[test_idx][features]
        y_train, y_test = df.iloc[train_idx][target], df.iloc[test_idx][target]

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_list.append(mean_absolute_error(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_list.append(r2_score(y_test, y_pred))
        pw10_list.append(pw10(y_test, y_pred))

        # Store for plotting and saving
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        sex_all.extend(X_test['sex'])
        est_all.extend(X_test['estimated_weight_lb'])

        # masks for this fold
        male_mask = X_test['sex'] == 'male'
        female_mask = X_test['sex'] == 'female'

        # male PW10
        if male_mask.sum() > 0:
            male_pw10_list.append(pw10(y_test[male_mask], y_pred[male_mask]))

        # female PW10
        if female_mask.sum() > 0:
            female_pw10_list.append(pw10(y_test[female_mask], y_pred[female_mask]))

    print("Cross-validated metrics (mean ± std):")
    print(f"PW10: {np.mean(pw10_list):.2f} ± {np.std(pw10_list):.2f} %")
    print("Male PW10:", np.mean(male_pw10_list), "±", np.std(male_pw10_list))
    print("Female PW10:", np.mean(female_pw10_list), "±", np.std(female_pw10_list))

    df_results = pd.DataFrame({
        'weight': y_true_all,
        'corrected_weight': y_pred_all,
        'estimated_weight': est_all,
        'sex': sex_all
    })

    df_results['height_cm'] = df.loc[df_results.index, 'height_cm']  # add height
    df_results_to_save = df_results[['weight', 'sex', 'height_cm', 'estimated_weight', 'corrected_weight']]
    df_results_to_save.to_csv("corrected_weights_kfold.csv", index=False)

    # save one final model
    final_model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
            ))
        ])

    X = df[features]
    y = df[target]

    final_model.fit(X, y)
    joblib.dump(final_model, "bias_correction_model.pkl")

def apply_bias_correction(configs, weight, sex, height):
    model_pkl = configs["paths"]["bias_correction_model"]
    model = joblib.load(model_pkl)

    # Ensure the input matches the feature names used in training
    df_input = pd.DataFrame([{
        "estimated_weight_lb": weight * 2.20462,  # convert kg -> lb if needed
        "height_cm": height,
        "sex": sex
    }])

    # Predict corrected weight
    corrected_weight_lbs = model.predict(df_input)[0]
    corrected_weight_kg = corrected_weight_lbs * 0.45359
    print(f"Final corrected weight is: {corrected_weight_kg} kg")

    return corrected_weight_kg

if __name__ == "__main__":
    paths, config = load_config(CONFIG_PATH)
    get_grad_boosting_model("/Users/adeleyounis/Desktop/Capstone/wAI/3D-processing/bias_correction/batched_all.csv")
    weight = 63.5
    apply_bias_correction(config, weight, "female", 155)