"""
Used Car Price Prediction
-------------------------
Predicts the selling price (in lakhs INR) of used cars from the CarDekho dataset.
Compares Linear Regression, Lasso, Random Forest and Gradient Boosting using
5-fold cross-validation plus a held-out test set.

Run:  python car_price_prediction.py
Needs: pandas, numpy, scikit-learn, matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- config
DATA_PATH = Path(__file__).parent / "car_data.csv"  # keep the CSV next to this script
OUTPUT_DIR = Path(__file__).parent / "outputs"
REFERENCE_YEAR = 2020  # the dataset was collected around 2020
RANDOM_STATE = 42

CATEGORICAL = ["Fuel_Type", "Seller_Type", "Transmission"]
SKEWED = ["Present_Price", "Kms_Driven"]  # long right tails -> log-transform
NUMERIC = ["Car_Age", "Owner"]
TARGET = "Selling_Price"


# ---------------------------------------------------------------- data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print("Missing values:", int(df.isnull().sum().sum()))

    # Car age is easier for a model to use than a raw year
    df["Car_Age"] = REFERENCE_YEAR - df["Year"]
    return df.drop(columns=["Car_Name", "Year"])


def build_preprocessor() -> ColumnTransformer:
    # One-hot encoding avoids implying a false order (e.g. CNG > Diesel > Petrol)
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), SKEWED),
        ("num", StandardScaler(), NUMERIC),
    ])


def build_models() -> dict:
    def pipe(model):
        # Prices are right-skewed, so the model learns log(price) and
        # predictions are converted back to lakhs automatically
        return TransformedTargetRegressor(
            regressor=Pipeline([("prep", build_preprocessor()), ("model", model)]),
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return {
        "Linear Regression": pipe(LinearRegression()),
        "Lasso": pipe(Lasso(alpha=0.001, max_iter=10_000)),
        "Random Forest": pipe(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)),
        "Gradient Boosting": pipe(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


# ---------------------------------------------------------------- evaluation
def evaluate(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "CV R2 (mean)": cv_r2.mean(),
            "CV R2 (std)": cv_r2.std(),
            "Train R2": metrics.r2_score(y_train, model.predict(X_train)),
            "Test R2": metrics.r2_score(y_test, pred),
            "Test MAE (lakhs)": metrics.mean_absolute_error(y_test, pred),
            "Test RMSE (lakhs)": metrics.mean_squared_error(y_test, pred) ** 0.5,
        })
    return pd.DataFrame(rows).set_index("Model").sort_values("Test R2", ascending=False)


def plot_predictions(models: dict, X_test, y_test, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    lim = [0, max(y_test.max(), 1) * 1.05]
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        pred = model.predict(X_test)
        ax.scatter(y_test, pred, alpha=0.7, edgecolor="k")
        ax.plot(lim, lim, "r--", label="Perfect prediction")
        ax.set(xlim=lim, ylim=lim, xlabel="Actual price (lakhs)", ylabel="Predicted price (lakhs)",
               title=f"{name}  (R² = {metrics.r2_score(y_test, pred):.3f})")
        ax.legend()
    fig.suptitle("Actual vs Predicted Selling Price (test set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=150)
    plt.show()


def plot_feature_importance(model, out_dir: Path) -> None:
    pipeline = model.regressor_
    names = pipeline.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    importance = pd.Series(pipeline.named_steps["model"].feature_importances_, index=names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot.barh(ax=ax, color="steelblue")
    ax.set(title="Feature importance (Gradient Boosting)", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------- prediction
def predict_price(model, *, present_price, kms_driven, year, fuel="Petrol",
                  seller="Dealer", transmission="Manual", owner=0) -> float:
    car = pd.DataFrame([{
        "Present_Price": present_price, "Kms_Driven": kms_driven, "Owner": owner,
        "Car_Age": REFERENCE_YEAR - year, "Fuel_Type": fuel,
        "Seller_Type": seller, "Transmission": transmission,
    }])
    return float(model.predict(car)[0])


# ---------------------------------------------------------------- main
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data(DATA_PATH)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = build_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)

    pd.set_option("display.float_format", "{:.3f}".format)
    print("\nModel comparison\n" + "=" * 90)
    print(results.to_string())
    results.to_csv(OUTPUT_DIR / "model_results.csv")

    plot_predictions(models, X_test, y_test, OUTPUT_DIR)
    plot_feature_importance(models["Gradient Boosting"], OUTPUT_DIR)

    best = results.index[0]
    price = predict_price(models[best], present_price=9.85, kms_driven=6900, year=2017)
    print(f"\nExample ({best}): a 2017 petrol manual, 6,900 km, new price 9.85 lakhs "
          f"-> predicted resale {price:.2f} lakhs")


if __name__ == "__main__":
    main()
