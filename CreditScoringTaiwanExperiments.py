# This script runs the full Taiwan credit scoring expermiments (LR + XGB).
# All outputs are contained in the folder Containment (to not overwrite the original results and models mentioned in our report)

# We included the same headings from the Report: Step I -- Step X, so it will be easier to understand (its a long pıpeline)

# This script uses a CLI parser (basically you can give it arguments to execute all or a subset of experiments, because they may take a long time to run)
# -> you can flip things on/off without editing code.
# Flags like --no-xgb or --overnight tell the script what to do.

# run it from terminal like these sample use cases:

#   -Full 8-experiment run (dubbed: "overnight" because it may take some time to trin and execute all models)
#    (all 8 configs, saves metrics + preds into Containment-Folder):
#       python CreditScoringTaiwanExperiments.py --overnight --score roc_auc --results-path experiment_results.csv --preds-dir preds

#   -Run without XGB (only Logistic Regression):
#       python CreditScoringTaiwanExperiments.py --no-xgb --score roc_auc

#   -Run without FE (raw features only):
#       python CreditScoringTaiwanExperiments.py --no-fe --score roc_auc

#   -Simple run w/o runnıng any models (just download + save the data; no training):
#       python CreditScoringTaiwanExperiments.py --show-head 5 --save-csv taiwan_credit.csv --no-logreg --no-xgb


import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from joblib import dump

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from xgboost import XGBClassifier


# Step I: Data acquisition and schema standardization.
# We load the Default of Credit Card Clients dataset directly from the UCI repository.

# root folder for ALL outputs: named Containment
rootFolder = Path("Containment")
(rootFolder / "results").mkdir(parents=True, exist_ok=True)
(rootFolder / "preds").mkdir(parents=True, exist_ok=True)
(rootFolder / "models").mkdir(parents=True, exist_ok=True)
(rootFolder / "xtest").mkdir(parents=True, exist_ok=True)
(rootFolder / "data").mkdir(parents=True, exist_ok=True)

# Column mapping / cleanup (X1..X23 -> official names)
columnMappingFromX_to_DescriptiveName = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names and map X1..X23 if present
    stripped = {c: str(c).strip() for c in df.columns}
    if any(k != v for k, v in stripped.items()):
        df = df.rename(columns=stripped)
    intersect = set(columnMappingFromX_to_DescriptiveName).intersection(df.columns)
    if intersect:
        df = df.rename(
            columns={k: columnMappingFromX_to_DescriptiveName[k] for k in intersect}
        )
    return df


def load_data():
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    X = standardize_columns(X)
    y_series = y.squeeze() if hasattr(y, "squeeze") else y
    y_series = pd.Series(y_series, name="target")
    return X, y_series


def preview_data(n: int = 5, save_csv_path: str = ""):
    X, y = load_data()
    df_full = X.copy()
    df_full["target"] = y.values

    if n and n > 0:
        print("\n=== Data Preview ===")
        print(df_full.head(n))
        print("\nShape:", df_full.shape)
        print("Columns:", list(df_full.columns))
        print("Target balance (value counts):")
        print(df_full["target"].value_counts())

    if save_csv_path:
        out_file = rootFolder / "data" / Path(save_csv_path).name
        df_full.to_csv(out_file, index=False)
        print(f"\nSaved full dataset to: {out_file}")


# Step II: Exploratory data analysis (EDA) and cleaning.
# Input variables are mapped to readable labels (e.g., education, marriage, sex). We compute
# quick descriptive statistics and class balance.

# (Explorative Data Analysis EDA is done in a separate module. here we keep it minimal via preview_data())


# Step III: Feature engineering (FE).
# Ratios and aggregated features, avoiding multicollinearity:
# - Utilization UT1…UT6 = BILL_AMTt / LIMIT_BAL
# - Repayment ratio RR1…RR6 = PAY_AMTt / BILL_AMTt (clipped/robust in viz elsewhere)
# - Delinquency stats (overdue payments): late_payment_count, max_delay
# - Aggregates: total billed, total repaid, coarse age bin
def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(X.copy())

    # Credit utilization ratios (bill / limit)
    for i in range(1, 7):
        df[f"utilization_{i}"] = df[f"BILL_AMT{i}"] / (df["LIMIT_BAL"] + 1e-6)

    # mean utilization (used later for diagnostics)
    df["utilization_mean"] = df[[f"utilization_{i}" for i in range(1, 7)]].mean(axis=1)

    # Repayment ratios (pay / bill)
    for i in range(1, 7):
        df[f"repayment_ratio_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"] + 1e-6)

    # Aggregates
    df["avg_bill_amt"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].mean(axis=1)
    df["var_bill_amt"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].var(axis=1)
    df["total_repaid"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["total_billed"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)

    # Payment behavior
    df["late_payment_count"] = (df[[f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]] > 0).sum(
        axis=1
    )
    df["max_delay"] = df[[f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]].max(axis=1)

    # Bucketing: Age group
    df["age_bin"] = pd.cut(df["AGE"], bins=[20, 30, 40, 50, 60, 100], labels=False)
    return df


# Step IV: Train/test split.
# 80/20 stratified split to preserve class balance. Scaling applied *after* the split
# (leakage-safe). XGB doesn’t need scaling, but we keep it uniform to simplify the pipe.


def scale_after_split(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_s, X_test_s


# Step V: hyperparameter tuning.
# LR: grid over C, penalty, class_weight (metric selectable via --score).
# XGB: randomized search over standard parameters; 3-fold CV.


def logistic_regression_model(X_train, y_train, scoring="roc_auc"):
    # grid-search LR over C/penalty/class_weight
    log_reg = LogisticRegression(max_iter=2000, solver="liblinear")
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    }
    scorer = make_scorer(f1_score, pos_label=1) if scoring == "f1" else scoring
    grid = GridSearchCV(
        log_reg, param_grid, cv=5, scoring=scorer, n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# Step VI: Class-imbalance strategies.
# LR toggles class_weight = 'balanced' per experiment. XGB uses scale_pos_weight = neg/pos


def xgboost_model(X_train, y_train, scoring="roc_auc", balance_aware=True):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = (neg / pos) if (pos > 0 and balance_aware) else 1.0

    xgb = XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=42,
    )

    param_dist = {
        "n_estimators": [150, 250, 350],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=12,
        scoring=scoring,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# Step VII: Training and model selection.
# Best estimators from tuning are refit on full training data
# handled by GridSearchCV/RandomizedSearchCV


# Step VIII: Evaluation on the hold-out set.
# We report Accuracy, ROC-AUC, Precision, Recall + confusion matrix counts.


def evaluate_model(model, X_test, y_test, model_name="Model"):
    # verbose eval for manual runs
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    f1d = f1_score(y_test, y_pred, pos_label=1)

    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")
    print(f"F1 (Default=1): {f1d:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {"accuracy": acc, "roc_auc": auc, "f1_default": f1d}


def evaluate_metrics(model, X_test, y_test):
    # compact eval for logging
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_default": f1_score(y_test, y_pred, pos_label=1),
        "precision_default": precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "recall_default": recall_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# Step IX: Experimental study (automated 8-run grid).
# FE (on, off) x Balance (on, off) x Model (LR, XGB)
# Saves: metrics CSV/JSON, per-config preds (y_true,y_prob), trained models, and X_test.
# Everything is contained under root folder "Containment"


def run_pipeline(
    run_logreg: bool = True,
    run_xgb: bool = True,
    use_fe: bool = True,
    scoring: str = "roc_auc",
):
    # quick single-branch runner
    X, y = load_data()
    label = "Feature Eng" if use_fe else "Baseline (no FE)"
    print(f"\n##### {label} #####")
    X_proc = feature_engineering(X) if use_fe else X

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s, X_test_s = scale_after_split(X_train, X_test)

    results = {}

    if run_logreg:
        log_model, log_params = logistic_regression_model(
            X_train_s, y_train, scoring=scoring
        )
        print("\nBest Logistic Regression Params:", log_params)
        results[f"log_reg_{label}"] = evaluate_model(
            log_model, X_test_s, y_test, f"Logistic Regression ({label})"
        )

    if run_xgb:
        xgb_model_best, xgb_params = xgboost_model(
            X_train_s, y_train, scoring=scoring, balance_aware=True
        )
        print("\nBest XGBoost Params:", xgb_params)
        results[f"xgb_{label}"] = evaluate_model(
            xgb_model_best, X_test_s, y_test, f"XGBoost ({label})"
        )

    if results:
        print("\n=== Summary ===")
        for name, metrics in results.items():
            auc_txt = (
                f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"
            )
            print(
                f"{name}: Acc={metrics['accuracy']:.4f}, AUC={auc_txt}, F1={metrics['f1_default']:.4f}"
            )
    else:
        print("\n(No models were selected to run.)")


def run_overnight_experiments(scoring="roc_auc", results_path=None, preds_dir="preds"):
    # force preds under Containment regardless of flag path
    preds_dir = (
        rootFolder / "preds" if not preds_dir else rootFolder / Path(preds_dir).name
    )
    preds_dir.mkdir(parents=True, exist_ok=True)

    """
    Runs 8 experiments:
      - Model in logreg, xgb
      - use_fe in False, True
      - balance_aware in False, True
    Saves CSV + JSON with metrics and best params (under Containment/results).
    """
    X, y = load_data()

    rows = []
    start_all = time.time()

    for model_name in ["logreg", "xgb"]:
        for use_fe in [False, True]:
            # features
            X_in = feature_engineering(X) if use_fe else X

            # split BEFORE scaling
            X_train, X_test, y_train, y_test = train_test_split(
                X_in, y, test_size=0.2, random_state=42, stratify=y
            )
            # scale AFTER split
            X_train_s, X_test_s = scale_after_split(X_train, X_test)

            for balance_aware in [False, True]:
                label = {
                    "model": model_name,
                    "use_fe": use_fe,
                    "balance_aware": balance_aware,
                    "scoring": scoring,
                }

                print(f"\n=== Running {label} ===")
                t0 = time.time()

                # train
                if model_name == "logreg":
                    # LR: class_weight fixed by balance_aware
                    cw = "balanced" if balance_aware else None
                    lr = LogisticRegression(
                        max_iter=2000, solver="liblinear", class_weight=cw
                    )
                    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]}
                    scorer = (
                        make_scorer(f1_score, pos_label=1)
                        if scoring == "f1"
                        else scoring
                    )
                    search = GridSearchCV(
                        lr, param_grid, cv=5, scoring=scorer, n_jobs=-1, refit=True
                    )
                    search.fit(X_train_s, y_train)
                    best_model, best_params = (
                        search.best_estimator_,
                        search.best_params_,
                    )
                else:
                    best_model, best_params = xgboost_model(
                        X_train_s, y_train, scoring=scoring, balance_aware=balance_aware
                    )

                # label matching plotting script
                label_str = config_label(model_name, use_fe, balance_aware)

                # Probabilities for positive class
                if hasattr(best_model, "predict_proba"):
                    y_prob = best_model.predict_proba(X_test_s)[:, 1]
                else:
                    y_prob = None

                # save preds for ROC/PR
                if y_prob is not None:
                    pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob}).to_csv(
                        preds_dir / f"{label_str}.csv", index=False
                    )

                # save model (important for some graphics)
                models_dir = rootFolder / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                dump(best_model, models_dir / f"{label_str}.joblib")

                xtest_dir = rootFolder / "xtest"
                xtest_dir.mkdir(parents=True, exist_ok=True)
                try:
                    X_test_s.to_parquet(xtest_dir / f"{label_str}.parquet")
                except Exception:
                    # fallback
                    X_test_s.to_csv(xtest_dir / f"{label_str}.csv", index=False)

                # evaluate
                metrics = evaluate_metrics(best_model, X_test_s, y_test)
                elapsed = time.time() - t0

                row = {
                    **label,
                    "best_params": json.dumps(best_params),
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "f1_default": metrics["f1_default"],
                    "precision_default": metrics["precision_default"],
                    "recall_default": metrics["recall_default"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                    "runtime_sec": round(elapsed, 2),
                }
                rows.append(row)

                print(f"Best params: {best_params}")
                print(
                    f"Metrics: Acc={metrics['accuracy']:.4f}, "
                    f"AUC={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}, "
                    f"F1={metrics['f1_default']:.4f}, "
                    f"Rec={metrics['recall_default']:.4f}, "
                    f"Prec={metrics['precision_default']:.4f} "
                    f"(runtime {elapsed / 60:.1f} min)"
                )

    # save results under Containment/results
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = rootFolder / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        (results_dir / Path(results_path).name)
        if results_path
        else (results_dir / f"overnight_results_{ts}.csv")
    )
    json_path = csv_path.with_suffix(".json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved results to: {csv_path} and {json_path}")
    print(f"Total wall time: {(time.time() - start_all) / 3600:.2f} hours")


def config_label(model_name: str, use_fe: bool, balance_aware: bool) -> str:
    mdl = "LR" if model_name == "logreg" else "XGB"
    fe = "FE" if use_fe else "NoFE"
    bal = "Bal" if balance_aware else "NoBal"
    return f"{mdl}-{fe}-{bal}"


# Step X — Interpretation and diagnostics.
# Saved models, X_test, and preds are used by EvaluationPlots.py


# CLI parser to customize the run (perform all 8 experiments: "overnight" / Select which experiments to omit)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring (Taiwan) experiments")
    parser.add_argument(
        "--no-logreg", action="store_true", help="Disable Logistic Regression"
    )
    parser.add_argument("--no-xgb", action="store_true", help="Disable XGBoost")
    parser.add_argument(
        "--no-fe", action="store_true", help="Disable Feature Engineering"
    )
    parser.add_argument(
        "--show-head", type=int, default=0, help="Print first N rows of the raw data"
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="",
        help="Save full dataset (features+target) to CSV (saved inside ./Containment/data)",
    )
    parser.add_argument(
        "--debug-cols",
        action="store_true",
        help="Print final feature columns after standardization and exit",
    )
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Run all experiments (LR & XGB) for FE/no-FE and balance/no-balance",
    )
    parser.add_argument(
        "--score",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "accuracy", "f1"],
        help="Scoring metric for tuning",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="",
        help="Optional CSV name for overnight results (saved inside ./Containment/results)",
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        default="preds",
        help="Subfolder name for per-config preds (saved inside ./Containment)",
    )
    args = parser.parse_args()

    # OBSOLETE (since we fixed the issue) debug of final column names
    if args.debug_cols:
        X_dbg, _ = load_data()
        X_dbg = standardize_columns(X_dbg)
        print("\nColumns (repr):", repr(list(X_dbg.columns)))
        import sys

        sys.exit(0)

    # Preview/export of header
    if args.show_head or args.save_csv:
        preview_data(n=args.show_head, save_csv_path=args.save_csv)

    # Overnight batch (8 runs)
    if args.overnight:
        rp = args.results_path if args.results_path else None
        run_overnight_experiments(
            scoring=args.score, results_path=rp, preds_dir=args.preds_dir
        )
        import sys

        sys.exit(0)

    # manual single-branch (toggle via flags)
    run_pipeline(
        run_logreg=not args.no_logreg,
        run_xgb=not args.no_xgb,
        use_fe=not args.no_fe,
        scoring=args.score,
    )
