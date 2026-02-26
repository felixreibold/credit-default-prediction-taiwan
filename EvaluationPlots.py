# This script does generates the vizualization for experiment outputs:
# PR-curve and ROC
# Confusıon Matrix for each of the 8 experiments
# Feature Importance and Logistic Regression coefficients

# run it via terminal command: python EvaluationPlots.py

# you can find the output figuers in the folder FiguresForEvaluation (<-- this is for submission)
# If you run this script the figures get saved to a folder named exp_figs2

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from glob import glob

from joblib import load as joblib_load
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

# configurations
resultsPath = "experiment_results.csv"  # reads overall metrics
outputpath = Path("exp_figs2")
predsPath = Path("preds")  # expects preds/<label>.csv with y_true,y_prob
modelsPath = Path("models")  # expects models/<label>.joblib
xtestPath = Path("xtest")  # expects xtest/<label>.parquet OR .csv
outputpath.mkdir(exist_ok=True, parents=True)

# color scheme
colorScheme = plt.cm.tab10

# two “best” perfomance models are these (we figured that out after some trıal and error :)
bestExperimentsLabels = ["LR-FE-Bal", "XGB-FE-Bal"]


# figure styles need to be same accross all fıgures
def set_style_of_figures():
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "axes.titlepad": 8,
        }
    )


# helper to lable bars (small numbers for the bar length on top, otherwise hard to quantify the differences)
def label_the_barcharts(ax, values, fmt="{:.3f}", dy=None):
    v = np.asarray(values, dtype=float)
    if dy is None:
        vmin, vmax = float(np.min(v)), float(np.max(v))
        dy = 0.02 * (vmax - vmin) if vmax > vmin else 0.01
    for i, val in enumerate(v):
        ax.text(i, val + dy, fmt.format(val), ha="center", va="bottom", fontsize=9)


def save_fig(fig, name):
    path = outputpath / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[Fig] Saved: {path}")
    plt.close(fig)


# data wrangling
def _label_row(r):
    model = "LR" if str(r["model"]).lower().startswith("log") else "XGB"
    fe = "FE" if bool(r["use_fe"]) else "NoFE"
    bal = "Bal" if bool(r["balance_aware"]) else "NoBal"
    return f"{model}-{fe}-{bal}"


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    for c in ["use_fe", "balance_aware"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    df["label"] = df.apply(_label_row, axis=1)

    # Balanced Accuracy from confusion columns
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = df["tp"] / (df["tp"] + df["fn"]).replace({0: np.nan})
        tnr = df["tn"] / (df["tn"] + df["fp"]).replace({0: np.nan})
    df["balanced_accuracy"] = ((tpr.fillna(0) + tnr.fillna(0)) / 2).astype(float)
    return df


###########
# PLOTS ### sınce we produce several plots of the same kınd i.e. confusıon matrices we encapsulate that functionality here
##########


def bar_plot(df, metric, title, filename, annotate=False, fmt="{:.3f}"):
    d = df.sort_values(metric, ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    vals = d[metric].values
    colors = colorScheme(np.arange(len(d)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"].tolist(), rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    if annotate:
        label_the_barcharts(ax, vals, fmt=fmt)
    save_fig(fig, filename)


def confusion_heatmap(tn, fp, fn, tp, title, filename):
    mat = np.array([[tn, fp], [fn, tp]])
    fig = plt.figure(figsize=(4.8, 4.3))
    ax = plt.gca()
    im = ax.imshow(mat, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)
    save_fig(fig, filename)


def maybe_load_preds(label):
    f = predsPath / f"{label}.csv"
    if not f.exists():
        return None, None
    d = pd.read_csv(f)
    if {"y_true", "y_prob"} <= set(d.columns):
        return d["y_true"].to_numpy(), d["y_prob"].to_numpy()
    return None, None


def plot_roc_all(labels):
    plt.figure(figsize=(7.5, 6))
    plotted = 0
    for lab in labels:
        y_true, y_prob = maybe_load_preds(lab)
        if y_true is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lab} (AUC={roc_auc:.3f})")
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outputpath / "07_roc_curves_all.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_all(labels):
    plt.figure(figsize=(7.5, 6))
    plotted = 0
    for lab in labels:
        y_true, y_prob = maybe_load_preds(lab)
        if y_true is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(rec, prec, label=f"{lab} (AUPRC={ap:.3f})")
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curves")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outputpath / "08_pr_curves_all.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep(label, filename):
    y_true, y_prob = maybe_load_preds(label)
    if y_true is None:
        return
    thresholds = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    fig = plt.figure(figsize=(7.5, 5.2))
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold sweep — {label}")
    plt.legend()
    save_fig(fig, filename)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return ece


def plot_calibration(label, filename, n_bins=10):
    y_true, y_prob = maybe_load_preds(label)
    if y_true is None:
        return
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    fig = plt.figure(figsize=(5.8, 5.2))
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.plot(mean_pred, frac_pos, marker="o", lw=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration — {label} (ECE={ece:.3f})")
    save_fig(fig, filename)


# LR & XGB interpretability analysıs
# for LR we plot the coefficients
# for XGB we use an interpretability plot


def load_model_for_label(label):
    f = modelsPath / f"{label}.joblib"
    return joblib_load(f) if f.exists() else None


def load_xtest_for_label(label):
    f_parq = xtestPath / f"{label}.parquet"
    f_csv = xtestPath / f"{label}.csv"
    if f_parq.exists():
        return pd.read_parquet(f_parq)
    if f_csv.exists():
        return pd.read_csv(f_csv)
    return None


def lr_coef_table_smx(X, y, feature_names):
    import statsmodels.api as sm  # import here to keep top tidy

    X_ = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_).fit(disp=False)
    coefs = model.params
    ses = model.bse
    lo = coefs - 1.96 * ses
    hi = coefs + 1.96 * ses
    or_ = np.exp(coefs)
    or_lo = np.exp(lo)
    or_hi = np.exp(hi)
    df = pd.DataFrame(
        {
            "feature": ["Intercept"] + feature_names,
            "beta": coefs.values,
            "se": ses.values,
            "ci_low": lo.values,
            "ci_high": hi.values,
            "odds_ratio": or_.values,
            "or_ci_low": or_lo.values,
            "or_ci_high": or_hi.values,
        }
    )
    return df[df["feature"] != "Intercept"].sort_values(
        "odds_ratio", key=np.abs, ascending=False
    )


def save_table_markdown(df, filename, top_k=15):
    if df is None or df.empty:
        return
    df_ = df.head(top_k).copy()
    md = "| feature | beta | odds_ratio | 95% CI (beta) |\n|---|---:|---:|---|\n"
    for r in df_.itertuples(index=False):
        md += f"| {r.feature} | {r.beta:.4f} | {r.odds_ratio:.3f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |\n"
    (outputpath / filename).write_text(md, encoding="utf-8")


def plot_lr_coef_importance(
    model, X=None, y=None, feature_names=None, base_label="LR", top_k=20
):
    assert feature_names is not None, "feature_names required"
    beta = np.ravel(model.coef_)
    order = np.argsort(np.abs(beta))[::-1][: min(top_k, len(beta))]
    names_top = [feature_names[i] for i in order]
    beta_top = beta[order]
    or_top = np.exp(beta_top)

    have_ci = False
    try:
        import statsmodels.api as sm

        if X is not None and y is not None:
            X_ = sm.add_constant(
                X.values if hasattr(X, "values") else X, has_constant="add"
            )
            m = sm.Logit(y, X_).fit(disp=False)
            params = pd.Series(m.params[1:].values, index=feature_names)
            ses = pd.Series(m.bse[1:].values, index=feature_names)
            beta_ci_lo = (params - 1.96 * ses).reindex(names_top).values
            beta_ci_hi = (params + 1.96 * ses).reindex(names_top).values
            have_ci = True
    except Exception:
        have_ci = False

    y_pos = np.arange(len(names_top))
    fig, ax = plt.subplots(figsize=(8.8, 0.44 * len(names_top) + 2))

    if have_ci:
        colors = np.where(beta_top >= 0, "#1f77b4", "#d62728")
        ax.barh(y_pos, beta_top, color=colors, alpha=0.25)
        xerr = np.vstack([beta_top - beta_ci_lo, beta_ci_hi - beta_top])
        ax.errorbar(beta_top, y_pos, xerr=xerr, fmt="o", color="k", capsize=3, zorder=3)
        title = f"LR coefficients (±95% CI) — {base_label}"
        min_x = float(np.min(beta_ci_lo))
        max_x = float(np.max(beta_ci_hi))
    else:
        ax.barh(y_pos, beta_top, color=colorScheme(np.arange(len(beta_top))))
        title = f"LR coefficients — {base_label}"
        min_x = float(np.min(beta_top))
        max_x = float(np.max(beta_top))

    span = max_x - min_x
    pad = 0.20 * span if span > 0 else 1.0
    ax.set_xlim(min_x - pad, max_x + pad)

    # Labels next to text next to bars: the LR coefficients can be useful for probabılıstic ınterpratation
    off = 0.02 * span if span > 0 else 0.05
    for i, b in enumerate(beta_top):
        txt = f"β={b:+.3f}  OR={or_top[i]:.2f}"
        ax.text(
            b + (off if b >= 0 else -off),
            y_pos[i],
            txt,
            va="center",
            ha="left" if b >= 0 else "right",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_top)
    ax.invert_yaxis()
    ax.axvline(0.0, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Coefficient (β) — features assumed standardized")
    ax.set_title(title)

    save_fig(fig, f"{base_label}_LR_coef_importance.png")


def xgb_shap_plots(model, X, feature_names, base_label="XGB"):
    try:
        import shap
    except Exception:
        print("[XGB] shap not available; skipping SHAP plots.")
        return

    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)

    plt.figure()
    shap.summary_plot(sv, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"Feature importance (SHAP) — {base_label}")
    plt.tight_layout()
    plt.savefig(
        outputpath / f"{base_label}_shap_summary_bar.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    for feat in ["PAY_0", "utilization_mean"]:
        if feat in feature_names:
            plt.figure()
            shap.dependence_plot(feat, sv, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(
                outputpath / f"{base_label}_shap_dep_{feat}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


#########
# MAIN ####
###########


def main():
    set_style_of_figures()

    if not Path(resultsPath).exists():
        raise FileNotFoundError(f"CSV not found: {resultsPath}")

    df = load_results(resultsPath)

    # 1. Bar charts for various accuracy metrics ROC-AUC, overall Accuracy, balanced Accuracy, Precision and Recall
    bar_plot(
        df,
        "roc_auc",
        "ROC-AUC by configuration",
        "01_auc_bars.png",
        annotate=True,
        fmt="{:.3f}",
    )
    bar_plot(
        df,
        "f1_default",
        "F1 (positive class) by configuration",
        "02_f1_bars.png",
        annotate=True,
        fmt="{:.3f}",
    )
    bar_plot(
        df,
        "precision_default",
        "Precision (positive) by configuration",
        "03_precision_bars.png",
        annotate=True,
        fmt="{:.3f}",
    )
    bar_plot(
        df,
        "recall_default",
        "Recall (positive) by configuration",
        "04_recall_bars.png",
        annotate=True,
        fmt="{:.3f}",
    )
    bar_plot(
        df,
        "accuracy",
        "Accuracy by configuration",
        "05_accuracy_bars.png",
        annotate=True,
        fmt="{:.3f}",
    )
    bar_plot(
        df,
        "balanced_accuracy",
        "Balanced Accuracy by configuration",
        "06_balacc_bars.png",
        annotate=True,
        fmt="{:.3f}",
    )
    print("[OK] Saved metric comparison bars.")

    # 2. Confusion matrices for ALL EXPERIMENTS
    for lab in bestExperimentsLabels:
        row = df.loc[df["label"] == lab]
        if row.empty:
            print(f"[Warn] config not found in CSV: {lab}")
            continue
        r = row.iloc[0]
        confusion_heatmap(
            int(r.tn),
            int(r.fp),
            int(r.fn),
            int(r.tp),
            title=f"Confusion Matrix — {lab}",
            filename=f"CM_{lab}.png",
        )
    print("[OK] Saved confusion matrices for best models.")

    for r in df.itertuples(index=False):
        confusion_heatmap(
            int(r.tn),
            int(r.fp),
            int(r.fn),
            int(r.tp),
            title=f"Confusion Matrix — {r.label}",
            filename=f"CM_{r.label}.png",
        )
    print("[OK] Saved confusion matrices for ALL configurations.")

    # 3. Curves (need per-sample preds)
    labels_all = df["label"].tolist()
    plot_roc_all(labels_all)
    plot_pr_all(labels_all)
    for lab in bestExperimentsLabels:
        plot_threshold_sweep(lab, f"09_threshold_sweep_{lab}.png")
        plot_calibration(lab, f"10_calibration_{lab}.png")
    print("[OK] ROC/PR/threshold/calibration plotted where preds/* exist.")

    # 4. Interpretability
    for lab in bestExperimentsLabels:
        model = load_model_for_label(lab)
        Xtest = load_xtest_for_label(lab)
        if model is None or Xtest is None:
            print(
                f"[Warn] Skipping interpretability for {lab} (missing model or X_test)."
            )
            continue
        feature_names = list(Xtest.columns)

        # LR coeff
        from sklearn.linear_model import LogisticRegression

        if isinstance(model, LogisticRegression):
            y_true, _ = maybe_load_preds(lab)

            plot_lr_coef_importance(
                model=model,
                X=Xtest,
                y=y_true,
                feature_names=feature_names,
                base_label=lab.replace("-", "_"),
                top_k=20,
            )

            try:
                ci_df = lr_coef_table_smx(Xtest.values, y_true, feature_names)
                save_table_markdown(ci_df, f"{lab}_LR_coef_table.md", top_k=15)
            except Exception:
                pass

        # XGB Importance
        else:
            xgb_shap_plots(
                model, Xtest, feature_names, base_label=lab.replace("-", "_")
            )
            print(f"[OK] Saved XGB SHAP plots for {lab}.")

    print("Finished")


if __name__ == "__main__":
    main()
