# This script does the initial explorative data vizualization

# run it via terminal command: python DataAnalysis.py

# you can find the output figures in the folder FiguresForDataExploration
# the code also generates console outputs representing the figures

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# here we specify where to get the data from and where to save the outputs
datapath = "taiwan_credit.csv"
outputpath = "FiguresForEvaluation2"
saveFigs = True
showFigsForDebug = False
os.makedirs(outputpath, exist_ok=True)


# color scheme
colorScheme = plt.cm.tab10


# figure should all have the same dimensions and font size (readability)
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


# helper function to put labels on top of bars
def label_the_barcharts(ax, fmt="int", offset=3):
    for p in ax.patches:
        h = p.get_height()
        if h is None or np.isnan(h):
            continue
        if fmt == "int":
            txt = f"{int(round(h)):,}"
        elif fmt == "pct":
            txt = f"{h * 100:.1f}%"
        else:
            txt = f"{h:.2f}"
        ax.annotate(
            txt,
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            xytext=(0, offset),
            textcoords="offset points",
        )


def save_fig(fig, name):
    if saveFigs:
        path = os.path.join(outputpath, name)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        print(f"[Fig] Saved: {path}")
    if showFigsForDebug:
        plt.show()
    else:
        plt.close(fig)


# data import (needs taiwan_credit.csv to be downloaded once, we do it in the main ML script CreditScoringTaiwanExperiments.py)
df = pd.read_csv(datapath)

expected_cols = {
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "target",
}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {sorted(missing)}")

# some of the data needs to be relabeled
sex_map = {1: "Male", 2: "Female"}
df["SEX_L"] = df["SEX"].map(sex_map).fillna("Unknown")

edu_map = {
    1: "Graduate School",
    2: "University",
    3: "High School",
    4: "Others",
    0: "Others",
    5: "Others",
    6: "Others",
}
df["EDUCATION_L"] = df["EDUCATION"].map(edu_map).fillna("Others")

mar_map = {1: "Married", 2: "Single", 3: "Others", 0: "Unknown"}
df["MARRIAGE_L"] = df["MARRIAGE"].map(mar_map).fillna("Others")

# we define what a delinquency is (basically a late payment in last month)
df["DELINQ_LAST_M"] = (df["PAY_0"] > 0).astype(int)
pay_cols = [
    c for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in df.columns
]
df["late_payment_count"] = (df[pay_cols] > 0).sum(axis=1)

# we vizualize utilization and repayment ratios
for i in range(1, 7):
    df[f"utilization_{i}"] = df[f"BILL_AMT{i}"] / (df["LIMIT_BAL"] + 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"].replace(0, np.nan)
    df[f"repayment_ratio_{i}"] = r.clip(0, 5).fillna(0)

df["utilization_mean"] = df[[f"utilization_{i}" for i in range(1, 7)]].mean(axis=1)
df["repayment_ratio_mean"] = df[[f"repayment_ratio_{i}" for i in range(1, 7)]].mean(
    axis=1
)

set_style_of_figures()

# print some information on the ratio of classes in the data (how many default cases vs. non-default)
N = len(df)
n1 = int(df["target"].sum())
p1 = 100 * df["target"].mean()
print("[Gist] Target distribution")
print(f"- Total samples: {N:,}")
print(f"- Defaults (1): {n1:,} ({p1:.1f}%)")
print(f"- Non-defaults (0): {N - n1:,} ({100 - p1:.1f}%)\n")


##################
# PLOTTING-TIME ##
#################


# 1. Class distribution (variable ıs called traget)
fig = plt.figure()
counts = df["target"].value_counts().sort_index()
ax = plt.gca()
ax.bar(
    counts.index.astype(str), counts.values, color=colorScheme(np.arange(len(counts)))
)
ax.set_title("Target Distribution (Default vs Non-Default)")
ax.set_xlabel("Default (1) / Non-Default (0)")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
label_the_barcharts(ax, fmt="int")
save_fig(fig, "01_target_distribution.png")

# 2. education distribution (below Fig. 3 shows the default rate per education)
fig = plt.figure()
order = ["Graduate School", "University", "High School", "Others"]
vals = df["EDUCATION_L"].value_counts().reindex(order).fillna(0)
ax = plt.gca()
ax.bar(vals.index, vals.values, color=colorScheme(np.arange(len(vals))))
ax.set_title("Education Level Distribution")
ax.set_xlabel("Education")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
label_the_barcharts(ax, fmt="int")
save_fig(fig, "02_education_distribution.png")

# 3. default rate by education
fig = plt.figure()
edu_rate = df.groupby("EDUCATION_L")["target"].mean().reindex(order)
ax = plt.gca()
ax.bar(edu_rate.index, edu_rate.values, color=colorScheme(np.arange(len(edu_rate))))
ax.set_title("Default Rate by Education")
ax.set_xlabel("Education")
ax.set_ylabel("Default Rate")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
label_the_barcharts(ax, fmt="pct")
save_fig(fig, "03_default_rate_by_education.png")

# 4. PAY_0 distribution (visualızes: how many people payed on tıme, how many were late, how many pre-payed?)
fig = plt.figure()
pay0_counts = df["PAY_0"].value_counts().sort_index()
ax = plt.gca()
ax.bar(
    pay0_counts.index.astype(str),
    pay0_counts.values,
    color=colorScheme(np.arange(len(pay0_counts))),
)
ax.set_title("PAY_0 (Recent Month Repayment Status) Distribution")
ax.set_xlabel("PAY_0 code (-1/0 = duly, >0 = months delayed)")
ax.set_ylabel("Count")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
label_the_barcharts(ax, fmt="int")
save_fig(fig, "04_PAY0_distribution.png")

# 5. LIMIT_BAL histogram
fig = plt.figure()
ax = plt.gca()
ax.hist(df["LIMIT_BAL"], bins=40, color="skyblue", edgecolor="black")
ax.set_title("Credit Limit (LIMIT_BAL) Histogram")
ax.set_xlabel("LIMIT_BAL (NT$)")
ax.set_ylabel("Count")
save_fig(fig, "05_limit_bal_hist.png")

# 6. Top correlations bar chart
num = df.select_dtypes(include=[np.number])
corrs = num.corr(numeric_only=True)["target"].drop("target")
corrs = corrs.drop(index="DELINQ_LAST_M", errors="ignore")
top9_idx = corrs.abs().sort_values(ascending=False).head(9).index
vals = corrs.loc[top9_idx]

plt.figure(figsize=(7, 4))
colors_scheme = colorScheme(np.arange(len(vals)))
ax = plt.bar(vals.index, vals.values, color=colors_scheme)
plt.title("Top Correlations with Default")
plt.ylabel("Pearson correlation (r)")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(outputpath, "06_top_corrs_excl_DELINQ_LAST_M.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 7. Default rate by gender
fig = plt.figure()
gender_rates = df.groupby("SEX_L")["target"].mean()
ax = plt.gca()
ax.bar(
    gender_rates.index,
    gender_rates.values,
    color=colorScheme(np.arange(len(gender_rates))),
)
ax.set_title("Default Rate by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Default Rate")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
label_the_barcharts(ax, fmt="pct")
save_fig(fig, "07_default_rate_by_gender.png")

# 8. Default rate by num of delinquent months
fig = plt.figure()
late_rates = df.groupby("late_payment_count")["target"].mean().sort_index()
ax = plt.gca()
ax.bar(
    late_rates.index.astype(str),
    late_rates.values,
    color=colorScheme(np.arange(len(late_rates))),
)
ax.set_title("Default Rate vs. Number of Delinquent Months")
ax.set_xlabel("Count of months with delay (PAY_* > 0)")
ax.set_ylabel("Default Rate")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
label_the_barcharts(ax, fmt="pct")
save_fig(fig, "08_default_rate_by_late_count.png")

# 9. Default rate by utilization decile (people get sorted according to their utilization we defined above in buckets/ranges of 10%)
df["util_decile"] = pd.qcut(df["utilization_mean"].clip(0, 1.5), 10, duplicates="drop")
rate = df.groupby("util_decile")["target"].mean().mul(100)
plt.figure()
ax = rate.plot(kind="bar", color=colorScheme(np.arange(len(rate))))
ax.set_title("Default Rate by Utilization Decile")
ax.set_xlabel("Utilization (% of limit)")
ax.set_ylabel("Default Rate (%)")
ax.set_xticklabels(
    [f"{iv.left:.0%}–{iv.right:.0%}" for iv in rate.index], rotation=15, ha="right"
)
plt.tight_layout()
plt.savefig(
    os.path.join(outputpath, "09_default_rate_by_utilization_decile.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Finished")
