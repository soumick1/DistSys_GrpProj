
#!/usr/bin/env python3
"""
viz_serverless.py — Visualizations & Ablations for Serverless Caching Policies (robust CSV loader)

This version matches the Simulator.dumpStats() format exactly:
  - Row 1: "MinMemoryReq,<float>"
  - Row 2: header: time,coldStartTime,memorySize,excutingTime,nColdStart,nExcution
  - Subsequent rows: time series

Usage
-----
# Parse all CSVs under ./log and write plots to ./viz
python viz_serverless.py --log-dirs ./log --out ./viz

# Compare multiple experiments (e.g., Representative, Rare, Random)
python viz_serverless.py --log-dirs ./log_rep ./log_rare ./log_rand --labels Representative Rare Random --out ./viz

# Only a subset of policies
python viz_serverless.py --log-dirs ./log --select-policies COSTSIZE LGD WTINYLFU_COSTSIZE

Constraints
-----------
- No seaborn, 1 chart per figure, no explicit colors.
"""

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CANON = {
    "time": "Time",
    "coldstarttime": "ColdStartTime",
    "memorysize": "MemorySize",
    "excutingtime": "ExcutingTime",  # matches simulator typo
    "ncoldstart": "NColdStart",
    "nexcution": "NExcution",        # matches simulator typo
}

REQUIRED = ["Time", "ColdStartTime", "MemorySize", "ExcutingTime", "NColdStart", "NExcution"]


def normalize_header(raw_header: List[str]) -> List[str]:
    out = []
    for h in raw_header:
        k = re.sub(r"[^A-Za-z0-9]", "", h.strip()).lower()
        out.append(CANON.get(k, h.strip()))
    return out


def load_policy_csv(path: str) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Load a policy CSV with the simulator format.
    Returns (df, min_memory_req). df has canonical column names.
    """
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            # skip fully empty rows
            if not r or all((str(x).strip() == "" for x in r)):
                continue
            rows.append([x.strip() for x in r])

    if not rows:
        return pd.DataFrame(columns=REQUIRED), None

    min_memory_req = None
    header_row_idx = 0

    # First line may be MinMemoryReq,<value>
    first = rows[0][0].strip().lower()
    if first == "minmemoryreq":
        try:
            min_memory_req = float(rows[0][1])
        except Exception:
            min_memory_req = None
        header_row_idx = 1

    if header_row_idx >= len(rows):
        return pd.DataFrame(columns=REQUIRED), min_memory_req

    # Normalize header to canonical names
    header_raw = rows[header_row_idx]
    header = normalize_header(header_raw)

    # Remaining rows are data
    data_rows = rows[header_row_idx + 1 : ]

    # Build DataFrame
    df = pd.DataFrame(data_rows, columns=header)

    # Keep only required columns (if present)
    keep = [c for c in header if c in REQUIRED]
    df = df[keep]

    # If some required columns missing, raise a readable error
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns: {missing} (header read: {header_raw})")

    # Convert numerics
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows where all required are NaN
    df = df.dropna(subset=REQUIRED, how="all").reset_index(drop=True)
    return df, min_memory_req


def find_policies(log_dir: str, select_policies: List[str] = None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    candidates = glob.glob(os.path.join(log_dir, "**", "*.csv"), recursive=True)
    for p in candidates:
        name = os.path.splitext(os.path.basename(p))[0]
        name = re.split(r"[-_]\d+$", name)[0]
        if select_policies and name not in select_policies:
            continue
        mapping[name] = p
    return mapping


def summarize_policy(df: pd.DataFrame, min_memory_req: Optional[float]) -> Dict[str, float]:
    if df.empty:
        return {}

    last = df.iloc[-1]
    N = float(last["NExcution"]) if "NExcution" in df.columns else np.nan
    cold_time = float(last["ColdStartTime"])
    exec_time = float(last["ExcutingTime"])
    n_cold = float(last["NColdStart"])

    avg_latency = exec_time / N if N and N > 0 else np.nan
    cold_rate = n_cold / N if N and N > 0 else np.nan
    avg_cold_cost = cold_time / n_cold if n_cold and n_cold > 0 else 0.0

    mem_mean_ts = float(df["MemorySize"].mean()) if "MemorySize" in df.columns else np.nan
    mem_peak_ts = float(df["MemorySize"].max()) if "MemorySize" in df.columns else np.nan

    # Use simulator's MinMemoryReq as PeakMemory (to match console print), fall back to ts peak
    peak_memory = float(min_memory_req) if (min_memory_req is not None) else mem_peak_ts

    return dict(
        N=N,
        ColdStartTime=cold_time,
        ExcutingTime=exec_time,
        NColdStart=n_cold,
        AvgLatencyMs=avg_latency,
        ColdStartRate=cold_rate,
        AvgColdStartCostMs=avg_cold_cost,
        MemMean=mem_mean_ts,
        MemPeakTS=mem_peak_ts,
        PeakMemory=peak_memory,
    )


def collect_experiment(log_dir: str, select_policies: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mapping = find_policies(log_dir, select_policies)
    summaries = []
    ts_frames = []
    for policy, path in sorted(mapping.items()):
        try:
            df, mmr = load_policy_csv(path)
        except Exception as e:
            print(f"[WARN] Could not load {path}: {e}")
            continue
        s = summarize_policy(df, mmr)
        if not s:
            continue
        s["Policy"] = policy
        s["CSVPath"] = path
        summaries.append(s)

        df = df.copy()
        df["Policy"] = policy
        ts_frames.append(df)

    if summaries:
        summary_df = pd.DataFrame(summaries).set_index("Policy").sort_values("AvgLatencyMs")
    else:
        summary_df = pd.DataFrame(columns=[
            "N","ColdStartTime","ExcutingTime","NColdStart",
            "AvgLatencyMs","ColdStartRate","AvgColdStartCostMs",
            "MemMean","MemPeakTS","PeakMemory","CSVPath"
        ])

    timeseries_df = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()
    return summary_df, timeseries_df


def bar(ax, x, y, title, ylabel, rotate=45):
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=rotate, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)


def scatter(ax, x, y, labels, title, xlabel, ylabel):
    ax.scatter(x, y)
    for xi, yi, name in zip(x, y, labels):
        ax.annotate(str(name), (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)


def savefig(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_core_suite(summary: pd.DataFrame, out_dir: str, tag: str = "") -> List[str]:
    saved = []
    if summary.empty:
        return saved

    s = summary.sort_values("AvgLatencyMs")
    policies = list(s.index)

    # 1) Avg latency
    fig, ax = plt.subplots()
    bar(ax, policies, s["AvgLatencyMs"].tolist(),
        f"Average Latency per Invocation {tag}".strip(), "ms")
    saved.append(savefig(fig, out_dir, f"bar_avg_latency{tag}.png"))

    # 2) Cold-start rate (%)
    fig, ax = plt.subplots()
    bar(ax, policies, (s["ColdStartRate"] * 100.0).tolist(),
        f"Cold-start Rate (%) {tag}".strip(), "%")
    saved.append(savefig(fig, out_dir, f"bar_coldstart_rate{tag}.png"))

    # 3) Avg cold-start cost
    fig, ax = plt.subplots()
    bar(ax, policies, s["AvgColdStartCostMs"].tolist(),
        f"Average Cold-start Cost {tag}".strip(), "ms")
    saved.append(savefig(fig, out_dir, f"bar_avg_coldstart_cost{tag}.png"))

    # 4) Total cold-start time
    fig, ax = plt.subplots()
    bar(ax, policies, s["ColdStartTime"].tolist(),
        f"Total Cold-start Time {tag}".strip(), "ms")
    saved.append(savefig(fig, out_dir, f"bar_total_coldstart_time{tag}.png"))

    # 5) Total executing time
    fig, ax = plt.subplots()
    bar(ax, policies, s["ExcutingTime"].tolist(),
        f"Total Executing Time {tag}".strip(), "ms")
    saved.append(savefig(fig, out_dir, f"bar_total_executing_time{tag}.png"))

    # 6) Memory mean
    fig, ax = plt.subplots()
    bar(ax, policies, s["MemMean"].tolist(),
        f"Average Memory Usage {tag}".strip(), "MB")
    saved.append(savefig(fig, out_dir, f"bar_mem_mean{tag}.png"))

    # 7) Peak memory (use simulator's MinMemoryReq to match console print)
    fig, ax = plt.subplots()
    bar(ax, policies, s["PeakMemory"].tolist(),
        f"Peak Memory Requirement (from MinMemoryReq) {tag}".strip(), "MB")
    saved.append(savefig(fig, out_dir, f"bar_mem_peak_req{tag}.png"))

    # 8) Scatter: latency vs cold-start rate
    fig, ax = plt.subplots()
    scatter(ax, s["ColdStartRate"].tolist(), s["AvgLatencyMs"].tolist(), policies,
            f"Latency vs Cold-start Rate {tag}".strip(), "Cold-start Rate", "Avg Latency (ms)")
    saved.append(savefig(fig, out_dir, f"scatter_latency_vs_coldstartrate{tag}.png"))

    # 9) Scatter: total exec vs total cold-start
    fig, ax = plt.subplots()
    scatter(ax, s["ColdStartTime"].tolist(), s["ExcutingTime"].tolist(), policies,
            f"Total Exec Time vs Cold-start Time {tag}".strip(), "Cold-start Time (ms)", "Total Exec Time (ms)")
    saved.append(savefig(fig, out_dir, f"scatter_exectime_vs_coldstart{tag}.png"))

    # 10) Stacked latency breakdown
    service_ms = (s["ExcutingTime"] - s["ColdStartTime"]) / s["N"]
    cold_ms = s["ColdStartTime"] / s["N"]
    fig, ax = plt.subplots()
    ax.bar(policies, service_ms.tolist(), label="Service")
    ax.bar(policies, cold_ms.tolist(), bottom=service_ms.tolist(), label="Cold-start")
    ax.set_title(f"Avg Latency Breakdown per Invocation {tag}".strip())
    ax.set_ylabel("ms")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    saved.append(savefig(fig, out_dir, f"stacked_latency_breakdown{tag}.png"))

    return saved


def plot_timeseries(timeseries: pd.DataFrame, summary: pd.DataFrame, out_dir: str, tag: str = "", topk: int = 3):
    if timeseries.empty or summary.empty:
        return []

    s = summary.sort_values("AvgLatencyMs")
    top_policies = list(s.index[:max(1, topk)])

    saved = []
    # Memory over time
    fig, ax = plt.subplots()
    for p in top_policies:
        g = timeseries[timeseries["Policy"] == p]
        ax.plot(g["Time"].values, g["MemorySize"].values, label=p)
    ax.set_title(f"Memory Usage over Time (Top-{len(top_policies)}) {tag}".strip())
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Memory (MB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    saved.append(savefig(fig, out_dir, f"ts_memory{tag}.png"))

    # Cumulative cold-start time over time
    fig, ax = plt.subplots()
    for p in top_policies:
        g = timeseries[timeseries["Policy"] == p]
        ax.plot(g["Time"].values, g["ColdStartTime"].values, label=p)
    ax.set_title(f"Cumulative Cold-start Time over Time (Top-{len(top_policies)}) {tag}".strip())
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Cold-start Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    saved.append(savefig(fig, out_dir, f"ts_coldstart{tag}.png"))

    return saved


def grouped_bar_over_experiments(summaries: List[pd.DataFrame], labels: List[str], metric: str, out_dir: str, fname: str):
    sets = [set(df.index) for df in summaries if not df.empty]
    if not sets:
        return None
    common = set.intersection(*sets)
    if not common:
        return None

    policies = sorted(list(common))
    data = []
    for df in summaries:
        data.append([float(df.loc[p, metric]) for p in policies])

    fig, ax = plt.subplots()
    x = np.arange(len(policies))
    width = 0.8 / len(data)
    for i, series in enumerate(data):
        ax.bar(x + i * width, series, width=width, label=labels[i] if i < len(labels) else f"Exp{i+1}")
    ax.set_title(f"{metric} across Experiments")
    ax.set_ylabel(metric)
    ax.set_xticks(x + width * (len(data)-1) / 2)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    return savefig(fig, out_dir, fname)


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description="Visualizations & Ablations for Serverless Caching Policies")
    ap.add_argument("--log-dirs", nargs="+", required=True, help="One or more log directories (each is an experiment)")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for experiments; length must match log-dirs")
    ap.add_argument("--select-policies", nargs="*", default=None, help="Optional subset of policy names to include")
    ap.add_argument("--out", default="./viz", help="Output directory for plots and summary CSVs")
    ap.add_argument("--topk", type=int, default=3, help="Top-k policies (by latency) for time-series plots")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.log_dirs):
        raise SystemExit("If --labels is provided, it must have the same length as --log-dirs.")

    os.makedirs(args.out, exist_ok=True)

    all_summaries = []
    all_timeseries = []

    for i, log_dir in enumerate(args.log_dirs):
        label = (args.labels[i] if args.labels else os.path.basename(os.path.abspath(log_dir))) or f"Exp{i+1}"
        summary, ts = collect_experiment(log_dir, args.select_policies)

        # Write per-experiment summary CSV
        out_sum = os.path.join(args.out, f"summary_{label}.csv")
        if not summary.empty:
            summary.to_csv(out_sum)

        tag = f"_{label}"
        plot_core_suite(summary, args.out, tag=tag)
        plot_timeseries(ts, summary, args.out, tag=tag, topk=args.topk)

        all_summaries.append(summary.assign(_label=label))
        all_timeseries.append(ts.assign(_label=label))

    valid_summaries = [s for s in all_summaries if not s.empty]
    if len(valid_summaries) >= 2:
        labels = [s["_label"].iloc[0] for s in valid_summaries]
        for metric, fname in [
            ("AvgLatencyMs", "grouped_avg_latency.png"),
            ("ColdStartRate", "grouped_coldstart_rate.png"),
            ("AvgColdStartCostMs", "grouped_avg_coldstart_cost.png"),
            ("MemMean", "grouped_mem_mean.png"),
        ]:
            grouped_bar_over_experiments(valid_summaries, labels, metric, args.out, fname)

    if valid_summaries:
        concat = pd.concat(valid_summaries, keys=[s["_label"].iloc[0] for s in valid_summaries], names=["Experiment", "Policy"])
        concat.to_csv(os.path.join(args.out, "summary_all_experiments.csv"))


if __name__ == "__main__":
    main()
