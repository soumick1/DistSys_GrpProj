
from __future__ import annotations
import argparse
import os
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

from Include import *
from Simulator import Simulator
from TraceGen import load_data, getDataset, dumpData
import config

ALL_POLICIES: List[str] = [
    "TTL", "LRU", "LFU", "GD", "LGD", "SIZE", "COST", "FREQ", "RAND",
    "FREQCOST", "FREQSIZE", "COSTSIZE",
    "GDSF", "SLRU", "TWOQ", "WTINYLFU_LRU", "WTINYLFU_COSTSIZE",
]

@dataclass
class Settings:
    memoryBudget: float
    timeLimit: int
    functionLimit: int

DEFAULT_SETTINGS: Dict[str, Settings] = {
    "Representative": Settings(memoryBudget=17e3, timeLimit=100, functionLimit=400),
    "Rare":           Settings(memoryBudget=6e4,  timeLimit=100, functionLimit=int(1.5e4)),
    "Random":         Settings(memoryBudget=4.5e4, timeLimit=100, functionLimit=400),
    "Temporal":       Settings(memoryBudget=17e3, timeLimit=100, functionLimit=400),
    "Bernoulli":      Settings(memoryBudget=17e3, timeLimit=100, functionLimit=400),
}

def _read_last_row(csv_path: str) -> Tuple[float, float, float, float, float, float]:
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        rows = [row for row in r if row]
    minMemoryReq = float(rows[0][1])
    time, coldStartTime, memorySize, excutingTime, nColdStart, nExcution = rows[-1]
    return (
        minMemoryReq,
        float(time),
        float(coldStartTime),
        float(memorySize),
        float(excutingTime),
        float(nColdStart),
        float(nExcution),
    )

def _ensure_dataset_slice(dataset: str, day: int, function_limit: int):
    base_fmap, base_imap = load_data(config.datasetLocation, day)
    try:
        fmap, imap = getDataset(base_fmap, base_imap, dataset, nFunction=function_limit)
    except Exception:
        return
    dumpData(fmap, imap, dataset, config.datasetLocation, day)

def run_worker(dataset: str,
               policy: str,
               day: int,
               base_log_dir: str,
               memoryBudget: float,
               timeLimit: int,
               functionLimit: int,
               logInterval: int,
               progressBar: bool=False,
               verbose: bool=False) -> Tuple[str, str, str]:
    if policy == "Baseline":
        memoryBudget = 1e7  # emulate always-warm

    out_dir = os.path.join(base_log_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    # Try loading the sliced dataset; if missing, create it.
    functionMap, invocationMap = load_data(config.datasetLocation, day, dataset)
    if not invocationMap:
        _ensure_dataset_slice(dataset, day, functionLimit)
        functionMap, invocationMap = load_data(config.datasetLocation, day, dataset)

    sim = Simulator(
        memoryBudget,
        functionMap,
        invocationMap,
        policy,
        timeLimit,
        functionLimit,
        logInterval,
        progressBar,
        verbose,
    )
    sim.run()
    sim.dumpStats(out_dir)

    csv_path = os.path.join(out_dir, f"{policy}.csv")
    return (dataset, policy, csv_path)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Async runner for serverless cache policies.")
    ap.add_argument("--datasets", nargs="+",
                    default=["Representative", "Rare", "Random"],
                    help="Dataset slices to run (Representative, Rare, Random, Temporal, Bernoulli).")
    ap.add_argument("--policies", nargs="+", default=["ALL"],
                    help="Policies to run or 'ALL' for the full suite.")
    ap.add_argument("--parallel", type=int, default=None,
                    help="Max parallel workers (default: min(num_jobs, CPU count)).")
    ap.add_argument("--day", type=int, default=1, help="Dataset day to use (Azure 2019 ranges 1..14).")
    ap.add_argument("--budget", type=float, default=None,
                    help="Override memory budget (MB) for ALL datasets; if omitted, per-dataset defaults are used.")
    ap.add_argument("--time-limit", type=int, default=None, help="Time horizon in minutes (default per dataset).")
    ap.add_argument("--function-limit", type=int, default=None, help="Max functions (default per dataset).")
    ap.add_argument("--log-interval", type=int, default=1000, help="Stats sampling interval in ms.")
    ap.add_argument("--log-dir", default="./log", help="Base directory for logs.")
    ap.add_argument("--no-progress", action="store_true", help="Disable per-worker progress bars.")
    ap.add_argument("--verbose", action="store_true", help="Verbose simulator logs.")
    return ap.parse_args()

def main():
    args = parse_args()

    policies = ALL_POLICIES if (len(args.policies) == 1 and args.policies[0].upper() == "ALL") else args.policies
    cpu = max(1, (os.cpu_count() or 1))
    num_jobs = len(policies) * len(args.datasets)
    max_workers = min(num_jobs, cpu) if args.parallel is None else max(1, args.parallel)

    print(f"Datasets: {args.datasets}")
    print(f"Policies: {policies}")
    print(f"Parallel workers: {max_workers}")
    print(f"Logs root: {args.log_dir}")
    print("")

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for dataset in args.datasets:
            st = DEFAULT_SETTINGS.get(dataset, DEFAULT_SETTINGS["Representative"])
            memoryBudget = args.budget if args.budget is not None else st.memoryBudget
            timeLimit = args.time_limit if args.time_limit is not None else st.timeLimit
            functionLimit = args.function_limit if args.function_limit is not None else st.functionLimit

            for policy in policies:
                fut = ex.submit(
                    run_worker,
                    dataset,
                    policy,
                    args.day,
                    args.log_dir,
                    float(memoryBudget),
                    int(timeLimit),
                    int(functionLimit),
                    int(args.log_interval),
                    (not args.no_progress),
                    args.verbose,
                )
                futures.append(fut)

        results: Dict[str, List[Tuple[str, str, str]]] = {d: [] for d in args.datasets}
        for fut in as_completed(futures):
            try:
                dataset, policy, csv_path = fut.result()
            except Exception as e:
                print(f"[error] worker failed: {e}")
                continue
            results[dataset].append((dataset, policy, csv_path))
            print(f"[done] {dataset}/{policy} -> {csv_path}")

    for dataset in args.datasets:
        print("")
        print(f"Dataset: {dataset}")
        header = "Policy, ColdStartTime, MemorySize, ExcutingTime, NColdStart, NExcution, PeakMemory"
        print(header)
        rows = []
        for _, policy, csv_path in sorted(results[dataset], key=lambda x: x[1]):
            try:
                (minMem, time, coldTime, memSize, execTime, nCold, nExec) = _read_last_row(csv_path)
                print([policy, coldTime, memSize, execTime, nCold, nExec, minMem])
                rows.append((policy, coldTime, memSize, execTime, nCold, nExec, minMem))
            except Exception as e:
                print(f"[warn] Could not read summary for {dataset}/{policy}: {e}")
        if rows:
            out_summary = os.path.join(args.log_dir, dataset, "summary.tsv")
            with open(out_summary, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["Policy", "ColdStartTime", "MemorySize", "ExcutingTime", "NColdStart", "NExcution", "PeakMemory"])
                for r in rows:
                    w.writerow(r)
            print(f"Summary written: {out_summary}")

if __name__ == "__main__":
    mp.freeze_support()  # Windows spawn protection
    main()
