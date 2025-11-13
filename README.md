# DistSys_GrpProj

dataset at https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
set the dataset location in config.py

---

## Contents
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Preprocessing (optional)](#preprocessing-optional)
- [Run the Simulator](#run-the-simulator)
- [Visualization & Ablations](#visualization--ablations)
- [Metrics & Notation](#metrics--notation)
- [Universal Eviction Framework](#universal-eviction-framework)
- [Policies (Formulas & Intuition)](#policies-formulas--intuition)
- [Tuning Knobs](#tuning-knobs)
- [Troubleshooting](#troubleshooting)
- [Recommended Report Structure](#recommended-report-structure)
- [At-a-Glance Policy Table](#at-a-glance-policy-table)

---

## Quick Start

**Requirements**
- Python 3.9+
- Install deps (add more if your env asks for them):
```bash
pip install numpy pandas matplotlib tqdm
```

**Point to the dataset** (edit `config.py`):
```python
datasetLocation = r"azurefunctions-dataset2019"
```

**Run** (executes many policies and writes logs to `./log/<POLICY>.csv`):
```bash
python Simulator.py
```

**Visualize** (writes plots + summary CSVs to `./viz`):
```bash
python viz_serverless.py --log-dirs ./log --out ./viz
```

---

## Dataset Setup
- Download the **Azure Functions 2019** dataset and place the raw CSVs under the folder you set in `config.py` as `datasetLocation`.
- The simulator will auto-generate preprocessed maps (`functionMap_*.csv`, `invocationMap_*.csv`) on first run.  
  On Windows, CSV read/write is opened with `newline=''` to prevent blank-line issues.

---

## Preprocessing (optional)
Generate curated slices once (faster iterative experiments):
```bash
python TraceGen.py
# Produces Representative / Rare / Random maps under your dataset folder
```

---

## Run the Simulator
`Simulator.py` runs a suite of policies by default and prints one summary line per policy, e.g.:

```
['COSTSIZE', ColdStartTime, MemorySize, ExcutingTime, NColdStart, NExcution, PeakMemory]
```

It also writes a **time series** CSV for each policy under `./log/<POLICY>.csv` in this format:

- Line 1: `MinMemoryReq,<float>` (used as “Peak Memory Requirement”)
- Line 2: header (lower/mixed case):  
  `time,coldStartTime,memorySize,excutingTime,nColdStart,nExcution`
- Lines 3+: periodic samples (controlled by `logInterval` inside `Simulator.py`)

---

## Visualization & Ablations

The script `viz_serverless.py` parses the simulator logs and produces figures for a report.

**Install (if needed):**
```bash
pip install numpy pandas matplotlib
```

**Single experiment:**
```bash
python viz_serverless.py --log-dirs ./log --out ./viz
```

**Subset of policies:**
```bash
python viz_serverless.py \
  --log-dirs ./log \
  --select-policies COSTSIZE LGD WTINYLFU_COSTSIZE \
  --out ./viz
```

**Compare multiple experiments** (e.g., Representative / Rare / Random or different budgets):
```bash
python viz_serverless.py \
  --log-dirs ./log_rep ./log_rare ./log_rand \
  --labels Representative Rare Random \
  --out ./viz
```

**Outputs (PNG + CSV):**
- `bar_avg_latency*.png` — average latency per invocation  
- `bar_coldstart_rate*.png` — cold-start rate (%)  
- `bar_avg_coldstart_cost*.png` — average cold-start cost (ms)  
- `bar_total_coldstart_time*.png`, `bar_total_executing_time*.png`  
- `bar_mem_mean*.png` — average memory usage (MB)  
- `bar_mem_peak_req*.png` — peak **requirement** (from `MinMemoryReq`)  
- `scatter_latency_vs_coldstartrate*.png`, `scatter_exectime_vs_coldstart*.png`  
- `stacked_latency_breakdown*.png` — service vs cold-start per invocation  
- `ts_memory*.png`, `ts_coldstart*.png` — time series for top-k policies by latency  
- CSVs: `viz/summary_<label>.csv`, `viz/summary_all_experiments.csv`

---

## Metrics & Notation

We process \(N\) invocations. Time is **ms**, memory is **MB**.

### Per-invocation variables
- \( i \in \{1,\dots,N\} \): invocation index  
- \( s_i \): **service time** (function runtime excluding cold start)  
- \( c_i \ge 0 \): **cold-start delay** (0 if warm)  
- \( \delta_i \in \{0,1\} \): cold-start **indicator** (1 if cold, else 0)  
- \( \ell_i = s_i + \delta_i\,c_i \): end-to-end **latency** of invocation \(i\)

### Cache/system variables
- \( B \): memory budget (MB)  
- \( m_f > 0 \): memory size of one warm container for function \( f \) (MB)  
- \( c_f \ge 0 \): cold-start cost for function \( f \) (ms)  
- \( n_f(t) \): observed **frequency** (hits so far) of \( f \) at time \( t \)  
- \( a_f(t) \): **last access time** of \( f \)  
- \( \ell_f(t) = \log\!\bigl(1+n_f(t)\bigr) + 1 \): log-smoothed frequency  
- \( L(t) \): **logical clock** (recency/aging counter updated by the simulator)  
- \( p_f(t) \): **priority score** for \( f \) at time \( t \) (larger = safer to keep)  
- \( M(t) = \sum_{g \in C(t)} m_g \): memory used by cache set \( C(t) \)

### Accumulators (what the simulator prints)

\[
\begin{aligned}
\mathbf{ColdStartTime} &= \sum_{i=1}^{N} \delta_i\,c_i \\\\
\mathbf{ExcutingTime} &= \sum_{i=1}^{N} \ell_i = \sum_{i=1}^{N}\bigl(s_i + \delta_i\,c_i\bigr) \\\\
\mathbf{NColdStart} &= \sum_{i=1}^{N} \delta_i \\\\
\mathbf{NExcution} &= N \\\\
\overline{m} &\approx \frac{1}{M_s} \sum_{k=1}^{M_s} m(t_k) \quad (\text{sampled mean memory}) \\\\
\mathbf{PeakMemory} &= \max_k m(t_k) \quad (\text{or MinMemoryReq from CSV})
\end{aligned}
\]

**Derived:**

\[
\text{AvgLatency}=\frac{\mathbf{ExcutingTime}}{N},\quad
\text{ColdStartRate}=\frac{\mathbf{NColdStart}}{N},\quad
\text{AvgColdCost}=\frac{\mathbf{ColdStartTime}}{\mathbf{NColdStart}}\;(>0)
\]

---

## Universal Eviction Framework

When an invocation for function \( f \) arrives and no idle warm container exists:

1. **Evict** entries with the **smallest** \( p_g(t) \) until \( M(t) + m_f \le B \).  
2. **Admission** (some policies): decide whether to actually keep the new warm container.  
3. **Insert / Update**: set \( p_f(t) \), update \( n_f, a_f, L \) as per the policy.

> Intuition: the policy is defined by \( p_f(t) \) and optional **admission** or **segmentation** rules.

---

## Policies (Formulas & Intuition)

Below, “base” means we add a recency baseline via the logical clock \( L(t) \). Eviction always removes **lowest** \( p_f \).

### TTL — Time-to-live (idle eviction)
Evict entries idle longer than a fixed \( \tau \):

\[
\text{if } t - a_g(t) > \tau \Rightarrow g \text{ is evicted (pre-check).}
\]

After pruning, act like LRU with \( p_f(t) = L(t) \).  
**Goal:** cull stale entries. Weak under bursty skew.

---

### LRU — Least Recently Used
Recency only:

\[
p_f(t) = L(t), \qquad \text{on access: } L \!\uparrow,\; p_f \!\leftarrow L
\]

**Goal:** favor recent; fragile under scans/long tails.

---

### LFU — Least Frequently Used
Frequency only:

\[
p_f(t) = n_f(t), \qquad \text{increment } n_f \text{ on hits}
\]

**Goal:** protect globally hot; slow to adapt; ignores cost/size.

---

### GD — GreedyDual (variant in code)
Cost- & size-aware with frequency modulation:

\[
p_f(t) = L(t) + n_f(t)\,\frac{c_f}{m_f}
\]

**Goal:** keep **frequent & expensive-per-MB** entries.

---

### LGD — Log-GreedyDual
GD with tempered frequency:

\[
p_f(t) = L(t) + \underbrace{\bigl(\log(1+n_f(t))+1\bigr)}_{\ell_f(t)}\,\frac{c_f}{m_f}
\]

**Goal:** robust to spikes; protects high \( c_f/m_f \) with gentle freq bias.

---

### SIZE — Favor small

\[
p_f(t) = L(t) + \frac{1}{m_f}
\]

**Goal:** maximize resident count; can evict few large but critical ones.

---

### COST — Favor costly

\[
p_f(t) = L(t) + c_f
\]

**Goal:** reduce total cold-start time regardless of size.

---

### FREQ — Frequency + aging

\[
p_f(t) = L(t) + n_f(t)
\]

**Goal:** like LFU but slightly more reactive via \( L \).

---

### RAND — Random
Random victim/score; sanity-check baseline.

---

### FREQCOST — Frequency × cost

\[
p_f(t) = L(t) + n_f(t)\,c_f
\]

**Goal:** protect frequent & expensive; ignores size.

---

### FREQSIZE — Frequency / size

\[
p_f(t) = L(t) + \frac{n_f(t)}{m_f}
\]

**Goal:** favor tiny hot items; may starve big ones.

---

### COSTSIZE — Cost / size (cost per MB)

\[
p_f(t) = L(t) + \frac{c_f}{m_f}
\]

**Goal:** **bang-for-buck**; excellent default in serverless.

---

### GDSF — GreedyDual-Size-Frequency (intended form)

\[
p_f(t) = L(t) + \frac{n_f(t)\,c_f}{m_f}
\]

> In current code GD and GDSF are algebraically identical; to differentiate, add a **global age \(H\)** (classic GreedyDual) and/or an **admission** step.

---

### SLRU — Segmented LRU (probation/protected)
Two LRU segments:
- State \( s_f \in \{\text{prob}, \text{prot}\} \).  
- Insert into **prob**; promote to **prot** on second hit.  
- Quotas: \( |\text{prot}| \approx \rho|C| \), \( |\text{prob}| \approx (1-\rho)|C| \) (e.g., \( \rho=0.6 \)).

**Eviction:** from **prob** first; if empty, demote LRU from **prot**→**prob**.  
**Priority view:** both use \( p_f(t)=L(t) \), but **protected** items get a large offset (rank above probation).  
**Goal:** block one-hit wonders; needs quotas to shine on long-tail traces.

---

### TWOQ — A1 window + Am main
Two queues:
- State \( q_f \in \{A_1, A_m\} \). Insert into \( A_1 \); promote to \( A_m \) on reuse.  
- Quotas: \( |A_1|=\alpha|C| \), \( |A_m|=(1-\alpha)|C| \) (e.g., \( \alpha=0.1 \)).

**Eviction:** from \( A_1 \) first; then \( A_m \).  
**Priority view:** \( p_f(t)=L(t) \) with a **boost** for \( A_m \) to rank above \( A_1 \).  
**Goal:** recency filter; very effective against pollution when tuned.

---

### WTINYLFU_LRU — TinyLFU admission + LRU eviction
Maintain a Count-Min Sketch estimate \( \phi_f(t) \) of recency-biased frequency.

**Admission** (inserting \( f \); would-be victim \( v \)):

\[
\text{admit } f \iff \phi_f(t) \ge \phi_v(t) \quad (\text{optionally } \phi_f \ge (1+\varepsilon)\phi_v)
\]

If admitted, eviction behaves as **LRU**: \( p_f(t)=L(t) \).  
**Goal:** block one-hit wonders from polluting the cache.

---

### WTINYLFU_COSTSIZE — TinyLFU admission + COSTSIZE eviction
Same admission; eviction score:

\[
p_f(t) = L(t) + \frac{c_f}{m_f}
\]

**Goal:** combine SOTA **admission** with serverless-friendly **cost-per-MB** eviction.

---

### Baseline
Reference run for comparisons. In current scripts it equals **TTL** (same numbers).

---

### Running the main file
**Main** (async run on datasets):
```bash
python TraceGen.py # ---? run it once if you haven't run it
python main.py # ---> runs on all datasets and on 16 workers
python main.py --datasets Representative Rare Random --parallel 8 # ---> specific datasets and workers
```

## Tuning Knobs

- **Memory budget** \( B \) (MB): chart tradeoff curves (latency vs. memory).  
- **SLRU / TWOQ quotas:** \( \rho \) (protected), \( \alpha \) (A1 window).  
- **TinyLFU:** sketch width/depth, **decay schedule**, admission **margin** \( \varepsilon \).  
- **GreedyDual family:** add global age \( H \) (set to evicted key) to separate GD from GDSF.  
- **TTL:** idle timeout \( \tau \).  
- **Logging cadence:** `logInterval` (ms) in `Simulator.py` for smoother time-series.

---

## Troubleshooting

- **Math shows raw delimiters**: Ensure display equations use `\[ ... \]` on their **own line** with blank lines around. Do not wrap `\[ ... \]` or `\( ... \)` inside code fences or blockquotes.
- **CSV “missing columns” in viz**: The visualizer expects:
  - First line: `MinMemoryReq,<float>`
  - Second line header: `time,coldStartTime,memorySize,excutingTime,nColdStart,nExcution`
  - Subsequent lines: numeric samples. If a CSV is truncated/corrupt, delete it and re-run.
- **Peak memory differs between console and plots**: Plots include **Peak Memory Requirement** from `MinMemoryReq` to match the console’s final column.

---

## Recommended Report Structure

1) **Dataset & setup** (day, slice, memory budget \( B \), `logInterval`).  
2) **Metrics & equations** (above).  
3) **Policy taxonomy & formulas** (above).  
4) **Overall comparison** (AvgLatency, ColdStartRate bars).  
5) **Breakdowns** (stacked latency; cold-start cost vs count).  
6) **Ablations** (vary \( B \); SLRU/TWOQ quotas; TinyLFU margin/decay; GD vs GDSF with global \(H\)).  
7) **Findings & recommendations** (e.g., **COSTSIZE / LGD** as strong baselines; **WTinyLFU** for robustness under long-tail churn).

---

## At-a-Glance Policy Table

| Policy | Priority \(p_f(t)\) | Extras |
|---|---|---|
| TTL | LRU after pruning idle \(> \tau\) | Idle timeout \(\tau\) |
| LRU | \(L(t)\) | Recency only |
| LFU | \(n_f(t)\) | Frequency only |
| GD | \(L(t) + n_f \frac{c_f}{m_f}\) | Cost/MB × freq |
| LGD | \(L(t) + (\log(1+n_f)+1)\frac{c_f}{m_f}\) | Tempered freq |
| SIZE | \(L(t) + \frac{1}{m_f}\) | Favor small |
| COST | \(L(t) + c_f\) | Favor costly |
| FREQ | \(L(t) + n_f\) | Freq + aging |
| RAND | random | Baseline |
| FREQCOST | \(L(t) + n_f c_f\) | Freq×cost |
| FREQSIZE | \(L(t) + \frac{n_f}{m_f}\) | Freq/size |
| **COSTSIZE** | \(L(t) + \frac{c_f}{m_f}\) | **Cost per MB** |
| GDSF | \(L(t) + \frac{n_f c_f}{m_f}\) | Size-norm. freq×cost |
| SLRU | \(L(t)\) (+protected boost) | Probation↔Protected, quota \(\rho\) |
| TWOQ | \(L(t)\) (+Am boost) | \(A_1\)↔\(A_m\), quota \(\alpha\) |
| WTINYLFU_LRU | \(L(t)\) | **Admission** via sketch \(\phi_f\) |
| WTINYLFU_COSTSIZE | \(L(t) + \frac{c_f}{m_f}\) | **Admission** + cost/MB |
| Baseline | (TTL in scripts) | Reference |

---
