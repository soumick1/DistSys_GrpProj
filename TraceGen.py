
import numpy as np
import csv
from Include import *
import heapq
import config


def _iter_csv_rows(path):
    """Yield non-empty, trimmed CSV rows; tolerate BOM and blank lines."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all((not str(c).strip() for c in row)):
                continue  # skip blanks
            yield [c.strip() for c in row]


# ---------- Raw Azure -> internal maps ----------

def parse_data(datasetPath: str, day: int = 1):
    strDay = f"{day:02}"
    durationFile   = f"{datasetPath}/function_durations_percentiles.anon.d{strDay}.csv"
    memoryFile     = f"{datasetPath}/app_memory_percentiles.anon.d{strDay}.csv"
    invocationFile = f"{datasetPath}/invocations_per_function_md.anon.d{strDay}.csv"

    # key: (HashOwner, HashApp)
    durationMap = {}
    memoryMap = {}
    invocationMap = {}

    # --- load duration percentiles ---
    try:
        for line in _iter_csv_rows(durationFile):
            if line[0] == "HashOwner":
                continue
            HashOwner, HashApp, HashFunction = line[0], line[1], line[2]
            Average = int(float(line[3]))
            Count   = int(float(line[4]))
            Minimum = float(line[5])
            Maximum = float(line[6])
            if (HashOwner, HashApp) not in durationMap:
                durationMap[(HashOwner, HashApp)] = {}
            if HashFunction not in durationMap[(HashOwner, HashApp)]:
                durationMap[(HashOwner, HashApp)][HashFunction] = Duration(
                    HashOwner, HashApp, HashFunction, Average, Count, Minimum, Maximum
                )
    except IOError:
        print("Could not read file: 1 ", durationFile)

    # --- load memory percentiles ---
    try:
        for line in _iter_csv_rows(memoryFile):
            if line[0] == "HashOwner":
                continue
            HashOwner, HashApp = line[0], line[1]
            SampleCount = int(float(line[2]))
            # Azure file may have float MB; keep as float
            AverageAllocatedMb = float(line[3])
            if (HashOwner, HashApp) not in memoryMap:
                memoryMap[(HashOwner, HashApp)] = Memory(HashOwner, HashApp, SampleCount, AverageAllocatedMb)
    except IOError:
        print("Could not read file: 2 ", memoryFile)

    # --- build function map using duration+memory ---
    functionMap = {}
    for (HashOwner, HashApp) in durationMap:
        if (HashOwner, HashApp) not in memoryMap:
            continue
        memory: Memory = memoryMap[(HashOwner, HashApp)]
        count_funcs = len(durationMap[(HashOwner, HashApp)])
        # average per-function memory for this app
        functionMemory = float(memory.AverageAllocatedMb) / max(1, count_funcs)
        for HashFunction, duration in durationMap[(HashOwner, HashApp)].items():
            durationTime = float(duration.Average)
            coldStartTime = float(duration.Maximum) - durationTime
            key = (HashOwner, HashApp, HashFunction)
            if key not in functionMap:
                functionMap[key] = Function(HashOwner, HashApp, HashFunction, coldStartTime, durationTime, functionMemory)

    # --- load invocations ---
    try:
        for line in _iter_csv_rows(invocationFile):
            if line[0] == "HashOwner":
                continue
            HashOwner, HashApp, HashFunction, Trigger = line[0], line[1], line[2], line[3]
            try:
                Counts = list(map(lambda x: int(float(x)), line[4:]))
            except ValueError:
                Counts = [int(float(x)) if str(x).strip() else 0 for x in line[4:]]
            key = (HashOwner, HashApp, HashFunction)
            if key in functionMap and key not in invocationMap:
                invocationMap[key] = Invocation(HashOwner, HashApp, HashFunction, Trigger, Counts)
    except IOError:
        print("Could not read file: 3 ", invocationFile)

    assert functionMap and invocationMap, "Data not loaded"
    print("Data loaded successfully")

    # dump maps to files (windows-safe, no extra blank lines)
    with open(f"{datasetPath}/functionMap_d{strDay}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "coldStartTime", "duration", "memory"])
        for key, function in functionMap.items():
            writer.writerow([function.HashOwner, function.HashApp, function.HashFunction,
                             function.coldStartTime, function.duration, function.memory])

    with open(f"{datasetPath}/invocationMap_d{strDay}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "Trigger", "Counts"])
        for key, invocation in invocationMap.items():
            writer.writerow([invocation.HashOwner, invocation.HashApp, invocation.HashFunction,
                             invocation.Trigger] + invocation.Counts)

    return functionMap, invocationMap


# ---------- Cached-map loader ----------

def load_data(datasetPath: str, day: int = 1, type: str | None = None) -> tuple[dict[tuple[int,int,int],Function], dict[tuple[int,int,int],Invocation]]:
    strDay = f"{day:02}"
    functionMap = {}
    invocationMap = {}
    type_suffix = f"_{type}" if type else ""

    functionFile = f"{datasetPath}/functionMap{type_suffix}_d{strDay}.csv"
    invocationFile = f"{datasetPath}/invocationMap{type_suffix}_d{strDay}.csv"

    try:
        for line in _iter_csv_rows(functionFile):
            if line[0] == "HashOwner":
                continue
            functionMap[(line[0], line[1], line[2])] = Function(line[0], line[1], line[2],
                                                                float(line[3]), float(line[4]), float(line[5]))
    except IOError:
        print("Could not read file: ", functionFile)

    try:
        for line in _iter_csv_rows(invocationFile):
            if line[0] == "HashOwner":
                continue
            invocationMap[(line[0], line[1], line[2])] = Invocation(line[0], line[1], line[2],
                                                                    line[3], list(map(lambda x: int(float(x)), line[4:])))
    except IOError:
        print("Could not read file: ", invocationFile)

    if len(functionMap) == 0:
        # Fall back to parsing raw Azure files and regenerate the base maps
        functionMap, invocationMap = parse_data(datasetPath, day)

    return functionMap, invocationMap


# ---------- Dataset slicing ----------

def getRareData(functionMap, invocationMap, nFunction: int = 100):
    heap = []
    for functionId in invocationMap:
        invocation = invocationMap[functionId]
        heapq.heappush(heap, (sum(invocation.Counts), functionId))
    # pick rarest quarter (or up to nFunction)
    rareFuncs = heapq.nsmallest(max(1, min(len(invocationMap)//4, nFunction)), heap)
    newInvocationMap = {}
    newfunctionMap = {}
    for _, functionId in rareFuncs:
        newInvocationMap[functionId] = invocationMap[functionId]
        newfunctionMap[functionId] = functionMap[functionId]
    return newfunctionMap, newInvocationMap


def getRandomData(functionMap, invocationMap, nFunction: int = 100):
    functionKeys = list(invocationMap.keys())
    n = min(nFunction, len(functionKeys))
    randomIndexes = np.random.choice(range(len(functionKeys)), n, replace=False)
    newfunctionMap = {}
    newInvocationMap = {}
    for index in randomIndexes:
        key = functionKeys[index]
        newfunctionMap[key] = functionMap[key]
        newInvocationMap[key] = invocationMap[key]
    return newfunctionMap, newInvocationMap


def getRepresentativeData(functionMap, invocationMap, nFunction: int = 100):
    allFuncs = []
    for functionId in invocationMap:
        invocation = invocationMap[functionId]
        allFuncs.append((sum(invocation.Counts), functionId))
    allFuncs.sort(reverse=True)
    allFuncs = [functionId for _, functionId in allFuncs]
    # sampled from n parts
    newfunctionMap = {}
    newInvocationMap = {}
    if not allFuncs:
        return newfunctionMap, newInvocationMap
    nSample = max(1, nFunction // 4)
    for i in range(4):
        start = i * (len(allFuncs) // 4)
        end = (i + 1) * (len(allFuncs) // 4) if i < 3 else len(allFuncs)
        if end <= start:
            continue
        randomIndex = np.random.choice(range(start, end), min(nSample, end - start), replace=False)
        for index in randomIndex:
            functionId = allFuncs[index]
            newfunctionMap[functionId] = functionMap[functionId]
            newInvocationMap[functionId] = invocationMap[functionId]
    return newfunctionMap, newInvocationMap


def getTemporalData(functionMap, invocationMap, nFunction: int = 100):
    temporalFeatures = []
    # Step 1: Compute temporal stats for each function
    for functionId, invocation in invocationMap.items():
        Counts = invocation.Counts
        total = sum(Counts)
        if total == 0:
            continue
        active_minutes = sum(c > 0 for c in Counts)
        burst_factor = max(Counts) / (np.mean(Counts) + 1e-9)
        hourly_sums = [sum(Counts[i:i+60]) for i in range(0, len(Counts), 60)]
        peak_hour = int(np.argmax(hourly_sums)) if hourly_sums else 0
        idle_ratio = 1 - active_minutes / len(Counts)
        temporalFeatures.append((functionId, total, active_minutes, burst_factor, peak_hour, idle_ratio))

    # Step 2: Categorize functions
    steady, bursty, sporadic, night = [], [], [], []
    for fid, total, active, burst, peak, idle in temporalFeatures:
        if burst < 2 and active > 1000:
            steady.append(fid)
        elif 2 <= burst <= 5 and 200 <= active <= 1000:
            bursty.append(fid)
        elif burst > 5 or active < 200:
            sporadic.append(fid)
        elif peak < 6:  # night-active
            night.append(fid)

    # Step 3: Sample from each group
    newfunctionMap, newInvocationMap = {}, {}
    groups = [steady, bursty, sporadic, night]
    nSample = max(1, nFunction // max(1, len(groups)))
    for group in groups:
        if len(group) == 0:
            continue
        sample_index = np.random.choice(range(0, len(group)), min(nSample, len(group)), replace=False)
        sampleIds = [group[ind] for ind in sample_index]
        for fid in sampleIds:
            newfunctionMap[fid] = functionMap[fid]
            newInvocationMap[fid] = invocationMap[fid]

    return newfunctionMap, newInvocationMap


def getBernoulliThinnedData(functionMap, invocationMap, p: float = 0.2, seed: int = 42):
    if not (0.0 <= p <= 1.0) or np.isnan(p):
        raise ValueError(f"Thinning probability p must be in [0,1]. Got p={p!r}")

    rng = np.random.default_rng(seed)
    newfunctionMap = functionMap
    newInvocationMap = {}

    for fid, inv in invocationMap.items():
        counts = np.asarray(inv.Counts, dtype=int)
        thinned = rng.binomial(counts, p).tolist()
        newInvocationMap[fid] = Invocation(
            inv.HashOwner, inv.HashApp, inv.HashFunction, inv.Trigger, thinned
        )

    return newfunctionMap, newInvocationMap


def getDataset(functionMap, invocationMap, datasetType: str, nFunction: int = 100):
    nFunction = int(nFunction)
    if datasetType == "Rare":
        return getRareData(functionMap, invocationMap, nFunction)
    elif datasetType == "Random":
        return getRandomData(functionMap, invocationMap, nFunction)
    elif datasetType == "Representative":
        return getRepresentativeData(functionMap, invocationMap, nFunction)
    elif datasetType == "Temporal":
        return getTemporalData(functionMap, invocationMap, nFunction)
    elif datasetType == "Bernoulli":
        return getBernoulliThinnedData(functionMap, invocationMap, p=0.7, seed=42)
    else:
        raise ValueError("Invalid dataset type")


def dumpData(functionMap, invocationMap, type: str, datasetPath: str, day: int = 1):
    strDay = f"{day:02}"
    with open(f"{datasetPath}/functionMap_{type}_d{strDay}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "coldStartTime", "duration", "memory"])
        for key in functionMap:
            function = functionMap[key]
            writer.writerow([function.HashOwner, function.HashApp, function.HashFunction,
                             function.coldStartTime, function.duration, function.memory])
    with open(f"{datasetPath}/invocationMap_{type}_d{strDay}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "Trigger", "Counts"])
        for key in invocationMap:
            invocation = invocationMap[key]
            writer.writerow([invocation.HashOwner, invocation.HashApp, invocation.HashFunction,
                             invocation.Trigger] + invocation.Counts)


if __name__ == "__main__":
    for type in ["Representative", "Rare", "Random", "Bernoulli", "Temporal"]:
        functionMap, invocationMap = load_data(config.datasetLocation, 1)
        functionMap, invocationMap = getDataset(functionMap, invocationMap, type, 400 if type != "Rare" else 1.5e4)
        print(f"{type} dataset, function count: {len(invocationMap)}")
        dumpData(functionMap, invocationMap, type, config.datasetLocation, 1)
