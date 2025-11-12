import numpy as np
import csv 
from Include import *
import heapq
import config

def parse_data(datasetPath:str, day:int=1):
    strDay = f"{day:02}"
    durationFile=f"{datasetPath}/function_durations_percentiles.anon.d{strDay}.csv"
    memoryFile=f"{datasetPath}/app_memory_percentiles.anon.d{strDay}.csv"
    invocationFile=f"{datasetPath}/invocations_per_function_md.anon.d{strDay}.csv"
    durationData = None
    memoryData = None
    invocationData = None
    try:
        with open(durationFile, 'r') as f:
            reader = csv.reader(f)
            durationData = list(reader)
    except IOError:
        print("Could not read file: ", durationFile)
    try:
        with open(memoryFile, 'r') as f:
            reader = csv.reader(f)
            memoryData = list(reader)
    except IOError:
        print("Could not read file: ", memoryFile)
    try:
        with open(invocationFile, 'r') as f:
            reader = csv.reader(f)
            invocationData = list(reader)
    except IOError:
        print("Could not read file: ", invocationFile)
    assert durationData and memoryData and invocationData, "Data not loaded"
    
    # key: (HashOwner, HashApp)
    durationMap = {}
    memoryMap = {}
    invocationMap = {}
    
    for line in durationData[1:]:
        HashOwner = line[0]
        HashApp = line[1]
        HashFunction = line[2]
        Average = int(line[3])
        Count = int(line[4])
        Minimum = float(line[5])
        Maximum = float(line[6])
        if (HashOwner, HashApp) not in durationMap:
            durationMap[(HashOwner, HashApp)] = {}
        if HashFunction not in durationMap[(HashOwner, HashApp)]:
            durationMap[(HashOwner, HashApp)][HashFunction] = Duration(HashOwner, HashApp, HashFunction, Average, Count, Minimum, Maximum)
    for line in memoryData[1:]:
        HashOwner = line[0]
        HashApp = line[1]
        SampleCount = int(line[2])
        AverageAllocatedMb = int(line[3])
        if (HashOwner, HashApp) not in memoryMap:
            memoryMap[(HashOwner, HashApp)] = Memory(HashOwner, HashApp, SampleCount, AverageAllocatedMb)

    # generate function data
    functionMap = {}
    for (HashOwner, HashApp) in durationMap:
        for HashFunction in durationMap[(HashOwner, HashApp)]:
            duration:Duration = durationMap[(HashOwner, HashApp)][HashFunction]
            durationTime = duration.Average
            coldStartTime = duration.Maximum-durationTime
            if (HashOwner, HashApp) not in memoryMap:
                continue
            memory:Memory = memoryMap[(HashOwner, HashApp)]
            count = len(durationMap[(HashOwner, HashApp)])
            functionMemory = memory.AverageAllocatedMb/count
            if (HashOwner, HashApp, HashFunction) not in functionMap:
                functionMap[(HashOwner, HashApp, HashFunction)] = Function(HashOwner, HashApp, HashFunction, coldStartTime, durationTime, functionMemory)
                
    for line in invocationData[1:]:
        HashOwner = line[0]
        HashApp = line[1]
        HashFunction = line[2]
        Trigger = line[3]
        Counts = list(map(int, line[4:]))
        if (HashOwner, HashApp, HashFunction) not in invocationMap and (HashOwner, HashApp, HashFunction) in functionMap:
            invocationMap[(HashOwner, HashApp, HashFunction)] = Invocation(HashOwner, HashApp, HashFunction, Trigger, Counts)
    print("Data loaded successfully")
    
    # dump maps to files
    with open(f"{datasetPath}/functionMap_d{strDay}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "coldStartTime", "duration", "memory"])
        for key in functionMap:
            function = functionMap[key]
            writer.writerow([function.HashOwner, function.HashApp, function.HashFunction, function.coldStartTime, function.duration, function.memory])
    with open(f"{datasetPath}/invocationMap_d{strDay}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "Trigger", "Counts"])
        for key in invocationMap:
            invocation = invocationMap[key]
            writer.writerow([invocation.HashOwner, invocation.HashApp, invocation.HashFunction, invocation.Trigger]+invocation.Counts)
    return functionMap, invocationMap

def load_data(datasetPath:str, day:int=1, type:str=None) -> tuple[dict[tuple[int,int,int],Function], dict[tuple[int,int,int],Invocation]]:
    strDay = f"{day:02}"
    functionMap = {}
    invocationMap = {}
    if type:
        type = f"_{type}"
    else:
        type = ""
    functionFile = f"{datasetPath}/functionMap{type}_d{strDay}.csv"
    invocationFile = f"{datasetPath}/invocationMap{type}_d{strDay}.csv"
    try:
        with open(functionFile, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if not line or not line[0].strip():  # <-- skip blank rows
                    continue
                if line[0] == "HashOwner":
                    continue
                functionMap[(line[0], line[1], line[2])] = Function(line[0], line[1], line[2], float(line[3]), float(line[4]), float(line[5]))
    except IOError:
        print("Could not read file: ", functionFile)
    try:
        with open(invocationFile, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if not line or not line[0].strip():  # <-- skip blank rows
                    continue
                if line[0] == "HashOwner":
                    continue
                invocationMap[(line[0], line[1], line[2])] = Invocation(line[0], line[1], line[2], line[3], list(map(int, line[4:])))
    except IOError:
        print("Could not read file: ", invocationFile)
    if len(functionMap) == 0:
        functionMap, invocationMap = parse_data(datasetPath, day)
    return functionMap, invocationMap

def getRareData(functionMap, invocationMap, nFunction:int=100):
    heap = []
    for functionId in invocationMap:
        invocation = invocationMap[functionId]
        heapq.heappush(heap, (sum(invocation.Counts), functionId))
    rareFuncs = heapq.nsmallest(len(invocationMap)//4, heap)
    newInvocationMap = {}
    newfunctionMap = {}
    for _, functionId in rareFuncs:
        newInvocationMap[functionId] = invocationMap[functionId]
        newfunctionMap[functionId] = functionMap[functionId]
    return newfunctionMap, newInvocationMap

def getRandomData(functionMap, invocationMap, nFunction:int=100):
    functionKeys = list(invocationMap.keys())
    randomIndexes = np.random.choice(range(len(functionKeys)), nFunction, replace=False)
    newfunctionMap = {}
    newInvocationMap = {}
    for index in randomIndexes:
        key = functionKeys[index]
        newfunctionMap[key] = functionMap[key]
        newInvocationMap[key] = invocationMap[key]
    return newfunctionMap, newInvocationMap

def getRepresentativeData(functionMap, invocationMap, nFunction:int=100):
    allFuncs = []
    for functionId in invocationMap:
        invocation = invocationMap[functionId]
        allFuncs.append((sum(invocation.Counts), functionId))
    allFuncs.sort(reverse=True)
    allFuncs = [functionId for _, functionId in allFuncs]
    # sampled from n parts
    nSample = nFunction//4
    newfunctionMap = {}
    newInvocationMap = {}
    for i in range(4):
        start = i*(len(allFuncs)//4)
        end = (i+1)*(len(allFuncs)//4)
        randomIndex = np.random.choice(range(start, end), nSample, replace=False)
        for index in randomIndex:
            functionId = allFuncs[index]
            newfunctionMap[functionId] = functionMap[functionId]
            newInvocationMap[functionId] = invocationMap[functionId]
    return newfunctionMap, newInvocationMap
    
def getDataset(functionMap, invocationMap, datasetType:str, nFunction:int=100):
    nFunction = int(nFunction)
    if datasetType == "Rare":
        return getRareData(functionMap, invocationMap, nFunction)
    elif datasetType == "Random":
        return getRandomData(functionMap, invocationMap, nFunction)
    elif datasetType == "Representative":
        return getRepresentativeData(functionMap, invocationMap, nFunction)
    else:
        raise ValueError("Invalid dataset type")
    
def dumpData(functionMap, invocationMap, type:str, datasetPath:str, day:int=1):
    strDay = f"{day:02}"
    with open(f"{datasetPath}/functionMap_{type}_d{strDay}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "coldStartTime", "duration", "memory"])
        for key in functionMap:
            function = functionMap[key]
            writer.writerow([function.HashOwner, function.HashApp, function.HashFunction, function.coldStartTime, function.duration, function.memory])
    with open(f"{datasetPath}/invocationMap_{type}_d{strDay}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["HashOwner", "HashApp", "HashFunction", "Trigger", "Counts"])
        for key in invocationMap:
            invocation = invocationMap[key]
            writer.writerow([invocation.HashOwner, invocation.HashApp, invocation.HashFunction, invocation.Trigger]+invocation.Counts)
    
if __name__ == "__main__":
    for type in ["Representative", "Rare", "Random"]:
        functionMap, invocationMap = load_data(config.datasetLocation, 1)
        functionMap, invocationMap = getDataset(functionMap, invocationMap, type, 400 if type!="Rare" else 1.5e4)
        print(f"{type} dataset, function count: {len(invocationMap)}")
        dumpData(functionMap, invocationMap, type, config.datasetLocation, 1)
        