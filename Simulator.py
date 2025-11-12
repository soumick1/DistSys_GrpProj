from Include import *
from queue import PriorityQueue
from tqdm import tqdm, trange
import heapq
import csv
from multiprocessing import current_process
from TraceGen import *
import config


# --- WTinyLFU: tiny Count-Min Sketch for admission ---
class CountMinSketch:
    def __init__(self, width=4096, depth=4, decay=0.01):
        self.width = width
        self.depth = depth
        self.tables = [[0] * width for _ in range(depth)]
        self.total = 0
        self.decay = decay

    def _h(self, key, i):
        # stable hash across runs for str/tuples
        return (hash((i, key)) & 0x7FFFFFFF) % self.width

    def add(self, key, c=1):
        for i in range(self.depth):
            self.tables[i][self._h(key, i)] += c
        self.total += c
        # periodic decay keeps counts fresh
        if self.total % (self.width) == 0:
            for i in range(self.depth):
                row = self.tables[i]
                for j in range(self.width):
                    # integer decay to avoid floats
                    row[j] = row[j] - (row[j] >> 3)  # ~= *0.875

    def estimate(self, key):
        return min(self.tables[i][self._h(key, i)] for i in range(self.depth))



class Stats:
    def __init__(
        self,
        time,
        coldStartTime: int = 0,
        memorySize: int = 0,
        excutingTime: int = 0,
        nColdStart: int = 0,
        nExcution: int = 0,
    ):
        self.time = time
        self.coldStartTime = coldStartTime
        self.memorySize = memorySize
        self.excutingTime = excutingTime
        self.nColdStart = nColdStart
        self.nExcution = nExcution





class Container:
    def __init__(self, priority: int, functionId: str, endTime: int):
        self.priority = priority
        self.functionId = functionId
        self.endTime = endTime
        self.frequency = 1
        # New: metadata for SLRU / 2Q / WTinyLFU
        self.segment = "prob"   # SLRU: "prob" or "prot"
        self.queue = "A1"       # 2Q:   "A1"  or "Am"
        self.admitted = True    # WTinyLFU admission decision
        self.newborn = True     # True only for the event that created it

    def __lt__(self, other):
        return self.priority < other.priority


class Simulator:
    def __init__(
        self,
        memoryBudget: float,
        functionMap: dict[tuple, Function],
        invocationMap: dict[tuple, Invocation],
        policy: str = "LRU",
        timeLimit: int = 0,
        functionLimit: int = 0,
        logInterval: int = 1000,
        progressBar: bool = False,
        verbose: bool = False,
    ):
        self.memoryBudget = memoryBudget  # MB
        self.functionMap = functionMap
        self.invocationMap = invocationMap
        self.policy = policy
        self.logInterval = logInterval
        self.TTL = min2ms(10)

        self.eventQueue = []
        self.memoryUsed = 0
        self.cache: list[Container] = []
        self.stats: list[Stats] = []
        self.curMin = 0
        self.coldStartTime = 0
        self.excutingTime = 0
        self.lastLogTime = 0
        self.nColdStart = 0
        self.nExcution = 0
        self.minMemoryReq = 0
        self.filename=f"{self.policy}"
        self.step = 0
        self.progressBar = progressBar
        self.verbose = verbose
        self.freqWeight = {}

        self._slru_prot_boost = 1_000_000_000_000  # keep protected well above probation
        self._slru_prot_frac = 0.5  # ~half of cache entries can be protected

        # 2Q: main-queue boost (A1 < Am so A1 evicts first)
        self._twoq_Am_boost = 500_000_000_000

        # WTinyLFU: admission filter
        self.cms = CountMinSketch()

        # FassCache actually uses a logical clock instead of a physical clock
        # The clock is updated by the the priority of the envicted cache item
        self.logicalClock = 0

        # init event queue
        functionCount = 0
        for functionId in self.invocationMap:
            functionCount += 1
            if functionLimit and functionLimit > 0 and functionCount > functionLimit:
                break
            counts = self.invocationMap[functionId].Counts
            for min, count in enumerate(counts):
                if timeLimit and min > timeLimit:
                    break
                if count == 0:
                    continue
                for i in range(count):
                    time = min2ms(min + i / count)
                    self.eventQueue.append((time, functionId))
        self.eventQueue.sort(key=lambda x: x[0])

    def log(self, msg: str, filename: str = "log.txt", newfile: bool = False):
        if not self.verbose:
            return
        if newfile:
            open(filename, "w").write("")
        open(filename, "a").write(msg)

    def run(self):
        self.log(f"Policy {self.policy}\n", filename=f"./log/{self.filename}.log", newfile=True)
        # log("Start simulation\n", newfile=True)
        if self.progressBar:
            # try:
            #     barPosition=current_process()._identity[0]
            # except:
            barPosition=0
            rangeObject=tqdm(range(len(self.eventQueue)), desc=f"{self.policy:10}",position=barPosition,leave=False)
        else:
            rangeObject=range(len(self.eventQueue))
        for _ in rangeObject:
            self.process_event()
            self.step += 1

    def setPolicy(self, policy: str):
        self.policy = policy

    def getFreq(self, functionId, time):
        return sum([container.frequency for container in self.cache if container.functionId == functionId])
    
        # real frequency, but perfroms bad
        freqWindow = min2ms(1)
        freq = 0
        step = self.step -1
        while step >= 0:
            eventTime, eventFunctionId = self.eventQueue[step]
            if time - eventTime > freqWindow:
                break
            if functionId == eventFunctionId:
                freq += 1
            step -= 1
        return freq
            
    def getWeight(self, functionId, time):
        # weight = 0
        # for i in range(len(self.freqWeight[functionId])):
        #     freq, t = self.freqWeight[functionId][i]
        #     if i == len(self.freqWeight[functionId]) - 1:
        #         t = time - t
        #     if t == 0:
        #         continue
        #     weight += freq / t
        # return weight
        funcCount = sum([container.frequency for container in self.cache if container.functionId == functionId])
        triggerCount = sum([container.frequency for container in self.cache if self.invocationMap[functionId].Trigger == self.invocationMap[container.functionId].Trigger])
        total = len(self.cache)
        if total == 0 or triggerCount == 0:
            return 1
        return triggerCount / total

    def _slru_enforce(self):
        # Demote oldest protected if protected set is too large
        prot = [c for c in self.cache if c.segment == "prot"]
        if len(self.cache) == 0:
            return
        target = int(max(1, len(self.cache) * self._slru_prot_frac))
        if len(prot) > target:
            victim = min(prot, key=lambda c: c.priority)
            victim.segment = "prob"
            victim.priority -= self._slru_prot_boost

    def _twoq_mark_promote(self, container):
        # on second hit: A1 -> Am
        if container.queue == "A1":
            container.queue = "Am"
            container.priority += self._twoq_Am_boost

    def getPriority(self, time, functionId):
        freq = self.getFreq(functionId, time)
        logFreq = np.log(freq + 1) + 1
        cost = self.functionMap[functionId].coldStartTime
        size = max(1e-6, self.functionMap[functionId].memory)  # avoid divide-by-zero
        priority = time

        # Existing policies (unchanged semantics)
        if self.policy == "LRU":
            priority = self.logicalClock
        elif self.policy == "LFU":
            priority = freq
        elif self.policy == "GD":
            priority = self.logicalClock + freq * (cost / size)
        elif self.policy == "FREQCOST":
            priority = self.logicalClock + freq * cost
        elif self.policy == "FREQSIZE":
            priority = self.logicalClock + freq / size
        elif self.policy == "COSTSIZE":
            priority = self.logicalClock + cost / size
        elif self.policy == "LGD":
            priority = self.logicalClock + (cost / size) * logFreq
        elif self.policy == "SIZE":
            priority = self.logicalClock + 1 / size
        elif self.policy == "COST":
            priority = self.logicalClock + cost
        elif self.policy == "FREQ":
            priority = self.logicalClock + freq
        elif self.policy == "TTL":
            priority = time
        elif self.policy == "RAND":
            priority = np.random.randint(10)

        # --- New policies (priority scale: larger = safer) ---
        elif self.policy == "GDSF":
            # GreedyDual-Size-Frequency: like GD but factors freq/size multiplicatively
            priority = self.logicalClock + (freq * cost) / size

        elif self.policy == "SLRU":
            # base on recency; segment boost applied in findAvailContainer/newContainer
            priority = self.logicalClock

        elif self.policy == "TWOQ":
            # base on recency; queue boost applied in findAvailContainer/newContainer
            priority = self.logicalClock

        elif self.policy == "WTINYLFU_LRU":
            # eviction behaves like LRU; admission decided elsewhere
            priority = self.logicalClock

        elif self.policy == "WTINYLFU_COSTSIZE":
            # eviction behaves like COSTSIZE; admission decided elsewhere
            priority = self.logicalClock + cost / size

        self.log(
            f"time {int(round(time))}, clock {self.logicalClock}, functionId {functionId}, "
            f"freq {freq}, cost {cost}, size {size}, Lfreq {logFreq}, priority {priority}\n",
            filename=f"./log/{self.filename}.log",
        )
        return priority

    def freeMemory(self, size, time):
        # assert (
        #     self.memoryBudget >= size
        # ), f"Memory budget too small, size {size}, memoryBudget {self.memoryBudget}"
        maxPriority = 0
        i = 0
        self.cache.sort(key=lambda x: x.priority)
        while self.memoryUsed + size > self.memoryBudget and i < len(self.cache):
            container = self.cache[i]
            # skip running container
            if container.endTime > time:
                i += 1
                continue
            # remove container
            self.memoryUsed -= self.functionMap[container.functionId].memory
            self.cache.pop(i)
            # update logical clock
            self.logicalClock = container.priority
            # self.logicalClock = max(maxPriority, container.priority)
            self.freqWeight[container.functionId][-1][1] = time - self.freqWeight[container.functionId][-1][1]
        self.minMemoryReq = max(self.minMemoryReq, self.memoryUsed + size)
        if self.memoryUsed + size > self.memoryBudget:
            # TODO: should delay the event
            pass

    def newContainer(self, time, functionId):
        if functionId not in self.freqWeight:
            self.freqWeight[functionId] = []
        self.freqWeight[functionId].append([0, time])

        endTime = time + self.functionMap[functionId].duration + self.functionMap[functionId].coldStartTime
        priority = self.getPriority(time, functionId)
        container = Container(priority, functionId, endTime)

        # Initialize for SLRU / 2Q
        if self.policy == "SLRU":
            container.segment = "prob"  # start in probation
            # no boost for probation (protected gets big boost in promotion)
        if self.policy == "TWOQ":
            container.queue = "A1"  # start in A1 (recent)

        # WTinyLFU admission (decide whether to keep it warm after serving)
        if self.policy in ("WTINYLFU_LRU", "WTINYLFU_COSTSIZE"):
            # choose likely victim by priority (not running)
            victim = None
            for c in sorted(self.cache, key=lambda x: x.priority):
                if c.endTime <= time:
                    victim = c
                    break
            f_new = self.cms.estimate(functionId)
            f_vic = self.cms.estimate(victim.functionId) if victim else 0
            # admit only if new estimated frequency >= victim's (classic WTinyLFU idea)
            container.admitted = (f_new >= f_vic)

        self.cache.append(container)
        self.memoryUsed += self.functionMap[functionId].memory
        return container

    def findAvailContainer(self, time, functionId):
        # TTL policy: evict idle
        if self.policy == "TTL":
            i = 0
            while i < len(self.cache):
                container = self.cache[i]
                if container.priority + self.TTL < time:
                    self.memoryUsed -= self.functionMap[container.functionId].memory
                    self.cache.pop(i)
                else:
                    i += 1

        # find available container and update
        for container in self.cache:
            if container.functionId == functionId and container.endTime < time:
                container.priority = self.getPriority(time, functionId)
                container.endTime = time + self.functionMap[functionId].duration
                container.frequency += 1
                self.freqWeight[functionId][-1][0] += 1

                # SLRU: on hit, promote probation -> protected
                if self.policy == "SLRU":
                    if container.segment == "prob":
                        container.segment = "prot"
                        container.priority += self._slru_prot_boost
                        self._slru_enforce()

                # 2Q: on second hit, A1 -> Am
                if self.policy == "TWOQ":
                    self._twoq_mark_promote(container)

                return container  # return container object
        return None

    def process_event(self):
        time, functionId = self.eventQueue[self.step]
        # track popularity for WTinyLFU
        self.cms.add(functionId, 1)

        functionInfo = self.functionMap[functionId]
        coldStartTime = 0
        excutingTime = functionInfo.duration

        # try to find an available (idle) container
        container = self.findAvailContainer(time, functionId)
        created_now = False
        if container is None:
            # free enough memory for a new container
            self.freeMemory(functionInfo.memory, time)
            # create a new container
            container = self.newContainer(time, functionId)
            created_now = True
            # cold start penalty
            coldStartTime = functionInfo.coldStartTime
            excutingTime += coldStartTime
            self.nColdStart += 1

        self.nExcution += 1

        # sync priority across same functionId instances
        for c in self.cache:
            if c.functionId == functionId:
                c.priority = container.priority

        # stats
        self.coldStartTime += coldStartTime
        self.excutingTime += excutingTime
        if time - self.lastLogTime > self.logInterval:
            self.lastLogTime = time
            self.stats.append(
                Stats(
                    time,
                    self.coldStartTime,
                    self.memoryUsed,
                    self.excutingTime,
                    self.nColdStart,
                    self.nExcution,
                )
            )
            self.log(
                f"time {int(round(time))}, coldStartTime {self.coldStartTime}, memoryUsed {self.memoryUsed}, "
                f"excutingTime {self.excutingTime}, nColdStart {self.nColdStart}, nExcution {self.nExcution}\n",
                filename=f"./log/{self.filename}.log",
            )

        # WTinyLFU admission: if not admitted, don't keep it warm
        if created_now and (self.policy in ("WTINYLFU_LRU", "WTINYLFU_COSTSIZE")) and (not container.admitted):
            # drop immediately after serving the current invocation
            self.memoryUsed -= self.functionMap[functionId].memory
            try:
                self.cache.remove(container)
            except ValueError:
                pass

    def dumpStats(self, location: str):
        csv_file = open(f"{location}/{self.filename}.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["MinMemoryReq", self.minMemoryReq])
        csv_writer.writerow(
            [
                "time",
                "coldStartTime",
                "memorySize",
                "excutingTime",
                "nColdStart",
                "nExcution",
            ]
        )
        for stat in self.stats:
            csv_writer.writerow(
                [
                    stat.time,
                    stat.coldStartTime,
                    stat.memorySize,
                    stat.excutingTime,
                    stat.nColdStart,
                    stat.nExcution,
                ]
            )


# if __name__ == "__main__":
#     day = 1
#     dataLocation = config.datasetLocation
#     policy="COSTSIZE"
#     memoryBudget = 10e3
#     # in min
#     timeLimit = 100
#     functionLimit = 400
#     # in ms
#     logInterval = 1e3
#
#     functionMap, invocationMap = load_data(dataLocation, day)
#     simulator = Simulator(memoryBudget, functionMap, invocationMap, policy, timeLimit, functionLimit, logInterval, True, True)
#     simulator.run()
#     simulator.dumpStats(f"./log")
#
#     print(" Policy, ColdStartTime, MemorySize, ExcutingTime, NColdStart, NExcution, PeakMemory")
#     with open(f"log/{policy}.csv", "r", newline='') as f:
#         r = csv.reader(f)
#         rows = [row for row in r if row]  # drop blank lines
#     minMemoryReq = float(rows[0][1])
#     time, coldStartTime, memorySize, excutingTime, nColdStart, nExcution = rows[-1]
#     print([policy, float(coldStartTime), float(memorySize), float(excutingTime), float(nColdStart), float(nExcution), minMemoryReq])

if __name__ == "__main__":
    day = 1
    dataLocation = config.datasetLocation
    memoryBudget = 10e3      # MB
    timeLimit = 100          # minutes
    functionLimit = 400
    logInterval = 1e3        # ms

    functionMap, invocationMap = load_data(dataLocation, day)

    policies = [
        "TTL",
        "LRU",
        "LFU",
        "GD",
        "LGD",
        "SIZE",
        "COST",
        "FREQ",
        "RAND",
        # existing cost-aware variants (supported in Simulator)
        "FREQCOST",
        "FREQSIZE",
        "COSTSIZE",
        # NEW policies
        "GDSF",
        "SLRU",
        "TWOQ",
        "WTINYLFU_LRU",
        "WTINYLFU_COSTSIZE",
        #"Baseline",
    ]

    print("Policy, ColdStartTime, MemorySize, ExcutingTime, NColdStart, NExcution, PeakMemory")
    for policy in policies:
        simulator = Simulator(memoryBudget, functionMap, invocationMap,
                              policy, timeLimit, functionLimit, logInterval,
                              progressBar=True, verbose=True)
        simulator.run()
        simulator.dumpStats("./log")

        with open(f"log/{policy}.csv", "r", newline="") as f:
            import csv
            rows = [r for r in csv.reader(f) if r]
        minMemoryReq = float(rows[0][1])  # first row second column
        time, coldStartTime, memorySize, excutingTime, nColdStart, nExcution = rows[-1]
        print([policy, float(coldStartTime), float(memorySize), float(excutingTime),
               float(nColdStart), float(nExcution), minMemoryReq])
