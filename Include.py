import numpy as np
from dataclasses import dataclass

@dataclass
class Duration:
    HashOwner: int
    HashApp: int
    HashFunction: int
    Average: float
    Count: int
    Minimum: float
    Maximum: float
    
# HashOwner,HashApp,SampleCount,AverageAllocatedMb
@dataclass
class Memory:
    HashOwner: int
    HashApp: int
    SampleCount: int
    AverageAllocatedMb: int
        
# HashOwner,HashApp,HashFunction,Trigger,1..1440
@dataclass
class Invocation:
    HashOwner: int
    HashApp: int
    HashFunction: int
    Trigger: str
    Counts: list[int]

@dataclass
class Function:
    HashOwner: int
    HashApp: int
    HashFunction: int
    coldStartTime: float
    duration: float
    memory: float
        
def min2ms(minute:float)->float:
    return minute*60*1000
def ms2min(ms:float)->float:
    return ms/(60*1000)