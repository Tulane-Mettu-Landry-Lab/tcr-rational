import numpy as np
from ._flow_collector import FlowCollector

class ResultAnalyzer(object):
    
    def __init__(self, collector:FlowCollector, benchmark, configs=[]):
        self.collector = collector
        self.benchmark = benchmark
        self.configs = configs
        self.results = {}
    
    def collect(self):
        for (method_, chain_), collect in zip(self.configs, self.collector.collects):
            if method_ not in self.results:
                self.results[method_] = {}
            # self.results[method_][chain_] = self.benchmark.fetch(self.align_collect(collect), chain_=chain_)
            self.results[method_][chain_] = self.benchmark.fetch(self.align_collect(collect)[:, 1:], chain_=chain_)
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.results[key]
        else:
            return self.results[key[0]][key[1]]
    
    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()
    
    def items(self):
        return self.results.items()
    
    def align_collect(self, collect):
        _max_len = max([m.shape[-1] for m in collect])
        _padded = [np.pad(c, pad_width=[0, _max_len-c.shape[-1]]) for c in collect]
        return np.concatenate(_padded, axis=0)