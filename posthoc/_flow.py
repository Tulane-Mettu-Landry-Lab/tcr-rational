from typing import Callable
from .methods import ModulePosthocReduce
class ModelFlow(object):
    
    def __init__(self, track, *args, collect:Callable=lambda x:x):
        self.flow_blocks = args
        self.track = track
        self.collect = collect
    
    def __len__(self):
        return len(self.flow_blocks)
    
    def __repr__(self):
        return f'{len(self)} track methods set'
    
    def set_flow(self, *args):
        self.flow_blocks = args
        
    def set_collect(self, collect:Callable=lambda x:x):
        self.collect = collect
    
    def solve(self):
        _df = self.track.table.sort_values('Backward Order')
        _out = None
        for idx, module_id in enumerate(_df.ID.values):
            _step_out = self.flow_blocks[idx](*self.track.hook[int(module_id)])
            if _out is None:
                _out = _step_out
            else:
                _out = ModulePosthocReduce(type(self.flow_blocks[idx-1]), type(self.flow_blocks[idx]))(_out, _step_out)
        return self.collect(_out)
    
    def __call__(self):
        return self.solve()