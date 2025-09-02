from typing import Optional
from ._hook import ModuleHook
from .module_node import ModuleNode
from ._stack import ModuleStack
class ModuleStackGroup(object):

    def __init__(self, module_node:ModuleNode, module_hook:Optional[ModuleHook]=None):
        self.node = module_node
        self.hook = module_hook
        self.stacks = []

    def __len__(self):
        return len(self.stacks)
    
    def __getitem__(self, key):
        return self.stacks[key]
    
    def __repr__(self):
        return f'Track {len(self)} groups of modules in {self.node.name}'
    
    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        else:
            _value = self.stacks[self._iter_idx]
            self._iter_idx += 1
            return _value
        
    def add(self, *args, stack:Optional[ModuleStack]=None):
        if stack is None:
            stack = ModuleStack(self.node, *args, module_hook=self.hook)
            self.stacks.append(stack)
        else:
            stack.set_node(self.node)
            stack.set_hook(self.hook)
            self.stacks.append(stack)