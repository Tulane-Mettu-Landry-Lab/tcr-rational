class SolverLibrary(object):
    
    _lib = {}
    
    def __class_getitem__(cls, item):
        return cls._lib[item]
    
    @classmethod
    def register(cls, name:str):
        def _register_wrapper(method):
            cls._lib[name] = method
            return method
        return _register_wrapper
    
    @classmethod
    def methods(cls):
        return cls._lib.keys()