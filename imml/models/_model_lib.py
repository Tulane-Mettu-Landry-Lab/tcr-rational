class ModelLibrary(object):
    _lib = {}
    
    def __class_getitem__(cls, item):
        return cls._lib[item]
    
    @classmethod
    def register(cls, name:str):
        def _register_wrapper(model):
            cls._lib[name] = model
            return model
        return _register_wrapper
    
    @classmethod
    def models(cls):
        return cls._lib.keys()