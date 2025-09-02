import json
import os
from typing import Optional, TextIO, OrderedDict, Union

from ..vis._html_wrapper import HTMLTreeWrapper

class IMMLConfiguration:
    
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and len(kwargs) > 0:
            raise TypeError('args and kwargs cannot work together.')
        elif len(args) > 0:
            self.configs = args
        else:
            self.configs = kwargs
    
    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)
    
    def _repr_html_(self):
        return self.render()
    
    def render(self):
        return HTMLTreeWrapper(self.to_dict()).render()
    
    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, key):
        return self.configs[key]
    
    def __setitem__(self, key, value):
        self.configs[key] = value
        
    def __getattr__(self, key):
        if hasattr(self.configs, key):
            return getattr(self.configs, key)
        elif key == '_repr_html_':
            return self._repr_html_
    
    def __iter__(self):
        self.__iter = iter(self.configs)
        return self
    
    def __next__(self):
        return next(self.__iter)
    
    def keys(self):
        if isinstance(self.configs, (list, tuple)):
            return range(len(self))
        else:
            return self.configs.keys()
    
    def values(self):
        if isinstance(self.configs, (list, tuple)):
            return self.configs
        else:
            return self.configs.values()
        
    def items(self):
        return zip(self.keys(), self.values())
    
    def to_dict(self):
        return OrderedDict(self.items())
    
    def to_list(self):
        return list(self.values())
    
    def jsonify(self):
        if isinstance(self.configs, (list, tuple)):
            return self.to_list()
        else:
            return self.to_dict()
    
    @classmethod
    def from_config(cls, readable_or_path:Union[str, TextIO]):
        if isinstance(readable_or_path, TextIO):
            _text = readable_or_path.read()
        elif isinstance(readable_or_path, str):
            if os.path.isfile(readable_or_path):
                with open(readable_or_path, 'r') as f_:
                    _text = f_.read()
            elif os.path.isfile(readable_or_path+'.json'):
                with open(readable_or_path+'.json', 'r') as f_:
                    _text = f_.read()
            else:
                _text = readable_or_path
        else:
            raise TypeError(f'{type(readable_or_path)} not support.')
        _config = json.loads(_text)
        if isinstance(_config, (list, tuple)):
            return cls(*_config)
        else:
            return cls(**_config)
    
    def save_config(self, path:Optional[str]=None):
        _config = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            _dir_path = os.path.dirname(path)
            os.makedirs(_dir_path, exist_ok=True)
            with open(path, 'w') as f_:
                f_.write(_config)
        else:
            return _config
        
    @classmethod
    def from_object(cls, obj:Union[str,dict,list,tuple]):
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict):
            return cls(**obj)
        elif isinstance(obj, (list, tuple)):
            return cls(*obj)
        else:
            raise TypeError(f'{type(obj)} is not supported.')
        
class IMMLConfigurationGroup(IMMLConfiguration):
    
    def __init__(self, **kwargs):
        self.configs = kwargs
        
    def to_dict(self):
        return {k:v.configs for k,v in self.items()}
    
    def to_list(self):
        return [v.configs for v in self.values()]
    
    def save_config(self, path:Optional[str]=None):
        _config = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            _dir_path = os.path.dirname(path)
            os.makedirs(_dir_path, exist_ok=True)
            with open(path, 'w') as f_:
                f_.write(_config)
        else:
            return _config
    
    @classmethod
    def from_config(cls, readable_or_path:Optional[Union[str, TextIO]]=None, **kwargs):
        if isinstance(readable_or_path, TextIO):
            _text = readable_or_path.read()
            _config = json.loads(_text)
            _config = {k:IMMLConfiguration.from_object(v) for k,v in _config.items()}
            return cls(**_config)
        elif isinstance(readable_or_path, str):
            if os.path.isfile(readable_or_path):
                with open(readable_or_path, 'r') as f_:
                    _text = f_.read()
            elif os.path.isfile(readable_or_path+'.json'):
                with open(readable_or_path+'.json', 'r') as f_:
                    _text = f_.read()
            else:
                _text = readable_or_path
            _config = json.loads(_text)
            _config = {k:IMMLConfiguration.from_object(v) for k,v in _config.items()}
            return cls(**_config)
        elif readable_or_path is None:
            _config = {k:IMMLConfiguration.from_config(v) for k,v in kwargs.items()}
            return cls(**_config)
        else:
            raise TypeError(f'{type(readable_or_path)} not support.')