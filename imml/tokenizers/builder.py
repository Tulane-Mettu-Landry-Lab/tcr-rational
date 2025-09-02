import json
import os
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE, WordLevel, WordPiece, Unigram, Model
from tokenizers.pre_tokenizers import Split, PreTokenizer
from tokenizers.trainers import BpeTrainer, Trainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from typing import Union, OrderedDict, Optional, Tuple, Any

from ..data import DataColumnDataset, DataFrameDataset

class TokenizerBuilder(object):
    
    _default_special_tokens = \
        dict(
            eos_token = '<EOS>',
            mask_token = '<MSK>',
            cls_token = '<CLS>',
            pad_token = '<PAD>',
            unk_token = '<MIS>',
            sep_token = '<SEP>',
            bos_token = '<BOS>',
        )
    
    
    def __init__(
        self,
        vocab_size:int=50,
        model:Union[str,Model]='BPE',
        special_tokens:Optional[dict]={},
        split_pattern:str='',
        split_behavior:str='isolated',
    ) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = self._default_special_tokens
        self.special_tokens.update(special_tokens)
        self.split_pattern = split_pattern
        self.split_behavior = split_behavior
        self.model = self.__fetch_model(model)
        self.pre_tokenizer = \
            Split(
                pattern=Regex(split_pattern),
                behavior=split_behavior
            )
        self.trainer = \
            self.__fetch_trainer(
                model = self.model
            )(
                vocab_size = vocab_size,
                special_tokens = list(self.special_tokens.values())
            )
            
    @classmethod
    def from_template(cls, template:Union[str, dict]):
        if isinstance(template, str):
            with open(template, 'r') as _template_file:
                template = json.load(_template_file)
        return cls(
            vocab_size = template.get('vocab_size', 50),
            model = template.get('model', 'BPE'),
            special_tokens = template.get('special_tokens', {}),
            split_pattern = template.get('split_pattern', ''),
            split_behavior = template.get('split_behavior', 'isolated'),
        )
            
    def __repr__(self) -> str:
        return f'{self.model.__qualname__} Tokenizer Builder: vocab_size={self.vocab_size}'
    
    @property
    def template(self) -> dict:
        return dict(
            vocab_size = self.vocab_size,
            model = self.model.__qualname__,
            special_tokens = self.special_tokens,
            split_pattern = self.split_pattern,
            split_behavior = self.split_behavior,
        )

    def __fetch_model(
        self,
        model:Union[str,Model]='BPE'
    ) -> Model:
        if isinstance(model, Model):
            return model
        else:
            model = {
                'BPE': BPE,
                'WordLevel': WordLevel,
                'WordPiece': WordPiece,
                'Unigram': Unigram,
            }[model]
            return model
    
    def __build_tokenizer(
        self,
        model:Model,
        special_tokens:dict,
        pre_tokenizer:PreTokenizer
    ) -> Tokenizer:
        _tokenizer =  Tokenizer(
            model(
                unk_token=special_tokens['unk_token']
            )
        )
        _tokenizer.pre_tokenizer = pre_tokenizer
        return _tokenizer
    
    def __fetch_trainer(
        self,
        model:Model,
    ) -> Trainer:
        _trainer = {
            BPE: BpeTrainer,
            WordLevel: WordLevelTrainer,
            WordPiece: WordPieceTrainer,
            Unigram: UnigramTrainer
        }[model]
        return _trainer
    
    def __to_pretrained(
        self,
        tokenizer:Tokenizer,
        special_tokens:dict,
    ) -> PreTrainedTokenizer:
        _pretrained_tokenizer = \
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                eos_token = special_tokens['eos_token'],
                mask_token = special_tokens['mask_token'],
                cls_token = special_tokens['cls_token'],
                pad_token = special_tokens['pad_token'],
                unk_token = special_tokens['unk_token'],
                sep_token = special_tokens['sep_token']
            )
        return _pretrained_tokenizer
        
    def train(
        self,
        data:Dataset
    ) -> PreTrainedTokenizer:
        tokenizer = \
            self.__build_tokenizer(
                model=self.model,
                special_tokens=self.special_tokens,
                pre_tokenizer=self.pre_tokenizer
            )
        tokenizer.train_from_iterator(data, self.trainer)
        
        return self.__to_pretrained(
            tokenizer=tokenizer,
            special_tokens=self.special_tokens
        )
    
    def __call__(
        self,
        data:Dataset
    ) -> PreTrainedTokenizer:
        return self.train(data=data)
    
def build_tokenizer(
    data:DataFrameDataset,
    columns:list[str],
    template:Union[str, dict],
    path:str,
    remove_na:bool=True,
    unique:bool=True,
    name:str='Tokenizer',
    desc:str='Tokenizer',
) -> PreTrainedTokenizer:
    os.makedirs(path, exist_ok=True)
    col_data = DataColumnDataset(data, columns, remove_na=remove_na, unique=unique)
    builder = TokenizerBuilder.from_template(template)
    tokenizer = builder(col_data)
    tokenizer.save_pretrained(path)
    readme = {
        'name': name,
        'desc': desc,
        'dataset': {
            'name': data.config['name'],
            'desc': data.config['desc'],
            'columns': columns,
            'size': len(col_data),
            'remove_na': remove_na,
            'unique': unique
        },
        'builder': builder.template,
        'tokenizer': {
            'vocab_size': tokenizer.vocab_size,
            'special_tokens': tokenizer.special_tokens_map
        }
    }
    with open(os.path.join(path, 'readme.json'), 'w') as _readme_file:
        json.dump(readme, _readme_file, indent=2)
    return tokenizer