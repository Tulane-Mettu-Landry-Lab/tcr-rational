from typing import Optional
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from ._rule_collator import RuleCollator


class RuleProcessor(object):
    
    def __init__(self, rules=[], tokenizers=[], primary_tokenizer=0):
        self._tokenizers = {}
        for config in tokenizers:
            self.add_tokenizer(**config)
        self.rules = rules
        if isinstance(primary_tokenizer, int):
            primary_tokenizer = list(self._tokenizers.keys())[primary_tokenizer]
        self.primary_tokenizer = self._tokenizers[primary_tokenizer]
        
    def template_render(self, template, vars, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.primary_tokenizer
        else:
            tokenizer = self._tokenizers[tokenizer]
        _special_tokens = {k.split('_')[0]:v for k, v in tokenizer.special_tokens_map.items()}
        return str(template).format(**vars, **_special_tokens)
    
    def add_tokenizer(self, name, path, single='<CLS> $A <EOS>', pair='<CLS> $A <SEP> $B:1 <EOS>:1'):
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=single,
            pair=pair,
            special_tokens=[
                (token, tokenizer.encode(token)[0])
                for token in tokenizer.special_tokens_map.values()
            ]
        )
        self._tokenizers[name] = tokenizer
        
    def rule_template(self, sample, template, tokenizer, mapping={}):
        _data = self.template_render(template=template, vars=sample, tokenizer=tokenizer)
        _data = self._tokenizers[tokenizer](_data)
        _data = {mapping.get(k,k):v for k,v in _data.items()}
        return _data
    
    def rule_forward(self, sample, keys, mapping={}):
        _data = {key:sample[key] for key in keys}
        _data = {mapping.get(k,k):v for k,v in _data.items()}
        return _data
    
    def rule_tokenize(self, sample, template, tokenizer, mapping={}):
        _data = self.template_render(template=template, vars=sample, tokenizer=tokenizer)
        _data = self._tokenizers[tokenizer](_data)['input_ids']
        _data = dict(input_ids = _data)
        _data = {mapping.get(k,k):v for k,v in _data.items()}
        return _data
    
    
    def process(self, sample):
        output = {}
        for i, rule in enumerate(self.rules):
            _rule = rule['rule']
            _prefix = rule['prefix']
            _param = {k:v for k, v in rule.items() if k not in ['rule', 'prefix']}
            _key = str(i) if _prefix is None else _prefix
            if _rule == 'template':
                output[_key] = self.rule_template(sample=sample, **_param)
            elif _rule == 'forward':
                output.update(self.rule_forward(sample=sample, **_param))
            elif _rule == 'tokenize':
                output.update(self.rule_tokenize(sample=sample, **_param))  
            else:
                raise Exception(f'{_rule} is invalid rule.')
        return output
    
    def __call__(self, sample):
        return self.process(sample)
    
    def colletor(
        self,
        mlm:bool = True,
        mlm_probability:float=0.15,
        mask_replace_prob:float=0.8,
        random_replace_prob:float=0.1,
        pad_to_multiple_of:Optional[int]=None,
        seed:Optional[int]=None,
    ) -> RuleCollator:
        return RuleCollator(
            rules=self.rules,
            tokenizer=self.primary_tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            mask_replace_prob=mask_replace_prob,
            random_replace_prob=random_replace_prob,
            pad_to_multiple_of=pad_to_multiple_of,
            seed=seed,
        )