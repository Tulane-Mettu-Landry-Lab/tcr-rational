import torch
from typing import Optional
from transformers.tokenization_utils_base import BatchEncoding
from transformers import DataCollatorForLanguageModeling

class RuleCollator:
    
    def __init__(
        self,
        rules:list[dict],
        tokenizer,
        mlm:bool = True,
        mlm_probability:float=0.15,
        mask_replace_prob:float=0.8,
        random_replace_prob:float=0.1,
        pad_to_multiple_of:Optional[int]=None,
        seed:Optional[int]=None,
    ) -> None:
        self._mlm_collator = \
            DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=mlm,
                mlm_probability=mlm_probability,
                mask_replace_prob=mask_replace_prob,
                random_replace_prob=random_replace_prob,
                pad_to_multiple_of=pad_to_multiple_of,
                seed=seed,
            )
        self.rules = rules
        self.rule_names = self.__fetch_rule_names(self.rules)
    
    def __fetch_rule_names(self, rules):
        return [
            i if rule['prefix'] is None else rule['prefix']
            for i, rule in enumerate(rules)
        ]
        
    def _debatch(self, batch):
        samples = {}
        for sample in batch:
            for key in sample:
                if key not in samples:
                    samples[key] = []
                samples[key].append(sample[key])
        return samples
    
    def __call__(self, batch):
        _samples = self._debatch(batch=batch)
        for key in _samples:
            if isinstance(_samples[key], list) and \
                isinstance(_samples[key][0], dict) and \
                'input_ids' in _samples[key][0]:
                _samples[key] = self._mlm_collator(_samples[key])
            else:
                _samples[key] = torch.tensor(_samples[key])
                
        _samples_flatten = {}
        for key, val in _samples.items():
            if isinstance(val, BatchEncoding):
                _samples_flatten.update({f'{key}_{k}':v for k, v in val.items()})
            else:
                _samples_flatten[key] = val
        return BatchEncoding(_samples_flatten)