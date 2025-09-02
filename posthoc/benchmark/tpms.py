from itertools import chain
import json
import numpy as np

class TCRpMHCSurfaceBenchmark(object):
    def __init__(self, path='datasets/itcr/STCRDab_xAI.json'):
        self.path = path
        with open(self.path, 'r') as f:
            self.data = json.load(f)
        self.distances = {
            'alpha':[], 'beta':[], 'epitope':[]
        }
        for sample in self.data:
            _d = np.stack([np.array(i).min(axis=-1) for k, i in sample['distance'].items() if k in ['cdra3', 'cdrb3']]).min(axis=0)
            # _d = np.stack([np.array(i).min(axis=-1) for i in sample['distance'].values()]).min(axis=0)
            self.distances['epitope'].append(_d)
            self.distances['alpha'].append(np.array(sample['distance']['cdra3']).min(axis=0))
            self.distances['beta'].append(np.array(sample['distance']['cdrb3']).min(axis=0))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.distances[key]

    def fetch(self, preds, chain_='epitope', flatten=False):
        r_preds = []
        for gt, pred in zip(self.distances[chain_], preds):
            r_preds.append(list(pred[:len(gt)]))
        if flatten:
            r_preds = np.array(list(chain(*r_preds)))
        return r_preds