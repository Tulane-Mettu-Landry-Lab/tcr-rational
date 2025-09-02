import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from scipy.special import softmax
from ._metric import Metric

@Metric.register()    
class MetricMLMAccuracy(Metric):
    name = 'mlm_accuracy'
    def metric(self, y_true, y_pred, ignore_id=-100, **kwargs):
        mask = (y_true != ignore_id)
        return accuracy_score(y_true[mask], np.argmax(y_pred[mask], axis=-1))

@Metric.register()    
class MetricMLMPrecision(Metric):
    name = 'mlm_precision'
    def metric(self, y_true, y_pred, ignore_id=-100, average='macro', zero_division=0, **kwargs):
        mask = (y_true != ignore_id)
        return precision_score(
            y_true[mask],
            np.argmax(y_pred[mask], axis=-1),
            labels=np.arange(y_pred.shape[-1]),
            average=average,
            zero_division=zero_division,
        )
        
@Metric.register()    
class MetricMLMRecall(Metric):
    name = 'mlm_recall'
    def metric(self, y_true, y_pred, ignore_id=-100, average='macro', zero_division=0, **kwargs):
        mask = (y_true != ignore_id)
        return recall_score(
            y_true[mask],
            np.argmax(y_pred[mask], axis=-1),
            labels=np.arange(y_pred.shape[-1]),
            average=average,
            zero_division=zero_division,
        )
        
@Metric.register()    
class MetricMLMROCAUC(Metric):
    name = 'mlm_rocauc'
    def metric(self, y_true, y_pred, ignore_id=-100, multi_class='ovo', norm=True, **kwargs):
        mask = (y_true != ignore_id)
        y_true_ = y_pred[mask]
        if norm:
            y_true_ = softmax(y_true_, axis=-1)
        return roc_auc_score(
            y_true[mask],
            y_true_,
            multi_class=multi_class,
            labels=np.arange(y_pred.shape[-1])
        )