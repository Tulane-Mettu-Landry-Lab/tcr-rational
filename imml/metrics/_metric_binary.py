import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from scipy.special import softmax
from ._metric import Metric

@Metric.register()    
class MetricBinaryAccuracy(Metric):
    name = 'binary_accuracy'
    def metric(self, y_true, y_pred, **kwargs):
        return accuracy_score(y_true, np.argmax(y_pred, axis=-1))

@Metric.register()    
class MetricBinaryPrecision(Metric):
    name = 'binary_precision'
    def metric(self, y_true, y_pred, norm=True, threshold=0.5, **kwargs):
        if norm:
            y_pred = softmax(y_pred, axis=-1)
        return precision_score(y_true, y_pred[:, -1] > threshold)
    
@Metric.register()    
class MetricBinaryRecall(Metric):
    name = 'binary_recall'
    def metric(self, y_true, y_pred, norm=True, threshold=0.5, **kwargs):
        if norm:
            y_pred = softmax(y_pred, axis=-1)
        return recall_score(y_true, y_pred[:, -1] > threshold)

@Metric.register()    
class MetricBinaryF1(Metric):
    name = 'binary_f1'
    def metric(self, y_true, y_pred, norm=True, threshold=0.5, **kwargs):
        if norm:
            y_pred = softmax(y_pred, axis=-1)
        return f1_score(y_true, y_pred[:, -1] > threshold)
    
@Metric.register()    
class MetricBinaryROCAUC(Metric):
    name = 'binary_rocauc'
    def metric(self, y_true, y_pred, norm=True, **kwargs):
        if norm:
            y_pred = softmax(y_pred, axis=-1)
        return roc_auc_score(y_true, y_pred[:, -1])