from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class SedAccScoreClip(Metric):
    def __init__(self, threshold=0.5, output_transform=lambda x: x):
        self._score = None
        self._num_items = None
        self.threshold = threshold
        super(SedAccScoreClip, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._score = 0.0
        self._num_items = 0
        super(SedAccScoreClip, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred, target = output

        ## NB: SIGMOID IS NOT APPLIED HERE

        pred = pred["clipwise_output"]
        pred = pred.max(axis=1)[0].detach().cpu().numpy()
        target = target[:,0,:].detach().cpu().numpy()
        
        for i, (p, t) in enumerate(zip(pred,target)):
            if t.sum()==0.0:
                if ((p>=self.threshold).sum())>0.0:
                    score = accuracy_score([1], [0], average='micro')
                else:
                    score = accuracy_score([1], [1], average='micro')
            else:
                score = accuracy_score([t], [p>=self.threshold], average='micro')
            self._score += score
            self._num_items +=1

    @sync_all_reduce("_score", "_num_items")
    def compute(self):
        return self._score / self._num_items