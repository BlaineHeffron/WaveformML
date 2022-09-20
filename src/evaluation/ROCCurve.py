#metric for computing ROC Curve from multiclass data
from torchmetrics import ConfusionMatrix, Metric
from torch import Tensor, zeros, int32
from copy import copy

class ROCCurve(Metric):
    def __init__(self,  class_index=0, class_name=None, n_thresh=100):
        """
        @param class_index: sets the class we are looking at
        @param class_name: sets the name of the class we are looking at
        @param n_thresh: sets the number of thresholds used
        """
        super().__init__()
        self.class_index = class_index
        self.class_name = class_name
        self.conf_matrices = []
        for i in range(n_thresh):
            self.conf_matrices.append(ConfusionMatrix(num_classes=2, threshold=(i+1.0)/n_thresh))

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: probabilities from model (after softmax is applied to model output logits)
            target: true class index
        """
        class_inds = target == self.class_index
        non_class_inds = target != self.class_index
        binary_target = copy(target)
        binary_target[class_inds] = 1
        binary_target[non_class_inds] = 0
        for i in range(len(self.conf_matrices)):
            self.conf_matrices[i].update(preds[:, self.class_index], binary_target)




    def compute(self) -> Tensor:
        """Computes confusion matrix.

        Returns:
            tensor of dim 2, shape (2,N), result[0,:] is true positive rate, result[1, :] is false positive rate
        """
        result = zeros(2, len(self.conf_matrices))
        for i in range(len(self.conf_matrices)):
            res = self.conf_matrices[i].compute()
            result[0, i] = res[0,0]/(res[0, 0] + res[0, 1])
            result[1, i] = res[1,0]/(res[1, 0] + res[1, 1])

        return result
