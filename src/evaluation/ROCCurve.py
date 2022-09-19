#metric for computing ROC Curve from multiclass data
from torchmetrics import ConfusionMatrix, Metric
from torch import Tensor, zeros, int32

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
        for i in range(len(self.conf_matrices)):
            self.conf_matrices[i].update(preds[:, self.class_index], target)




    def compute(self) -> Tensor:
        """Computes confusion matrix.

        Returns:
            If ``multilabel=False`` this will be a ``[n_classes, n_classes]`` tensor and if ``multilabel=True``
           this will be a ``[n_classes, 2, 2]`` tensor.
        """
        result = zeros(2, len(self.conf_matrices))
        for i in range(len(self.conf_matrices)):
            res = self.conf_matrices[i].compute()
            result[0, i] = res[0,0]/(res[0, 0] + res[0, 1])
            result[1, i] = res[1,0]/(res[1, 0] + res[1, 1])

        return result
