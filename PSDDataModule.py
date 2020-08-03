import pytorch_lightning as pl
from util import DictionaryUtility

class PSDDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ntype = len(self.config.paths)
        self.total_train = self.config.n_train * self.ntype
        self.dataset_class = self.modules.retrieve_class(self.config.dataset_class)

    def prepare_data(self):
        # called only on 1 GPU
        if not hasattr(self,"train_dataset"):
            self.train_dataset = self.dataset_class(self.config,
                                                    self.config.n_train,
                                                    **DictionaryUtility.to_dict(self.config.dataset_params))
        elif not hasattr(self,"test_dataset"):
            self.test_dataset = self.dataset_class(self.config,
                                                   self.config.n_test,
                                                   self.train_set.get_file_list(),
                                                   **DictionaryUtility.to_dict(self.config.dataset_params))

        elif not hasattr(self,"val_dataset"):
            if hasattr(self.config,"n_validate"):
                self.n_validate = self.config.n_validate
            else:
                self.n_validate = self.config.n_test
            self.val_dataset = self.dataset_class(self.config,
                                                  self.config.n_validate,
                                                  self.train_set.get_file_list() + self.test_dataset.get_file_list(),
                                                  **DictionaryUtility.to_dict(self.config.dataset_params))

    def setup(self):
        # called on every GPU
        return

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset