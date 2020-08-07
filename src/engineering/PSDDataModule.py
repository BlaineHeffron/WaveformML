import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.utils.util import DictionaryUtility, ModuleUtility


class PSDDataModule(pl.LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.ntype = len(self.config.paths)
        self.total_train = self.config.n_train * self.ntype
        self.modules = ModuleUtility(self.config.imports)
        self.dataset_class = self.modules.retrieve_class(self.config.dataset_class)
        self.device = device

    def prepare_data(self):
        # called only on 1 GPU
        if not hasattr(self, "train_dataset"):
            self.train_dataset = self.dataset_class(self.config,
                                                    self.config.n_train,
                                                    self.device,
                                                    **DictionaryUtility.to_dict(self.config.dataset_params))

        if not hasattr(self, "val_dataset"):
            if hasattr(self.config, "n_validate"):
                n_validate = self.config.n_validate
            else:
                n_validate = self.config.n_test
            self.val_dataset = self.dataset_class(self.config,
                                                  n_validate,
                                                  self.device,
                                                  file_excludes=self.train_dataset.get_file_list(),
                                                  **DictionaryUtility.to_dict(self.config.dataset_params))

        if not hasattr(self, "test_dataset"):
            self.test_dataset = self.dataset_class(self.config,
                                                   self.config.n_test,
                                                   self.device,
                                                   file_excludes=self.train_dataset.get_file_list() + self.val_dataset.get_file_list(),
                                                   **DictionaryUtility.to_dict(self.config.dataset_params))

    def setup(self):
        # called on every GPU
        pass

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            self.prepare_data()
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          **DictionaryUtility.to_dict(self.config.dataloader_params))

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.prepare_data()
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataloader_params))

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            self.prepare_data()
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataloader_params))
