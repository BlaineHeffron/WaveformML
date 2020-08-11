import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.utils.util import DictionaryUtility, ModuleUtility
from os.path import join
import logging


class PSDDataModule(pl.LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.config = config
        self.ntype = len(self.config.dataset_config.paths)
        self.total_train = self.config.dataset_config.n_train * self.ntype
        self.modules = ModuleUtility(self.config.dataset_config.imports)
        self.dataset_class = self.modules.retrieve_class(self.config.dataset_config.dataset_class)
        self.device = device
        self.dataset_shuffle_map = {}

    def prepare_data(self):
        # called only on 1 GPU
        if not hasattr(self, "train_dataset"):
            self.train_dataset = self.dataset_class(self.config, "train",
                                                    self.config.dataset_config.n_train,
                                                    self.device,
                                                    **DictionaryUtility.to_dict(self.config.dataset_config.dataset_params))
            self.log.info("Training dataset generated.")

        if not hasattr(self, "val_dataset"):
            if hasattr(self.config.dataset_config, "n_validate"):
                n_validate = self.config.dataset_config.n_validate
            else:
                n_validate = self.config.dataset_config.n_test
            self.val_dataset = self.dataset_class(self.config, "validate",
                                                  n_validate,
                                                  self.device,
                                                  file_excludes=self.train_dataset.get_file_list(),
                                                  **DictionaryUtility.to_dict(self.config.dataset_config.dataset_params))
            self.log.info("Validation dataset generated.")

        if not hasattr(self, "test_dataset"):
            self.test_dataset = self.dataset_class(self.config, "test",
                                                   self.config.dataset_config.n_test,
                                                   self.device,
                                                   file_excludes=self.train_dataset.get_file_list() + self.val_dataset.get_file_list(),
                                                   **DictionaryUtility.to_dict(self.config.dataset_config.dataset_params))
            self.log.info("Test dataset generated.")

    def setup(self):
        # called on every GPU
        if self.config.dataset_config.data_prep == "shuffle":
            self.train_dataset.write_shuffled()  # might need to make this call configurable

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            self.prepare_data()
        return DataLoader(self.train_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.prepare_data()
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            self.prepare_data()
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))
