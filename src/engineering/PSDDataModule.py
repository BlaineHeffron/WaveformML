import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.utils.util import DictionaryUtility, ModuleUtility
from torch.utils.data import get_worker_info
import logging


class PSDDataModule(pl.LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.config = config
        self.device = device
        self.ntype = len(self.config.dataset_config.paths)
        self.total_train = self.config.dataset_config.n_train * self.ntype
        self.modules = ModuleUtility(self.config.dataset_config.imports)
        self.dataset_class = self.modules.retrieve_class(self.config.dataset_config.dataset_class)
        self.dataset_shuffle_map = {}

    def prepare_data(self):
        # called only on 1 GPU
        pass


    def setup(self, stage=None):
        # called on every GPU
        if stage == 'fit' or stage is None:
            if not hasattr(self, "train_dataset"):
                self.train_dataset = self.dataset_class(self.config, "train",
                                                        self.config.dataset_config.n_train,
                                                        self.device,
                                                        **DictionaryUtility.to_dict(
                                                            self.config.dataset_config.dataset_params))
                self.train_excludes = self.train_dataset.get_file_list()
                self.log.info("Training dataset generated.")
            worker_info = get_worker_info()
            if hasattr(self.config.dataset_config, "data_prep"):
                if self.config.dataset_config.data_prep == "shuffle":
                    if worker_info is None:
                        self.log.info("Main process beginning to shuffle dataset.")
                    else:
                        self.log.info("Worker process {} beginning to shuffle dataset.".format(worker_info.id))
                    self.train_dataset.write_shuffled()  # might need to make this call configurable

        if stage == 'test' or stage is None:
            if not hasattr(self, "val_dataset"):
                if hasattr(self.config.dataset_config, "n_validate"):
                    n_validate = self.config.dataset_config.n_validate
                else:
                    n_validate = self.config.dataset_config.n_test
                self.val_dataset = self.dataset_class(self.config, "validate",
                                                      n_validate,
                                                      self.device,
                                                      file_excludes=self.train_excludes,
                                                      **DictionaryUtility.to_dict(
                                                          self.config.dataset_config.dataset_params))
                self.log.info("Validation dataset generated.")

            if not hasattr(self, "test_dataset"):
                self.test_dataset = self.dataset_class(self.config, "test",
                                                       self.config.dataset_config.n_test,
                                                       self.device,
                                                       file_excludes=self.train_excludes + self.val_dataset.get_file_list(),
                                                       **DictionaryUtility.to_dict(
                                                           self.config.dataset_config.dataset_params))
                self.log.info("Test dataset generated.")

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            self.setup("train")
        return DataLoader(self.train_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.setup("test")
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            self.setup("test")
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))
