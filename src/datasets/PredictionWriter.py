import torch

from src.datasets.PulseDataset import dataset_class_type_map
from src.datasets.HDF5IO import P2XTableWriter, H5Input
from src.utils.SparseUtils import swap_sparse_from_dense
from src.utils.XMLUtils import XMLWriter
from src.utils.util import get_config, ModuleUtility, get_file_md5
from os.path import exists


class PredictionWriter(P2XTableWriter):
    """
    base class for writing predictions out
    subclass this and implement swap_values
    """

    def __init__(self, path, input_path, config, checkpoint, **kwargs):
        """
        @param path: path to output .h5 file
        @param input_path: path to input file for conversion
        @param config: path to config file used for model
        @param checkpoint: path to model checkpoint
        """
        super(PredictionWriter, self).__init__(path)
        self.XMLW = XMLWriter()
        self.checkpoint_path = checkpoint
        self.config_path = config
        self.config = get_config(config)
        self.model = None
        self.data_type = None
        self.input = H5Input(input_path)
        self.n_buffer_rows = 1024 * 16
        self.n_rows_per_read = 2048
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.retrieve_model()
        self.set_datatype()

    def retrieve_model(self):
        modules = ModuleUtility(self.config.run_config.imports)
        self.model = modules.retrieve_class(self.config.run_config.run_class).load_from_checkpoint(self.checkpoint_path,
                                                                                                   config=self.config)

    def set_datatype(self):
        modules = ModuleUtility(self.config.dataset_config.imports)
        dataset_class = modules.retrieve_class(self.config.dataset_config.dataset_class)
        self.data_type = dataset_class_type_map(dataset_class)

    def write_predictions(self):
        self.copy_chanmap(self.input)
        self.input.setup_table(self.data_type.name, self.data_type.type, self.data_type.event_index_name,
                               event_index_coord=self.data_type.event_index_coord)
        nrows = self.input.h5f[self.data_type.name].shape[0]
        self.create_table(self.data_type.name, (nrows,), self.data_type.type)
        n_current_buffer = 0
        self.copy_p2x_attrs(self.input, self.data_type.name)
        data = self.input.next_chunk(self.n_rows_per_read)
        n_current_buffer += self.n_rows_per_read
        self.swap_values(data, self.model)
        self.add_rows(self.data_type.name, data)
        while data is not None:
            data = self.input.next_chunk(self.n_rows_per_read)
            if data is not None:
                self.swap_values(data, self.model)
                self.add_rows(self.data_type.name, data)
                n_current_buffer += self.n_rows_per_read
                if n_current_buffer >= self.n_buffer_rows:
                    self.flush(self.data_type.name)
        self.flush()
        self.input.close()
        self.close()

    def swap_values(self, data, model):
        raise NotImplementedError()

    def set_xml(self):
        """
        set XMLW.step_settings
        @return: none
        SUBCLASS override this
        """
        settings = {"model_checkpoint": self.checkpoint_path,
                    "model_checkpoint_hash": get_file_md5(self.checkpoint_path),
                    "model_config": self.config_path,
                    "model_config_hash": get_file_md5(self.checkpoint_path)}
        for key, val in settings.items():
            self.XMLW.step_settings[key] = val

    def write_XML(self):
        self.XMLW.input_file = self.input.path + ".xml"
        self.XMLW.output_file = self.path
        self.XMLW.step_name = str(type(self).__name__)
        self.set_xml()
        self.XMLW.write_xml(self.path + ".xml")


class ZPredictionWriter(PredictionWriter):

    def __init__(self, path, input_path, config, checkpoint, **kwargs):
        super().__init__(path, input_path, config, checkpoint, **kwargs)
        self.phy_index_replaced = 4

    def swap_values(self, data, model):
        coords = torch.tensor(data["coord"], dtype=torch.int32, device=model.device)
        vals = torch.tensor(data["pulse"], dtype=torch.float32, device=model.device)
        output = model.model([coords, vals]).detach().cpu().numpy().squeeze(1)
        swap_sparse_from_dense(data["phys"][:, model.evaluator.z_index], output, data["coord"])
        self.phy_index_replaced = model.evaluator.z_index

    def set_xml(self):
        super().set_xml()
        self.XMLW.step_settings["phys_index_replaced"] = str(self.phy_index_replaced)
