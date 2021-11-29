import torch

from src.datasets.HDF5Dataset import MAX_RANGE
from src.datasets.PulseDataset import dataset_class_type_map
from src.datasets.HDF5IO import P2XTableWriter, H5Input
from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator
from src.utils.SQLiteUtils import get_gains
from src.utils.SparseUtils import swap_sparse_from_dense, swap_sparse_from_event, normalize_waveforms
from src.utils.XMLUtils import XMLWriter
from src.utils.util import get_config, ModuleUtility, get_file_md5
from numpy import zeros, divide, full, float32
import os


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
        self.map_location = None
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.retrieve_model()
        self.set_datatype()

    def retrieve_model(self):
        modules = ModuleUtility(self.config.run_config.imports)
        args = {}
        if self.map_location is not None:
            args["map_location"] = self.map_location
        self.model = modules.retrieve_class(self.config.run_config.run_class).load_from_checkpoint(self.checkpoint_path,
                                                                                                   config=self.config,
                                                                                                   **args)
        self.model.eval()
        self.model.freeze()

    def set_datatype(self):
        modules = ModuleUtility(self.config.dataset_config.imports)
        dataset_class = modules.retrieve_class(self.config.dataset_config.dataset_class)
        self.data_type = dataset_class_type_map(dataset_class)

    def write_predictions(self):
        # TODO: get dead segment list from xml file or accept list from command line input (or config file or something)
        self.copy_chanmap(self.input)
        self.input.setup_table(self.data_type.name, self.data_type.type, self.data_type.event_index_name,
                               event_index_coord=self.data_type.event_index_coord)
        nrows = self.input.h5f[self.data_type.name].shape[0]
        self.create_table(self.data_type.name, (nrows,), self.data_type.type)
        n_current_buffer = 0
        self.copy_p2x_attrs(self.input, self.data_type.name)
        with torch.no_grad():
            data = self.input.next_chunk(self.n_rows_per_read)
            n_current_buffer += data.shape[0]
            self.swap_values(data)
            self.add_rows(self.data_type.name, data)
            while data is not None:
                data = self.input.next_chunk(self.n_rows_per_read)
                if data is not None:
                    self.swap_values(data)
                    self.add_rows(self.data_type.name, data)
                    n_current_buffer += data.shape[0]
                    if n_current_buffer >= self.n_buffer_rows:
                        n_current_buffer = 0
                        self.flush(self.data_type.name)
        self.flush()
        self.input.close()
        self.close()

    def swap_values(self, data):
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

    def write_XML(self, runtime):
        self.XMLW.input_file = self.input.path + ".xml"
        self.XMLW.output_file = self.path
        self.XMLW.step_name = str(type(self).__name__)
        self.set_xml()
        self.XMLW.write_xml(self.path + ".xml", runtime)


class ZPredictionWriter(PredictionWriter, SingleEndedEvaluator):

    def __init__(self, path, input_path, config, checkpoint, **kwargs):
        PredictionWriter.__init__(self, path, input_path, config, checkpoint, **kwargs)
        SingleEndedEvaluator.__init__(self, None, **kwargs)
        self.phy_index_replaced = 4
        if "calgroup" in kwargs.keys():
            gains = get_gains(os.environ["PROSPECT_CALDB"], kwargs["calgroup"])
            if "scale_factor" in kwargs.keys():
                self.gains = divide(full((self.nx, self.ny, 2), kwargs["scale_factor"] * 690.0 / MAX_RANGE, dtype=float32), gains)
            else:
                self.gains = divide(full((self.nx, self.ny, 2), 690.0 / MAX_RANGE), gains)
        else:
            self.gains = None

    def swap_values(self, data):
        coords = torch.tensor(data["coord"], dtype=torch.int32, device=self.model.device)
        coords[:, -1] = coords[:, -1] - coords[0, -1]  # make sure always starts with 0
        if "waveform" in data.keys():
            if self.gains is None:
                raise IOError("Must pass calgroup argument in order to normalize WaveformPairCal data before passing to model")
            vals = zeros(data["waveform"].shape, dtype=float32)
            normalize_waveforms(data["coord"], data["waveform"], self.gains, vals)
            vals = torch.tensor(vals, dtype=torch.float32, device=self.model.device)
        else:
            vals = torch.tensor(data["pulse"], dtype=torch.float32, device=self.model.device)
        output = (self.model([coords, vals]).detach().cpu().numpy().squeeze(1) - 0.5) * self.z_scale
        swap_sparse_from_dense(data["EZ"][:, 1], output, data["coord"])
        """
        if self.hascal:
            phys = torch.tensor(data["phys"], dtype=torch.float32, device=self.model.device)
            dense_phys = self.get_dense_matrix(phys, coords)
            dense_E = stack((dense_phys[:, self.E_index] * self.E_scale, dense_phys[:, self.PE0_index] * self.PE_scale,
                             dense_phys[:, self.PE1_index] * self.PE_scale), axis=1)
            cal_E_pred = zeros(dense_E[:, 0].shape)
            E_basic_prediction_dense(dense_E, output, self.blind_detl, self.blind_detr,
                                     self.calibrator.light_pos_curves,
                                     self.calibrator.light_sum_curves, cal_E_pred)
            swap_sparse_from_dense(data["EZ"][:, 0], cal_E_pred, data["coord"])
        """

    def set_xml(self):
        super().set_xml()
        self.XMLW.step_settings["EZ_index_replaced"] = [1]


class IRNPredictionWriter(PredictionWriter):

    def __init__(self, path, input_path, config, checkpoint, **kwargs):
        PredictionWriter.__init__(self, path, input_path, config, checkpoint, **kwargs)
        self.phys_index_replaced = 4

    def swap_values(self, data):
        coords = torch.tensor(data["coord"], dtype=torch.int32, device=self.model.device)
        coords[:, -1] = coords[:, -1] - coords[0, -1]  # make sure always starts with 0
        vals = torch.tensor(data["pulse"], dtype=torch.float32, device=self.model.device)
        output = self.model([coords, vals]).detach().cpu().numpy()
        swap_sparse_from_event(data["phys"][:, self.phys_index_replaced:], output, data["coord"])

    def set_xml(self):
        super().set_xml()
        self.XMLW.step_settings["phys_index_replaced"] = [4, 5, 6]


class IRNIMPredictionWriter(PredictionWriter):

    def __init__(self, path, input_path, config, checkpoint, **kwargs):
        PredictionWriter.__init__(self, path, input_path, config, checkpoint, **kwargs)
        self.phys_index_replaced = 2
        if "output_is_sparse" in kwargs:
            self.output_is_sparse = kwargs["output_is_sparse"]
        else:
            self.output_is_sparse = True

    def swap_values(self, data):
        coords = torch.tensor(data["coord"], dtype=torch.int32, device=self.model.device)
        coords[:, -1] = coords[:, -1] - coords[0, -1]  # make sure always starts with 0
        vals = torch.tensor(data["pulse"], dtype=torch.float32, device=self.model.device)
        output = self.model([coords, vals]).detach().cpu().numpy()
        if self.output_is_sparse:
            data["phys"][:, self.phys_index_replaced:] = output
        else:
            swap_sparse_from_dense(data["phys"][:, self.phys_index_replaced:], output, data["coord"])

    def set_xml(self):
        super().set_xml()
        self.XMLW.step_settings["phys_index_replaced"] = [2, 3, 4, 5, 6]
