import torch

from src.datasets.H5CompoundTypes import WaveformPairCal, PhysPulse, extension_type_map
from src.datasets.HDF5Dataset import MAX_RANGE
from src.datasets.PulseDataset import dataset_class_type_map
from src.datasets.HDF5IO import P2XTableWriter, H5Input
from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator
from src.utils.SQLiteUtils import get_gains
from src.utils.SparseUtils import swap_sparse_from_dense, swap_sparse_from_event, normalize_waveforms, \
    convert_wf_phys_SE_classifier
from src.utils.XMLUtils import XMLWriter
from src.utils.util import get_config, ModuleUtility, get_file_md5
from numpy import zeros, divide, full, float32, copy
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
        self.input_type = extension_type_map(input_path)
        self.n_buffer_rows = 1024 * 16
        self.n_rows_per_read = 2048
        self.map_location = None
        self.swap = True
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.retrieve_model()
        if "datatype" in kwargs.keys():
            if kwargs["datatype"] == "WaveformPairCal":
                self.data_type = WaveformPairCal()
            elif kwargs["datatype"] == "PhysPulse":
                self.data_type = PhysPulse()
            else:
                raise IOError("unrecognized datatype: {}, did you mean 'WaveformPairCal' or 'PhysPulse'?".format(kwargs["datatype"]))
        else:
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
        self.input.setup_table(self.input_type.name, self.input_type.type, self.input_type.event_index_name,
                               event_index_coord=self.input_type.event_index_coord)
        nrows = self.input.h5f[self.input_type.name].shape[0]
        self.create_table(self.data_type.name, (nrows,), self.data_type.type)
        n_current_buffer = 0
        self.copy_p2x_attrs(self.input, self.data_type.name, self.input_type.name)
        with torch.no_grad():
            data = self.input.next_chunk(self.n_rows_per_read)
            n_current_buffer += data.shape[0]
            if self.swap:
                self.swap_values(data)
            else:
                data = self.convert_values(data)
            self.add_rows(self.data_type.name, data)
            while data is not None:
                data = self.input.next_chunk(self.n_rows_per_read)
                if data is not None:
                    if self.swap:
                        self.swap_values(data)
                    else:
                        data = self.convert_values(data)
                    self.add_rows(self.data_type.name, data)
                    n_current_buffer += data.shape[0]
                    if n_current_buffer >= self.n_buffer_rows:
                        n_current_buffer = 0
                        self.flush(self.data_type.name)
        self.flush(self.data_type.name)
        self.input.close()
        self.close()

    def swap_values(self, data):
        raise NotImplementedError()

    def convert_values(self, data):
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
                    "model_config_hash": get_file_md5(self.config_path)}
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
        if 'waveform' in data.dtype.names:
            if self.gains is None:
                raise IOError("Must pass calgroup argument in order to normalize WaveformPairCal data before passing to model")
            vals = zeros(data["waveform"].shape, dtype=float32)
            coords = copy(data["coord"])
            normalize_waveforms(coords, data["waveform"], self.gains, vals)
            vals = torch.tensor(vals, dtype=torch.float32, device=self.model.device)
            coords = torch.tensor(coords, dtype=torch.int32, device=self.model.device)
        else:
            vals = torch.tensor(data["pulse"], dtype=torch.float32, device=self.model.device)
            coords = torch.tensor(data["coord"], dtype=torch.int32, device=self.model.device)
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


class IRNIMPredictionWriter(PredictionWriter, SingleEndedEvaluator):

    def __init__(self, path, input_path, config, checkpoint, **kwargs):
        PredictionWriter.__init__(self, path, input_path, config, checkpoint, **kwargs)
        SingleEndedEvaluator.__init__(self, None, **kwargs)
        self.phys_index_replaced = 2
        if isinstance(self.data_type, PhysPulse):
            self.swap = False
        if "output_is_sparse" in kwargs:
            self.output_is_sparse = kwargs["output_is_sparse"]
        else:
            self.output_is_sparse = True
        if "calgroup" in kwargs.keys():
            gains = get_gains(os.environ["PROSPECT_CALDB"], kwargs["calgroup"])
            if "scale_factor" in kwargs.keys():
                self.gains = divide(full((self.nx, self.ny, 2), kwargs["scale_factor"] * 690.0 / MAX_RANGE, dtype=float32), gains)
            else:
                self.gains = divide(full((self.nx, self.ny, 2), 690.0 / MAX_RANGE), gains)
        else:
            self.gains = None

    def swap_values(self, data):
        if 'waveform' in data.dtype.names:
            if self.gains is None:
                raise IOError("Must pass calgroup argument in order to normalize WaveformPairCal data before passing to model")
            vals = zeros(data["waveform"].shape, dtype=float32)
            coords = copy(data["coord"])
            normalize_waveforms(coords, data["waveform"], self.gains, vals)
            vals = torch.tensor(vals, dtype=torch.float32, device=self.model.device)
            coords = torch.tensor(coords, dtype=torch.int32, device=self.model.device)
        else:
            vals = torch.tensor(data["pulse"], dtype=torch.float32, device=self.model.device)
            coords = torch.tensor(data["coord"], dtype=torch.int32, device=self.model.device)
        #coords = torch.tensor(data["coord"], dtype=torch.int32, device=self.model.device)
        #coords[:, -1] = coords[:, -1] - coords[0, -1]  # make sure always starts with 0
        #vals = torch.tensor(data["pulse"], dtype=torch.float32, device=self.model.device)
        output = self.model([coords, vals]).detach().cpu().numpy()
        if self.output_is_sparse:
            data["phys"][:, self.phys_index_replaced:] = output
        else:
            swap_sparse_from_dense(data["phys"][:, self.phys_index_replaced:], output, data["coord"])

    def convert_values(self, data):
        if 'waveform' in data.dtype.names:
            if self.gains is None:
                raise IOError("Must pass calgroup argument in order to normalize WaveformPairCal data before passing to model")
            vals = zeros(data["waveform"].shape, dtype=float32)
            coords = copy(data["coord"])
            normalize_waveforms(coords, data["waveform"], self.gains, vals)
            vals = torch.tensor(vals, dtype=torch.float32, device=self.model.device)
            coords = torch.tensor(coords, dtype=torch.int32, device=self.model.device)
        else:
            vals = torch.tensor(data["pulse"], dtype=torch.float32, device=self.model.device)
            coords = torch.tensor(data["coord"], dtype=torch.int32, device=self.model.device)
        output = self.model([coords, vals]).detach().cpu().numpy()
        phys = zeros((coords.shape[0],), dtype=self.data_type.type)
        phys["evt"] = data["evt"]
        phys["t"] = data["t"]
        phys["PE"] = data["PE"]
        phys["seg"] = data["coord"][:, 0] + data["coord"][:, 1]*14
        phys["PID"] = data["PID"]
        convert_wf_phys_SE_classifier(data["coord"], data["E"], phys["E"], phys["rand"], data["dt"], phys["dt"], data["z"], phys["y"], data["PSD"], phys["PSD"],
                                          phys["E_SE"], phys["y_SE"], phys["Esmear_SE"], phys["PSD_SE"], data["EZ"][:, 1], output, self.blind_detl, self.blind_detr)
        return phys



    def set_xml(self):
        super().set_xml()
        if self.swap:
            self.XMLW.step_settings["phys_index_replaced"] = [2, 3, 4, 5, 6]
        else:
            self.XMLW.step_settings["classifier_score_ioni_placement"] = "E"
            self.XMLW.step_settings["classifier_score_recoil_placement"] = "rand"
            self.XMLW.step_settings["classifier_score_ncap_placement"] = "dt"
            self.XMLW.step_settings["classifier_score_ingress_placement"] = "y"
            self.XMLW.step_settings["classifier_score_muon_placement"] = "PSD"


class ZAndClassWriter(PredictionWriter, SingleEndedEvaluator):
    def __init__(self, path, input_path, zconfig, zcheckpoint, classconfig, classcheckpoint, **kwargs):
        self.scale_factor_z = 1.0
        self.scale_factor_class = 1.0
        if "datatype" in kwargs.keys() and kwargs["datatype"] != "PhysPulse":
            raise IOError("datatype must be physpulse for ZAndClassWriter")
        kwargs["datatype"] = "PhysPulse"
        PredictionWriter.__init__(self, path, input_path, zconfig, zcheckpoint, **kwargs)
        SingleEndedEvaluator.__init__(self, None, **kwargs)
        self.phys_index_replaced = 2
        self.swap = False
        if "output_is_sparse" in kwargs:
            self.output_is_sparse = kwargs["output_is_sparse"]
        else:
            self.output_is_sparse = True
        if "calgroup" in kwargs.keys():
            gains = get_gains(os.environ["PROSPECT_CALDB"], kwargs["calgroup"])
            self.gains = divide(full((self.nx, self.ny, 2), 690.0 / MAX_RANGE), gains)
        else:
            self.gains = None
        if "scale_factor" in kwargs.keys():
            raise IOError("Must specify scale factor for z or classifier (scale_factor_z or scale_factor_class)")
        self.class_config_path = classconfig
        self.class_config = get_config(classconfig)
        self.class_checkpoint_path = classcheckpoint
        modules = ModuleUtility(self.class_config.run_config.imports)
        args = {}
        if self.map_location is not None:
            args["map_location"] = self.map_location
        self.class_model = modules.retrieve_class(self.class_config.run_config.run_class).load_from_checkpoint(self.class_checkpoint_path,
                                                                                                               config=self.class_config,
                                                                                                               **args)
        self.class_model.eval()
        self.class_model.freeze()

    def convert_values(self, data):
        if self.gains is None:
            raise IOError(
                "Must pass calgroup argument in order to normalize WaveformPairCal data before passing to model")
        vals = zeros(data["waveform"].shape, dtype=float32)
        coords = copy(data["coord"])
        if hasattr(self, "scale_factor_class"):
            normalize_waveforms(coords, data["waveform"], self.gains * self.scale_factor_class, vals)
        else:
            normalize_waveforms(coords, data["waveform"], self.gains, vals)
        vals = torch.tensor(vals, dtype=torch.float32, device=self.model.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.model.device)
        class_out = self.class_model([coords, vals]).detach().cpu().numpy()
        if self.scale_factor_z / self.scale_factor_class != 1.0:
            z_out = (self.model([coords, vals * (self.scale_factor_z / self.scale_factor_class)]).detach().cpu().numpy().squeeze(1) - 0.5) * self.z_scale
        else:
            z_out = (self.model([coords, vals]).detach().cpu().numpy().squeeze(1) - 0.5) * self.z_scale
        swap_sparse_from_dense(data["EZ"][:, 1], z_out, data["coord"])
        phys = zeros((coords.shape[0],), dtype=self.data_type.type)
        phys["evt"] = data["evt"]
        phys["t"] = data["t"]
        phys["PE"] = data["PE"]
        phys["seg"] = data["coord"][:, 0] + data["coord"][:, 1] * 14
        phys["PID"] = data["PID"]
        convert_wf_phys_SE_classifier(data["coord"], data["E"], phys["E"], phys["rand"], data["dt"], phys["dt"], data["z"],
                                      phys["y"], data["PSD"], phys["PSD"],
                                      phys["E_SE"], phys["y_SE"], phys["Esmear_SE"], phys["PSD_SE"], data["EZ"][:, 1],
                                      class_out, self.blind_detl, self.blind_detr)
        return phys


    def set_xml(self):
        super().set_xml()
        self.XMLW.step_settings["ML_z_placement"] = "y_SE"
        self.XMLW.step_settings["classifier_score_ioni_placement"] = "E"
        self.XMLW.step_settings["classifier_score_recoil_placement"] = "rand"
        self.XMLW.step_settings["classifier_score_ncap_placement"] = "dt"
        self.XMLW.step_settings["classifier_score_ingress_placement"] = "y"
        self.XMLW.step_settings["classifier_score_muon_placement"] = "PSD"
        settings = {"model_z_checkpoint": self.checkpoint_path,
                    "model_z_checkpoint_hash": get_file_md5(self.checkpoint_path),
                    "model_z_config": self.config_path,
                    "model_z_config_hash": get_file_md5(self.config_path),
                    "model_classifier_checkpoint": self.class_checkpoint_path,
                    "model_classifier_checkpoint_hash": get_file_md5(self.class_checkpoint_path),
                    "model_classifier_config": self.class_config_path,
                    "model_classifier_config_hash": get_file_md5(self.class_config_path),
                    "scale_factor_z": self.scale_factor_z,
                    "scale_factor_class": self.scale_factor_class}
        for key, val in settings.items():
            self.XMLW.step_settings[key] = val
