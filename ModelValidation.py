DIMX = 14
DIMY = 11

DIM = "DIMENSION"
NIN = "N_INPUT_CHANNELS"
NOUT = "N_OUTPUT_CHANNELS"
FS = "FILTER_SIZE"  # assumes square filter
STR = "FILTER_STRIDE"
PAD = "FILTER_PADDING"
DIL = "FILTER_DILATION"
alg_map = {
    "sparseconvnet.Convolution": [DIM, NIN, NOUT, FS, STR],
    "nn.Linear": [NIN, NOUT]
}
type_map = {
    "convolution": [DIM, NIN, NOUT, FS, STR, PAD, DIL],
    "linear": [NIN, NOUT]
}


class ModelValidation(object):
    """
    class ModelValidation
    checks convolutional layers to ensure they match with dense layers
    """

    @staticmethod
    def validate(config):
        if not hasattr(config, "algorithm"):
            raise IOError("Error: config file must have an algorithm specified.")
        if config.net_config.net_type == "2DConvolution":
            current_dim = [DIMX, DIMY, config.system_config.n_samples * 2]
        else:
            raise IOError(
                "Error: model validation not currently configured for net type {0}".format(config.net_config.net_type))
        current_alg = ""
        prev_alg = ""
        for alg in config.algorithm:
            if isinstance(alg, str):
                prev_alg = current_alg
                current_alg = alg
            elif isinstance(alg, list):
                algtype = ModelValidation._get_type(current_alg)
                inputs = ModelValidation._parse_function_inputs(current_alg, alg, algtype)
                if algtype == "convolution":
                    current_dim = ModelValidation._calc_output_size(inputs, current_dim)
                elif algtype == "todense":
                    continue
                elif algtype == "linear":
                    if inputs[0] != current_dim[2]:
                        raise IOError("Error: dimension mismatch between layer {0} and {1}. Expecting the input "
                                      "dimensions to be {2}".format(prev_alg, current_alg, current_dim[2]))
                    current_dim[2] = inputs[1]

    @staticmethod
    def _parse_function_inputs(current_alg, args_list, alg_type):
        """
        returns array [DIM, NIN, NOUT, FS, STR, PAD, DIL]
        0 is placeholder for no parameter
        """
        if alg_type not in type_map.keys():
            return args_list
        match = type_map[alg_type]
        output = [0] * len(match)
        if current_alg in alg_map.keys():
            for i, m in enumerate(match):
                for j, typename in enumerate(alg_map[current_alg]):
                    if typename == m:
                        output[i] = args_list[j]
                        break
        return output

    @staticmethod
    def _calc_output_size(arg_list, current_dim):
        """
        expects array [DIM, NIN, NOUT, FS, STR, PAD, DIL]
        0 is placeholder for no parameter
        calculates the output, returns [x,y] [width, height]
        only 2d for now, only square filters
        """
        # w = (nw - fw + 2p ) / s + 1 TODO: add dilation
        # h = (nh - fh + 2p ) / s + 1
        if current_dim[2] != NIN:
            raise IOError("")
        h = (current_dim[1] - arg_list[3] + 2 * arg_list[5]) / arg_list[4] + 1
        w = (current_dim[0] - arg_list[3] + 2 * arg_list[5]) / arg_list[4] + 1
        return [w, h, NOUT]

    @staticmethod
    def _get_type(alg):
        if alg:
            lowalg = alg.lower()
            if "conv" in lowalg:
                return "convolution"
            elif "todense" in lowalg:
                return "todense"
            elif "linear" == lowalg:
                return "linear"
            else:
                return "other"
        else:
            return "none"
