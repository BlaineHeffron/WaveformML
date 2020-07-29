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
    "sparseconvnet.SubmanifoldConvolution": [DIM, NIN, NOUT, FS],
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
        if not hasattr(config.net_config, "algorithm"):
            raise IOError("Error: config file must have an algorithm specified.")
        if config.net_config.net_type == "2DConvolution":
            current_dim = [DIMX, DIMY, config.system_config.n_samples * 2]
        else:
            raise IOError(
                "Error: model validation not currently configured for net type {0}".format(config.net_config.net_type))
        current_alg = ""
        prev_alg = ""
        for alg in config.net_config.algorithm:
            if isinstance(alg, str):
                prev_alg = current_alg
                current_alg = alg
            elif isinstance(alg, list):
                algtype = ModelValidation._get_type(current_alg)
                #print("algtype is {}".format(algtype))
                inputs = ModelValidation._parse_function_inputs(current_alg, alg, algtype)
                if algtype == "convolution":
                    current_dim = ModelValidation._calc_output_size(inputs, current_dim,
                            current_alg, prev_alg)
                    #print("new current dim is {0}".format(current_dim))
                elif algtype == "flatten":
                    newdim = 0
                    for dim in current_dim:
                        if newdim == 0:
                            newdim = dim
                        else:
                            newdim = newdim * dim
                    current_dim = [newdim]
                elif algtype == "linear":
                    if inputs[0] != current_dim[-1]:
                        raise IOError("Error: dimension mismatch between layer {0} and {1}. Expecting the input "
                                      "dimensions to be {2}, got {3}".format(prev_alg, current_alg,
                                          current_dim[-1], inputs[0]))
                    current_dim[-1] = inputs[1]

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
    def _calc_output_size(arg_list, current_dim, ca, pa):
        """
        expects array [DIM, NIN, NOUT, FS, STR, PAD, DIL]
        0 is placeholder for no parameter
        calculates the output, returns [x,y] [width, height]
        only 2d for now, only square filters
        """
        # w = (nw - fw + 2p ) / s + 1 TODO: add dilation
        # h = (nh - fh + 2p ) / s + 1
        if current_dim[2] != arg_list[1]:
            raise IOError("Error between layers {0} and {1}: \n"
                    "Input feature dimension {2} does not match "
                    "previous output feature dimension {3}.".format(pa,ca,arg_list[1],current_dim[2]))
        if arg_list[4] == 0:
            arg_list[4] = 1
        h = (current_dim[1] - arg_list[3] + 2 * arg_list[5]) / arg_list[4] + 1
        w = (current_dim[0] - arg_list[3] + 2 * arg_list[5]) / arg_list[4] + 1
        return [int(w), int(h), int(arg_list[2])]

    @staticmethod
    def _get_type(alg):
        if alg:
            lowalg = alg.lower()
            name = lowalg.split('.')[1]
            if "convolution" in name:
                return "convolution"
            elif "todense" in name:
                return "todense"
            elif "linear" == name:
                return "linear"
            elif "flatten" == name:
                return "flatten"
            else:
                return "other"
        else:
            return "none"
