from math import floor

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
    "sparseconvnet.FullConvolution": [DIM, NIN, NOUT, FS, STR],
    "sparseconvnet.SubmanifoldConvolution": [DIM, NIN, NOUT, FS],
    "nn.Linear": [NIN, NOUT],
    "nn.Conv1d": [NIN, NOUT, FS, STR, PAD, DIL],
    "nn.Conv2d": [NIN, NOUT, FS, STR, PAD, DIL],
    "nn.Conv3d": [NIN, NOUT, FS, STR, PAD, DIL],
    "nn.Conv4d": [NIN, NOUT, FS, STR, PAD, DIL],
    "spconv.SparseConv1d": [NIN, NOUT, FS, STR, PAD, DIL],
    "spconv.SparseConv2d": [NIN, NOUT, FS, STR, PAD, DIL],
    "spconv.SparseConv3d": [NIN, NOUT, FS, STR, PAD, DIL],
    "spconv.SparseConv4d": [NIN, NOUT, FS, STR, PAD, DIL]
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
        DIMT = config.system_config.n_samples
        if config.net_config.net_type == "2DConvolution":
            current_dim = [DIMX, DIMY, DIMT * 2]
        elif config.net_config.net_type == "3DConvolution":
            current_dim = [DIMX, DIMY, DIMT, 2]
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
                # print("algtype is {}".format(algtype))
                inputs = ModelValidation._parse_function_inputs(current_alg, alg, algtype)
                if algtype == "convolution":
                    ndim = ModelValidation._get_conv_dim(current_alg, inputs)
                    current_dim = ModelValidation._calc_output_size(inputs, current_dim,
                                                                    current_alg, prev_alg, ndim)
                    # print("new current dim is {0}".format(current_dim))
                elif algtype == "flatten":
                    newdim = 0
                    for dim in current_dim:
                        if newdim == 0:
                            newdim = dim
                        else:
                            newdim = newdim * dim
                    current_dim = [newdim]
                elif algtype == "linear":
                    if inputs[NIN] != current_dim[-1]:
                        raise IOError("Error: dimension mismatch between layer {0} and {1}. Expecting the input "
                                      "dimensions to be {2}, got {3}".format(prev_alg, current_alg,
                                                                             current_dim[-1], inputs[NIN]))
                    current_dim[-1] = inputs[NOUT]

    @staticmethod
    def _parse_function_inputs(current_alg, args_list, alg_type):
        """
        returns dict {DIM, NIN, NOUT, FS, STR, PAD, DIL}
        0 is placeholder for no parameter
        """
        if alg_type not in type_map.keys():
            return args_list
        match = type_map[alg_type]
        output = {m : 0 for m in match}
        if current_alg in alg_map.keys():
            for i, m in enumerate(match):
                for j, typename in enumerate(alg_map[current_alg]):
                    if typename == m:
                        if isinstance(args_list[j], list):
                            output[m] = args_list[j]
                        else:
                            if i > 2:
                                output[m] = [args_list[j]] * 4
                            else:
                                output[m] = args_list[j]
                        break
        if FS in match and not output[FS]:
            output[FS] = [0]*4
        if STR in match and not output[STR]:
            output[STR] = [1]*4
        if PAD in match and not output[PAD]:
            output[PAD] = [0]*4
        if DIL in match and not output[DIL]:
            output[DIL] = [0]*4
        return output

    @staticmethod
    def _calc_output_size_1d(current, arg_dict, ind):
        return (current[ind] + 2 * arg_dict[PAD][ind] - arg_dict[FS][ind] -
                (arg_dict[FS][ind] - 1) * (arg_dict[DIL][ind] - 1)) / arg_dict[STR][ind] + 1

    @staticmethod
    def _calc_output_size(arg_dict, current_dim, ca, pa, ndim):
        """
        arglist is expected to be an array [DIM, NIN, NOUT, FS, STR, PAD, DIL]
        0 is placeholder for no parameter
        calculates the output, returns [x,y] [width, height]
        ndim is the number of dimensions of the convolutional layer
        """
        # o = floor((i + 2p - k - (k-1)*(d-1))/s) + 1
        # i is the initial length along the dimension
        # o is the output length
        # k is the filter width
        # d is the dilation
        # s is the stride
        # p is the padding
        if len(current_dim) > 1:
            if len(current_dim) != ndim+1:
                if ndim == 1 and len(current_dim) == 4:
                    # special case, the 1d convolution is on the channel data
                    f = ModelValidation._calc_output_size_1d(current_dim, arg_dict, 3)
                    return [current_dim[0], current_dim[1], current_dim[2], f]
                else:
                    raise IOError("Dimensionality of the dataset is {0}, network layer is for {1} dimensional inputs.".format(len(current_dim)-1,ndim))
        if current_dim[-1] != arg_dict[NIN]:
            raise IOError("Error between layers {0} and {1}: \n"
                          "Input feature dimension {2} does not match "
                          "previous output feature dimension {3}.".format(pa, ca, arg_dict[NIN], current_dim[2]))
        if arg_dict[STR] == 0:
            arg_dict[STR] = 1

        w = ModelValidation._calc_output_size_1d(current_dim, arg_dict, 0)
        if ndim == 1:
            return [int(w), int(arg_dict[NOUT])]
        h = ModelValidation._calc_output_size_1d(current_dim, arg_dict, 1)
        if ndim == 2:
            return [int(w), int(h), int(arg_dict[NOUT])]
        z = ModelValidation._calc_output_size_1d(current_dim, arg_dict, 2)
        if ndim == 3:
            return [int(w), int(h), int(z), int(arg_dict[NOUT])]
        t = ModelValidation._calc_output_size_1d(current_dim, arg_dict, 3)
        if ndim == 4:
            return [int(w), int(h), int(z), int(t), int(arg_dict[FS])]
        else:
            raise IOError("only 4d or fewer convolutions are supported")

    @staticmethod
    def _get_type(alg):
        if alg:
            lowalg = alg.lower()
            name = lowalg.split('.')[1]
            if "conv" in name:
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

    @staticmethod
    def _get_conv_dim(alg, inputs):
        name = alg.split(".")[1].lower()
        if NIN in alg_map[alg]:
            return inputs[alg_map[alg].index(NIN)]
        else:
            if "1d" in name:
                return 1
            elif "2d" in name:
                return 2
            elif "3d" in name:
                return 3
            elif "4d" in name:
                return 4


