import importlib
import time
import os
import json
import pickle
from collections.abc import Mapping, Sequence
from collections import OrderedDict
from os.path import abspath, normpath, join
from pathlib import Path
import git
import sys
import logging
import re
import collections
import numpy as np

log = logging.getLogger(__name__)


class DictionaryUtility:
    """
    Utility methods for dealing with dictionaries.
    """

    @staticmethod
    def to_object(item):
        """
        Convert a dictionary to an object (recursive).
        """

        def convert(item):
            if isinstance(item, dict):
                return type('jo', (), {k: convert(v) for k, v in item.items()})
            if isinstance(item, list):
                def yield_convert(item):
                    for index, value in enumerate(item):
                        yield convert(value)

                return list(yield_convert(item))
            else:
                return item

        return convert(item)

    def to_dict(obj):
        """
         Convert an object to a dictionary (recursive).
         """

        def convert(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            result = {}
            for key, val in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                element = []
                if isinstance(val, list):
                    for item in val:
                        element.append(convert(item))
                else:
                    element = convert(val)
                result[key] = element
            return result

        return convert(obj)


class ModuleUtility:
    """
    Utility methods for loading modules and chaining their functions
    """

    def __init__(self, modlist):
        self.modules = {}
        self.classes = {}
        self.modList = modlist
        modNames = [m.split('.')[-1] for m in modlist]
        for i, m in enumerate(modlist):
            self.modules[modNames[i]] = importlib.import_module(m)

    def retrieve_module(self, name):
        if name in self.modules:
            return self.modules[name]
        else:
            raise IOError("{0} is not in the module list.".format(name))

    def retrieve_class(self, class_string):
        if class_string in self.classes:
            return self.classes[class_string]
        if "." in class_string:
            info = class_string.split(".")
            self.classes[class_string] = getattr(self.retrieve_module(info[0]), info[1])
            return self.classes[class_string]
        else:
            for name in self.modules:
                if hasattr(self.modules[name], class_string):
                    self.classes[class_string] = getattr(self.modules[name], class_string)
                    return self.classes[class_string]
            raise IOError("{0} is not a valid class path.\n"
                          " Must be formatted <module name>.<class name>".format(class_string))

    def create_class_instances(self, classes):
        instances = []
        last_type = None
        prev_object = None
        for class_name in classes:
            if isinstance(class_name, str):
                if last_type is str:
                    instances.append(self.retrieve_class(prev_object))
                prev_object = class_name
                last_type = str
            elif isinstance(class_name, list):
                if last_type is str:
                    instances.append(self.retrieve_class(prev_object)(*class_name))
                else:
                    raise IOError("Argument list must be preceded by a string of the class path.\n"
                                  "Errored at input: ", str(class_name))
                last_type = list
                prev_object = None
            elif isinstance(class_name, dict):
                if len(class_name.keys()) > 1:
                    print("Warning: Multiple keyed dictionaries might not get entered in proper order.\n"
                          " Please use an array of dictionaries with a single key instead to specify the object. \n"
                          " Or a list alternating key (string) arguments (list) pairs.")
                for key in class_name:
                    instances.append(self.retrieve_class(class_name[key])(**classes[class_name[key]]))
                last_type = dict
                prev_object = None
        if last_type is str:
            instances.append(self.retrieve_class(prev_object))
        return instances


def path_split(a_path):
    return os.path.normpath(a_path).split(os.path.sep)


def path_create(a_path):
    if not os.path.exists(a_path):
        os.mkdir(os.path.expanduser(os.path.normpath(a_path)))


def save_path(log_folder, model_name, exp_name, get_regex=False):
    if exp_name:
        if model_name:
            if get_regex:
                return model_name + "_" + exp_name + r'_epoch=(\d*)-val_checkpoint_on=(\d*\.\d*).ckpt'
            else:
                return os.path.join(log_folder, model_name + "_" + exp_name + "_{epoch:02d}-{val_checkpoint_on:.3f}")
        else:
            raise IOError("No model name given. Set model_name property in net_config.")
    else:
        raise IOError("No experiment name given. Set exp_name property in run_config.")


def get_tb_logdir_version(log_folder):
    pattern = re.compile(r'version_(\d*)')
    for m in re.finditer(pattern, log_folder):
        if m:
            return int(m.group(1))
    return None


def retrieve_model_checkpoint(log_folder, model_name, exp_name):
    p = Path(log_folder)
    if not p.is_dir():
        raise RuntimeError("{0} is not a valid directory.".format(str(p.resolve())))
    assert (p.is_dir())
    files = p.glob('**/*.ckpt')  # recursive search in all version folders
    pattern = re.compile(save_path(log_folder, model_name, exp_name, get_regex=True))
    vals = []
    fs = []
    for f in files:
        for m in re.finditer(pattern, str(f.name)):
            if m:
                vals.append(m.group(2))
                fs.append(str(f))
    best = 1000000
    bestind = -1
    fallback = []
    for i, v in enumerate(vals):
        v = float(v)
        if v == 0:
            fallback.append(i)
        else:
            if v < best:
                best = v
                bestind = i
    if bestind == -1:
        if fallback:
            return fs[fallback[0]]
        else:
            return ""
    return fs[bestind]


def save_object_to_file(obj, path, fmt="json"):
    if fmt == "json":
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)
    else:
        with open(path, 'w') as f:
            pickle.dump(obj, f)


def read_object_from_file(path, fmt="json"):
    if fmt == "json":
        obj = json.load(path, indent=2)
    else:
        obj = pickle.load(path)
    return obj


def save_config(config, log_folder, exp_name, postfix, is_dict=False):
    with open('{0}_{1}.json'.format(os.path.join(log_folder, exp_name), postfix), 'w') as outfile:
        if is_dict:
            json.dump(config, outfile, indent=2)
        else:
            json.dump(DictionaryUtility.to_dict(config), outfile, indent=2)


def set_default_trainer_args(trainer_args, config):
    if hasattr(config.system_config, "half_precision"):
        if config.system_config.half_precision:
            log.info("Trainer is using 16 bit precision.")
            trainer_args["precision"] = 16
    if hasattr(config.system_config, "gpu_enabled"):
        if config.system_config.gpu_enabled:
            trainer_args["gpus"] = 1  # TODO add config option for multiple gpus
            if "num_nodes" in trainer_args.keys():
                if trainer_args["num_nodes"] > 1:
                    trainer_args["distributed_backend"] = 'ddp'
    trainer_args["max_epochs"] = config.optimize_config.total_epoch
    if hasattr(config.optimize_config, "validation_freq"):
        trainer_args["check_val_every_n_epoch"] = config.optimize_config.validation_freq


def unique_path_combine(pathlist):
    common = []
    i = 1
    pathlist = [abspath(normpath(p)) for p in pathlist]
    for path in pathlist:
        path_array = path_split(path)
        if common:
            while common[0:i] == path_array[0:i]:
                i += 1
            common = common[0:i - 1]
        else:
            common = path_array
    output_string = ""
    if len(common) > 0:
        for path in pathlist:
            path_array = path_split(path)
            i = 0
            while i < len(common) and path_array[i] == common[i]:
                i += 1
            if output_string != "":
                output_string += "__{0}".format('_'.join(path_array[i:]))
            else:
                output_string = '_'.join(path_array[i:])
    else:
        for path in pathlist:
            path_array = path_split(path)
            if output_string != "":
                output_string += "__{0}".format('_'.join(path_array))
            else:
                output_string = '_'.join(path_array)
    return output_string


class ValidateUtility:

    @staticmethod
    def _check_path(c, p, o, t, default=""):
        if p not in c[o]:
            if default:
                c[o][p] = default
            else:
                raise IOError("Config file does not have the property {0} specified "
                              "in the {1}. Format as a {2}.".format(p, o, t))

    @staticmethod
    def validate_config(config, validate_file):
        with open(validate_file) as json_data_file:
            validate = json.load(json_data_file)
        for config_path in validate:
            if config_path.startswith("_"):
                continue

            if config_path not in config:
                raise IOError("Config file does not have a {}.".format(config_path))
            for property_path in validate[config_path]:
                if property_path.startswith("_"):
                    continue
                if isinstance(validate[config_path][property_path], str):
                    if validate[config_path][property_path] == "":
                        ValidateUtility._check_path(config, property_path, config_path, "string")
                    else:
                        ValidateUtility._check_path(config, property_path, config_path, "string",
                                                    default=validate[config_path][property_path])

                elif isinstance(validate[config_path][property_path], int) or isinstance(
                        validate[config_path][property_path], float):
                    if validate[config_path][property_path] == 0:
                        ValidateUtility._check_path(config, property_path, config_path, "number")
                    else:
                        ValidateUtility._check_path(config, property_path, config_path, "number",
                                                    default=validate[config_path][property_path])

                elif isinstance(validate[config_path][property_path], list):
                    if not validate[config_path][property_path]:
                        ValidateUtility._check_path(config, property_path, config_path, "list")
                    elif validate[config_path][property_path] == [""]:
                        ValidateUtility._check_path(config, property_path, config_path, "list of strings")
                    elif validate[config_path][property_path] == [{}]:
                        ValidateUtility._check_path(config, property_path, config_path, "list of objects")
                    else:
                        ValidateUtility._check_path(config, property_path, config_path, "list",
                                                    default=validate[config_path][property_path])
                elif isinstance(validate[config_path][property_path], dict):
                    if validate[config_path][property_path] == {}:
                        ValidateUtility._check_path(config, property_path, config_path, "object")
                    else:
                        ValidateUtility._check_path(config, property_path, config_path, "object",
                                                    default=validate[config_path][property_path])
                else:
                    ValidateUtility._check_path(config, property_path, config_path, "object")


class OrderlyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Mapping):
            return OrderedDict(o)
        elif isinstance(o, Sequence):
            return list(o)
        return json.JSONEncoder.default(self, o)


def write_run_info(mydir):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    timestr = time.strftime("%Y-%m-%d %H:%M:%S")
    info = {"datetime": timestr,
            "args": sys.argv,
            "github_hexsha": sha}
    save_config(info, mydir, "run", "info", True)


def config_equals(path, config):
    with open(path, 'r') as f:
        data = json.load(f)
        for key in data.keys():
            if data[key] != config[int(key)]:
                return False
    return True


def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def check_config(config_file, config_dir):
    orig = config_file
    if not config_file.endswith(".json"):
        config_file = "{}.json".format(config_file)
    if not os.path.isabs(config_file):
        config_file = join(config_dir, config_file)
        if not os.path.exists(config_file):
            config_file = join(os.getcwd(), config_file)
            if not os.path.exists(config_file):
                raise IOError("Could not find config file {0}. search in"
                              " {1}".format(orig, config_file))
    return config_file


def setup_logger(args):
    log_level = logging.ERROR
    if args.verbosity:
        if args.verbosity == 1:
            log_level = logging.CRITICAL
        elif args.verbosity == 2:
            log_level = logging.ERROR
        elif args.verbosity == 3:
            log_level = logging.WARNING
        elif args.verbosity == 4:
            log_level = logging.INFO
        elif args.verbosity >= 5:
            log_level = logging.DEBUG

    # set up logging to file - see previous section for more details
    logargs = {}
    if args.logfile:
        logargs["filename"] = args.logfile
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filemode='a', **logargs)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(log_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(console)
    logger.info("Logging verbosity set to {}".format(args.verbosity))

    # create a file handler if specified in command args
    if args.logfile:
        fh = logging.FileHandler(args.logfile)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Logging to file {}".format(args.logfile))
    return logger


def get_config(c):
    with open(c) as json_data_file:
        config = json.load(json_data_file)
    return DictionaryUtility.to_object(config)


def get_model_folder(config):
    if hasattr(config.system_config, "model_name"):
        model_name = config.system_config.model_name
    else:
        model_name = unique_path_combine(config.dataset_config.paths)
    if hasattr(config.system_config, "model_base_path"):
        model_dir = config.system_config.model_base_path
    else:
        model_dir = "./model"
        setattr(config.system_config, "model_base_path", model_dir)
    if not os.path.exists(model_dir):
        path_create(model_dir)
    model_folder = join(abspath(model_dir), model_name)
    if not os.path.exists(model_folder):
        path_create(model_folder)
    return model_name, model_folder


def list_matches(mylist, match):
    return np.asarray(mylist == match).nonzero()


def extract_values(all_values, labels, criterion):
    inds = list_matches(labels, criterion)
    if inds[0].size > 0:
        return all_values[inds]
    else:
        return np.array([])


def replace_file_pattern(fname, pattern, newpattern):
    pattern1 = pattern.split("*")
    pattern2 = newpattern.split("*")
    newstr = fname
    if len(pattern1) != len(pattern2):
        raise ValueError("incompatible patterns: {0} and {1}".format(pattern1, pattern2))
    for p1, p2 in zip(pattern1, pattern2):
        if p1 == p2 == '':
            continue
        newstr = newstr.replace(p1, p2, 1)
    return newstr


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def get_bins(start, stop, n):
    width = (stop - start) / n
    return np.arange(start, stop + width/2, width)
