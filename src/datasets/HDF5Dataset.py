from re import compile

from src.utils.HDF5Utils import H5FileHandler
from pathlib import Path
from os.path import getmtime, normpath, basename, join, exists
import torch
from torch.utils import data
from numpy import where, int32 as npint32, float32 as npfloat32
from os.path import dirname, abspath
from src.utils.util import read_object_from_file, save_object_to_file, json_load, replace_file_pattern
from src.datasets.H5CompoundTypes import H5CompoundType
import logging

FILENAME_SORT_REGEX = compile(r'_(\d+)')
N_CHANNELS = 14
MAX_RANGE = 2 ** N_CHANNELS - 1
MAX_RANGE_INV = 1. / MAX_RANGE


def _sort_pattern(name):
    nums = FILENAME_SORT_REGEX.findall(str(name))
    if nums:
        return int(nums[0])
    else:
        return name


def _needs_more_data(tally, n, all_files):
    for i, val in enumerate(tally):
        if val < n:
            if len(all_files[i]) > 0:
                return True
    return False


class HDF5Dataset(data.Dataset):
    """Represents an HDF5 dataset for coordinate-value type data.
    __getitem__ returns 3 tensors, one for the coordinate, feature, and label
    data in a tuple of the form
        [coordinates, features], labels

    note if the optional parameter label_name is set, the hdf5 file should contain a datset by that name
    containing the labels. If not, the program uses the directories given as the labels.
    
    Input params:
        file_paths: list of paths to the folders containing the dataset (one or multiple HDF5 files).
        file_pattern: file pattern to match, e.g. *WaveformPairSim.h5
        data_name: string indicating the name of the dataset
        coordinate_name: string indicating the name of the coordinates vector within the dataset
        feature_name: string indicating the name of the features vector within the dataset
        events_per_dir: number of events to be included in the dataset per given directory
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        file_excludes: list of file to be excluded from dataset
        label_name: string indicating the name of the labels dataset if applicable
        label_file_pattern: string indicating the file pattern used to obtain labels if they are in a separte file
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
    """

    @classmethod
    def retrieve_config(cls, config_path, device, use_half=False):
        """
        @param config_path: path to dataset configuration json file
        @param device: device upon which to map tensors
        @return: returns an object of type HDF5Dataset initialized with the dataset configuration
        """
        conf = json_load(config_path)
        cls.log = logging.getLogger(__name__)
        cls.data_cache = {}
        cls.data_cache_map = {}
        cls.device = device
        cls.file_paths = conf["file_paths"]
        cls.num_dirs = len(cls.file_paths)
        cls.info = {"file_paths": conf["file_paths"], "data_info": conf["data_info"],
                    "data_cache_size": conf["data_cache_size"], "data_name": conf["data_name"],
                    "coord_name": conf["coord_name"], "feat_name": conf["feat_name"],
                    "label_name": conf["label_name"], "events_per_dir": ["events_per_dir"],
                    "label_file_pattern": conf["label_file_pattern"], "file_pattern": conf["file_pattern"]}
        cls.group_mode = False
        cls.ordered_file_set = [fp["file_path"] for fp in cls.info["data_info"]]
        cls.half_precision = use_half
        return cls

    def __init__(self, file_paths,
                 file_pattern,
                 data_name,
                 coordinate_name,
                 feature_name,
                 events_per_dir,
                 device,
                 recursive=False,
                 load_data=False,
                 file_excludes=None,
                 label_name=None,
                 label_file_pattern=None,
                 data_cache_size=1,
                 normalize=False,
                 use_half=False):
        super().__init__()
        self.info = {}
        self.log = logging.getLogger(__name__)
        self.num_dirs = len(file_paths)
        self.normalize = normalize
        self.data_cache = {}
        self.data_cache_map = {}
        self.n_events = [0] * self.num_dirs  # each element indexed to the file_paths list
        self.half_precision = use_half
        if self.half_precision:
            self.log.debug("Using half precision")
        self.device = device
        self.file_paths = [normpath(abspath(f)) for f in file_paths]
        self.info["file_paths"] = self.file_paths
        self.info["data_info"] = []
        self.info["data_cache_size"] = data_cache_size
        self.info["data_name"] = data_name
        self.info["coord_name"] = coordinate_name
        self.info["feat_name"] = feature_name
        self.info["label_name"] = label_name
        self.info["label_file_pattern"] = label_file_pattern
        self.info["file_pattern"] = file_pattern
        self.info["events_per_dir"] = events_per_dir
        # self.log.debug("file excludes is {}".format(file_excludes))
        self.group_mode = False
        self.ordered_file_set = []
        # Search for all h5 files
        all_files = []
        for i, file_path in enumerate(file_paths):
            p = Path(file_path)
            if not p.is_dir():
                raise RuntimeError("{0} is not a valid directory.".format(str(p.resolve())))
            assert (p.is_dir())
            if recursive:
                files = sorted(p.glob('**/{0}'.format(file_pattern)), key=lambda e: _sort_pattern(e))
            else:
                files = sorted(p.glob(file_pattern), key=lambda e: _sort_pattern(e))
            if file_excludes:
                files = [f for f in files if str(f.resolve()) not in file_excludes]
            if len(files) < 1:
                raise RuntimeError('No hdf5 datasets found')
            all_files.append(files)

        if len(all_files) == 1:
            for h5dataset_fp in all_files[0]:
                dir_index = 0
                if self.n_events[dir_index] >= self.info['events_per_dir']:
                    continue
                self.ordered_file_set.append(str(h5dataset_fp.resolve()))
                self._add_data_infos(str(h5dataset_fp.resolve()), dir_index, load_data)
        else:
            # reorder according to events in each dir
            tally = [0] * len(all_files)  # event tally for each directory
            ordered_file_set = []
            while sum([len(all_files[i]) for i in range(len(all_files))]) > 0 \
                    and _needs_more_data(tally, events_per_dir, all_files):
                for i, file_set in enumerate(all_files):
                    while len(file_set) > 0 and tally[i] < events_per_dir:
                        # self.log.debug("about to add {} to dataset".format(str(file_set[0].resolve())))
                        ordered_file_set.append(file_set.pop(0))
                        tally[i] += self._get_event_num(ordered_file_set[-1])
                        if not tally[i] < max(tally):
                            break

            # print("file excludes ",file_excludes)
            # print(ordered_file_set)
            for h5dataset_fp in ordered_file_set:
                dir_index = self.file_paths.index(dirname(h5dataset_fp))
                if self.n_events[dir_index] >= self.info['events_per_dir']:
                    continue
                self.ordered_file_set.append(str(h5dataset_fp.resolve()))
                self._add_data_infos(str(h5dataset_fp.resolve()), dir_index, load_data)
        # self.log.debug("ordered file set is {}".format(self.ordered_file_set))

    def __getitem__(self, index):
        if self.group_mode:
            di = self.get_data_infos(self.info['coord_name'])[index]
        else:
            di = self.get_data_infos(self.info['data_name'])[index]
        # get data
        if self.group_mode:
            # self.log.debug("getting coord and feat in group mode for index {}".format(index))
            coords = self.get_data(self.info['coord_name'], index)
            vals = self.get_data(self.info['feat_name'], index)
        else:
            if self.info['label_name'] is not None:
                if self.info["label_file_pattern"] is not None:
                    coords, vals = self.get_data(self.info["data_name"], index)
                    coords, vals, y = self._concat_range(index, coords, vals, di)
                else:
                    coords, vals, y = self.get_data(self.info['data_name'], index)
                    coords, vals, y = self._concat_range(index, coords, vals, di, y)
            else:
                coords, vals = self.get_data(self.info['data_name'], index)
        if self.group_mode or self.info['label_name'] is None:
            coords, vals, y = self._concat_range(index, coords, vals, di)

        # coords = torch.tensor(coords, device=self.device, dtype=torch.int32)
        # vals = torch.tensor(vals, device=self.device, dtype=torch.float32) # is it slow converting to tensor here? had to do it here to fix an issue, but this may not be optimal
        # self.log.debug("now coords size is {}".format(coords.size))
        # self.log.debug("now vals size is {}".format(vals.size))
        # self.log.debug("y size is {}".format(y.size))
        # self.log.debug("shape of coords: {}".format(coords.shape))
        # self.log.debug("shape of features: {} ".format(vals.shape))
        # self.log.debug("shape of labels: {} ".format(y.shape))
        return [coords, vals], y

    def __len__(self):
        if self.group_mode:
            return len(self.get_data_infos(self.info['coord_name']))
        else:
            return len(self.get_data_infos(self.info['data_name']))

    def _concat_range(self, index, coords, vals, di, y=None):
        # this is only meant to be called in __getitem__ because it accesses file here
        valtype = torch.int16 if self.half_precision else torch.float32
        second_ind = 0
        first_ind = 0
        if di['event_range'][1] + 1 < di['n_events']:
            second_ind = where(coords[:, 2] == di['event_range'][1] + 1)[0][0]
        if di['event_range'][0] > 0:
            first_ind = where(coords[:, 2] == di['event_range'][0])[0][0]
        if second_ind > 0:
            coords = torch.tensor(coords[first_ind:second_ind, :], dtype=torch.int32, device=self.device)
            vals = torch.tensor(vals[first_ind:second_ind, :], dtype=valtype, device=self.device)
        elif first_ind > 0:
            coords = torch.tensor(coords[first_ind:, :], dtype=torch.int32, device=self.device)
            vals = torch.tensor(vals[first_ind:, :], dtype=valtype, device=self.device)
        else:
            coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
            vals = torch.tensor(vals, dtype=valtype, device=self.device)

        if y is None:
            if self.info['label_name'] is None:
                y = torch.Tensor.new_full(torch.tensor(di['event_range'][1] + 1 - di['event_range'][0], ),
                                          (di['event_range'][1] + 1 - di['event_range'][0],),
                                          di['dir_index'], dtype=torch.int64, device=self.device)
            # y = full((di['event_range'][1] - di['event_range'][0] + 1,), fill_value=di['dir_index'], dtype=int8)
            else:
                if second_ind > 0:
                    y = self.get_data(self.info['label_name'], index)[di['event_range'][0]:di['event_range'][1] + 1]
                elif first_ind > 0:
                    y = self.get_data(self.info['label_name'], index)[di['event_range'][0]:]
                else:
                    y = self.get_data(self.info['label_name'], index)
                y = torch.tensor(y, device=self.device, dtype=torch.int64)
        else:
            if second_ind > 0:
                y = torch.tensor(y[first_ind:second_ind], dtype=torch.float32, device=self.device)
            elif first_ind > 0:
                y = torch.tensor(y[first_ind:], dtype=torch.float32, device=self.device)
            else:
                y = torch.tensor(y, dtype=torch.float32, device=self.device)
            # vals = torch.tensor(vals, device=self.device, dtype=torch.float32) # is it slow converting to tensor here? had to do it here to fix an issue, but this may not be optimal
            # self.log.debug("now coords size is ", coords.size())
        #if (self.info["label_file_pattern"]):
        #    y[y > 1] = self.num_dirs #set neutron captures to the last class
        #    y[y <= 1] = di["dir_index"] #set others to the directory index

        if self.normalize and not self.half_precision:
            vals *= MAX_RANGE_INV
        return coords, vals, y

    def _add_data_block(self, dataset, dataset_name, file_path, load_data, num_events, dir_index, n_file_events,
                        modified, label_file=False):
        # if data is not loaded its cache index is -1
        idx = -1
        if load_data:
            # add data to the data cache
            idx = self._add_to_cache(dataset, file_path, label_file)

        if self.group_mode:
            if not file_path in self.data_cache_map:
                self.data_cache_map[file_path] = {dataset_name: idx}
            else:
                self.data_cache_map[file_path][dataset_name] = idx
        else:
            self.data_cache_map[file_path] = idx
        self.info['data_info'].append({'file_path': file_path,
                                       'name': dataset_name,
                                       'modified': modified,
                                       'n_events': int(n_file_events),
                                       'event_range': [0, int(num_events) - 1],
                                       'dir_index': dir_index})

    def _get_event_num(self, file_path):
        with H5FileHandler(file_path, 'r') as h5_file:
            event_num = h5_file[self.info['data_name']].attrs.get('nevents')[0]  # the number of events in the file
        return event_num
        # return h5_file[self.info['data_name']][self.info['coord_name']][-1][2] + 1

    def _add_data_infos(self, file_path, dir_index, load_data):
        with H5FileHandler(file_path, 'r') as h5_file:
            modified = getmtime(file_path)
            n_file_events = h5_file[self.info['data_name']].attrs.get('nevents')[0]  # the number of events in the file
            # n = h5_file[self.info['data_name']][self.info['coord_name']][-1][2] + 1  # the number of events in the file
            # a = len(unique(h5_file[self.info['data_name']][self.info['coord_name']][:,2]))
            # print("nevents is {0}, length of dataset is {1} for file "
            #      " {2}".format(n,a,file_path))
            n = n_file_events
            if self.info['events_per_dir'] - self.n_events[dir_index] < n:
                n = self.info['events_per_dir'] - self.n_events[dir_index]
            self.n_events[dir_index] += n

            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                if hasattr(group, "items"):
                    self.group_mode = True
                    for dname, ds in group.items():
                        self._add_data_block(ds, dname, file_path, load_data, n, dir_index, n_file_events, modified)
                else:
                    self.group_mode = False
                    self._add_data_block(group, gname, file_path, load_data, n, dir_index, n_file_events, modified)
        if self.info["label_file_pattern"]:
            fdir = dirname(file_path)
            fname = basename(file_path)
            label_file = replace_file_pattern(fname, self.info["file_pattern"], self.info["label_file_pattern"])
            label_file = join(fdir, label_file)
            if not exists(label_file):
                raise RuntimeError(
                    "No corresponding label file found for file {0}, tried {1}".format(file_path, label_file))
            with H5FileHandler(label_file, 'r') as h5_file:
                n_file_events = h5_file[self.info['data_name']].attrs.get('nevents')[
                    0]  # the number of events in the file
                n = n_file_events
                if self.info['events_per_dir'] - self.n_events[dir_index] < n:
                    n = self.info['events_per_dir'] - self.n_events[dir_index]
                modified = getmtime(file_path)
                # Walk through all groups, extracting datasets
                for gname, group in h5_file.items():
                    self._add_data_block(group, gname, file_path, load_data, n, dir_index, n_file_events, modified,
                                         label_file=True)

    def _get_dir_index(self, file_path):
        return self.file_paths.index(file_path)

    def _load_data(self, file_path, label_file=False):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with H5FileHandler(file_path, 'r', ) as h5_file:
            for gname, group in h5_file.items():
                if hasattr(group, "items"):
                    for dname, ds in group.items():
                        # add data to the data cache and retrieve
                        # the cache index
                        # self.log.debug("adding {0} to cache from file {1}".format(dname,file_path))
                        # if ds:
                        #    self.log.debug("size of dataset: {}".format(ds.size))
                        idx = self._add_to_cache(ds, file_path, label_file)

                        # find the beginning index of the hdf5 file we are looking for
                        file_idx = next(i for i, v in enumerate(self.info['data_info'])
                                        if v['file_path'] == file_path)

                        # the data info should have the same index since we loaded it in the same way
                        if not file_path in self.data_cache_map:
                            self.data_cache_map[self.info['data_info'][file_idx + idx]['file_path']] = {dname: idx}
                        else:
                            self.data_cache_map[self.info['data_info'][file_idx + idx]['file_path']][dname] = idx
                else:
                    idx = self._add_to_cache(group, file_path, label_file)
                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.info['data_info'])
                                    if v['file_path'] == file_path)
                    # the data info should have the same index since we loaded it in the same way
                    self.data_cache_map[self.info['data_info'][file_idx + idx]['file_path']] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.info['data_cache_size']:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            if self.group_mode:
                for key in self.data_cache_map[removal_keys[0]].keys():
                    self.data_cache_map[removal_keys[0]][key] = -1
            else:
                self.data_cache_map[removal_keys[0]] = -1

    def _add_to_cache(self, data, file_path, label_file=False):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            if label_file:
                if not self.info['label_name']:
                    raise ValueError("if label file pattern used, must also specify label name")
                self.data_cache[file_path] = [data[self.info['label_name']]]
            elif self.group_mode:
                self.data_cache[file_path] = [data[()]]
            else:
                if self.info['label_name'] is not None:
                    self.data_cache[file_path] = [(data[self.info['coord_name']], data[self.info['feat_name']],
                                                   data[self.info['label_name']])]
                else:
                    self.data_cache[file_path] = [(data[self.info['coord_name']], data[self.info['feat_name']])]
        else:
            if label_file:
                if not self.info['label_name']:
                    raise ValueError("if label file pattern used, must also specify label name")
                self.data_cache[file_path].append(data[self.info['label_name']])
            elif self.group_mode:
                self.data_cache[file_path].append(data[()])
            else:
                if self.info['label_name'] is not None:
                    self.data_cache[file_path].append((data[self.info['coord_name']], data[self.info['feat_name']],
                                                       data[self.info['label_name']]))
                else:
                    self.data_cache[file_path].append((data[self.info['coord_name']], data[self.info['feat_name']]))
        return len(self.data_cache[file_path]) - 1

    def get_path_info(self, file_path):
        for di in self.info['data_info']:
            if di['file_path'].strip() == file_path.strip():
                return di

    def get_data_infos(self, data_type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.info['data_info'] if di['name'] == data_type]
        return data_info_type

    def get_data(self, data_type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        if(data_type == self.info["label_name"] and self.info["label_file_pattern"] is not None):
            label_file = True
        else:
            label_file = False
        fp = self.get_data_infos(data_type)[i]['file_path']
        # self.log.debug("file path is {}".format(fp))
        if fp not in self.data_cache:
            self._load_data(fp, label_file)

        # get new cache_idx assigned by _load_data_info
        if self.group_mode:
            cache_idx = self.data_cache_map[self.get_data_infos(data_type)[i]['file_path']][data_type]
        else:
            cache_idx = self.data_cache_map[self.get_data_infos(data_type)[i]['file_path']]
        return self.data_cache[fp][cache_idx]

    def get_file_list(self):
        return [di['file_path'] for di in self.info['data_info']]

    def __str__(self):
        return str(self.info)

    def save_info_to_file(self, fpath):
        save_object_to_file(self.info, fpath)

    def read_info_from_file(self, fpath):
        self.info = read_object_from_file(fpath)

    def construct_dtype(self, feat_size, feat_type, coord_size=3, coord_type=npint32, label_size=1,
                        label_type=npfloat32):
        names = [self.info["coord_name"], self.info["feat_name"]]
        types = [coord_type, feat_type]
        lengths = [coord_size, feat_size]
        if self.info["label_name"]:
            names.append(self.info["label_name"])
            types.append(label_type)
            lengths.append(label_size)
        return H5CompoundType(types, lengths, names).type
