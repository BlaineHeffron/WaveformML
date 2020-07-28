from re import findall

import h5py
from numpy import full, uint8
from pathlib import Path
import torch
from torch.utils import data


def _sort_pattern(name):
    nums = findall(r'\d+', name)
    if nums:
        return int(nums[0])
    else:
        return name


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_paths: list of paths to the folders containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        file_pattern: file pattern to match, e.g. *WaveformPairSim.h5
        data_name: string indicating the name of the dataset
        events_per_dir: number of events to be included in the dataset per given directory
        file_excludes: list of file to be excluded from dataset
        label_name: string indicating the name of the labels
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        use_pinned: boolean whether or not to return batches in pinned memory (for use by the GPU)
    """

    def __init__(self, file_paths, recursive, load_data,
                 file_pattern, data_name, events_per_dir,
                 file_excludes=None,
                 label_name=None,
                 data_cache_size=3,
                 use_pinned=False):
        super().__init__(pin_memory=use_pinned)
        self.file_paths = file_paths
        self.file_excludes = file_excludes
        self.num_dirs = len(file_paths)
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.data_name = data_name
        self.label_name = label_name
        self.n_events = [0] * self.num_dirs  # each element indexed to the file_paths list
        self.events_per_dir = events_per_dir

        # Search for all h5 files
        for i, file_path in enumerate(file_paths):
            p = Path(file_path)
            assert (p.is_dir())
            if recursive:
                files = sorted(p.glob('**/{0}'.format(file_pattern)), key=lambda e: _sort_pattern(file_pattern))
            else:
                files = sorted(p.glob(file_pattern), key=lambda e: _sort_pattern(file_pattern))
            if len(files) < 1:
                raise RuntimeError('No hdf5 datasets found')

            for h5dataset_fp in files:
                if self.n_events[i] >= self.events_per_dir:
                    break
                self._add_data_infos(str(h5dataset_fp.resolve()), i, load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data(self.data_name, index)
        coords = []
        vals = []
        for data in x:
            coords.append(data[0])
            vals.append(data[2])
        coords = torch.LongTensor(coords)
        vals = torch.FloatTensor(vals)

        # get label
        if self.label_name is None:
            y = full(x.shape[0], self.get_data_infos(self.label_name)[index]['dir_index'], dtype=uint8)
        else:
            y = self.get_data(self.label_name, index)
        y = torch.from_numpy(y)

        return [coords, vals], y

    def __len__(self):
        return len(self.get_data_infos(self.data_name))

    def _add_data_block(self, dataset, dataset_name, file_path, load_data, n_events, dir_index):
        # if data is not loaded its cache index is -1
        idx = -1
        if load_data:
            # add data to the data cache
            idx = self._add_to_cache(dataset[()], file_path)

        # type is derived from the name of the dataset; we expect the dataset
        # name to have a name such as 'data' or 'label' to identify its type
        # we also store the shape of the data in case we need it
        self.data_info.append({'file_path': file_path,
                               'type': dataset_name,
                               'shape': dataset[()].shape,
                               'cache_idx': idx,
                               'event_range': [0, n_events - 1],
                               'dir_index': dir_index})

    def _add_data_infos(self, file_path, dir_index, load_data):
        if self.file_excludes:
            if file_path in self.file_excludes:
                return
        with h5py.File(file_path, 'r') as h5_file:
            n_events = h5_file[self.data_name].attrs['nevents']  # the number of events to retrieve
            if self.events_per_dir - self.n_events[dir_index] < n_events:
                n_events = self.events_per_dir - self.n_events[dir_index]
            self.n_events[dir_index] += n_events

            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                if hasattr(group, "items"):
                    for dname, ds in group.items():
                        self._add_data_block(ds, dname, file_path, load_data, n_events, dir_index)
                else:
                    self._add_data_block(group, gname, file_path, load_data, n_events, dir_index)

    def _get_dir_index(self, file_path):
        return self.file_paths.index(file_path)

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, 'r') as h5_file:
            for gname, group in h5_file.items():
                if hasattr(group, "items"):
                    for dname, ds in group.items():
                        # add data to the data cache and retrieve
                        # the cache index
                        idx = self._add_to_cache(ds.value, file_path)

                        # find the beginning index of the hdf5 file we are looking for
                        file_idx = next(i for i, v in enumerate(self.data_info)
                                        if v['file_path'] == file_path)

                        # the data info should have the same index since we loaded it in the same way
                        self.data_info[file_idx + idx]['cache_idx'] = idx
                else:
                    idx = self._add_to_cache(group[()], file_path)
                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info)
                                    if v['file_path'] == file_path)
                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = \
                [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1}
                 if di['file_path'] == removal_keys[0]
                 else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, data_type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == data_type]
        return data_info_type

    def get_data(self, data_type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(data_type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(data_type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

    def get_file_list(self):
        return [di['file_path'] for di in self.data_info]
