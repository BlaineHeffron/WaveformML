import os
from copy import copy
from torch.utils.data import get_worker_info
from src.utils.util import config_equals
import logging
from numpy import where as npwhere, concatenate, empty, array, int8

from src.datasets.HDF5Dataset import *
from src.utils.util import DictionaryUtility, save_object_to_file, read_object_from_file

"""
mydl = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                  collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                  multiprocessing_context=None)

"""


# note we will need to use pinned memory since this runs on GPU

# extend torch.utils.data.IterableDataset for iterating over the dataset files (specifically to get parallelization working)

def _has_files(fset):
    for key in fset.keys():
        if len(fset[key]):
            return True
    return False


def _needs_ids(data_info, last_id_retrieved, current_file_indices):
    for cat in data_info.keys():
        if current_file_indices[cat] >= len(data_info[cat]):
            continue
        if last_id_retrieved[cat] < data_info[cat][current_file_indices[cat]][1][1]:
            return True
    return False


class PulseDataset(HDF5Dataset):

    def __init__(self, config, dataset_type,
                 n_per_dir, device,
                 file_mask, dataset_name,
                 coord_name, feat_name,
                 file_excludes=None,
                 label_name=None,
                 data_cache_size=3,
                 batch_index=2):
        """
        Args:
            config: configuration file object
            n_per_dir: number of events to use per directory
            device: the device to initialize the tensors on
            file_mask: the filter used when selecting files
            dataset_name: the hdf5 dataset name
            coord_name: the name of the coordinates vector in the dataset
            feat_name: the name of the feature vector in the dataset
            file_excludes: list of file paths to exclude from dataset
            label_name: name of the label dataset, if not given it will be assumed from
                        directories in dataset_config.paths
            data_cache_size: number of file to hold in memory
            batch_index: index of the batch number for the coordinates vector
        """
        self.file_mask = file_mask
        self.config = config.dataset_config
        self.batch_index = batch_index
        paths = [os.path.join(self.config.base_path, path) for path in self.config.paths]
        super().__init__(paths,
                         file_mask, dataset_name,
                         coord_name, feat_name,
                         n_per_dir,
                         device,
                         file_excludes=file_excludes,
                         label_name=label_name,
                         data_cache_size=data_cache_size)

        model_dir = os.path.join(config.system_config.model_base_path, config.system_config.model_name)
        self.data_dir = os.path.join(os.path.dirname(config.system_config.model_base_path), "data")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.dataset_dir = os.path.join(model_dir, "datasets")
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        self.dataset_type = dataset_type
        if hasattr(self.config, "name"):
            self.file_path = os.path.join(self.dataset_dir, self.config.name + "_dataset.json")
        else:
            self.file_path = os.path.join(self.dataset_dir,
                                          "{0}_{1}_{2}_dataset.json".format(dataset_type, dataset_name, str(n_per_dir)))
        if not hasattr(self.config, "chunk_size"):
            self.chunk_size = 16384
        else:
            self.chunk_size = self.config.chunk_size
        if not hasattr(self.config, "shuffled_size"):
            # number of events in a file
            self.shuffled_size = 16384
        else:
            self.shuffled_size = self.config.shuffled_size
        self.log = logging.getLogger(__name__)
        if hasattr(self.config, "data_prep") and self.dataset_type == "train":
            if self.config.data_prep == "shuffle":
                self.log.info("Preparing to shuffle the dataset, alternating directory.")
                self._gen_shuffle_map()
                self.log.debug("Shuffle queue is {}".format(self.shuffle_queue))
            else:
                self.log.warning(
                    "{0} was given for dataset_config.data_prep but {1} does not know how to process".format(
                        self.config.data_prep, __name__))
        else:
            self.log.info("Writing dataset information to {}.".format(self.file_path))
            self.save_info_to_file()

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def save_info_to_file(self):
        info_dict = self.info
        info_dict["dataset_config"] = DictionaryUtility.to_dict(self.config)
        save_object_to_file(info_dict, self.file_path)

    def _gen_shuffle_map(self):
        self.shuffle_queue = []
        self.log.debug("Generating shuffle map for paths {}".format(self.config.paths))
        ordered_paths_by_dir = {i: [] for i in range(len(self.config.paths))}
        n_categories = len(self.config.paths)
        n_per_category = int(self.shuffled_size / n_categories)
        current_total = [0] * n_categories
        category_map = {os.path.normpath(os.path.join(self.config.base_path, p)): i for i, p in
                        enumerate(self.config.paths)}
        self.log.debug("category map: {}".format(category_map))
        for fp in self.ordered_file_set:
            ordered_paths_by_dir[category_map[os.path.normpath(os.path.dirname(fp))]].append(fp)
        for cat in ordered_paths_by_dir.keys():
            cur_file = 0
            for fp in ordered_paths_by_dir[cat]:
                # self.log.debug("determining where to place events from file {0}".format(fp))
                di = self.get_path_info(fp)
                n_events = di['event_range'][1] - di['event_range'][0] + 1
                if len(self.shuffle_queue) <= cur_file:
                    self.shuffle_queue.append({c: [] for c in ordered_paths_by_dir.keys()})
                if n_events <= n_per_category - current_total[cat]:
                    # self.log.debug("appending ({0}, {1}) to file {2} for category {3}".format(fp, di['event_range'],
                    #                                                                          cur_file, cat))
                    self.shuffle_queue[cur_file][cat].append([fp, copy(di['event_range']), di['modified']])
                    current_total[cat] += n_events
                else:
                    subrange = [di['event_range'][0], n_per_category - 1 - current_total[cat]]
                    while subrange[1] < di['event_range'][1]:
                        if len(self.shuffle_queue) <= cur_file:
                            self.shuffle_queue.append({c: [] for c in ordered_paths_by_dir.keys()})
                        # self.log.debug(
                        #    "appending ({0}, {1}) to file {2} for category {3}".format(fp, subrange, cur_file, cat))
                        self.shuffle_queue[cur_file][cat].append([fp, copy(subrange), di['modified']])
                        cur_file += 1
                        subrange[0] = subrange[1] + 1
                        subrange[1] = di['event_range'][1] if di['event_range'][1] - subrange[0] + 1 <= n_per_category \
                            else subrange[0] + n_per_category - 1
                        current_total[cat] = 0
                    if subrange[1] >= di['event_range'][1]:
                        subrange[1] = copy(di['event_range'][1])
                        if len(self.shuffle_queue) <= cur_file:
                            self.shuffle_queue.append({c: [] for c in ordered_paths_by_dir.keys()})
                        # self.log.debug(
                        #    "appending ({0}, {1}) to file {2} for category {3}".format(fp, subrange, cur_file, cat))
                        self.shuffle_queue[cur_file][cat].append([fp, copy(subrange), di['modified']])
                        current_total[cat] = subrange[1] - subrange[0] + 1

    def _read_chunk(self, file_info, dataset, columns):
        with h5py.File(file_info[0], 'r', ) as h5_file:
            ds = h5_file[dataset]
            coords = ds[columns[0]]
            features = ds[columns[1]]
            info = self.get_path_info(file_info[0])
            if info['n_events'] - 1 != file_info[1][1]:
                if file_info[1][0] > 0:
                    inds = npwhere((coords[:, self.batch_index] >= file_info[1][0]) & (
                                coords[:, self.batch_index] <= file_info[1][1]))
                else:
                    inds = npwhere(coords[:, self.batch_index] <= file_info[1][1])
            else:
                if file_info[1][0] > 0:
                    inds = npwhere(coords[:, self.batch_index] >= file_info[1][0])
                else:
                    inds = None
        if inds:
            return coords[inds], features[inds]
        else:
            return coords, features

    def _to_hdf(self, data, labels, fname, dataset_name, columns, event_counter):
        coords, features = data
        with h5py.File(fname, mode='a') as h5f:
            if dataset_name not in h5f.keys():
                dc = h5f.create_dataset(dataset_name + "/" + columns[0], compression="gzip", compression_opts=6,
                                        data=coords, chunks=True)
                dc.flush()
                df = h5f.create_dataset(dataset_name + "/" + columns[1], compression="gzip", compression_opts=6,
                                   data=features, chunks=True)
                df.flush()
                dl = h5f.create_dataset(dataset_name + "/" + "labels", compression="gzip", compression_opts=6,
                                        data=labels, chunks=True, dtype=int8)
                dl.flush()
                h5f[dataset_name].attrs.create("nevents", array([event_counter + 1]))
                h5f.flush()
            else:
                h5f[dataset_name][columns[0]].resize((h5f[dataset_name][columns[0]].shape[0] + coords.shape[0]), axis=0)
                h5f[dataset_name][columns[0]][-coords.shape[0]:, :] = coords
                h5f[dataset_name][columns[1]].resize((h5f[dataset_name][columns[1]].shape[0] + features.shape[0]),
                                                     axis=0)
                h5f[dataset_name][columns[1]][-features.shape[0]:, :] = features
                h5f[dataset_name].attrs.create("nevents", array([event_counter + 1]))
                h5f.flush()
        self.log.debug("File {} written".format(fname))

    def _get_index(self, data, idx, offset):
        if not data:
            return data
        inds = npwhere(data[0][:, self.batch_index] == idx + offset)
        return data[0][inds], data[1][inds]

    def _not_empty(self, chunk):
        if isinstance(chunk, int):
            return False
        return len(chunk) > 0 and chunk[0].size

    def _concat(self, d1, d2):
        if not len(d1):
            return d2
        return concatenate((d1[0], d2[0])), concatenate((d1[1], d2[1]))

    def _get_coord_feat_len(self, file_info):
        with h5py.File(file_info[0], 'r', ) as h5_file:
            coords = h5_file[self.info['data_name']][self.info['coord_name']]
            feats = h5_file[self.info['data_name']][self.info['feat_name']]
            return coords.shape[1], feats.shape[1], coords.dtype, feats.dtype

    def _get_length(self, file_info):
        with h5py.File(file_info[0], 'r', ) as h5_file:
            coords = h5_file[self.info['data_name']][self.info['coord_name']]
            info = self.get_path_info(file_info[0])
            if info['n_events'] - 1 != file_info[1][1]:
                if file_info[1][0] > 0:
                    return npwhere((coords[:, self.batch_index] >= file_info[1][0]) &
                                       coords[:, self.batch_index] <= file_info[1][1])[0].shape[0]
                else:
                    return npwhere(coords[:, self.batch_index] <= file_info[1][1])[0].shape[0]
            else:
                if file_info[1][0] > 0:
                    return npwhere(coords[:, self.batch_index] >= file_info[1][0])[0].shape[0]
                else:
                    return coords.shape[0]

    def _init_shuffled_dataset(self, data_info):
        total_rows = 0
        dtypecoord, dtypefeat, coord_len, feat_len = 0, 0, 0, 0
        for key in data_info.keys():
            for data in data_info[key]:
                total_rows += self._get_length(data)
        for key in data_info.keys():
            for data in data_info[key]:
                coord_len, feat_len, dtypecoord, dtypefeat = self._get_coord_feat_len(data)
                break
            break
        return empty((total_rows, coord_len), dtype=dtypecoord), empty((total_rows, feat_len), dtype=dtypefeat)

    def _write_shuffled(self, data_info, fname):
        self.log.debug("working on shuffling the following data into file {0}: {1}".format(fname, data_info))
        if os.path.exists(fname[0:-3] + ".json"):
            if config_equals(fname[0:-3] + ".json", data_info):
                self.log.info("Already found a valid combined file: {}, skipping.".format(fname))
                return
        labels = []
        out_df = self._init_shuffled_dataset(data_info)
        n_categories = len(data_info.keys())
        data_queue = [[]] * n_categories
        data_queue_offsets = [0] * n_categories
        last_id_grabbed = [-1] * n_categories
        current_file_indices = [0] * n_categories
        current_row_index = 0
        columns = [self.info['coord_name'], self.info['feat_name']]
        event_counter = -1
        ignore_cats = [False]*n_categories
        while _needs_ids(data_info, last_id_grabbed, current_file_indices):
            for cat in data_info.keys():
                if not data_info[cat]:
                    continue
                if not data_queue[cat] and current_file_indices[cat] < len(data_info[cat]):
                    self.log.debug(
                        "attempting to read chunk from file {}".format(data_info[cat][current_file_indices[cat]][0]))
                    self.log.debug("using dataset name {}".format(self.info['data_name']))
                    chunk = self._read_chunk(data_info[cat][current_file_indices[cat]], "/" + self.info['data_name'],
                                             columns)
                    if self._not_empty(chunk):
                        data_queue[cat] = chunk
                        ignore_cats[cat] = False
                        data_queue_offsets[cat] = data_info[cat][current_file_indices[cat]][1][0]
                    else:
                        self.log.warning("Unable to retrieve data from file {}".format(data_info[cat][current_file_indices[cat]][0]))
                        ignore_cats[cat] = True
                        data_queue[cat] = 0
                        data_queue_offsets[cat] = 0
                        current_file_indices[cat] += 1
                else:
                    if current_file_indices[cat] >= len(data_info[cat]):
                        ignore_cats[cat] = True

            all_chunks_have_data = True
            while all_chunks_have_data:
                for cat in data_info.keys():
                    tempdf = self._get_index(data_queue[cat], last_id_grabbed[cat] + 1, data_queue_offsets[cat])
                    if self._not_empty(tempdf):
                        last_id_grabbed[cat] += 1
                        event_counter += 1
                        tempdf[0][:, self.batch_index] = event_counter
                        out_df[0][current_row_index:current_row_index + tempdf[0].shape[0]] = \
                            tempdf[0]
                        out_df[1][current_row_index:current_row_index + tempdf[0].shape[0]] = \
                            tempdf[1]
                        current_row_index += tempdf[0].shape[0]
                        labels.append(cat)
                    else:
                        data_queue[cat] = 0
                        data_queue_offsets[cat] = 0
                        current_file_indices[cat] += 1
                        if not ignore_cats[cat]:
                            all_chunks_have_data = False
        self._to_hdf(out_df, labels, fname, self.info['data_name'], columns, event_counter)
        if event_counter != len(labels) - 1:
            self.log.error("Labels length is not the same as event counter")
        # save metadata regarding the events in the combined file
        save_object_to_file(data_info, fname[0:-3] + ".json")
        self.log.debug("finished shuffling data into file {}".format(fname))

    def write_shuffled(self):
        while len(self.shuffle_queue):
            shuffle_length = len(self.shuffle_queue)
            if "*" in self.file_mask:
                fname = "Combined_{0}_{1}".format(shuffle_length - 1, self.file_mask[self.file_mask.index("*") + 1:])
            else:
                fname = "Combined_{0}_{1}".format(shuffle_length - 1, self.file_mask)
            self._write_shuffled(self.shuffle_queue.pop(), os.path.join(self.data_dir, fname))
        worker_info = get_worker_info()
        self.log.debug("Worker info: {}".format(worker_info))
        self.log.info("Shuffling dataset finished. Setting the dataset to the new directory")
        super().__init__([self.data_dir],
                         self.file_mask, self.info['data_name'],
                         self.info['coord_name'], self.info['feat_name'],
                         self.info['events_per_dir'],
                         self.device,
                         label_name='labels',
                         data_cache_size=self.info['data_cache_size'])
        self.log.info("Writing shuffled dataset information to {}.".format(self.file_path))
        self.save_info_to_file()


class PulseDataset2D(PulseDataset):
    """Pulse data in the form of ChannelData of size [N,nsamples*2],
    where N is the number of PMTs fired for the M = batch size events"""

    def __init__(self, config, dataset_type, n_per_dir, device,
                 file_excludes=None,
                 label_name=None,
                 data_cache_size=3):
        """
        Args:
            config: configuration file object
            n_per_dir: number of events to use per directory
            file_excludes: list of file paths to exclude from dataset
            label_name: name of the table
            data_cache_size: number of file to hold in memory
            use_pinned: whether to use pinned memory (for GPU loading)
        """
        super().__init__(config, dataset_type,
                         n_per_dir, device,
                         "*WaveformPairSim.h5", "WaveformPairs",
                         "coord", "waveform",
                         file_excludes=file_excludes,
                         label_name=label_name,
                         data_cache_size=data_cache_size)

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class PulseDataset3D(PulseDataset):
    """Pulse data in the form of ChannelData of size [N,nsamples*2]
    where N is the number of PMTs fired for the M = batch size events"""

    def __init__(self, config, dataset_type, n_per_dir, device,
                 file_excludes=None,
                 label_name=None,
                 data_cache_size=3):
        """
        Args:
            config: configuration file object
            n_per_dir: number of events to use per directory
            file_excludes: list of file paths to exclude from dataset
            label_name: name of the table
            data_cache_size: number of file to hold in memory
            use_pinned: whether to use pinned memory (for GPU loading)
        """

        super().__init__(config, n_per_dir, device, dataset_type,
                         "*Waveform3DPairSim.h5", "Waveform3DPairs",
                         "coord", "waveform",
                         batch_index=3,
                         file_excludes=file_excludes,
                         label_name=label_name,
                         data_cache_size=data_cache_size)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
