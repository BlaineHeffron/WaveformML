import os
from copy import copy
from torch.utils.data import get_worker_info
import logging
from numpy import int8, array
from pandas import DataFrame, concat as pdconcat, read_hdf

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
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(config.system_config.model_base_path)), "data")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.dataset_dir = os.path.join(model_dir, "datasets")
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        self.dataset_type = dataset_type
        if hasattr(self.config, "name"):
            self.file_path = os.path.join(model_dir, self.config.name + "_dataset.json")
        else:
            self.file_path = os.path.join(model_dir,
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
        if hasattr(self.config, "data_prep"):
            if self.config.data_prep == "shuffle":
                self.log.info("Preparing to shuffle the dataset, alternating directory.")
                self._gen_shuffle_map()
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
        ordered_paths_by_dir = {os.path.normpath(p): [] for p in self.config.paths}
        n_categories = len(self.config.paths)
        n_per_category = int(self.shuffled_size / n_categories)
        current_total = [0] * n_categories
        current_file_info = {}
        next_file_info = {}
        next_total = [0] * n_categories
        category_map = {os.path.normpath(os.path.dirname(p)): i for i, p in enumerate(self.config.paths)}
        for fp in self.ordered_file_set:
            ordered_paths_by_dir[category_map[os.path.normpath(os.path.dirname(fp))]].append(fp)
        while _has_files(ordered_paths_by_dir):
            for category in ordered_paths_by_dir.keys():
                while current_total[category] < n_per_category and len(ordered_paths_by_dir[category]):
                    fp = ordered_paths_by_dir[category].pop(0)
                    di = self.get_path_info(fp)
                    event_range = copy(di['event_range'])
                    if event_range[1] - event_range[0] + 1 > (n_per_category - current_total[category]):
                        if not next_file_info[category]:
                            next_file_info[category] = []
                        event_range[1] = n_per_category - current_total - 1 + event_range[0]
                        next_file_info[category].append((fp, [event_range[1] + 1, di['event_range'][1]]))
                        next_total[category] += di['event_range'][1] - event_range[1]
                    current_total[category] += event_range[1] - event_range[0] + 1
                    if not current_file_info[category]:
                        current_file_info[category] = []
                    current_file_info[category].append((fp, event_range))
            self.shuffle_queue.append(copy(current_file_info))
            current_file_info = copy(next_file_info)
            next_file_info = {}
            current_total = copy(next_total)
            next_total = [0] * len(self.config.paths)
        if sum(next_total) > 0:
            self.shuffle_queue.append(next_file_info)

    def _get_where(self, file_info):
        info = self.get_path_info(file_info[0])
        where = None
        if info['n_events'] - 1 != file_info[1][1]:
            where = '{0}[{1}] <= {2}'.format(self.info['coord_name'], str(self.batch_index), str(file_info[1][1]))
        if file_info[1][0] > 0:
            if where:
                where += ' & {0}[{1}] >= {2}'.format(self.info['coord_name'],
                                                     str(self.batch_index), str(file_info[1][0]))
            else:
                where = '{0}[{1}] >= {2}'.format(self.info['coord_name'],
                                                 str(self.batch_index), str(file_info[1][0]))
        return where

    def _write_shuffled(self, data_info, fname):
        self.log.debug("working on shuffling the following data into file {0}: {1}".format(fname, data_info))
        out_df = DataFrame()
        labels = []
        n_categories = len(data_info.keys())
        data_queue = [DataFrame()] * n_categories
        last_id_grabbed = [-1] * n_categories
        current_file_indices = [0]*n_categories
        chunksize = int(self.chunk_size/n_categories)
        current_file_chunk = [0]*n_categories
        prepend_last_data = [False]*n_categories
        last_data = [0]*n_categories
        event_counter = -1
        while _needs_ids(data_info, last_id_grabbed, current_file_indices):
            for cat in data_info.keys():
                if data_queue[cat]:
                    continue
                chunk = read_hdf(data_info[cat][current_file_indices[cat]][0], self.info['data_name'],
                                  chunksize=chunksize,
                                  where = self._get_where(data_info[cat][current_file_indices[cat]]),
                                 start = current_file_chunk[cat]*chunksize)
                if chunk.size:
                    data_queue[cat] = chunk
                    current_file_chunk[cat] += 1
                else:
                    current_file_indices[cat] += 1
                    current_file_chunk[cat] = 0
                    chunk = read_hdf(data_info[cat][current_file_indices[cat]][0], self.info['data_name'],
                                     chunksize=chunksize,
                                     where=self._get_where(data_info[cat][current_file_indices[cat]]),
                                     start=current_file_chunk[cat] * chunksize)
                    if not chunk.size:
                        self.log.warning("no chunk for category {0} found")
                        data_queue[cat] = None
                    else:
                        data_queue[cat] = chunk
                        current_file_chunk[cat] += 1

            all_chunks_have_data = True
            while all_chunks_have_data:
                for cat in data_info.keys():
                    if prepend_last_data[cat]:
                        tempdf = data_queue[cat].query("{0}[{1}] == {2}".format(
                            self.info['coord_name'],
                            str(self.batch_index), last_id_grabbed[cat] ))
                    else:
                        tempdf = data_queue[cat].query("{0}[{1}] == {2}".format(
                            self.info['coord_name'],
                            str(self.batch_index), last_id_grabbed[cat] + 1))
                    if tempdf.size:
                        if prepend_last_data[cat]:
                            tempdf[:, self.info['coord_name'], self.batch_index] = event_counter
                            tempdf = pdconcat([last_data[cat], tempdf], ignore_index=True)
                            last_data[cat] = 0
                            prepend_last_data[cat] = False
                        else:
                            last_id_grabbed[cat] += 1
                            event_counter += 1
                            tempdf[:, self.info['coord_name'], self.batch_index] = event_counter
                        if last_data[cat]:
                            out_df = pdconcat([out_df, last_data[cat]], ignore_index=True)
                            labels.append(cat)
                        last_data[cat] = tempdf
                    else:
                        data_queue[cat] = 0
                        prepend_last_data[cat] = True
                        all_chunks_have_data = False
            for cat in data_info.keys():
                if last_data[cat]:
                    out_df = pdconcat([out_df, last_data[cat]], ignore_index=True)
                    labels.append(cat)
            out_df.to_hdf(fname, self.info['data_name'], complevel=9, mode='a', append=True)
            out_df = out_df.iloc[0:0]  # clear the dataframe
        if event_counter != len(labels):
            self.log.error("Labels length is not the same as event counter")
        h5f = h5py.File(fname, 'a')
        ds = h5f.create_dataset('labels', data=labels, compression="zlib", compression_opts=9, chunk=True, dtype=int8)
        ds.attrs["nevents"] = event_counter
        h5f.close()
        self.log.debug("finished shuffling data into file {}".format(fname))

    def write_shuffled(self):
        while len(self.shuffle_queue):
            l = len(self.shuffle_queue)
            if "*" in self.file_mask:
                fname = "Combined_{0}_{1}".format(l - 1, self.file_mask[self.file_mask.index("*") + 1:])
            else:
                fname = "Combined_{0}_{1}".format(l - 1, self.file_mask)
            self._write_shuffled(self.shuffle_queue.pop(), os.path.join(self.data_dir, fname))
        if not get_worker_info():
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
                         file_excludes=file_excludes,
                         label_name=label_name,
                         data_cache_size=data_cache_size)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
