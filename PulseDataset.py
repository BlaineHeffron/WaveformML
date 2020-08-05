import os
from HDF5Dataset import *

"""
mydl = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                  collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                  multiprocessing_context=None)

"""


# note we will need to use pinned memory since this runs on GPU

# extend torch.utils.data.IterableDataset for iterating over the dataset files (specifically to get parallelization working)

class PulseDataset2D(HDF5Dataset):
    """Pulse data in the form of ChannelData of size [N,nsamples*2]
    where N is the number of PMTs fired for the M = batch size events"""

    def __init__(self, config, n_per_dir,
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
        paths = [os.path.join(config.base_path, path) for path in config.paths]

        super().__init__(paths,
                         "*WaveformPairSim.h5", "WaveformPairs",
                         "coord", "waveform",
                         n_per_dir,
                         file_excludes=file_excludes,
                         label_name=label_name,
                         data_cache_size=data_cache_size)

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class PulseDataset3D(HDF5Dataset):
    """Pulse data in the form of ChannelData of size [N,nsamples*2]
    where N is the number of PMTs fired for the M = batch size events"""

    def __init__(self, config, n_per_dir,
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
        paths = [os.path.join(config.base_path, path) for path in config.paths]

        super().__init__(paths,
                         "*Waveform3DPairSim.h5", "Waveform3DPairs",
                         "coord", "waveform",
                         n_per_dir,
                         file_excludes=file_excludes,
                         label_name=label_name,
                         data_cache_size=data_cache_size)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
