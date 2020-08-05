# WaveformML
Machine learning tools for waveform analysis.

To use, first install the following dependencies.
I used the conda package manager to install these.
conda install -c pytorch pytorch=1.6
conda install -c conda-forge optuna
conda install json
conda install yaml
conda install hdf5

To run:

python main.py <name of config.json file>

Requirements for the config file are shown in config_requirements.json. This can be used as a template.
A full example config file is found in config/examples

Some useful command line arguments:

--log_gpu_memory  // logs the gpu usage
--overfit_batches 0.001 // overfits on a small percentage of the data. Useful for debugging a network
--auto_lr_find // starts the training session with a learning rate finder algorithm, prints results / saves to log folder





