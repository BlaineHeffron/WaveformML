# WaveformML
Machine learning tools for waveform analysis.

To use, first install the following dependencies.
I used the conda package manager to install these.
conda install -c pytorch pytorch=1.6
conda install json
conda install yaml
conda install hdf5

To run:

python main.py <name of config.json file>

Requirements for the config file are shown in config_requirements.json. This can be used as a template.
A full example config file is found in config/examples

Currently working on automating hyperparameter optimization using optuna
