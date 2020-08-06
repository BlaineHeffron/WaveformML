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

output is logged in model/<model name>/runs/<experiment name>/version_<n> where n is the (n-1)th run of the experiment

Requirements for the config file are shown in config_requirements.json. This can be used as a template.
A full example config file is found in config/examples

Some useful command line arguments:

--log_gpu_memory true // logs the gpu usage
--overfit_batches 0.001 // overfits on a small percentage of the data. Useful for debugging a network
--auto_lr_find true // starts the training session with a learning rate finder algorithm, prints results / saves to log folder
--profiler true // profiles the program, showing time spent in each section. output written in log dir found in <model folder>/runs/<experiment name>/version_<n> where n is the (n-1)th run of the experiment
--auto_scale_batchsize binsearch // automatically scales batch size until it finds the largest that fits in memory


see https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags
for a complete list of arguments


For performance tweaking on a GPU, set --profile=true, then slowly increase the num_workers
(under dataset_config/dataloader_params) until you find optimal performance.

Once optimal data loading performance is found, then tune your learning rate. You can set
--auto_lr_find=true to find an optimal learning rate for your learning scheduler.

Once a good learning rate is chosen, set up a hyperparameter optimization config and set
-oc <name of config file or path to config file>.
The search path for the optimize config is the same for model config files. Alternatively, you
can add an optuna_config section in the config file.


## Creating your own modules

Create your own modules by extending the LitPSD class. See
https://pytorch-lightning.readthedocs.io/en/latest/child_modules.html for more information.







