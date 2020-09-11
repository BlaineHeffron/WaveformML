# WaveformML
Machine learning tools for waveform analysis.

## Install
Recommended to use conda installation manager. https://docs.conda.io/projects/conda/en/latest/user-guide/install/

First, install prerequisites:

    conda install numpy
    conda install gitpython
    conda install yaml
    conda install hdf5

Then install pytorch version 1.6 or higher, see https://pytorch.org/get-started/locally/

On linux with GPU support, cuda version 10.2, this command is

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

With no GPU support, the command is

    conda install pytorch torchvision cpuonly -c pytorch

Then, install pytorch-lightning and optuna

    conda install -c conda-forge optuna

Pytorch lightning requires version 0.9 or higher:
    
    pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade
    
To run the sparse convolutional network code, install spconv or SparseConvNet

##### spconv: https://github.com/traveller59/spconv

install boost:
    
    sudo apt-get install libboost-all-dev
make sure cmake >= 3.13.2 is installed then install spconv:

    git clone https://github.com/traveller59/spconv --recursive
    cd spconv
    python setup.py bdist_wheel
    cd ./dist
    pip install <name of .whl file>
    
##### SparseConvNet 

follow instructions here: https://github.com/facebookresearch/SparseConvNet

##### install things with pip:
Using pip (example is for linux, see pytorch install guide for your system)

    pip install numpy
    pip install h5py
    pip install gitpython
    pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install optuna
    pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade


## Usage

    python main.py <name of config.json file>

output is logged in 

    {model folder}/runs/<experiment name>/version_<n> 
    
where {model folder} defaults to 

    ./model/<model name>
    
and `<n>` is the run number of the experiment starting from 0

### Selecting data

Set dataset_config.paths to the directories you 
would like to retrieve hdf5 data from. Set the 
number of events to draw from this directory with 
n_train, n_validate, and n_test. 

Metadata of events selected will be logged to 

    {model folder}/datasets/<dataset name>_<dataset type>_config.json
Where `<dataset name>` is constructed from the directory names or can be specified via
`dataset_config.name`. The `<dataset type>` is one of `train`, `val`, or `test`.
Constructed datasets will be saved on disk to 
    
    ./data/<model_name>/<dataset name>.json

To set the training, test, or validation data sets to use a specific metadata file, use
`dataset_config.train_config`, `.test_config`, and `.val_config` - point these to
the json metadata file.

Requirements for the config file are shown in config_requirements.json. This can be used as a template.
A full example config file is found in config/examples

## Some useful command line arguments:

    --overfit_batches 0.001 // overfits on a small percentage of the data. Useful for debugging a network
    --profiler true // profiles the program, showing time spent in each section. output written in log dir found in <model folder>/runs/<experiment name>/version_<n> where n is the (n-1)th run of the experiment
    --limit_test_batches n // if int, limits number of batches used for testing to n batches. If float < 1, uses that fraction of the test batches.
    --limit_val_batches n
    --limit_train_batches n
    --log_gpu_memory all | min_max // logs the gpu usage, set to min_max for only max and min usage logging
    --terminate_on_nan true // terminates when nan loss is returned
    --auto_lr_find true // starts the training session with a learning rate finder algorithm, prints results / saves to log folder
    --row_log_interval n // records tensorboard logs every n batches (default 1)
    --log_save_interval n // only writes out tensorboard logs every n batches


see https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags
for a complete list of arguments


## Performance tweaking and optimization

For performance tweaking on a GPU, set 
    
    --profiler=true
then slowly increase the num_workers (under dataset_config/dataloader_params) until you find optimal performance.

Once optimal data loading performance is found, then tune your learning rate. You can set
--auto_lr_find=true to find an optimal learning rate for your learning scheduler.

### Hyperparameter optimization

Once a good learning rate is chosen, set up a hyperparameter optimization config
and set 

    -oc <name of config file or path to config file>.

The search path for the optimize config is the same for model config files. Alternatively, you
can add an optuna_config section in the config file.

Results are in 

    ./studies/<experiment name>
    
Each trial's logs and model checkpoints are saved to the 

    ./studies/<experiment name>/trial_<n> 
folder.

#### Optuna Configuration

You can prune unpromising trials by adding the `-p` flag to the command.

You can choose your pruner by setting the `pruner` value in your optuna config file to 
one of the pruners listed here: https://optuna.readthedocs.io/en/stable/reference/pruners.html

You can pass parameters to the pruner with the `pruner_params` config options.

You can choose your sampler by setting the `sampler` value in your optuna config file to 
one of the samplers listed here: https://optuna.readthedocs.io/en/stable/reference/samplers.html

You can pass parameters to the pruner with the `pruner_params` config options.

The default sampler is `TPESampler` and the default pruner is `MedianPruner`

## Viewing logs:

In the console, run the command `tensorboard --logdir <log path>` 

This will serve the log data to http://localhost:6006




## Creating your own modules

Create your own modules by extending the LitPSD class. See
https://pytorch-lightning.readthedocs.io/en/latest/child_modules.html for more information.









