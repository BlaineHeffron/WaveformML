# WaveformML
Machine learning tools for waveform analysis.

## Install
Recommended to use conda installation manager. https://docs.conda.io/projects/conda/en/latest/user-guide/install/

First, install prerequisites:

    conda install gitpython
    conda install json
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


## Usage

    python main.py <name of config.json file>

output is logged in 

    {model folder}/runs/<experiment name>/version_<n> 
    
where {model folder} defaults to 

    ./model/<model name>
    
and \<n> is the run number of the experiment starting from 0

### Selecting data

Set dataset_config.paths to the directories you 
would like to retrieve hdf5 data from. Set the 
number of events to draw from this directory with 
n_train, n_validate, and n_test. 

Metadata of events selected will be logged to 

    {model folder}/datasets/<dataset name>_config.json

Constructed datasets will be saved on disk to 
    
    ./data/<model_name>/<dataset name>.json


   
To train with a spe

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

Here are some that aren't currently working for some reason but look quite useful:

    --auto_lr_find true // starts the training session with a learning rate finder algorithm, prints results / saves to log folder
    --auto_scale_batchsize binsearch // automatically scales batch size until it finds the largest that fits in memory

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



## Creating your own modules

Create your own modules by extending the LitPSD class. See
https://pytorch-lightning.readthedocs.io/en/latest/child_modules.html for more information.









