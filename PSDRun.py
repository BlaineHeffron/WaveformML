from os.path import join, abspath, exists, isfile
from os import mkdir
from util import *
from numpy import log
import torch
import importlib
from torch.utils.tensorboard import SummaryWriter
from math import exp
import time
import sparseconvnet as scn
import json
import util

class PSDRun:
    def __init__(self, config, load_cp=False):
        self.config = config
        if not hasattr(config, "system_config"): raise IOError("Config file must contain system_config")
        if not hasattr(config, "dataset_config"): raise IOError("Config file must contain dataset_config")
        if not hasattr(config.dataset_config, "paths"): raise IOError("Dataset config must contain paths list")
        if hasattr(config.system_config, "model_name"):
            self.model_name = config.system_config.model_name
        else:
            self.model_name = unique_path_combine(self.config.dataset_config.paths)
        if hasattr(config, "run_config"):
            if hasattr(config.run_config, "exp_name"):
                self.exp_name = config.run_config.exp_name
            else:
                counter = 1
                self.exp_name = "experiment_{0}".format(counter)
                while exists(self.save_path()):
                    counter += 1
                    self.exp_name = "experiment_{0}".format(counter)
        self.model_folder = join(abspath("./model"), self.model_name)
        if not exists(self.model_folder):
            mkdir(self.model_folder)
        # save config for record
        self.log_folder = join(self.model_folder, "runs")
        with open('{}_config.json'.format(join(self.log_folder, self.exp_name)), 'w') as outfile:
            json.dump(util.DictionaryUtility.to_dict(config), outfile, indent=2)
        #tensorboard writer for logging
        self.writer = SummaryWriter(log_dir=self.log_folder)
        self.modules = ModuleUtility(config.net_config.imports + config.dataset_config.imports)
        self.model_class = self.modules.retrieve_class(config.net_config.net_class)
        self.model = self.model_class(config)
        self.dataset_class = self.modules.retrieve_class(config.dataset_config.dataset_class)
        self.train_set = self.dataset_class(config.dataset_config,
                                            config.dataset_config.n_train,
                                            **DictionaryUtility.to_dict(config.dataset_config.dataset_params))
        self.test_set = self.dataset_class(config.dataset_config,
                                           config.dataset_config.n_test,
                                           self.train_set.get_file_list(),
                                           **DictionaryUtility.to_dict(config.dataset_config.dataset_params))
        self.total_train = config.dataset_config.n_train * len(config.dataset_config.paths)
        self.criterion_class = self.modules.retrieve_class(config.net_config.criterion_class)
        self.criterion = self.criterion_class(*config.net_config.criterion_params)
        self._lr_decay = log(config.optimize_config.lr_begin / config.optimize_config.lr_end)
        self._use_cp = load_cp
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            if hasattr(config.system_config, "gpu_enabled"):
                self._use_cuda = config.system_config.gpu_enabled
        self.dtype = 'torch.cuda.FloatTensor' if self._use_cuda else 'torch.FloatTensor'
        self.dtypei = 'torch.cuda.ByteTensor' if self._use_cuda else 'torch.ByteTensor'
        if self._use_cuda:
            self.model.cuda()
            self.criterion.cuda()

        opt_class = config.optimize_config.optimizer_class.split('.')[-1]
        optimizer = getattr(importlib.import_module(config.optimize_config.optimizer_class[0:-len(opt_class) - 1]),
                            opt_class)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=config.optimize_config.lr_begin,
                                   **DictionaryUtility.to_dict(config.optimize_config.optimizer_params))
        if self._use_cp and isfile(self.save_path(False)):
            self._epoch = torch.load(self.save_path(False)) + 1
            print('Restarting at epoch ' +
                  str(self._epoch) +
                  ' from model.pth ..')
            self.model.load_state_dict(torch.load(self.save_path()))
        else:
            self._epoch = 1
        print('#parameters', sum([x.nelement() for x in self.model.parameters()]))

    def save_path(self, ismodel=True):
        if hasattr(self, "exp_name"):
            if hasattr(self, "model_name"):
                if ismodel:
                    return join(self.model_folder, self.model_name + "_" + self.exp_name + ".model")
                else:
                    return join(self.model_folder, self.model_name + "_" + self.exp_name + ".epoch")
            else:
                raise Exception("No model name")
        else:
            raise Exception("No experiment name")

    def run(self):
        for epoch in range(self._epoch, self.config.optimize_config.total_epoch + 1):
            self.run_step(epoch)

    def run_step(self, epoch):
        self.model.train()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.optimize_config.lr_begin * \
                                exp((1 - epoch) * self._lr_decay)
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        start = time.time()
        for batch in self.train_set:
            self.optimizer.zero_grad()
            if self._use_cuda:
                batch[0][1] = batch[0][1].type(self.dtype)
                batch[1] = batch[1].type(self.dtypei)
            predictions = self.model(batch[0])
            loss = self.criterion.forward(predictions, batch[1])
            self.writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            self.optimizer.step()
        print('train epoch', epoch, 'time=', time.time() - start, 's')

        torch.save(epoch, self.save_path(False))
        torch.save(self.model.state_dict(), self.save_path())

        if epoch % int(self.config.optimize_config.freq_display) == 0:
            self.model.eval()
            # stats = {}
            start = time.time()
            losses = []
            for rep in range(1, 1 + 3):
                for batch in self.test_set:
                    if self._use_cuda:
                        batch[0][1] = batch[0][1].type(self.dtype)
                        batch[1] = batch[1].type(self.dtypei)
                    predictions = self.model(batch['x'])
                    loss = self.criterion.forward(predictions, batch[1])
                    self.writer.add_scalar("Loss/valid", loss, epoch)
                print('valid epoch', epoch, rep, 'time=', time.time() - start, 's')
