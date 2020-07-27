import json
import time
import math
from PulseDataset import *
from os.path import join
from net import *
import argparse
import sparseconvnet as scn

basedir = "/home/blaine/projects/orthopositronium/sim"
config_file= "config/OPs3ns_SCNet.json"

pos = HDF5Dataset(join(basedir,"Positron"),False,True,1)
ops =  HDF5Dataset(join(basedir,"OrthoPositronium"),False,True,1)

import ipdb; ipdb.set_trace()

def path_split(a_path):
    return os.path.normpath(a_path).split(os.path.sep)

def unique_path_combine(pathlist):
    common = []
    i = 1
    for path in pathlist:
        path_array = path_split(path)
        if common:
            while common[0:i] == path_array[0:i]:
                i +=1
            common = common[0:i-1]
        else:
            common = path_array
    output_string = ""
    if len(common) > 0:
        for path in pathlist:
            path_array = path_split(path)
            i = 0
            while path_array[i] == common[i]:
                i += 1
            if output_string != "":
                output_string += "__{0}".format('_'.join(path_array[i:]))
            else:
                output_string = '_'.join(path_array[i:])
    else:
        for path in pathlist:
            path_array = path_split(path)
            if output_string != "":
                  output_string += "__{0}".format('_'.join(path_array))
            else:
                output_string = '_'.join(path_array)
    return output_string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load_cp", help="load a training check point")
    args = parser.parse_args()

    with open(config_file) as json_data_file:
        config = json.load(json_data_file)

    exp_name = config.run_config.exp_name
    model_name = config.system_config.model_name
    model_folder = join(os.path.abspath("./model"), model_name)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    file_cache_size = 1
    model = SCNet(config)
    train_set = PulseDataset2D(config.dataset_config,
                               config.dataset_config.n_train,
                               data_cache_size=file_cache_size)
    test_set = PulseDataset2D(config.dataset_config,
                              config.dataset_config.n_test,
                              train_set.get_file_list(),
                              data_cache_size=file_cache_size)

    criterion = nn.CrossEntropyLoss()
    _lr_decay = np.log(config.optimize_config.lr_begin / config.optimize_config.lr_end)
    _use_cp = args.load_cp
    _use_cuda = torch.cuda.is_available()
    if _use_cuda:
        if hasattr(config.system_config,"gpu_enabled"):
            _use_cuda = config.system_config.gpu_enabled
    dtype = 'torch.cuda.FloatTensor' if _use_cuda else 'torch.FloatTensor'
    dtypei = 'torch.cuda.ByteTensor' if _use_cuda else 'torch.ByteTensor'
    if _use_cuda:
        model.cuda()
        criterion.cuda()
    opt_class = config.optimize_config.optimizer_class.split('.')[-1]
    optimizer = getattr(importlib.import_module(config.optimize_config.optimizer_class[0:-len(opt_class)-1]),opt_class)
    optimizer = optimizer(model.parameters(),
                          lr=config.optimize_config.lr_begin,
                          **config.optimize_config.optimizer_params)
    if _use_cp and os.path.isfile('epoch.pth'):
        _epoch = torch.load('epoch.pth') + 1
        print('Restarting at epoch ' +
              str(_epoch) +
              ' from model.pth ..')
        model.load_state_dict(torch.load('model.pth'))
    else:
        _epoch = 1
    print('#parameters', sum([x.nelement() for x in model.parameters()]))

    def store(stats,batch,predictions,loss):
        ctr=0
        for nP,f,classOffset,nClasses in zip(batch['nPoints'],batch['xf'],batch['classOffset'],batch['nClasses']):
            categ,f=f.split('/')[-2:]
            if not categ in stats:
                stats[categ]={}
            if not f in stats[categ]:
                stats[categ][f]={'p': 0, 'y': 0}
            #print(predictions[ctr:ctr+nP,classOffset:classOffset+nClasses].abs().max().item())
            stats[categ][f]['p']+=predictions.detach()[ctr:ctr+nP,classOffset:classOffset+nClasses].cpu().numpy()
            stats[categ][f]['y']=batch['y'].detach()[ctr:ctr+nP].cpu().numpy()-classOffset
            ctr+=nP

    def inter(pred, gt, label):
        assert pred.size == gt.size, 'Predictions incomplete!'
        return np.sum(np.logical_and(pred.astype('int') == label, gt.astype('int') == label))

    def union(pred, gt, label):
        assert pred.size == gt.size, 'Predictions incomplete!'
        return np.sum(np.logical_or(pred.astype('int') == label, gt.astype('int') == label))


    for epoch in range(_epoch, config.optimize_config.total_epoch + 1):
        model.train()
        stats = {}
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.optimize_config.lr_begin * \
                                math.exp((1 - epoch) * _lr_decay)
        scn.forward_pass_multiplyAdd_count=0
        scn.forward_pass_hidden_states=0
        start = time.time()
        for batch in train_set:
            optimizer.zero_grad()
            if _use_cuda:
                batch[0][1] = batch[0][1].type(dtype)
                batch[1] = batch[1].type(dtypei)
            predictions = model(batch[0])
            loss = criterion.forward(predictions, batch['y'])
            store(stats,batch,predictions,loss)
            loss.backward()
            optimizer.step()
        print('train epoch',epoch,1,'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/r['nmodels_sum']/1e6, 'MegaHidden',scn.forward_pass_hidden_states/r['nmodels_sum']/1e6,'time=',time.time() - start,'s')

        if _use_cp:
            torch.save(epoch, 'epoch.pth')
            torch.save(model.state_dict(),'model.pth')

        if epoch % int(config.optimize_config.freq_display) == 0:
            model.eval()
            stats = {}
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1, 1+3):
                for batch in test_set:
                    if _use_cuda:
                        batch[0][1] = batch[0][1].type(dtype)
                        batch[1] = batch[1].type(dtypei)
                    predictions = model(batch['x'])
                    loss = criterion.forward(predictions, batch[1])
                    store(stats, batch, predictions, loss)
                print('valid epoch',epoch,rep, 'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/r['nmodels_sum']/1e6, 'MegaHidden',scn.forward_pass_hidden_states/r['nmodels_sum']/1e6,'time=',time.time() - start,'s')
