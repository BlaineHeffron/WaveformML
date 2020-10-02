from os.path import dirname, basename

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from src.engineering.LitCallbacks import LoggingCallback
from src.engineering.LitPSD import LitPSD, PSDDataModule
from src.utils.util import get_config, get_tb_logdir_version, ModuleUtility, set_default_trainer_args


class TBHelper:
    def __init__(self, f):
        self.f = f
        self.ea = event_accumulator.EventAccumulator(f,
                                            size_guidance={ # see below regarding this argument
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                event_accumulator.IMAGES: 4,
                                                event_accumulator.AUDIO: 4,
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.HISTOGRAMS: 1,
                                            })
        self.ea.Reload() # loads events from file

    def get_best_value(self,scalar_name):
        if scalar_name not in self.ea.Tags()['scalars']:
            raise RuntimeError("No {0} foound in file {1}".format(scalar_name,self.f))

        else:
            best = 100000.
            for row in self.ea.Scalars('epoch_val_loss'):
                if row.value < best:
                    best = row.value
            return best



def run_evaluation(log_folder,config,ckpt):
    config = get_config(config)
    vnum = get_tb_logdir_version(str(ckpt))
    logger = TensorBoardLogger(dirname(dirname(log_folder)), name=basename(dirname(log_folder)), version=vnum)
    print("Creating new log file in directory {}".format(logger.log_dir))
    modules = ModuleUtility(config.run_config.imports)
    runner = modules.retrieve_class(config.run_config.run_class).load_from_checkpoint(ckpt, config)
    trainer_args = {"logger": logger}
    trainer_args["callbacks"] = [LoggingCallback()]
    set_default_trainer_args(trainer_args, config)
    model = LitPSD.load_from_checkpoint(ckpt, config)
    #model.set_logger(logger)
    data_module = PSDDataModule(config, runner.device)

    trainer = Trainer(**trainer_args)
    trainer.test(model, datamodule=data_module)

