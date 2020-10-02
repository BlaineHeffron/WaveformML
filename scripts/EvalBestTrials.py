import argparse
from pathlib import Path
from os.path import join, exists, basename, dirname
from src.utils.SQLUtils import OptunaDB
from src.utils.TensorBoardUtils import TBHelper, run_evaluation
from src.utils.util import get_config, get_model_folder

def get_best_logfile(logfiles):
    best = 1000.
    best_file = ''
    for f in logfiles:
        t = TBHelper(str(f.resolve()))
        a = t.get_best_value()
        if a < best:
            best = a
            best_file = str(f.resolve())
    return best_file

def get_corresponding_config_ckpt(logfile):
    bn = basename(logfile)
    dn = dirname(logfile)
    dp = Path(dn)
    ckpt = dp.glob("*.ckpt")
    conf = dp.glob("*_config.json")
    if len(ckpt) > 0 and len(conf) > 0:
        return conf[0], ckpt[0]
    else:
        raise RuntimeError("Couldnt find both a checkpoint file and a config file for {}".format(logfile))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", help="directory or directories to recurse in and find logs to determine lowest epoch_loss to evaluate",
                        nargs='*')
    parser.add_argument("--config", "-c",
                        help="Use a config file to find best trials.",
                        type=str)
    args = parser.parse_args()
    mydirs = []
    if(args.dir):
        mydirs = [args.dir]
    elif args.config:
        conf = get_config(args.config)
        model_name, model_folder = get_model_folder(conf)
        tb_folder = join(model_folder, "runs")
        study_folder = join(model_folder, "studies")
        if exists(tb_folder):
            mydirs.append(join(tb_folder, conf.run_config.exp_name))
        if exists(study_folder):
            mydirs.append(join(study_folder,conf.run_config.exp_name))

    for dir in mydirs:
        p = Path(args.mydir)
        sqlfiles = p.glob("**/*.sql")
        if len(sqlfiles) > 0:
            for s in sqlfiles:
                optdb = OptunaDB(str(s.resolve()))
                logdir = "trial_{}".format(optdb.get_best_trial())
                optdb.close()
                opt_path = Path(logdir)
                logfiles = opt_path.glob("**/*events.out.tfevents.*")
                cpfiles = p.glob("**/*.ckpt")
                if len(logfiles) == 0:
                    print("no log files found for {0}".format(logdir))

                best_file = get_best_logfile(logfiles)
                conf,ckpt =  get_corresponding_config_ckpt(best_file)
                print("Evaluating best trial from {}".format(s.resolve()))
                run_evaluation(logdir, conf, ckpt)

        else:
            print("not implemented yet")
            #cpfiles = p.glob("**/*.ckpt")
