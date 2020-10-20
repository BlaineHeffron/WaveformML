import argparse
import itertools
from pathlib import Path
from os.path import join, exists, basename, dirname, realpath
import sys
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utils.SQLUtils import OptunaDB
from src.utils.TensorBoardUtils import TBHelper, run_evaluation
from src.utils.util import get_config, get_model_folder


def get_best_logfile(logfiles):
    best = 1000.
    best_file = None
    for f in logfiles:
        t = TBHelper(str(f.resolve()))
        a = t.get_best_value("epoch_val_loss")
        if a is not None:
            if a < best:
                best = a
                best_file = str(f.resolve())
    return best_file

def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)

def get_corresponding_config_ckpt(logfile):
    bn = basename(logfile)
    dn = dirname(logfile)
    dp = Path(dn)
    ckpt = dp.glob("*.ckpt")
    conf = dp.glob("*_config.json")
    myckpt = peek(ckpt)
    myconf = peek(conf)
    if myckpt is not None and myconf is not None:
        return str(myconf[0].resolve()), str(myckpt[0].resolve())
    else:
        raise RuntimeError("Couldnt find both a checkpoint file and a config file for {}".format(logfile))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d",
                        help="directory or directories to recurse in and find logs to determine lowest epoch_loss to evaluate",
                        nargs='*')
    parser.add_argument("--config", "-c",
                        help="Use a config file to find best trials.",
                        type=str)
    args = parser.parse_args()
    mydirs = []
    if (args.dir):
        mydirs = args.dir
    elif args.config:
        conf = get_config(args.config)
        model_name, model_folder = get_model_folder(conf)
        tb_folder = join(model_folder, "runs")
        study_folder = join(model_folder, "studies")
        if exists(tb_folder):
            mydirs.append(join(tb_folder, conf.run_config.exp_name))
        if exists(study_folder):
            mydirs.append(join(study_folder, conf.run_config.exp_name))

    for directory in mydirs:
        p = Path(directory)
        sqlfiles = p.glob("**/*.db")
        for s in sqlfiles:
            optdb = OptunaDB(str(s.resolve()))
            logdir = join(dirname(str(s.resolve())), "trial_{}".format(optdb.get_best_trial()))
            optdb.close()
            opt_path = Path(logdir)
            logfiles = opt_path.glob("**/*events.out.tfevents.*")
            best_file = get_best_logfile(logfiles)
            if best_file is None:
                print("no log files found for {0}".format(logdir))
            conf, ckpt = get_corresponding_config_ckpt(best_file)
            print("Evaluating best trial from {}".format(s.resolve()))
            run_evaluation(logdir, conf, ckpt)

        #print("not implemented yet")
        # cpfiles = p.glob("**/*.ckpt")


if __name__ == "__main__":
    main()
