"""

Some simple loggers.

This module refers to OpenAI spinningup.
Link: https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py#L13

"""
import os, signal, time
from os import path as osp
import json
import warnings
import joblib
import torch
import numpy as np

from utils import colorize
from serialization import convertJson


class Logger():
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='process.txt', exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/currentdatetime``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        self.output_dir = output_dir or osp.join('tmp', 'experiments', time.strftime('%m-%d-%H-%M', time.localtime()))
        if osp.exists(self.output_dir):
            print(f"Warning: Log dir {self.output_dir} already exists! Storing info there anyway.")
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        for sig in [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, self.output_file.close)
        print(colorize(f"Logging data to {self.output_file.name}", 'green', bold=True))

        self.first_row = True
        self.log_headers = []
        self.log_cur_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        print(colorize(msg, color, bold=True))
        
    def saveConfig(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convertJson(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        print(colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)

    def logTabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``logTabular`` to store values for each diagnostic,
        make sure to call ``dumpTabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, f"Trying to introduce a new key {key} that you didn't include in the first iteration"
        assert key not in self.log_cur_row, f"You already set {key} this iteration. Maybe you forgot to call dumpTabular()"
        self.log_cur_row[key] = val

    def dumpTabular(self, stdout=True):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        if stdout:
            print("-"*n_slashes)
            for key in self.log_headers:
                val = self.log_cur_row.get(key, "")
                valstr = "%8.3g"%val if hasattr(val, "__float__") else val
                print(fmt%(key, valstr))
                vals.append(val)
            print("-"*n_slashes, flush=True)

        column_width = max_key_len + 2
        if self.output_file is not None:
            if self.first_row:
                header = "".join(map(lambda s: ('%-'+'%d'%column_width+'s')%s, self.log_headers))
                self.output_file.write(header+"\n")
            valstr = ["%.4g"%val if hasattr(val, '__float__') else val for val in vals]
            valstr = "".join(map(lambda s: ('%-'+'%d'%column_width+'s')%s, valstr))
            self.output_file.write(valstr+"\n")
            self.output_file.flush()
        self.log_cur_row.clear()
        self.first_row=False

    def saveState(self, state_dict, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_pytorch_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        fpath = osp.join(self.output_dir, 'pyt_save', 'weights')
        os.makedirs(fpath, exist_ok=True)
        fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
        try:
            joblib.dump(state_dict, osp.join(fpath, fname))
        except:
            self.log('Warning: could not pickle state_dict.', color='red')
        if hasattr(self, 'pytorch_saver_elements'):
            self._pytorch_simple_save(itr)
    
    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        assert hasattr(self, 'pytorch_saver_elements'), \
            "First have to setup saving with self.setup_pytorch_saver"
        fpath = osp.join(self.output_dir, 'pyt_save', 'models')
        os.makedirs(fpath, exist_ok=True)
        fname = 'model' + ('%d'%itr if itr is not None else '') + '.pt'
        fname = osp.join(fpath, fname)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We are using a non-recommended way of saving PyTorch models,
            # by pickling whole objects (which are dependent on the exact
            # directory structure at the time of saving) as opposed to
            # just saving network weights. This works sufficiently well
            # for the purposes of Spinning Up, but you may want to do 
            # something different for your personal PyTorch project.
            # We use a catch_warnings() context to avoid the warnings about
            # not being able to save the source code.
            torch.save(self.pytorch_saver_elements, fname)

class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

    .. code-block:: python

        epoch_logger.logTabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def logTabular(self, key, val=None, with_min_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().logTabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = self.statistics(vals, with_min_max)
            super().logTabular(key if average_only else 'Avg-' + key, stats[0])
            if not(average_only):
                super().logTabular('Std-'+key, stats[1])
            if with_min_max:
                super().logTabular('Max-'+key, stats[3])
                super().logTabular('Min-'+key, stats[2])
        self.epoch_dict[key] = []

    def getStats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        stats = self.statistics(vals, True)
        return stats

    def statistics(self, x, with_min_max=False):
        """
        Get mean/std and optional min/max of scalar x.

        Args:
            x: An array containing samples of the scalar to produce statistics
                for.

            with_min_max (bool): If true, return min and max of x in 
                addition to mean and std.
        """
        x = np.asarray(x, dtype=np.float32)
        mean = np.mean(x)
        std = np.std(x)
        if with_min_max:
            return mean, std, np.min(x), np.max(x)
        return mean, std