import os, sys
import logging, yaml
import random

from datetime import datetime
import numpy as np

import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, val, num=1):
        self.total += val * num
        self.count += num

    def item(self):
        return self.total/self.count

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            if isinstance(value, dict):
                value = AttrDict(value)
            elif isinstance(value, list):
                value = AttrDict.parselist(value)
            else:
                pass

            self[key] = value

    @staticmethod
    def parselist(obj):
        l = []
        for i in obj:
            if isinstance(i, dict):
                l.append(AttrDict(i))
            elif isinstance(i, list):
                l.append(AttrDict.parselist(i))
            else:
                l.append(i)
        return l

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        elif isinstance(value, list):
            value = AttrDict.parselist(value)
        else:
            pass
        self[key] = value

    def get(self, key, default=None):
        key = key.strip().strip(".").strip()

        if key == "":
            return self
        if "." not in key:
            return getattr(self, key) if hasattr(self, key) else default
        route = key.split(".")
        obj = self
        for i, k in enumerate(route):
            if isinstance(obj, list):
                obj = obj[int(k)]
            elif isinstance(obj, AttrDict):
                obj = getattr(obj, k)
            else:
                pass
        return obj

    def set(self, key, value):
        key = key.strip().strip(".").strip()
        if key == "":
            return
        *route, key = key.split(".")
        route = ".".join(route)
        setattr(self.get(route), key, value)
    
    def state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                v.state_dict(destination, prefix+k)
            else:
                destination[(prefix+'.'+k) if prefix else k] = v
        return destination



def override_config(config, options=None):
    if options is None: return config
    for opt in options:
        assert isinstance(opt, str), f"option({opt}) should be string"
        assert ("=" in opt) and (len(opt.split("=")) == 2), f"option({opt}) should have and only have one '=' to distinguish between key and value"

        key, value = opt.split("=")
        config.set(key, eval(value))

    return config


def get_config(cfg_file, overrides=""):
    assert os.path.exists(cfg_file) and os.path.isfile(cfg_file), f"config file {cfg_file} not exists or not file!"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)
    if overrides and "=" in overrides:
        config = override_config(config, overrides.split("|"))
    return config


def colorstr(*inputs):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])  # color arguments, string
    string = str(string)
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def get_logger(name, output, level=logging.INFO, fmt="%(asctime)s [%(levelname)s @ %(name)s] %(message)s", rank=-1):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # output to file
    file_handler = logging.FileHandler(os.path.join(output, f"{'dev' if 'dev' in output else time_str()}_{rank}.log"), mode='w')
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    # output to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)

    return logger


def setup_seed(seed):
    """ set seed for the whole program for removing randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model