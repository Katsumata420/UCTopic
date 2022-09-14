import gc
import json
import string
import orjson
import torch
import pickle
import shutil
import time
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from termcolor import colored

class IO:
    @staticmethod
    def is_valid_file(filepath):
        filepath = Path(filepath)
        return filepath.exists() and filepath.stat().st_size > 0

    def load(path):
        raise NotImplementedError

    def dump(data, path):
        raise NotImplementedError

class JsonLine(IO):
    @staticmethod
    def load(path, use_tqdm=False, datasize=-1):
        print(f"load {path}...")

        with open(path) as rf:
            if use_tqdm:
                lines = tqdm(rf, ncols=100, desc='Load JsonLine')

            if datasize == -1:
                return [json.loads(l) for l in lines]
            else:
                pooled_data = []
                for line in lines:
                    if len(pooled_data) == datasize:
                        break
                    pooled_data.append(json.loads(line))
                return pooled_data

    @staticmethod
    def dump(instances, path, flush_steps=1000000):
        assert type(instances) == list
        with open(path, 'w') as wf:
            for idx, instance in tqdm(enumerate(instances)):
                instance = json.dumps(instance, ensure_ascii=False)
                wf.write(instance + "\n")
                if idx % flush_steps:
                    wf.flush()
