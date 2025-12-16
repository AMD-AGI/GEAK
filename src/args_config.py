# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import yaml
from argparse import Namespace

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)