import yaml
from argparse import Namespace

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)