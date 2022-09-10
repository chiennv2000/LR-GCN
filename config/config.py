import yaml

class Configuration(object):
    def __init__(self):
        pass
    
    @staticmethod
    def get_config(file_path: str) -> dict:
        with open(file_path, 'r') as f:
            return yaml.load(f)