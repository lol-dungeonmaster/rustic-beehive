import os
from . import log

class UserSecretsClient:
    @classmethod
    def set_secret(cls, id: str, value: str):
        os.environ[id] = value
    @classmethod
    def get_secret(cls, id: str):
        try:
            return os.environ[id]
        except KeyError as e:
            log.warning(f"KeyError: authentication token for {id} is undefined")