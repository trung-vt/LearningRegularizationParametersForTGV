import os
from pathlib import Path
from typing import Union


def makepath(*args: Union[str, Path]) -> Path:
    """
    Make a path from the given arguments.
    """
    path = os.path.normpath(os.path.join(*args))
    return Path(path)
