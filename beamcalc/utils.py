import sys
import json

import matplotlib.pyplot as plt

from pathlib import Path
from loguru import logger


plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.grid"] = True

CONSOLE_FMT = (
    "<green>{time:DD.MM.YYYY | HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)
FILE_FMT = (
    "{time:DD.MM.YYYY | HH:mm:ss} | "
    "{level: <8} | "
    "{name}:{function}:{line} - {message}"
)

class SingletonMeta(type):
    """
    A metaclass that implements the Singleton pattern.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Settings(metaclass=SingletonMeta):
    def __init__(self, settings_path):
        self.settings = self.import_settings(settings_path)

        self.VERBOSE = self.settings["INTEGRATION"]["verbose"]
        self.MAXNODES = self.settings["INTEGRATION"]["maxnodes"]
        self.TOL = self.settings["INTEGRATION"]["res_tolerance"]
        self.BCTOL = self.settings["INTEGRATION"]["bc_tolerance"]

        self.DZ = self.settings["PRINTING"]["dz"]

    @staticmethod
    def import_settings(file_path):
        with open(file_path, encoding="utf-8") as file:
            settings = json.load(file)
        
        plt.rcParams['font.size'] = settings["PRINTING"]["fontsize"]

        log_path = Path(settings["LOGGING"]["path"])
        log_dir = log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.remove()

        logger.add(
            sys.stdout,
            level=settings["LOGGING"]["console_level"],
            colorize=True,
            format=CONSOLE_FMT,
            enqueue=False,
            backtrace=False,
            diagnose=True,
        )

        logger.add(
            log_path,
            level=settings["LOGGING"]["file_level"],
            format=FILE_FMT,
            encoding="utf-8",
            rotation=settings["LOGGING"]["rotation"], 
            retention=settings["LOGGING"]["retention"], 
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        return settings
