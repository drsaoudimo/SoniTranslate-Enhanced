import logging
import sys
import warnings
import os

def configure_logging_libs(debug=False):
    warnings.filterwarnings(action="ignore", category=UserWarning, module="pyannote")

    modules = [
        "numba", "httpx", "markdown_it", "speechbrain", "fairseq", "pyannote",
        "faiss",
        "pytorch_lightning.utilities.migration.utils",
        "pytorch_lightning.utilities.migration",
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch.utilities.migration.utils",
    ]

    try:
        for module in modules:
            logging.getLogger(module).setLevel(logging.WARNING)
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" if not debug else "1"

        # Fix verbose pyannote audio
        def fix_verbose_pyannote(*args, **kwargs):
            pass

        # Lazy import pyannote to avoid unnecessary memory usage
        import pyannote.audio.core.model  # noqa
        pyannote.audio.core.model.check_version = fix_verbose_pyannote

    except Exception as error:
        logger.error(str(error))


def setup_logger(name_log):
    logger = logging.getLogger(name_log)
    logger.setLevel(logging.INFO)

    default_handler = logging.StreamHandler(sys.stderr)
    default_handler.flush = sys.stderr.flush
    logger.addHandler(default_handler)

    logger.propagate = False

    formatter = logging.Formatter("[%(levelname)s] >> %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger


logger = setup_logger("sonitranslate")
logger.setLevel(logging.INFO)


def set_logging_level(verbosity_level):
    logging_level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    logger.setLevel(logging_level_mapping.get(verbosity_level, logging.INFO))


# Explicitly run garbage collection after configuring logging to free up memory
import gc
gc.collect()
