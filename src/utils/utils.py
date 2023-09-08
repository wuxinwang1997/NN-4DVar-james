import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List
import argparse
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)
import functools
import dapper
from dapper.tools import progressbar as progbar
from mpl_tools import is_notebook_or_qt as nb
import dapper.tools.progressbar as pb
import torch
import numpy as np


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def simulate(self, desc='Truth & Obs'):
    """Generate synthetic truth and observations."""
    Dyn, Obs, tseq, X0 = self.Dyn, self.Obs, self.tseq, self.X0

    # Init
    xx = np.zeros((tseq.K + 1, Dyn.M))
    yy = np.zeros((tseq.Ko + 1, Obs.M))
    yy_ = np.zeros((tseq.K + 1, Obs.M))

    x = X0.sample(1)
    if len(x.shape) == 2:
        x = np.squeeze(x, axis=0)
    xx[0] = x

    # Loop
    for k, ko, t, dt in pb.progbar(tseq.ticker, desc):
        x = Dyn(x, t - dt, dt)
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        yy_[k] = Obs(x, t) + Obs.noise.sample(1)
        if ko is not None:
            yy[ko] = yy_[k]
        xx[k] = x

    return xx, yy, yy_

def direct_obs_matrix(Nx, obs_inds):
    """Generate matrix that "picks" state elements `obs_inds` out of `range(Nx)`.

    Parameters
    ----------
    Nx: int
        Length of state vector
    obs_inds: ndarray
        Indices of elements of the state vector that are (directly) observed.

    Returns
    -------
    H: ndarray
        The observation matrix for direct partial observations.
    """
    Ny = len(obs_inds)
    print(Ny)
    H = torch.zeros((Ny, Nx))
    H[range(Ny), obs_inds] = 1
    print(H.dtype)
    # One-liner:
    # H = np.array([[i==j for i in range(M)] for j in jj],float)

    return H

def with_recursion(func, prog=False):
    """Make function recursive in its 1st arg.

    Return a version of `func` whose 2nd argument (`k`)
    specifies the number of times to times apply func on its output.

    .. warning:: Only the first argument to `func` will change,
        so, for example, if `func` is `step(x, t, dt)`,
        it will get fed the same `t` and `dt` at each iteration.

    Parameters
    ----------
    func : function
        Function to recurse with.

    prog : bool or str
        Enable/Disable progressbar. If `str`, set its name to this.

    Returns
    -------
    fun_k : function
        A function that returns the sequence generated by recursively
        running `func`, i.e. the trajectory of system's evolution.
    Examples
    --------
    >>> def dxdt(x):
    ...     return -x
    >>> step_1  = with_rk4(dxdt, autonom=True)
    >>> step_k  = with_recursion(step_1)
    >>> x0      = np.arange(3)
    >>> x7      = step_k(x0, 7, t0=np.nan, dt=0.1)[-1]
    >>> x7_true = x0 * np.exp(-0.7)
    >>> np.allclose(x7, x7_true)
    True
    """
    def fun_k(x0, k, *args, **kwargs):
        xx = np.zeros((k+1,)+x0.shape)
        xx[0] = x0

        # Prog. bar name
        if prog == False:
            desc = None
        elif prog == None:
            desc = "Recurs."
        else:
            desc = prog

        for i in progbar(range(k), desc):
            xx[i+1] = func(xx[i], *args, **kwargs).detach().cpu().numpy()

        return xx

    return fun_k


def ens_compatible(func):
    """Decorate to transpose before and after, i.e. `func(input.T).T`.

    This is helpful to make functions compatible with both 1d and 2d ndarrays.

    .. note:: This is not `the_way™` -- other tricks (ref `dapper.mods`)
        are sometimes more practical.

    Examples
    --------
    `dapper.mods.Lorenz63.dxdt`, `dapper.mods.DoublePendulum.dxdt`

    See Also
    --------
    np.atleast_2d, np.squeeze, np.vectorize
    """
    @functools.wraps(func)
    def wrapr(x, *args, **kwargs):
        return func(x.T, *args, **kwargs).T
    return wrapr