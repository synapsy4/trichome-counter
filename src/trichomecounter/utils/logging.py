"""
Class for training logging (local + tensorboard) + state saving
"""

import json
import yaml
import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from trichomecounter.utils import utils
from trichomecounter.paths import MODELS_DIR, TB_LOGS_DIR 

class TrainingLogger:
    """
    Tracks training progress across epochs and runs.

    This class manages model checkpointing, training history, run metadata,
    and TensorBoard logging. It maintains a persistent overview file that
    tracks cumulative training statistics across multiple training sessions,
    including the best validation score and total number of completed epochs.

    Parameters
    ----------
    cfg : dict
        Training config dictionary.

    Attributes
    ----------
    cfg : dict
        Training config.
    model_name : str
        Name of the model.
    model_dir : Path
        Directory where checkpoints and training logs are stored.
    tb_dir : Path
        TensorBoard log directory.
    overview_path : Path
        Path to the overview JSON file.
    epoch_offset : int
        Number of epochs completed in previous training runs.
    global_epoch : int
        Current epoch number incuding previous training runs.
    epoch : int
        Epoch number within the current run.
    run : int
        Current training run index.
    run_name : str
        Name of the current run directory.
    """
    def __init__(self, 
                 cfg: dict[str, Any]
                 ) -> None:
        """
        Initialize the training logger.

        Parameters
        ----------
        cfg : dict
            Training config dictionary.
        """

        self.cfg = cfg
        self.model_name = cfg["model"]["model_name"]

        # Define directories + overview path
        self.model_dir = MODELS_DIR / self.model_name
        self.tb_dir = TB_LOGS_DIR / self.model_name
        self.overview_path = self.model_dir / "overview.json"

        # Get overview (first run if non existent)
        overview = self._load_overview()
        first_run = True if overview is None else False

        # Get number of global epochs (from passed runs)
        self.epoch_offset = 0 if first_run else overview["epochs"] 
        self.global_epoch = self.epoch_offset
        self.epoch = 0
        
        # Set run number + init name fro run-specific dir
        self.run = 1 if first_run else overview["training_runs"] + 1
        self.run_name = f"ep{self.epoch_offset+1}-{self.epoch_offset+1}"
        self.last_run_name = self.run_name

        # Init tensorboard writers
        self.tb_summarywriter = None
        self.tb_hparamwriter = None
    
    def log_epoch(self, 
                  epoch: int, 
                  metrics: dict[str, Any], 
                  model: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer
                  ) -> None:
        """
        Log the results of a completed training epoch.

        Updates training metadata, saves model checkpoints, records metrics,
        updates TensorBoard logs, and maintains run-specific config files.

        Parameters
        ----------
        epoch : int
            Current epoch number within the active training run.
        metrics : dict[str, Any]
            Dictionary containing training and validation metrics.
        model : torch.nn.Module
            Model being trained.
        optimizer : torch.optim.Optimizer
            Optimizer associated with the model.
        """
        
        # Update global epoch + run name
        self.epoch = epoch
        self.global_epoch = self.epoch_offset + epoch
        self.run_name = f"ep{self.epoch_offset+1}-{self.global_epoch}"

        # Make sure model dir exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Update overview + check for new best epoch
        new_best, overview = self._update_overview(metrics)
        
        # Save model + optimizer states
        self._save_states(model, optimizer, new_best)
        
        # Update history json + log cfg to run dir
        self._update_history(metrics)
        self._log_run()
        
        # Update tensorboard
        self._tb_update_summary(metrics)
        self._tb_update_run(overview)

        # Update last run name
        self.last_run_name = self.run_name
    
    def close(self) -> None:
        """
        Close all active TensorBoard writers.
        """
        self.tb_summarywriter.close()
        self.tb_hparamwriter.close()

    def _load_overview(self) -> dict[str, Any] | None:
        """
        Load the training overview file.

        Returns
        -------
        dict[str, Any] or None
            Contents of the overview JSON file if it exists,
            otherwise ``None``.
        """
         
        if self.overview_path.exists():
            with open(self.overview_path, "r") as f:
                return json.load(f)
        else:
            return None
        
    def _update_overview(self, 
                         metrics: dict[str, Any]
                         ) -> tuple[bool, dict[str, Any]]:
        """
        Update overview file.

        Maintains global statistics across training runs, including total
        epochs completed, best validation MAE, and best-performing run.

        Parameters
        ----------
        metrics : dict[str, Any]
            Metrics of the current epoch.

        Returns
        -------
        new_best: bool
            Whether the current epoch achieved a new best
            validation MAE.
        overview: dict
            Updated overview dictionary.
        """
         
        # Get old overview if exists
        overview = self._load_overview()
        first_epoch = True if overview is None else False

        # Update overview content + check if epoch is new best
        if first_epoch:
            overview = {
                "model_name": self.model_name,
                "epochs": 1,
                "best_epoch": 1,
                "training_runs": 1,
                "best_run": 1,
                "best_val_mae": metrics["val_mae"]
            }
            new_best = True # n_epochs=1 => best epoch
        else:
            overview["epochs"] = self.global_epoch
            overview["training_runs"] = self.run
            
            if metrics["val_mae"] < overview["best_val_mae"]:
                new_best = True
                overview["best_val_mae"] = metrics["val_mae"]
                overview["best_epoch"] = self.global_epoch
                overview["best_run"] = self.run
            else:
                new_best = False

        # Save overview
        with open(self.overview_path, "w") as f:
            json.dump(overview, f, indent=4)

        return new_best, overview
        
    def _save_states(self, 
                     model: torch.nn.Module, 
                     optimizer: torch.optim.Optimizer,
                     new_best: bool
                     ) -> None:
        """
        Save model and optimizer checkpoints.

        Always saves the latest model and optimizer state. If the current
        epoch achieves the best validation performance so far, an additional
        best-model checkpoint is written.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose parameters should be saved.
        optimizer : torch.optim.Optimizer
            Optimizer whose state should be saved.
        new_best : bool
            Whether the current epoch achieved a new best validation score.
        """
        
        # Save last model + optimizer states
        model_save_path = self.model_dir / "last_cp.pth"
        optim_save_path = self.model_dir / "optim_cp.pth"
        model_state = model.state_dict()
        torch.save(obj=model_state, f=model_save_path)
        torch.save(obj=optimizer.state_dict(), f=optim_save_path)

        # If new best model state => save it
        if new_best:
            model_save_path = self.model_dir / "best_cp.pth"
            torch.save(obj=model_state, f=model_save_path) 


    def _update_history(self, 
                        metrics: dict[str, Any]
                        ) -> None:
        """
        Append epoch metrics to the training history log.

        Stores a JSON Lines record containing metrics and selected training
        configuration values for the current epoch.

        Parameters
        ----------
        metrics : dict[str, Any]
            Metrics of the current epoch.
        """

        # Write epoch metric overview
        epoch_metrics = {
                    "epoch": self.global_epoch,
                    "train_loss": round(metrics["train_loss"], 3),
                    "val_loss": round(metrics["val_loss"], 3),
                    "train_mae": round(metrics["train_mae"], 3),
                    "val_mae": round(metrics["val_mae"], 3),
                    "lr": self.cfg["training"]["lr"],
                    "wd": self.cfg["training"]["weight_decay"],
                    "loss_args": self.cfg["loss"]["loss_args"],
                    "target_map_cfg": utils.flatten_dict(self.cfg["target_map"])
                    }
        
        # Save epoch metrics
        history_path = self.model_dir / "history.jsonl"
        with open(history_path, "a") as f:
            f.write(json.dumps(epoch_metrics) + "\n")
 
    def _log_run(self) -> None:
        """
        Create or update the current run directory.

        Maintains run-specific directories and stores a copy of the training
        configuration with the number of completed epochs recorded.
        """

        last_run_dir = self.model_dir / self.last_run_name
        run_dir = self.model_dir / self.run_name

        # Rename or make run dir
        if last_run_dir.exists():
            last_run_dir.rename(run_dir)
        else:
            run_dir.mkdir(parents=True,
                          exist_ok=True)
            
        # Save copy of cfg with passed epochs
        cfg_copy = copy.deepcopy(self.cfg)
        cfg_copy["training"]["epochs"] = self.epoch
        cfg_save_path = run_dir / "config.yaml"
        with open(cfg_save_path, "w") as f:
            yaml.dump(cfg_copy, f)
                
    def _tb_update_summary(self, 
                           metrics: dict[str, Any]
                           ) -> None:
        """
        Update TensorBoard scalar summaries.

        Logs training and validation loss as well as MAE values for the
        current epoch.

        Parameters
        ----------
        metrics : dict[str, Any]
            Metrics of the current epoch.
        """

        if self.tb_summarywriter is None:
            self.tb_summarywriter = SummaryWriter(log_dir=self.tb_dir) 
        
        self.tb_summarywriter.add_scalars("Loss", {"train": metrics["train_loss"], "val": metrics["val_loss"]}, self.global_epoch)
        self.tb_summarywriter.add_scalars("MAE",  {"train": metrics["train_mae"],  "val": metrics["val_mae"]},  self.global_epoch)


    def _tb_update_run(self,
                       overview: dict[str, Any]
                       ) -> None:
        """
        Update TensorBoard hyperparameter logs.

        Creates or updates the TensorBoard run associated with the current
        training session and records configuration parameters together with
        summary statistics such as the best validation MAE.

        Parameters
        ----------
        overview : dict[str, Any]
            Updated overview containing cumulative statistics.
        """
        
        last_run_dir = self.tb_dir / "hparams" / self.last_run_name
        run_dir = self.tb_dir / "hparams" / self.run_name

        # First epoch of new run: create run dir
        if self.tb_hparamwriter is None:
            run_dir.mkdir(parents=True, exist_ok=True)
        # Else: close writer + rename dir
        else:
            self.tb_hparamwriter.close()
            last_run_dir.rename(run_dir)
            
        # Init hparam writer
        self.tb_hparamwriter = SummaryWriter(log_dir=str(run_dir))

        # Log cfg + hparams to tensorboard
        cfg_copy = copy.deepcopy(self.cfg)
        cfg_copy["training"]["epochs"] = self.epoch
        self.tb_hparamwriter.add_hparams(utils.flatten_dict(cfg_copy), {
            "hparam/best_val_mae": overview["best_val_mae"],
            "hparam/best_epoch":   overview["best_epoch"]
            })

