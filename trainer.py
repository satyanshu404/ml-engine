"""Script to train a model"""

import warnings

warnings.simplefilter("ignore")

import wandb
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from getpass import getpass
import random
import os
import logging
from typing import Dict, Callable, Optional, List
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from ml_engine.utils import set_device, set_gloabl_seed, read_file
from ml_engine.contract import (
    TrainerConfig,
    GlobalConfig,
    TrainReturnContract,
    TestReturnContract,
)
from ml_engine.constants import (
    GLOBAL_CONFIG,
    TRAINER_CONFIG,
    YAML,
    TRAIN_LOADER,
    VAL_LOADER,
    MAX_INT,
)

logger = logging.getLogger(__name__)


def read_global_config() -> GlobalConfig:
    """Read global config"""
    configs = read_file(YAML, "ml_engine/config.yaml")
    assert GLOBAL_CONFIG in configs, f"{GLOBAL_CONFIG} not found in config.yaml"
    return GlobalConfig(**configs.get(GLOBAL_CONFIG))


global_config = read_global_config()

# for deterministic ops
if global_config.deterministic:
    set_gloabl_seed(global_config.seed)
else:
    logger.warning("Deterministic ops are disabled, results may not be reproducible...")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data_loader: Dict[str, DataLoader],
        optimizer,
        scheduler: Optional,
        criterion,
        evaluate: Optional[Callable] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the trainer
        """
        assert TRAIN_LOADER in data_loader, f"{TRAIN_LOADER} not found in data_loader"
        assert VAL_LOADER in data_loader, f"{VAL_LOADER} not found in data_loader"

        self.config = self.load_config(config)
        self.device, self.autocast_device_type = set_device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.evaluate = evaluate
        self.train_loader = data_loader.get(TRAIN_LOADER)
        self.val_loader = data_loader.get(VAL_LOADER)
        self.train_losses, self.val_losses = [], []
        self.eval_best_value = (
            float(-MAX_INT)
            if self.config.save_best_model_on == "metric"
            else float(MAX_INT)
        )
        self.stop_early = False
        self.patience_counter = 0
        self.load_config(config)
        self.model_save_path = self.define_model_save_path()

        self.init_wandb()

    def init_wandb(self):
        """Initialize wandb"""
        if global_config.project_name is None:
            global_config.project_name = f"ml_engine_{self.model.__class__.__name__}"

        if self.config.run_name is None:
            self.config.run_name = f"{self.model.__class__.__name__}_{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}"

        # Check if W&B API key already exists
        if wandb.api.api_key is None:
            key = getpass(
                "Enter your W&B API key (you can get it from https://wandb.ai/authorize):"
            ).strip()
            wandb.login(key=key)
        else:
            wandb.login()

        wandb.init(
            project=global_config.project_name,
            name=self.config.run_name,
            config=self.config,
            reinit=True,
            mode="disabled" if not self.config.is_wandb else "online",
        )

        wandb.watch(self.model, self.criterion, log="all", log_freq=10)

    def train_epoch(self) -> TrainReturnContract:
        """
        Train the model for one epoch
        Returns:
            Dict[str, Optional[np.ndarray]]: Dictionary containing loss, targets and predictions
        """
        self.model.train()
        total_loss, total_samples = 0, 0
        all_preds, all_targets = [], []

        for batch_x, batch_y, _ in tqdm(
            self.train_loader, desc="Training", leave=False
        ):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()

            with torch.autocast(
                device_type=self.autocast_device_type,
                enabled=not self.config.use_autocast,
            ):
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
            all_preds.append(preds.detach().cpu())
            all_targets.append(batch_y.detach().cpu())

        all_preds_tensor = torch.cat(all_preds)
        all_targets_tensor = torch.cat(all_targets)

        if all_preds_tensor.is_cuda:
            all_preds_tensor = all_preds_tensor.cpu()
            all_targets_tensor = all_targets_tensor.cpu()

        all_preds_np = all_preds_tensor.numpy()
        all_targets_np = all_targets_tensor.numpy()
        all_preds_np = np.nan_to_num(all_preds_np, nan=0.0, posinf=1e6, neginf=-1e6)
        all_targets_np = np.nan_to_num(all_targets_np, nan=0.0, posinf=1e6, neginf=-1e6)

        avg_loss = total_loss / total_samples
        return TrainReturnContract(
            loss=avg_loss,
            targets=all_targets_np,
            predictions=all_preds_np,
        )

    def validate_epoch(self) -> TrainReturnContract:
        """
        Validate the model for one epoch
        Returns:
            Dict[str, Optional[np.ndarray]]: Dictionary containing loss, targets and predictions
        """
        self.model.eval()
        total_loss, total_samples = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_x, batch_y, _ in tqdm(
                self.val_loader, desc="Validating", leave=False
            ):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)
                all_preds.append(preds.detach().cpu())
                all_targets.append(batch_y.detach().cpu())

        all_preds_tensor = torch.cat(all_preds)
        all_targets_tensor = torch.cat(all_targets)

        if all_preds_tensor.is_cuda:
            all_preds_tensor = all_preds_tensor.cpu()
            all_targets_tensor = all_targets_tensor.cpu()

        all_preds_np = all_preds_tensor.numpy()
        all_targets_np = all_targets_tensor.numpy()
        all_preds_np = np.nan_to_num(all_preds_np, nan=0.0, posinf=1e6, neginf=-1e6)
        all_targets_np = np.nan_to_num(all_targets_np, nan=0.0, posinf=1e6, neginf=-1e6)

        avg_loss = total_loss / total_samples
        return TrainReturnContract(
            loss=avg_loss,
            targets=all_targets_np,
            predictions=all_preds_np,
        )

    def train(self):
        """Train the model"""
        self.log__start_training_info()
        logger.info("Starting training...")
        for epoch in tqdm(range(self.config.epochs), desc="Training", leave=False):
            logger.info(f"\n[Epoch {epoch + 1}/{self.config.epochs}]")
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            self.train_losses.append(train_metrics.loss)
            self.val_losses.append(val_metrics.loss)

            if self.scheduler:
                self.scheduler.step(val_metrics.loss)

            log_train_metric, log_val_metric = "", ""
            if self.evaluate:
                metric_func_name = self.evaluate.__name__
                train_metric_value = self.evaluate(
                    train_metrics.targets, train_metrics.predictions
                )
                val_metric_value = self.evaluate(
                    val_metrics.targets, val_metrics.predictions
                )
                log_train_metric = f"| {metric_func_name}: {train_metric_value:.6f}"
                log_val_metric = f"| {metric_func_name}: {val_metric_value:.6f}"

            logger.info(f"Train - Loss: {train_metrics.loss:.6f} {log_train_metric}")
            logger.info(f"Val   - Loss: {val_metrics.loss:.6f} {log_val_metric}")

            assert self.config.save_best_model_on in [
                "loss",
                "metric",
                None,
            ], "save_best_model_on must be 'loss' or 'metric' or None"

            if self.config.save_best_model_on == "metric" and self.evaluate:
                self.save_best_model(value=val_metric_value, mode="max")
            else:
                self.save_best_model(value=val_metrics.loss, mode="min")

            wandb.log(
                {
                    "train_loss": train_metrics.loss,
                    "val_loss": val_metrics.loss,
                    "train_accuracy": train_metric_value if self.evaluate else None,
                    "val_accuracy": val_metric_value if self.evaluate else None,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

            if self.stop_early:
                logger.info("Early stopping...")
                break

        wandb.finish()
        logger.info("Training completed...")

    def test_model(self, test_loader) -> TestReturnContract:
        """Test the model and return comprehensive metrics
        Args:
            test_loader (DataLoader): Test data loader
        Returns:
            TestReturnContract: Test metrics contract
        """
        logger.info("Testing...")
        self.model.eval()
        all_preds, all_targets, all_metadata = [], [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_meta in tqdm(
                test_loader, desc="Testing", leave=False
            ):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                preds = self.model(batch_x)

                all_preds.append(preds.cpu())
                all_targets.append(batch_y.cpu())
                all_metadata.extend(batch_meta)

        all_preds, all_targets = (
            torch.cat(all_preds).numpy().flatten(),
            torch.cat(all_targets).numpy().flatten(),
        )

        return TestReturnContract(
            metric=self.evaluate(all_targets, all_preds),
            targets=all_targets,
            predictions=all_preds,
            metadata=all_metadata,
        )

    def save_best_model(self, value: float, mode: str):
        if mode == "min":
            pct_change = (self.eval_best_value - value) / abs(self.eval_best_value)
            improved = pct_change > self.config.epsilon
        elif mode == "max":
            pct_change = (value - self.eval_best_value) / abs(self.eval_best_value)
            improved = pct_change > self.config.epsilon

        logger.info(f"pct_change: {pct_change}, improved: {improved}")
        if improved:
            logger.info(f"Model is improved, saving best model...")
            self.eval_best_value = value
            self.patience_counter = 0
            self.save_model()
        else:
            logger.info(
                f"Model is not improved, patience counter: {self.patience_counter}"
            )
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                self.stop_early = True

    def define_model_save_path(self) -> str:
        """Define model save path"""
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        if self.config.model_save_name is None:
            logger.info("Model save name not specified, creating a default name...")
            self.config.model_save_name = (
                f"best_model_{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}.pt"
            )
        return os.path.join(self.config.model_save_dir, self.config.model_save_name)

    def save_model(self):
        """Save model state dict to a path"""
        torch.save(self.model.state_dict(), self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")

    def load_config(self, config: Dict = None) -> TrainerConfig:
        """Load config for trainer"""
        _config = read_file(YAML, "ml_engine/config.yaml")
        training_config = _config.get(TRAINER_CONFIG)

        training_config = (
            {**training_config, **config}
            if config
            else logger.warning("No config provided, loading default config")
        )
        return TrainerConfig(**training_config)

    def log__start_training_info(self):
        logger.info(f"{'=' * 20} TRAINING INFO {'=' * 20}")
        logger.info(f"Training model set to: {self.model.__class__.__name__}")
        logger.info(f"Set seed to: {global_config.seed}")
        logger.info(f"Set epochs to: {self.config.epochs}")
        logger.info(f"Set device to: {self.device}")
        logger.info(f"Set learning rate to: {self.config.learning_rate}")
        logger.info(f"Set batch size to: {self.config.batch_size}")
        logger.info(f"Set patience to: {self.config.patience}")
        logger.info(f"Set epsilon to: {self.config.epsilon}")
        logger.info(f"Set is_wandb to: {self.config.is_wandb}")
        logger.info(f"Set deterministic to: {global_config.deterministic}")
        logger.info(f"Set use_autocast to: {self.config.use_autocast}")
        logger.info(f"Set Save Best Model on: {self.config.save_best_model_on}")
        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Using scheduler: {self.scheduler.__class__.__name__}")
        logger.info(f"Using criterion: {self.criterion.__class__.__name__}")
        logger.info(f"Using evaluate: {self.evaluate.__name__}")
        logger.info(f"Using project Name: {global_config.project_name}")
        logger.info(f"Using run Name: {self.config.run_name}")
        logger.info(f"Using save model path: {self.model_save_path}")
        logger.info(f"{'=' * 55}")
