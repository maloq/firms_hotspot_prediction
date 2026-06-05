"""Lightning modules that wrap sequence + static feature models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn

from .architectures import build_model


class SequenceStaticLightningModule(pl.LightningModule):
    """Binary classification module for dynamic + static feature models."""

    def __init__(
        self,
        model_name: str,
        model_config: Dict,
        learning_rate: float = 0.005,
        decay_rate: float = 0.5,
        decay_steps: int = 1000,
        l2: float = 0.001,
        clip_gradient_norm: float = 5.0,
        scheduler_config: Dict | None = None,
        log_train_ap: bool = True,
        loss_name: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
    ) -> None:
        super().__init__()
        model_config = dict(model_config or {})
        scheduler_config = dict(scheduler_config or {})
        loss_name = str(loss_name or "bce").strip().lower()
        if loss_name not in {"bce", "focal"}:
            raise ValueError("loss_name must be one of 'bce' or 'focal'")
        self.save_hyperparameters({
            "model_name": model_name,
            "model_config": model_config,
            "learning_rate": learning_rate,
            "decay_rate": decay_rate,
            "decay_steps": decay_steps,
            "l2": l2,
            "clip_gradient_norm": clip_gradient_norm,
            "scheduler_config": scheduler_config,
            "log_train_ap": log_train_ap,
            "loss_name": loss_name,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
        })
        self.model: nn.Module = build_model(model_name, **model_config)
        self.training_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.validation_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.test_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.scheduler_config = scheduler_config
        self.log_train_ap = bool(log_train_ap)
        self._train_ap_last: float | None = None

    @staticmethod
    def _metric_numpy(tensor: torch.Tensor):
        return tensor.detach().float().cpu().numpy()

    @staticmethod
    def _best_f1_from_scores(targets: torch.Tensor, probs: torch.Tensor) -> float:
        try:
            precision, recall, thresholds = precision_recall_curve(
                SequenceStaticLightningModule._metric_numpy(targets),
                SequenceStaticLightningModule._metric_numpy(probs),
            )
            if thresholds.size == 0:
                return 0.0
            f1 = (2 * precision * recall) / (precision + recall + 1e-12)
            f1 = f1[:-1]
            if f1.size == 0:
                return 0.0
            return float(f1.max())
        except Exception:
            return float("nan")

    def forward(self, dyn: torch.Tensor, stat: torch.Tensor, cat: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(dyn, stat, cat)

    def _loss(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        if self.hparams.loss_name == "focal":
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            prob = torch.sigmoid(logits)
            p_t = prob * target + (1.0 - prob) * (1.0 - target)
            loss = bce * torch.pow(1.0 - p_t, float(self.hparams.focal_gamma))
            if self.hparams.focal_alpha is not None:
                alpha = float(self.hparams.focal_alpha)
                alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
                loss = loss * alpha_t
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        if weight is not None:
            loss = loss * weight
        return loss.mean()

    def _shared_step(self, batch):
        dyn, stat, cat, y, loss_target, weight = batch
        logits = self(dyn, stat, cat if cat is not None and cat.numel() else None)
        loss = self._loss(logits, loss_target, weight)
        probs = torch.sigmoid(logits)
        return loss, probs, y

    def training_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        if self.log_train_ap:
            self.training_outputs.append((probs.detach(), y.detach()))
        batch_size = batch[0].shape[0]
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def on_train_epoch_start(self):
        self.training_outputs.clear()

    def on_train_epoch_end(self):
        if not self.training_outputs:
            return
        probs = torch.cat([out[0] for out in self.training_outputs], dim=0)
        targets = torch.cat([out[1] for out in self.training_outputs], dim=0)
        try:
            train_ap = float(
                average_precision_score(
                    self._metric_numpy(targets),
                    self._metric_numpy(probs),
                )
            )
        except Exception:
            train_ap = float('nan')
        self._train_ap_last = train_ap
        # log as epoch metric
        self.log("train_ap", train_ap, prog_bar=True, sync_dist=True)

    def on_validation_epoch_start(self):
        self.validation_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        batch_size = batch[0].shape[0]
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.validation_outputs.append((probs.detach(), y.detach()))
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        probs = torch.cat([out[0] for out in self.validation_outputs], dim=0)
        targets = torch.cat([out[1] for out in self.validation_outputs], dim=0)
        val_ap = float(
            average_precision_score(
                self._metric_numpy(targets),
                self._metric_numpy(probs),
            )
        )
        self.log("val_ap", val_ap, prog_bar=True, sync_dist=True)
        val_f1 = self._best_f1_from_scores(targets, probs)
        self.log("val_f1", val_f1, prog_bar=True, sync_dist=True)
        # selection metric: sum of train_ap (from this epoch) and val_ap
        sel_ap = val_ap
        if self._train_ap_last is not None and not (self._train_ap_last != self._train_ap_last):  # not NaN
            sel_ap = float(self._train_ap_last) + float(val_ap)
        self.log("sel_ap", sel_ap, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        self.test_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        batch_size = batch[0].shape[0]
        self.log("test_loss", loss, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.test_outputs.append((probs.detach(), y.detach()))
        return loss

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        probs = torch.cat([out[0] for out in self.test_outputs], dim=0)
        targets = torch.cat([out[1] for out in self.test_outputs], dim=0)
        test_ap = average_precision_score(
            self._metric_numpy(targets),
            self._metric_numpy(probs),
        )
        self.log("test_ap", test_ap, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Build parameter groups to apply weight decay only to weight tensors
        # (exclude biases and normalization parameters)
        def _param_groups_for_weight_decay():
            decay_params = []
            no_decay_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                is_bias = name.endswith('.bias')
                is_norm = ('.bn' in name.lower()) or ('batchnorm' in name.lower()) or ('.norm' in name.lower())
                if param.ndim == 1 or is_bias or is_norm:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            return [
                {"params": decay_params, "weight_decay": float(self.hparams.l2)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

        l2 = float(self.hparams.l2)
        if l2 > 0.0:
            optimizer = torch.optim.AdamW(
                _param_groups_for_weight_decay(),
                lr=self.hparams.learning_rate,
                weight_decay=0.0,  # weight decay set via groups
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=0.0,
            )

        scheduler_cfg = dict(self.scheduler_config or {})

        def default_lambda_scheduler():
            decay_steps = max(int(self.hparams.decay_steps), 1)
            decay_rate = float(self.hparams.decay_rate)

            def lr_lambda(step: int):
                return decay_rate ** (step / decay_steps)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        if not scheduler_cfg:
            lr_scheduler_dict = default_lambda_scheduler()
        else:
            scheduler_type = str(scheduler_cfg.get("type", "lambda")).lower()
            params = dict(scheduler_cfg.get("params", {}))
            interval = scheduler_cfg.get("interval", "epoch")
            frequency = int(scheduler_cfg.get("frequency", 1))
            monitor = scheduler_cfg.get("monitor")

            if scheduler_type == "lambda":
                decay_steps = max(int(params.get("decay_steps", self.hparams.decay_steps)), 1)
                decay_rate = float(params.get("decay_rate", self.hparams.decay_rate))

                def lr_lambda(step: int):
                    return decay_rate ** (step / decay_steps)

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
                interval = scheduler_cfg.get("interval", "step")
            elif scheduler_type == "step":
                step_size = max(int(params.get("step_size", self.hparams.decay_steps)), 1)
                gamma = float(params.get("gamma", self.hparams.decay_rate))
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=gamma,
                )
            elif scheduler_type == "cosine":
                t_max = params.get("T_max", params.get("t_max"))
                if t_max is None:
                    t_max = max(int(self.hparams.decay_steps), 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(int(t_max), 1),
                    eta_min=float(params.get("eta_min", 0.0)),
                )
            elif scheduler_type in {"reduce_on_plateau", "reducelronplateau"}:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=params.get("mode", "max"),
                    factor=float(params.get("factor", 0.5)),
                    patience=int(params.get("patience", 5)),
                    threshold=float(params.get("threshold", 1e-4)),
                    verbose=bool(params.get("verbose", True)),
                    min_lr=float(params.get("min_lr", 0.0)),
                )
                interval = "epoch"
                monitor = monitor or "val_ap"
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

            lr_scheduler_dict = {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": frequency,
            }
            if monitor is not None:
                lr_scheduler_dict["monitor"] = monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_dict,
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip_val = float(self.hparams.clip_gradient_norm)
        if clip_val <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm="norm",
        )
