"""Main code for executing the basic VIT architecture for videos."""
from typing import Any, Dict, Tuple, Optional

import pytorch_lightning as L
import random
import ipdb
import math

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ChainedScheduler
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision.transforms import v2
#from src.models.static_token_utils import batched_find_idxs_to_keep
from src.models.tokenizer import Tokenizer
from src.models.tokenizer_utils import (RandomErasingConfig, TokenizerConfig)
from src.utils.model_utils import mae_load_state_dict
from src.models.mixup import Mixup
from src.models.random_erasing import RandomErasing
from src.models.optim_utils import *

class VITModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler_cfg: CosineSchedulerConfig = None,
        tokenizer_cfg: TokenizerConfig = None,
        compile: bool = False,
        pretrain: Optional[str] = None,
        finetune: Optional[str] = None
    ):
        super().__init__()
        # Can't do auto-opt if we want to use a scheduler.
        self.automatic_optimization = False
         # Initialize the underlying model.
        self.model = model
        # We define the transforms for the tokenizer. We don't do this
        # on CPU because it's slow; we keep transform + tokenization on GPU.
       

        if pretrain is not None:
            state_dict = torch.load(pretrain)
            if "module" in state_dict:
                state_dict = state_dict["module"]
            elif "state_dict" in state_dict:
                # HACK: replace this with a more principled thing using lightning
                state_dict = state_dict["state_dict"]
                new_dict = dict()
                for k in state_dict:
                    if k.startswith("model."): 
                        new_key = k.replace("model.", "")
                        new_dict[new_key] = state_dict[k]
                state_dict = new_dict
            self.model.load_state_dict(state_dict, strict=False)
        elif finetune is not None:
            self._load_pretrained_ckpt(finetune)

        # Initialize the optimizer.
        self.hparams.optimizer = optimizer
        self.hparams.scheduler_cfg = scheduler_cfg
        
    
        self.compile = compile
        # Set up loss functions and metrics
        # metric objects for calculating and averaging accuracy across batches
        # For finetuning.
        self.val_criterion = torch.nn.CrossEntropyLoss()
        # Use this if no mixup. If mixup is enabled, tokenizer
        # will set the criterion to SoftTargetCrossEntropy.
        self.train_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.hparams.tokenizer_cfg = tokenizer_cfg
        self.setup_tokenizer()

        self.train_acc = Accuracy(task="multiclass", num_classes=model.num_classes)
        self.train_top5_acc = Accuracy(task="multiclass", num_classes=model.num_classes, top_k=5)
        self.top1_acc = Accuracy(task="multiclass", num_classes=model.num_classes)
        self.top5_acc = Accuracy(task="multiclass", num_classes=model.num_classes, top_k=5)
        # self.test_acc = Accuracy(task="multiclass", num_classes=model.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def setup_tokenizer(self) -> None:
        """
        Initialize tokenizer according to specified config.
        """
        # No resizing; this is taking care of in dataloder.
        self.transform = torch.nn.Sequential(
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        )
        # Move these to config
        mixup_fn = None
        random_erase_fn = None
        rand_aug_fn = None
        
        assert self.hparams.tokenizer_cfg is not None
        cfg = self.hparams.tokenizer_cfg

        if cfg.re_config is not None:
            random_erase_fn = RandomErasing(
                probability=cfg.re_config.probability,
                mode=cfg.re_config.mode,
                min_count=cfg.re_config.min_count,
                device=cfg.re_config.device
            )
        if cfg.mixup_config is not None:
            mixup_fn = Mixup(
                mixup_alpha=cfg.mixup_config.mixup_alpha,
                cutmix_alpha=cfg.mixup_config.cutmix_alpha,
                cutmix_minmax=None,
                prob=cfg.mixup_config.prob,
                switch_prob=cfg.mixup_config.switch_prob,
                mode=cfg.mixup_config.mode,
                label_smoothing=cfg.mixup_config.label_smoothing,
                num_classes=cfg.mixup_config.num_classes
            )
        # Set up random augment
        if cfg.ra_config is not None:
            rand_aug_fn = v2.RandAugment(
                num_ops=cfg.ra_config.num_ops,
                magnitude=cfg.ra_config.magnitude,
            )

        self.tokenizer = Tokenizer(drop_policy=cfg.drop_policy,
                                   drop_param=cfg.drop_param,
                                   embed_dims=self.model.embed_dims,
                                   transform=self.transform,
                                   encode_length=cfg.encode_length,
                                   mixup_fn=mixup_fn,
                                   random_erase_fn=random_erase_fn,
                                   rand_aug_fn=rand_aug_fn)
        
        # Have to use diff loss fn if we enable mixup.
        if self.tokenizer.mixup_fn is not None:
            self.train_criterion = SoftTargetCrossEntropy()

    def _load_pretrained_ckpt(self, ckpt_path: str) -> None:
        checkpoint_model = torch.load(ckpt_path)['model']
        
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                #del checkpoint_model[k]
        all_keys = list(checkpoint_model.keys())
        new_dict = dict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                if key.startswith('encoder.norm'):
                    new_key = key.replace('encoder.norm', 'fc_norm')
                    new_dict[new_key] = checkpoint_model[key]
                else:
                    new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        mae_load_state_dict(self.model, new_dict)
        
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        
        self.val_loss.reset()
        self.train_acc.reset()
        self.train_top5_acc.reset()
        self.top1_acc.reset()
        self.top5_acc.reset()
        self.val_acc_best.reset()


    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        if "frames" not in batch:
            print("Missing frames in batch, path: ", batch["video_path"])
        frames = batch["frames"]
        targets = batch["label"]
        batch_size = frames.shape[0]
        with torch.no_grad():
            #ipdb.set_trace()
            output_dict = self.tokenizer(frames, targets, is_training=self.training)
        split_crops = output_dict["split_crops"].detach()
        targets = output_dict["targets"].detach()
        num_tokens = output_dict["num_tokens"].detach()
        pos_embed = output_dict["pos_embeds"].detach()
        token_lengths = None
        if "token_lengths" in output_dict:
            if output_dict["token_lengths"] is not None:
                token_lengths = output_dict["token_lengths"].detach()
            else:
                token_lengths = None
        preds = self.model(split_crops, num_tokens, pos_embed, batch_size, token_lengths)
       
        return preds, targets

    def training_step(self, batch, batch_idx):
       # TODO: gradient clippign?
       opt = self.optimizers()
       opt.zero_grad()
       # Avoid hack from soft target; this should be fixed..
       orig_targets = batch["label"]
       preds, soft_targets = self.model_step(batch)
       #loss = self.criterion(preds, targets
       assert preds.shape[1] == 1
       preds = preds.squeeze(1)
       
       loss = self.train_criterion(preds, soft_targets)
       self.manual_backward(loss)
       opt.step()
       scheduler = self.lr_schedulers()
       scheduler.step()

       pred_classes = F.softmax(preds, dim=-1)
           
       self.train_loss(loss)
       self.train_acc(pred_classes, orig_targets)
       self.train_top5_acc(pred_classes, orig_targets)

       #self.log("train/learning_rate", cur_lr, on_step=True, on_epoch=True, prog_bar=True)
       self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
       self.log("train/top1_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
       self.log("train/top5_acc", self.train_top5_acc, on_step=True, on_epoch=True, prog_bar=True)
       return loss

    def validation_step(self, batch, batch_idx):
        """
        TODO: add visualiation for some random batch idxs, include in wandb
        TODO; add attention viz for some random batch idx, include in wandb
        """
        preds, targets = self.model_step(batch)
        assert preds.shape[1] == 1
        preds = preds.squeeze(1)
        loss = self.val_criterion(preds, targets)
        preds = F.softmax(preds, dim=-1)
        
        # update and log metrics
        self.val_loss(loss)
        self.top1_acc(preds, targets)
        self.top5_acc(preds, targets)
        # self.val_acc(preds, targets)
        
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/top1_acc", self.top1_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/top5_acc", self.top5_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        TODO: add visualiation for some random batch idxs, include in wandb
        TODO; add attention viz for some random batch idx, include in wandb
        """
        preds, targets = self.model_step(batch)
        #ipdb.set_trace()
        #loss = self.val_criterion(preds, targets)
        preds = F.softmax(preds, dim=-1)
        # Mean across "crop" axis.
        preds = preds.mean(1)
        # update and log metrics
        #self.test_loss(loss)
        self.top1_acc(preds, targets)
        self.top5_acc(preds, targets)
        #self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        #self.log("test/inference_time", end_time - start_time, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/top1_acc", self.top1_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/top5_acc", self.top5_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Set up optimizers.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be
            used for training.
        """
        LD = 0.75

        num_layers = self.model.depth
        values = list(LD** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(values)

        opt_params = setup_layer_decay(self.model, 
                                   ld_assigner=assigner, 
                                   weight_decay=0.05)

        optimizer = self.hparams.optimizer(params=opt_params)
        
        config = self.hparams.scheduler_cfg
        num_training_steps = self.trainer.estimated_stepping_batches
        batch_size = self.trainer.train_dataloader.batch_size
        config.total_steps = num_training_steps
        config.batch_size = batch_size
        scheduler = cosine_scheduler(optimizer=optimizer, config=config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        #return {"optimizer": optimizer}
