import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmcv.runner.hooks import HOOKS
from mmcv.utils import build_from_cfg
from mmdet.core import EvalHook
from mmdet.datasets import build_dataset, replace_ImageToTensor

from detr_ssod.datasets import build_dataloader
from detr_ssod.utils import find_latest_checkpoint, get_root_logger, patch_runner
from detr_ssod.utils.hooks import DistEvalHook


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            sampler_cfg=cfg.data.get("sampler", {}).get("train", {}),
        )
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if "runner" not in cfg:
        cfg.runner = {"type": "EpochBasedRunner", "max_epochs": cfg.total_epochs}
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )
    else:
        if "total_epochs" in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = eval_cfg.get(
            "by_epoch", cfg.runner["type"] != "IterBasedRunner"
        )
        if "type" not in eval_cfg:
            eval_hook = DistEvalHook if distributed else EvalHook
            eval_hook = eval_hook(val_dataloader, **eval_cfg)

        else:
            eval_hook = build_from_cfg(
                eval_cfg, HOOKS, default_args=dict(dataloader=val_dataloader)
            )

        runner.register_hook(eval_hook, priority=80)

    # user-defined hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    runner = patch_runner(runner)
    # resume_from = None
    # if cfg.get("auto_resume", True):
    #     resume_from = find_latest_checkpoint(cfg.work_dir)
    # if resume_from is not None:
    #     cfg.resume_from = resume_from

    # Note: do the checkpoint check, somtimes we need to resume 
    # a checkpoint which has different network structure, so we
    # need to check the parameters in the optimizer to make sure
    # it can load correct!

    # device_id = torch.cuda.current_device()
    # pre_checkpiont = runner.load_checkpoint(crg.resume_from, map_location=lambda storage, loc:storage.cuda(device_id))
    # if 

    if cfg.resume_from:
        try:
            runner.resume(cfg.resume_from)
        except:
            # Note: when resume the model with the optimizer error(mostly caused by the optimizer group number mismatch!)
            # when this massage throw out, you cat take two methods to solve this problem: (1) don't resume the optimizer
            # status, which is the default setting; (2) change the optimizer group dict manually to make them aligned.
            logger.warning('==='*30)
            logger.warning("!!!! Don't reuse the pretrained optimizer state in this resume prcess !!!!")
            logger.warning('==='*30)
            runner.resume(cfg.resume_from, resume_optimizer=False)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # breakpoint()
    runner.run(data_loaders, cfg.workflow)
