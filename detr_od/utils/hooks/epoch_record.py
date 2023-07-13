from bisect import bisect_right

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import IterBasedRunner, EpochBasedRunner


@HOOKS.register_module()
class EpochRecord(Hook):
    def __init__(
        self,
        normalize=True,
        name="curr_step"
    ):
        self.name = name
    def before_train_iter(self, runner):
        iter_based = True
        if isinstance(runner, IterBasedRunner):
            TypeError('EpochRecord only support the EpochBasedRunner, but got IterBasedRunner!')
        else:
            curr_step = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model.bbox_head, self.name)
        setattr(model.bbox_head, self.name, curr_step)
