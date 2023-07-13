from bisect import bisect_right

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class StepRecord(Hook):
    def __init__(
        self,
        normalize=True,
        name="curr_step"
    ):
        self.normalize = normalize
        self.name = name
    def before_train_iter(self, runner):
        iter_based = True
        try:
            curr_step = runner.iter
        except:
            curr_step = runner.epoch
            iter_based = False
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, self.name)
        setattr(model, self.name, curr_step/10000 if self.normalize and iter_based else curr_step)
