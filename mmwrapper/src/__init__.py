from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class HSABasicLogger(Hook):
    def __init__(self, interval, callback=None):
        super().__init__()
        self.callback = callback
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (runner.iter + 1) % self.interval == 0:
            log_vars = {k: v.clone().detach() for k, v in outputs.items()}
            # runner.inner_iter + 1 #only epoch
            runner.iter + 1
            log_vars["iter"] = runner.iter + 1
            # log_vars['lr'] = runner.current_lr()[0]
            if self.callback:
                self.callback(log_vars)
