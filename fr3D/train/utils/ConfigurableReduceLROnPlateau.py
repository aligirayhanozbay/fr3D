import tensorflow as tf

class ConfigurableReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, *args, optimizer_name='optimizer', **kwargs):
        self.optimizer_name=optimizer_name
        super().__init__(*args, **kwargs)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optim = getattr(self.model, self.optimizer_name)
        logs['lr'] = backend.get_value(optim.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(optim.lr)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        backend.set_value(optim.lr, new_lr)
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f'\nEpoch {epoch +1}: '
                                f'ReduceLROnPlateau reducing learning rate to {new_lr}.')
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
