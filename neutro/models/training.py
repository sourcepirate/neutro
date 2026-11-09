"""The shared training loop for :class:`neutro.models.Model`."""

import numpy as np
from tqdm import tqdm

from ..autograd import GradientTape, Tensor as AutoTensor
from ..callbacks import History


def train_on_batch(model, x_batch, y_batch, use_autograd=False):
    """Run one forward/backward/optimizer step."""
    if use_autograd:
        with GradientTape() as tape:
            for param in model.trainable_params:
                tape.watch(param)
            output = model.forward(x_batch, training=True)
            loss = _autograd_loss(model, y_batch, output)

        batch_loss = float(loss.data) if isinstance(loss, AutoTensor) else float(loss)
        tape.gradient(loss, model.trainable_params)
        model.optimizer.step(model._get_all_layers())
        model._zero_all_grads()
        return batch_loss, output

    output = model.forward(x_batch, training=True)
    batch_loss = model._compute_loss(y_batch, output)

    if model._is_mimo(model.outputs):
        grads = [
            model.loss_fn.gradient(y_batch[index], output[index])
            for index in range(len(model.outputs))
        ]
        model.backward(grads)
    else:
        model.backward(model.loss_fn.gradient(y_batch, output))

    model.optimizer.step(model._get_all_layers())
    return batch_loss, output


def _autograd_loss(model, y_batch, output):
    if not isinstance(model.outputs, list) or len(model.outputs) <= 1:
        return model.loss_fn(y_batch, output)
    if isinstance(y_batch, list):
        return sum(
            model.loss_fn(y_batch[index], output[index])
            for index in range(len(model.outputs))
        )
    return sum(model.loss_fn(y_batch, item) for item in output)


def update_validation_logs(model, logs, validation_data):
    """Evaluate on ``(val_x, val_y)`` and write ``val_*`` entries into logs."""
    val_x, val_y = validation_data
    val_output = model.predict(val_x)
    is_mimo = model._is_mimo(model.outputs)
    logs['val_loss'] = model._compute_loss(val_y, val_output)
    for metric in model.metrics:
        logs[f'val_{metric.get_name()}'] = model._eval_metric(
            metric, val_y, val_output, is_mimo
        )


def _uses_generator(x):
    return (
        not isinstance(x, list)
        and hasattr(x, '__iter__')
        and not isinstance(x, np.ndarray)
    )


def _sample_count(x, is_mimo):
    return x[0].shape[0] if is_mimo else x.shape[0]


def _shuffle(x, y, is_mimo_x, is_mimo_y, count):
    indices = np.arange(count)
    np.random.shuffle(indices)
    x_shuffled = [item[indices] for item in x] if is_mimo_x else x[indices]
    y_shuffled = [item[indices] for item in y] if is_mimo_y else y[indices]
    return x_shuffled, y_shuffled


def _batch(values, start, end, is_mimo):
    return [item[start:end] for item in values] if is_mimo else values[start:end]


def _run_epoch(model, x, y, is_mimo_x, is_mimo_y, use_generator,
               n_samples, batch_size, use_autograd, callbacks, logs,
               verbose, epoch, epochs):
    epoch_loss = 0.0
    epoch_metrics = {metric.get_name(): 0.0 for metric in model.metrics}
    total_seen = 0

    if use_generator:
        num_batches = len(x)
        data_iter = iter(x)
        x_shuffled = y_shuffled = None
    else:
        x_shuffled, y_shuffled = _shuffle(x, y, is_mimo_x, is_mimo_y, n_samples)
        num_batches = int(np.ceil(n_samples / batch_size))

    pbar = tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}") if verbose == 1 else None

    for index in range(num_batches):
        if use_generator:
            x_batch, y_batch = next(data_iter)
        else:
            start = index * batch_size
            end = min((index + 1) * batch_size, n_samples)
            x_batch = _batch(x_shuffled, start, end, is_mimo_x)
            y_batch = _batch(y_shuffled, start, end, is_mimo_y)

        batch_size_actual = x_batch[0].shape[0] if is_mimo_x else len(x_batch)
        total_seen += batch_size_actual

        for callback in callbacks:
            callback.on_batch_begin(index, logs)

        batch_loss, output = train_on_batch(model, x_batch, y_batch, use_autograd)
        epoch_loss += batch_loss * batch_size_actual

        for metric in model.metrics:
            value = model._eval_metric(metric, y_batch, output, model._is_mimo(model.outputs))
            epoch_metrics[metric.get_name()] += value * batch_size_actual

        for callback in callbacks:
            callback.on_batch_end(index, logs)

        if pbar is not None:
            pbar.set_postfix(_progress(epoch_loss, epoch_metrics, total_seen))
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return epoch_loss, epoch_metrics, total_seen


def _progress(loss, metrics, seen):
    values = {'loss': f"{loss / seen:.4f}"}
    values.update({name: f"{value / seen:.4f}" for name, value in metrics.items()})
    return values


def _epoch_logs(loss, metrics, seen):
    return {'loss': loss / seen, **{name: value / seen for name, value in metrics.items()}}


def _print_epoch(model, logs, verbose, epoch, epochs, has_validation):
    if verbose == 0:
        return
    if verbose == 1:
        if has_validation:
            print(f" - val_loss: {logs['val_loss']:.4f}" + model._metric_summary(logs, 'val_'))
        return

    message = f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f}" + model._metric_summary(logs)
    if has_validation:
        message += f" - val_loss: {logs['val_loss']:.4f}" + model._metric_summary(logs, 'val_')
    print(message)


def fit_loop(model, x, y=None, epochs=1, batch_size=32, verbose=1,
             validation_data=None, callbacks=None, use_autograd=False):
    """Run the epoch/batch training loop for ``model``."""
    is_mimo_x = isinstance(x, list)
    is_mimo_y = isinstance(y, list)
    use_generator = _uses_generator(x)
    n_samples = None if use_generator else _sample_count(x, is_mimo_x)

    history = History()
    all_callbacks = [history] + (callbacks or [])
    for callback in all_callbacks:
        callback.set_model(model)

    logs = {}
    for callback in all_callbacks:
        callback.on_train_begin(logs)

    for epoch in range(epochs):
        if model.stop_training:
            break

        for callback in all_callbacks:
            callback.on_epoch_begin(epoch, logs)

        epoch_loss, epoch_metrics, seen = _run_epoch(
            model, x, y, is_mimo_x, is_mimo_y, use_generator,
            n_samples, batch_size, use_autograd, all_callbacks, logs,
            verbose, epoch, epochs,
        )
        logs = _epoch_logs(epoch_loss, epoch_metrics, seen)

        if validation_data:
            update_validation_logs(model, logs, validation_data)

        for callback in all_callbacks:
            callback.on_epoch_end(epoch, logs)

        _print_epoch(model, logs, verbose, epoch, epochs, bool(validation_data))

    for callback in all_callbacks:
        callback.on_train_end(logs)
    return history
