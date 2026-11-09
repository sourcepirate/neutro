"""The shared training loop for :class:`neutro.models.Model`.

Keeping the loop in its own module makes ``base_model.py`` easier to read:
``Model`` handles graph construction and a single training step, while this
module handles epochs, batching, callbacks, validation, and progress bars.
"""

import numpy as np
from tqdm import tqdm

from ..autograd import GradientTape, Tensor as AutoTensor
from ..callbacks import History


def train_on_batch(model, x_batch, y_batch, use_autograd=False):
    """Run one forward/backward/optimizer step.

    Returns:
        ``(batch_loss, output)``.
    """
    if use_autograd:
        with GradientTape() as tape:
            for p in model.trainable_params:
                tape.watch(p)
            output = model.forward(x_batch, training=True)
            if isinstance(model.outputs, list) and len(model.outputs) > 1:
                if isinstance(y_batch, list):
                    loss = sum(
                        model.loss_fn(y_batch[j], output[j])
                        for j in range(len(model.outputs))
                    )
                else:
                    loss = sum(model.loss_fn(y_batch, o) for o in output)
            else:
                loss = model.loss_fn(y_batch, output)

        batch_loss = float(loss.data) if isinstance(loss, AutoTensor) else float(loss)
        tape.gradient(loss, model.trainable_params)
        model.optimizer.step(model._get_all_layers())
        model._zero_all_grads()
    else:
        output = model.forward(x_batch, training=True)
        is_mimo_out = model._is_mimo(model.outputs)
        batch_loss = model._compute_loss(y_batch, output)

        if is_mimo_out:
            grads = [
                model.loss_fn.gradient(y_batch[j], output[j])
                for j in range(len(model.outputs))
            ]
            model.backward(grads)
        else:
            model.backward(model.loss_fn.gradient(y_batch, output))

        model.optimizer.step(model._get_all_layers())

    return batch_loss, output


def update_validation_logs(model, logs, validation_data):
    """Evaluate on ``(val_x, val_y)`` and write ``val_*`` entries into logs."""
    val_x, val_y = validation_data
    val_output = model.predict(val_x)
    is_mimo_val_out = model._is_mimo(model.outputs)
    logs['val_loss'] = model._compute_loss(val_y, val_output)
    for m in model.metrics:
        m_val = model._eval_metric(m, val_y, val_output, is_mimo_val_out)
        logs[f'val_{m.get_name()}'] = m_val


def fit_loop(model, x, y=None, epochs=1, batch_size=32, verbose=1,
             validation_data=None, callbacks=None, use_autograd=False):
    """Run the epoch/batch training loop for ``model``.

    Args:
        model: A ``Model`` instance that has been compiled.
        x, y: Arrays, lists of arrays (MIMO), or a generator.
        epochs: Number of passes over the data.
        batch_size: Batch size for array inputs.
        verbose: ``0`` quiet, ``1`` progress bar, ``2`` one line per epoch.
        validation_data: Optional ``(val_x, val_y)`` pair.
        callbacks: Optional list of ``Callback`` instances.
        use_autograd: If true, use one whole-model ``GradientTape`` for each
            batch (``Model.autograd_fit``); otherwise call each layer's
            ``backward`` (``Model.fit``).

    Returns:
        A ``History`` with ``history['loss']`` and one list per metric.
    """
    is_mimo_x = isinstance(x, list)
    is_mimo_y = isinstance(y, list)

    use_generator = False
    if not is_mimo_x and hasattr(x, '__iter__') and not isinstance(x, np.ndarray):
        use_generator = True
        n_samples = len(x) * x.batch_size if hasattr(x, 'batch_size') else len(x)
    else:
        if is_mimo_x:
            n_samples = x[0].shape[0]
        else:
            n_samples = x.shape[0]

    history = History()
    history.set_model(model)

    all_callbacks = [history] + (callbacks or [])
    for cb in all_callbacks:
        cb.set_model(model)

    logs = {}
    for cb in all_callbacks:
        cb.on_train_begin(logs)

    for epoch in range(epochs):
        if model.stop_training:
            break
        for cb in all_callbacks:
            cb.on_epoch_begin(epoch, logs)

        epoch_loss = 0
        epoch_metrics = {m.get_name(): 0 for m in model.metrics}

        if use_generator:
            num_batches = len(x)
            data_iter = iter(x)
        else:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            if is_mimo_x:
                x_shuffled = [xi[indices] for xi in x]
            else:
                x_shuffled = x[indices]

            if is_mimo_y:
                y_shuffled = [yi[indices] for yi in y]
            else:
                y_shuffled = y[indices]

            num_batches = int(np.ceil(n_samples / batch_size))

        total_seen = 0
        pbar = None
        if verbose == 1:
            pbar = tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}")

        for i in range(num_batches):
            if use_generator:
                x_batch, y_batch = next(data_iter)
            else:
                start, end = i * batch_size, min((i + 1) * batch_size, n_samples)

                if is_mimo_x:
                    x_batch = [xi[start:end] for xi in x_shuffled]
                else:
                    x_batch = x_shuffled[start:end]

                if is_mimo_y:
                    y_batch = [yi[start:end] for yi in y_shuffled]
                else:
                    y_batch = y_shuffled[start:end]

            batch_size_actual = x_batch[0].shape[0] if is_mimo_x else len(x_batch)
            total_seen += batch_size_actual

            for cb in all_callbacks:
                cb.on_batch_begin(i, logs)

            batch_loss, output = model._train_on_batch(x_batch, y_batch, use_autograd)
            epoch_loss += batch_loss * batch_size_actual

            for m in model.metrics:
                m_val = model._eval_metric(
                    m, y_batch, output, model._is_mimo(model.outputs))
                epoch_metrics[m.get_name()] += m_val * batch_size_actual

            for cb in all_callbacks:
                cb.on_batch_end(i, logs)

            if pbar is not None:
                postfix = {'loss': f"{epoch_loss / total_seen:.4f}"}
                for m in model.metrics:
                    name = m.get_name()
                    postfix[name] = f"{epoch_metrics[name] / total_seen:.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        logs = {
            'loss': epoch_loss / total_seen,
            **{k: v / total_seen for k, v in epoch_metrics.items()}
        }

        if validation_data:
            model._update_validation_logs(logs, validation_data)

        for cb in all_callbacks:
            cb.on_epoch_end(epoch, logs)

        if verbose:
            if verbose == 1:
                if validation_data:
                    print(f" - val_loss: {logs['val_loss']:.4f}"
                          + model._metric_summary(logs, 'val_'))
            else:
                msg = (f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f}"
                       + model._metric_summary(logs))
                if validation_data:
                    msg += (f" - val_loss: {logs['val_loss']:.4f}"
                            + model._metric_summary(logs, 'val_'))
                print(msg)

    for cb in all_callbacks:
        cb.on_train_end(logs)
    return history
