import numpy as np
from .tensor import Tensor
from .function import Function, _Ctx
from .utils import broadcast_backward
from .tape import get_active_tape


class Gather(Function):
    @staticmethod
    def forward(ctx, table, indices):
        ctx.save_for_backward(indices=indices.astype(int), table_shape=table.shape)
        return table[indices.astype(int)]

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_data['indices']
        table_grad = np.zeros(ctx.saved_data['table_shape'])
        np.add.at(table_grad, indices, grad_output)
        return table_grad, None


class QuantizeSTE(Function):
    @staticmethod
    def forward(ctx, x, bits):
        Q_b = 2 ** (bits - 1)
        abs_max = np.max(np.abs(x))
        if abs_max < 1e-15:
            abs_max = 1e-15
        scale = abs_max / Q_b
        x_scaled = x / scale if scale > 0 else x
        x_quant = np.clip(np.round(x_scaled), -Q_b, Q_b - 1)
        return x_quant * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Conv2DFunction(Function):
    @staticmethod
    def forward(ctx, inputs, W, b, kernel_size, strides, padding):
        from neutro.utils.conv_utils import im2col_indices
        batch, h, w, c = inputs.shape
        kh, kw = kernel_size
        sh, sw = strides
        pad = 0 if padding == 'valid' else (kh - 1) // 2

        x = inputs.transpose(0, 3, 1, 2)
        x_cols = im2col_indices(x, kh, kw, padding=pad, stride=sh)
        f = W.shape[-1]
        W_reshaped = W.reshape(f, -1)
        oh = (h + 2 * pad - kh) // sh + 1
        ow = (w + 2 * pad - kw) // sw + 1

        out = W_reshaped @ x_cols
        if b is not None:
            out = out + b.reshape(-1, 1)
        out = out.reshape(f, oh, ow, batch).transpose(3, 1, 2, 0)

        ctx.save_for_backward(
            x_shape=inputs.shape, x_cols=x_cols, W=W.copy(),
            kh=kh, kw=kw, sh=sh, sw=sw, pad=pad, c=c, f=f, h=h, w=w
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from neutro.utils.conv_utils import col2im_indices
        x_shape = ctx.saved_data['x_shape']
        x_cols = ctx.saved_data['x_cols']
        W = ctx.saved_data['W']
        kh, kw = ctx.saved_data['kh'], ctx.saved_data['kw']
        sh, sw = ctx.saved_data['sh'], ctx.saved_data['sw']
        pad = ctx.saved_data['pad']
        c, f = ctx.saved_data['c'], ctx.saved_data['f']
        h, w = ctx.saved_data['h'], ctx.saved_data['w']

        dout = grad_output.transpose(0, 3, 1, 2).reshape(f, -1)
        dW = dout @ x_cols.T
        dW = dW.reshape(f, c, kh, kw).transpose(2, 3, 1, 0)
        db = np.sum(grad_output, axis=(0, 1, 2))

        W_reshaped = W.transpose(3, 2, 0, 1).reshape(f, -1)
        dx_cols = W_reshaped.T @ dout
        batch = x_shape[0]
        dx = col2im_indices(dx_cols, (batch, c, h, w), kh, kw, padding=pad, stride=sh)
        dx = dx.transpose(0, 2, 3, 1)

        return dx, dW, db, None, None, None


class MaxPool2DFunction(Function):
    @staticmethod
    def forward(ctx, inputs, pool_size, strides):
        from neutro.utils.conv_utils import im2col_indices
        batch, h, w, c = inputs.shape
        ph, pw = pool_size
        sh, sw = strides
        oh = (h - ph) // sh + 1
        ow = (w - pw) // sw + 1

        x = inputs.transpose(0, 3, 1, 2).reshape(-1, 1, h, w)
        x_cols = im2col_indices(x, ph, pw, padding=0, stride=sh)
        arg_max = np.argmax(x_cols, axis=0)
        out = x_cols[arg_max, np.arange(arg_max.size)]
        out = out.reshape(oh, ow, batch, c).transpose(2, 0, 1, 3)

        ctx.save_for_backward(arg_max=arg_max, x_cols_shape=x_cols.shape,
                              oh=oh, ow=ow, batch=batch, c=c, h=h, w=w,
                              ph=ph, pw=pw, sh=sh, sw=sw)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from neutro.utils.conv_utils import col2im_indices
        arg_max = ctx.saved_data['arg_max']
        x_cols_shape = ctx.saved_data['x_cols_shape']
        oh, ow = ctx.saved_data['oh'], ctx.saved_data['ow']
        batch, c = ctx.saved_data['batch'], ctx.saved_data['c']
        h, w = ctx.saved_data['h'], ctx.saved_data['w']
        ph, pw = ctx.saved_data['ph'], ctx.saved_data['pw']
        sh, sw = ctx.saved_data['sh'], ctx.saved_data['sw']

        dout = grad_output.transpose(1, 2, 0, 3).flatten()
        dx_cols = np.zeros(x_cols_shape)
        dx_cols[arg_max, np.arange(arg_max.size)] = dout
        dx = col2im_indices(dx_cols, (batch * c, 1, h, w), ph, pw, padding=0, stride=sh)
        dx = dx.reshape(batch, c, h, w).transpose(0, 2, 3, 1)
        return dx, None, None


class Conv1DFunction(Function):
    @staticmethod
    def forward(ctx, inputs, W, b, kernel_size, strides, padding):
        from neutro.utils.conv_utils import im2col_indices
        batch, steps, c = inputs.shape
        k = kernel_size[0]
        s = strides[0]
        f = W.shape[-1]
        pad = 0 if padding == 'valid' else (k - 1) // 2

        x = inputs[:, :, None, :].transpose(0, 3, 1, 2)
        x_cols = im2col_indices(x, k, 1, padding=(pad, 0), stride=(s, 1))
        W_reshaped = W[:, None, :, :].transpose(3, 2, 0, 1).reshape(f, -1)
        out_steps = (steps + 2 * pad - k) // s + 1

        out = W_reshaped @ x_cols
        if b is not None:
            out = out + b.reshape(-1, 1)
        out = out.reshape(f, out_steps, 1, batch).transpose(3, 1, 2, 0).squeeze(2)

        ctx.save_for_backward(
            x_cols=x_cols, W=W.copy(), k=k, c=c, f=f, s=s,
            batch=batch, steps=steps, pad=pad, kernel_size=kernel_size, strides=strides
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from neutro.utils.conv_utils import col2im_indices
        x_cols = ctx.saved_data['x_cols']
        W = ctx.saved_data['W']
        k, c, f = ctx.saved_data['k'], ctx.saved_data['c'], ctx.saved_data['f']
        s = ctx.saved_data['s']
        batch, steps = ctx.saved_data['batch'], ctx.saved_data['steps']
        pad = ctx.saved_data['pad']

        dout_4d = grad_output[:, :, None, :]
        dout = dout_4d.transpose(3, 1, 2, 0).reshape(f, -1)
        dW = dout @ x_cols.T
        dW = dW.reshape(f, c, k, 1).transpose(2, 3, 1, 0).squeeze(1)
        db = np.sum(grad_output, axis=(0, 1))

        W_reshaped = W[:, None, :, :].transpose(3, 2, 0, 1).reshape(f, -1)
        dx_cols = W_reshaped.T @ dout
        dx = col2im_indices(dx_cols, (batch, c, steps, 1), k, 1, padding=(pad, 0), stride=(s, 1))
        dx = dx.transpose(0, 2, 3, 1).squeeze(2)
        return dx, dW, db, None, None, None

