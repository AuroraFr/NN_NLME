import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array, Float
import jax.lax as lax
import jax.random as jr

# class Antibody_3dose_ode(eqx.Module):

#     def __call__(self, time, y0, args):
        
#         death_rate_S_cell, vaccine_autigen_decline_rate, Ab_production_init_acc, F2, F3, Ab_degrad_rate = args

#         S, Ab = y0
#         fold_change = 1.0 
#         second_dose = 30
#         third_dose = 250
        
#         values = jnp.select(
#                  [time < second_dose, (time >= second_dose) & (time < third_dose), time >= third_dose],
#                  [jnp.array([fold_change, time]),  jnp.array([F2, time-second_dose]), jnp.array([F3, time-third_dose])],
#                 default=jnp.array([1.0 ,time]))

#         dSdt =  1 * (values[0] * jnp.exp(-vaccine_autigen_decline_rate *  values[1]) - death_rate_S_cell * S)
#         dAbdt = 1 * (Ab_production_init_acc * S - Ab_degrad_rate * Ab)

#         return jnp.array([dSdt, dAbdt])

class MaskedAttentionPooling(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, hidden_dim, key):
        self.linear = eqx.nn.Linear(hidden_dim, 1, key=key)

    def __call__(self, y, mask):
        # y: (T, H), mask: (T,) or (T, 1)
        print("attention pooling", y.shape, mask.shape)
        logits = jax.vmap(self.linear)(y).squeeze(-1)              # (T,)
        logits = jnp.where(mask > 0, logits, -1e9)   # Masked attention
        weights = jax.nn.softmax(logits, axis=0)     # (T,)
        pooled = jnp.sum(weights[:, None] * y, axis=0)  # (H,)
        return pooled

import equinox as eqx
import jax
import jax.numpy as jnp


class GRUDCell(eqx.Module):
    """Single-subject GRU-D cell: time-decay of hidden state + optional input decay."""
    gru: eqx.nn.GRUCell
    lin_gamma_x: eqx.nn.Linear
    lin_gamma_h: eqx.nn.Linear
    x_mean: jnp.ndarray  # (D,)

    def __init__(self, input_dim: int, hidden_dim: int, x_mean, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.gru = eqx.nn.GRUCell(input_dim, hidden_dim, key=k1)
        self.lin_gamma_x = eqx.nn.Linear(1, input_dim, key=k2)   # Δt -> D
        self.lin_gamma_h = eqx.nn.Linear(1, hidden_dim, key=k3)  # Δt -> H
        self.x_mean = jnp.asarray(x_mean)                        # (D,)

    def __call__(self, carry, inputs):
        """
        carry:  (h_prev, x_last)
          - h_prev: (H,)
          - x_last: (D,)

        inputs: (x_t, delta_t)
          - x_t:     (D,)
          - delta_t: scalar or (1,)
        """
        h_prev, x_last = carry
        x_t, delta_t = inputs

        # shape (1,) for Linear(1 -> *)
        if delta_t.ndim == 0:
            delta_t = delta_t[None]

        # gamma_x, gamma_h in (0,1), depend on Δt
        gamma_x = jnp.exp(-jax.nn.relu(self.lin_gamma_x(delta_t))).squeeze(0)  # (D,)
        gamma_h = jnp.exp(-jax.nn.relu(self.lin_gamma_h(delta_t))).squeeze(0)  # (H,)

        # Decay last input toward mean (you can skip if you only care about hidden decay)
        x_decay = gamma_x * x_last + (1.0 - gamma_x) * self.x_mean
        # If no missingness, just use the observed x_t; if you later add a mask m_t:
        # x_imputed = m_t * x_t + (1.0 - m_t) * x_decay
        x_imputed = x_t

        # Decay hidden state
        h_decay = gamma_h * h_prev

        # Standard GRU on (x_imputed, decayed hidden)
        h_new = self.gru(x_imputed, h_decay)

        # New "last input" is the imputed input
        x_last_new = x_imputed

        new_carry = (h_new, x_last_new)
        # We also return h_new as the per-step output
        return new_carry, h_new


class Encoder(eqx.Module):

    use_batchnorm:bool
    use_conv:bool
    use_rnn:bool
    use_dropout:bool
    use_data_aug:bool
    use_mask: bool
    conv:eqx.nn.Conv1d
    norm: eqx.nn.BatchNorm
    linear: eqx.nn.Linear
    mean: eqx.nn.Linear
    logstd: eqx.nn.Linear
    attn_pool: MaskedAttentionPooling
    cell: eqx.nn.GRUCell
    hidden_dim: int
    input_dim: int
    dropout_prob: float
    data_aug_method:str

    def __init__(self, latent_shape, input_dim, hidden_dim, key, variance_init_bias=-10, 
                 conv=True, input_channel=1, out_channel=1, kernel_size=3, use_mask=False, rnn=False, batchnorm=False, dropout=False, dropout_prob=0.2,
                 data_aug=False, data_aug_method=None, padding=0, irregular=True):
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        # seed = 42
        # key = jax.random.key(seed)
        key1, key2, key3, key4, key_attn = jax.random.split(key,5)

        self.use_batchnorm = batchnorm
        self.use_rnn = rnn
        self.use_conv = conv
        self.use_data_aug = data_aug
        self.use_dropout = dropout
        self.dropout_prob = dropout_prob
        self.data_aug_method = data_aug_method
        self.use_mask = use_mask
        
        if conv:
            #1D Convolution Output Length:
            # output_length = floor((input_length + 2 * padding - kernel_size) / stride) + 1
            output_length = input_dim - kernel_size + 1 
            if input_channel > 1:
                self.conv = eqx.nn.Conv1d(input_channel, out_channel, kernel_size=3, key=key1)
            else:
                self.conv = eqx.nn.Conv1d(input_channel, out_channel, kernel_size=kernel_size, padding=padding, key=key1)
            # if output_length < self.hidden_dim and (not use_mask) and (not irregular):
            #     self.linear = None
            #     self.mean = eqx.nn.Linear(output_length, latent_shape, key=key3)
            #     self.logstd = eqx.nn.Linear(output_length, latent_shape, key=key4, use_bias=True)
            # else:
            if (not use_mask) and (not irregular):
                self.linear = eqx.nn.Linear(output_length, self.hidden_dim, key=key2)
                self.attn_pool = None
            elif irregular and (not use_mask):
                self.attn_pool = MaskedAttentionPooling(hidden_dim=out_channel, key=key_attn)
                self.linear = eqx.nn.Linear(out_channel, self.hidden_dim, key=key2)
            elif use_mask:
                self.attn_pool = MaskedAttentionPooling(hidden_dim=out_channel, key=key_attn)
                self.linear = eqx.nn.Linear(out_channel, self.hidden_dim, key=key2)
                    
            self.mean = eqx.nn.Linear(self.hidden_dim, latent_shape, key=key3)
            self.logstd = eqx.nn.Linear(self.hidden_dim, latent_shape, key=key4, use_bias=True)
            self.cell = None
        elif rnn:
            self.attn_pool = None
            if data_aug:
                self.cell = eqx.nn.GRUCell(2, self.hidden_dim, key=key1)
            else:
                self.cell = eqx.nn.GRUCell(input_channel, self.hidden_dim, key=key1)

            self.linear = eqx.nn.Linear(self.hidden_dim, self.hidden_dim, key=key2)
            self.conv = None
            self.mean = eqx.nn.Linear(self.hidden_dim, latent_shape, key=key3)
            self.logstd = eqx.nn.Linear(self.hidden_dim, latent_shape, key=key4, use_bias=True)
        else:
            self.linear = eqx.nn.Linear(self.input_dim, self.hidden_dim, key=key2)
            self.mean = eqx.nn.Linear(self.hidden_dim, latent_shape, key=key3)
            self.logstd = eqx.nn.Linear(self.hidden_dim, latent_shape, key=key4, use_bias=True)
            self.conv = None
            self.cell = None
            self.attn_pool = None
        
        if batchnorm:
            self.norm = eqx.nn.BatchNorm(input_size=1, axis_name="batch")
        else:
            self.norm = None
        
        wkey1, _ = jax.random.split(key, 2)
        custom_std_fc_layer = 0.001
        
        linear_weight = lambda l: l.weight
        out, in_ = self.logstd.weight.shape

        mean_new_weight = jax.random.normal(wkey1, (out, in_)) * custom_std_fc_layer
        self.mean = eqx.tree_at(linear_weight, self.mean, mean_new_weight)
        logvar_new_weight = jnp.zeros((out, in_))
        self.logstd = eqx.tree_at(linear_weight, self.logstd, logvar_new_weight)

        linear_bias = lambda l : l.bias
        new_bias = jnp.full(latent_shape, variance_init_bias, dtype=float) 
        self.logstd = eqx.tree_at(linear_bias, self.logstd, new_bias)

    
    def __call__(self, y, mask=None, key=None, state=None) -> Float[Array, "2"]:

        if self.use_conv:
            if self.use_data_aug:
                if self.data_aug_method == 'diff':
                    diff = jnp.diff(y)
                    diff = jnp.insert(diff, 0,0)
                    y = jnp.hstack((y, diff))
                y = y.reshape(self.conv.in_channels, -1, self.input_dim)

            else:
                y = y.reshape(self.conv.in_channels, -1)
                y = self.conv(y)
                if self.use_mask:
                    y = y.T
                    print('attention pooling with mask')
                    y = self.attn_pool(y, mask)  # shape: (C_out,)
                    key = jr.split(key, 2)[0]
                elif self.attn_pool:
                    mask = jnp.ones((y.shape[1],))
                    y = y.T
                    print("print('attention pooling without mask')", mask.shape)
                    y = self.attn_pool(y, mask)
                    print('after mask', y.shape)
                else:
                    y = jnp.ravel(y)

        elif self.use_rnn:
            if self.use_data_aug and self.data_aug_method == 'diff':
                diff = jnp.diff(y)
                diff = jnp.insert(diff, 0,0)
                y = jnp.hstack((y, diff)).reshape(self.input_dim, 2)
            else:
                y = y.reshape(self.input_dim, -1)
            
            # #2layers GRU
            # h1_0 = jnp.zeros((self.hidden_dim,))
            # h2_0 = jnp.zeros((self.hidden_dim,))
            # init_carry = (h1_0, h2_0)
            # def f(carry, inp):
            #     h1, h2 = carry
            #     # first "layer"
            #     h1_new = self.cell(inp, h1)
            #     # second "layer" with the SAME GRU parameters
            #     h2_new = self.cell(h1_new, h2)
            #     return (h1_new, h2_new), None
            # (h1_T, h2_T), _ = lax.scan(f, init_carry, y)
            # y = h2_T.squeeze()

            hidden = jnp.zeros((self.hidden_dim,))

            def f(carry, inp):
                return self.cell(inp, carry), None
            
            y, _ = lax.scan(f, hidden, y)
            
            y = y.squeeze()
            print("after GRU", y.shape)

        if self.linear:
            y = self.linear(y)
        
        if self.use_batchnorm:

            y = y.reshape(1, -1)
            y, state = self.norm(y, state)
            y = jnp.ravel(y)

        y = jax.nn.gelu(y)

        if self.use_dropout:
            key, subkey = jr.split(key, 2)
            y = eqx.nn.Dropout(self.dropout_prob)(y, key=subkey)
        
        mean = self.mean(y)
        post_std = self.logstd(y)

        return mean, post_std, state