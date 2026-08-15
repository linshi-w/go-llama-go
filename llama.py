import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import triton
import triton.language as tl
from safetensors.torch import load_file


@dataclasses.dataclass
class ModelConfig:
    head_dim: int

    hidden_size: int

    intermediate_size: int

    num_attention_heads: int

    num_hidden_layers: int

    num_key_value_heads: int

    rms_norm_eps: float

    rope_theta: float

    torch_dtype: str

    vocab_size: int

    use_qk_norm: bool = False


@triton.jit
def _rmsnorm_fwd(
    x_ptr,
    y_ptr,
    w_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    var = tl.sum(acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = x.to(tl.float32) * rstd * w.to(tl.float32)
        y = y.to(x.dtype)
        tl.store(y_ptr + cols, y, mask=mask)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def forward(self, input):
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])
        M, N = x.shape
        y = torch.empty_like(x)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _rmsnorm_fwd[(M,)](
            x,
            y,
            self.weight,
            x.stride(0),
            N,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)


@triton.jit
def _residual_rmsnorm_fwd(
    x_ptr,
    residual_ptr,
    w_ptr,
    out_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    residual_ptr += row * stride
    out_ptr += row * stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        r = tl.load(residual_ptr + cols, mask=mask, other=0.0)
        s = x.to(tl.float32) + r.to(tl.float32)
        tl.store(residual_ptr + cols, s.to(x.dtype), mask=mask)
        acc += s * s
    var = tl.sum(acc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        s = tl.load(residual_ptr + cols, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = s.to(tl.float32) * rstd * w.to(tl.float32)
        tl.store(out_ptr + cols, y.to(s.dtype), mask=mask)


def residual_rmsnorm(x, residual, weight, eps):
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    residual = residual.reshape(-1, orig_shape[-1])
    M, N = x.shape
    out = torch.empty_like(x)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _residual_rmsnorm_fwd[(M,)](
        x,
        residual,
        weight,
        out,
        x.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return residual.reshape(orig_shape), out.reshape(orig_shape)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _fused_swiglu_fwd(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    out_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_wn,
    stride_om,
    USE_FP8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        cols = k + offs_k
        k_mask = cols < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + cols[None, :],
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        gate_w = tl.load(
            gate_w_ptr + offs_n[None, :] * stride_wn + cols[:, None],
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        )
        up_w = tl.load(
            up_w_ptr + offs_n[None, :] * stride_wn + cols[:, None],
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        )
        if USE_FP8:
            x = x.to(tl.float8e4nv)
        gate_acc = tl.dot(x, gate_w, gate_acc)
        up_acc = tl.dot(x, up_w, up_acc)

    hidden = gate_acc * tl.sigmoid(gate_acc) * up_acc
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        hidden.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_swiglu_projection(x, gate_weight, up_weight):
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1]).contiguous()
    M, K = x.shape
    N = gate_weight.shape[0]
    gate_weight = gate_weight.contiguous()
    up_weight = up_weight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    use_fp8 = gate_weight.dtype == getattr(torch, "float8_e4m3fn", None)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _fused_swiglu_fwd[grid](
        x,
        gate_weight,
        up_weight,
        out,
        M=M,
        K=K,
        N=N,
        stride_xm=K,
        stride_wn=K,
        stride_om=N,
        USE_FP8=use_fp8,
    )
    return out.reshape(*orig_shape[:-1], N)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.register_buffer("gate_proj_fp8", None, persistent=False)
        self.register_buffer("up_proj_fp8", None, persistent=False)

    def _prepare_fp8(self, input):
        if self.gate_proj_fp8 is not None or not input.is_cuda:
            return
        if (8, 9) <= torch.cuda.get_device_capability(input.device) < (10, 0):
            self.gate_proj_fp8 = self.gate_proj.weight.detach().to(torch.float8_e4m3fn)
            self.up_proj_fp8 = self.up_proj.weight.detach().to(torch.float8_e4m3fn)

    def forward(self, input):
        self._prepare_fp8(input)
        gate_weight = (
            self.gate_proj_fp8 if self.gate_proj_fp8 is not None else self.gate_proj.weight
        )
        up_weight = (
            self.up_proj_fp8 if self.up_proj_fp8 is not None else self.up_proj.weight
        )
        hidden = fused_swiglu_projection(input, gate_weight, up_weight)
        return self.down_proj(hidden)


@triton.jit
def _rope_qk_fwd(
    query_ptr,
    key_ptr,
    sin_ptr,
    cos_ptr,
    out_q_ptr,
    out_k_ptr,
    query_rows,
    query_heads,
    key_heads,
    sequence_length,
    width: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
):
    combined_row = tl.program_id(0)
    offs = tl.arange(0, HALF_BLOCK)
    half = width // 2
    mask = offs < half

    is_query = combined_row < query_rows
    row = tl.where(is_query, combined_row, combined_row - query_rows)
    heads = tl.where(is_query, query_heads, key_heads)
    position = (row // heads) % sequence_length
    source = tl.where(is_query, query_ptr, key_ptr)
    destination = tl.where(is_query, out_q_ptr, out_k_ptr)

    base = row * width
    left = tl.load(source + base + offs, mask=mask, other=0.0).to(tl.float32)
    right = tl.load(source + base + half + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + position * half + offs, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + position * half + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        destination + base + offs,
        (left * cos - right * sin).to(query_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        destination + base + half + offs,
        (left * sin + right * cos).to(query_ptr.dtype.element_ty),
        mask=mask,
    )


def apply_rope_qk(query, key, sin_table, cos_table):
    query = query.contiguous()
    key = key.contiguous()
    out_q = torch.empty_like(query)
    out_k = torch.empty_like(key)
    query_rows = query.numel() // query.shape[-1]
    key_rows = key.numel() // key.shape[-1]
    _rope_qk_fwd[(query_rows + key_rows,)](
        query,
        key,
        sin_table,
        cos_table,
        out_q,
        out_k,
        query_rows,
        query.shape[2],
        key.shape[2],
        query.shape[1],
        width=query.shape[-1],
        HALF_BLOCK=triton.next_power_of_2(query.shape[-1] // 2),
    )
    return out_q, out_k


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    H_KV,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h * H_KV // H

    q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_base = K + off_z.to(tl.int64) * stride_kz + off_h_kv.to(tl.int64) * stride_kh
    v_base = V + off_z.to(tl.int64) * stride_vz + off_h_kv.to(tl.int64) * stride_vh
    o_base = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = (q * (sm_scale * 1.4426950408889634)).to(Q.dtype.element_ty)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        k_ptrs = (
            k_base
            + offs_d[:, None] * stride_kk
            + (start_n + offs_n[None, :]) * stride_kn
        )
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)

        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk = tl.where(
            offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf")
        )
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = (
            v_base
            + (start_n + offs_n[:, None]) * stride_vn
            + offs_d[None, :] * stride_vk
        )
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    l_i = 1.0 / l_i
    acc = acc * l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def _flash_attn_forward(query, key, value, sm_scale):
    Z, H, N_CTX, Dh = query.shape
    H_KV = key.shape[1]
    out = torch.empty_like(query)

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _attn_fwd[grid](
        query,
        key,
        value,
        out,
        sm_scale,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        H_KV,
        N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=Dh,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, sin_table, cos_table):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).permute(0, 2, 1, 3)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states, key_states = apply_rope_qk(
            query_states, key_states, sin_table, cos_table
        )
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)

        attn_output = _flash_attn_forward(
            query_states,
            key_states,
            value_states,
            1.0 / math.sqrt(self.head_dim),
        )

        return self.o_proj(
            attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        )


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, sin_table, cos_table):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), sin_table, cos_table
        )

        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states


@triton.jit
def _rope_tables_fwd(
    sin_ptr,
    cos_ptr,
    size,
    half_dim,
    freq_scale_log2,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    positions = offs // half_dim
    dims = offs % half_dim
    theta = tl.exp2(dims.to(tl.float32) * freq_scale_log2)
    angles = positions.to(tl.float32) * theta
    tl.store(sin_ptr + offs, tl.sin(angles), mask=mask)
    tl.store(cos_ptr + offs, tl.cos(angles), mask=mask)


def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    half_dim = emb_dim // 2
    sin_table = torch.empty((seq_len, half_dim), dtype=dtype, device=device)
    cos_table = torch.empty_like(sin_table)
    size = seq_len * half_dim
    freq_scale_log2 = -2.0 * math.log2(base) / emb_dim
    _rope_tables_fwd[(triton.cdiv(size, 256),)](
        sin_table,
        cos_table,
        size,
        half_dim,
        freq_scale_log2,
        BLOCK=256,
    )
    return sin_table, cos_table


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_hidden_layers = config.num_hidden_layers

        self.rms_norm_eps = config.rms_norm_eps

        self.rope_theta = config.rope_theta

        self.torch_dtype = config.torch_dtype

        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )

        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        seq_len = hidden_states.shape[1]

        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len,
            self.head_dim,
            base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype),
            device=input_ids.device,
        )

        attention_input = self.layers[0].input_layernorm(hidden_states)
        for i in range(self.num_hidden_layers):
            layer = self.layers[i]
            attention_output = layer.self_attn(attention_input, sin_table, cos_table)
            hidden_states, mlp_input = residual_rmsnorm(
                hidden_states,
                attention_output,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
            )
            mlp_output = layer.mlp(mlp_input)
            if i + 1 < self.num_hidden_layers:
                next_norm = self.layers[i + 1].input_layernorm
            else:
                next_norm = self.norm
            hidden_states, attention_input = residual_rmsnorm(
                hidden_states,
                mlp_output,
                next_norm.weight,
                next_norm.eps,
            )

        return attention_input


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = torch.argmax(logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(
            **{
                key: value
                for key, value in config.items()
                if key in ModelConfig.__annotations__
            }
        )

        state_dict = load_file(model_path / "model.safetensors")
        config.use_qk_norm = any("q_norm" in k for k in state_dict)

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)

        return model
