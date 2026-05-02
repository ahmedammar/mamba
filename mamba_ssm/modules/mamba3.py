# Copyright (c) 2026, Dao AI Lab, Goombalab.

import math
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, rms_norm_ref

try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
except ImportError:
    mamba3_mimo_combined = None

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import apply_rotary_qk_inference_fwd

try:
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
except ImportError:
    mamba3_step_fn = None


def _apply_rotary_qk_inference_cpu(q, k, angle_state, angle_proj, dt, bias_q, bias_k, rotate_pairwise):
    """Pure-PyTorch port of apply_rotary_qk_inference_fwd for CPU/MPS.

    Mirrors the triton kernel's contract: returns (q_rot, k_rot, new_angle_state)
    with the rotary portion rotated by the cumulative data-dependent RoPE angle
    and any pass-through portion left untouched.
    """
    batch, R, nheads, d = q.shape
    rotary_dim = angle_state.shape[-1] * 2

    new_angle = (
        angle_state.to(torch.float32)
        + torch.tanh(angle_proj.to(torch.float32))
        * dt.to(torch.float32)[:, :, None]
        * math.pi
    )
    angle_b = new_angle.unsqueeze(1).expand(-1, R, -1, -1)
    cos = torch.cos(angle_b)
    sin = torch.sin(angle_b)

    if bias_q is not None:
        q = q + bias_q[None]
    if bias_k is not None:
        k = k + bias_k[None]

    if rotate_pairwise:
        q_rot = q[..., :rotary_dim].to(torch.float32)
        q_pass = q[..., rotary_dim:].to(torch.float32)
        k_rot = k[..., :rotary_dim].to(torch.float32)
        k_pass = k[..., rotary_dim:].to(torch.float32)

        q_rot = q_rot.reshape(batch, R, nheads, rotary_dim // 2, 2)
        q0, q1 = q_rot[..., 0], q_rot[..., 1]
        k_rot = k_rot.reshape(batch, R, nheads, rotary_dim // 2, 2)
        k0, k1 = k_rot[..., 0], k_rot[..., 1]
        qor = torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1).reshape(batch, R, nheads, rotary_dim)
        kor = torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1).reshape(batch, R, nheads, rotary_dim)

        if rotary_dim < d:
            q_out = torch.cat([qor, q_pass], dim=-1)
            k_out = torch.cat([kor, k_pass], dim=-1)
        else:
            q_out, k_out = qor, kor
    else:
        half = d // 2
        rdim_half = rotary_dim // 2
        if half > rdim_half:
            pad = list(cos.shape)
            pad[-1] = half - rdim_half
            cos = torch.cat([cos, torch.ones(pad, device=cos.device, dtype=cos.dtype)], dim=-1)
            sin = torch.cat([sin, torch.zeros(pad, device=sin.device, dtype=sin.dtype)], dim=-1)
        q0, q1 = q[..., :half].to(torch.float32), q[..., half:].to(torch.float32)
        k0, k1 = k[..., :half].to(torch.float32), k[..., half:].to(torch.float32)
        q_out = torch.cat([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1)
        k_out = torch.cat([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1)

    return q_out.to(q.dtype), k_out.to(k.dtype), new_angle.to(angle_state.dtype)

class Mamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        # ----------------------------------------
        # Mamba-3 configs
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        #-------------------------------------------
        # Fused kernel and sharding options
        chunk_size=64, # Recommended: 64 for SISO, 64/mimo_rank for MIMO
        dropout=0.0,  # Just to absorb the kwarg
        layer_idx=None,  # Absorb kwarg for general module
        n_layer=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm=is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        if not self.is_mimo:
            self.mimo_rank = 1
        else:
            assert mamba3_mimo_combined is not None, "Fails to import Mamba-3 MIMO kernels. Please ensure you installed the necessary dependencies, such as TileLang."

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        
        # RoPE flags
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2/rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads * self.mimo_rank + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt_bias parameterization        
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True
        
        # B and C biases
        self.B_bias = nn.Parameter(1+torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
        self.C_bias = nn.Parameter(1+torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
                                                       
        # RMS Norm for B and C
        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        if self.is_mimo:
            # Initialize up/down MIMO projection (for x and z)
            mimo_x_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            mimo_z_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
            mimo_o_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank

            self.mimo_x = nn.Parameter(mimo_x_init_weights, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init_weights, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init_weights, requires_grad=True)
    
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                **factory_kwargs
            )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)


    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        angle_dt_state, ssm_state, k_state, v_state  = None, None, None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            angle_dt_state, ssm_state, k_state, v_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _, _, _ = self.step(u, angle_dt_state, ssm_state, k_state, v_state)
                return out

        # Apply in_proj
        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner, 
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads, 
                self.num_rope_angles
            ],
            dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        # Compute ADT, DT
        _A = -F.softplus(dd_A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)            
        DT = F.softplus(dd_dt + self.dt_bias) # (B, L, N)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Compute angle — cast to float32 as required by the MIMO/SISO kernels
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32) # (B, L, N, S)

        # Apply RMS Norm on B and C
        B = self.B_norm(B)
        C = self.C_norm(C)
        
        # Apply Mamba-3 kernel
        if self.is_mimo:
            y = mamba3_mimo_combined(
                Q=C,
                K=B,
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o if not self.is_outproj_norm else None,
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=ssm_state is not None,
                cu_seqlens=cu_seqlens,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k, last_v, *rest = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_state.copy_(last_k)
                v_state.copy_(last_v)
            if self.is_outproj_norm:
                z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z = rearrange(z, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z)
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=None,
                return_final_states=ssm_state is not None,
                cu_seqlens=cu_seqlens,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k, last_v, *rest = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_state.copy_(last_k.unsqueeze(1))
                v_state.copy_(last_v)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z)
        
        out = self.out_proj(y.to(x.dtype))
        return out
    

    def _preprocess(self, A_proj, dd_dt, B, C, x, z, trap_proj, angle_proj):
        _A = -F.softplus(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        rank = self.mimo_rank if self.is_mimo else 1
        B = rearrange(B, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)

        B = self.B_norm(B)
        C = self.C_norm(C)

        B = B.expand(-1, -1, self.nheads, -1) # (B, R, N, S)
        C = C.expand(-1, -1, self.nheads, -1) # (B, R, N, S)
    
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, trap, _A, angles

    def _postprocess(self, y, outpj, z, zpj, headdim):
        # y: (batch, R, H, D) — apply mimo_z to z, then norm, then mimo_o
        z_r = torch.einsum("bhp,rhp->brhp", z.float(), zpj)  # (batch, R, H, D)
        z_r = rearrange(z_r, "b r h p -> b r (h p)")
        y = rearrange(y, "b r h p -> b r (h p)").float()
        y = self.norm(y, z_r)
        y = rearrange(y, "b r (h p) -> b r h p", p=headdim)
        y = torch.einsum("brhp,rhp->bhp", y, outpj)  # (batch, H, D)
        return y

    def step(self, u, angle_state, ssm_state, k_state, v_state, **kwargs):
        """
        Decode function using CuteDSL kernel from mamba3_step_fn.py.
        Also modify the state vars in-place for the next step.

        NOTE: Only tested on H100. Compatibility with other hardware
        will be made available in the future. Non-CUDA devices (CPU, MPS)
        are dispatched to a pure-PyTorch reference implementation.

        Args:
            u: (batch, d_model)
            angle_state: (batch, nheads, num_rope_angles)
            ssm_state: (batch, nheads, headdim, d_state)
            k_state: (batch, R, nheads, d_state), where R = mimo_rank (R=1 if not MIMO)
            v_state: (batch, nheads, headdim)
            **kwargs: ignored
        Returns:
            out: (batch, d_model)
            nxt_angle_state: (batch, nheads, num_rope_angles)
            state_out: (batch, nheads, headdim, d_state)
            nxt_k_state: (batch, R, nheads, d_state), where R = mimo_rank (R=1 if not MIMO)
            nxt_v_state: (batch, nheads, headdim)
        """
        if u.device.type != "cuda":
            return self._step_cpu(u, angle_state, ssm_state, k_state, v_state)

        assert mamba3_step_fn is not None, "Cute Mamba-3 step function is not available. Please ensure you installed the necessary dependencies, such as nvidia-cutlass-dsl and quack-kernels."

        # in_proj
        zxBCdt = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1)

        DT, B, C, x, z, trap, A, angles = self._preprocess(
            dd_A, dd_dt, B, C, x, z, trap, angles)

        bias_q = rearrange(self.C_bias, "h r n -> r h n")
        bias_k = rearrange(self.B_bias, "h r n -> r h n")

        # NOTE: MIMO calls the Tilelang kernel, 
        # which permute the blockwise rotation matrix so that
        # the i-th entry is paired with the i+N//2-th entry:
        rotate_pairwise = not self.is_mimo
        C, B, nxt_angle_state = apply_rotary_qk_inference_fwd(
            q=C, k=B, angle_state=angle_state, 
            angle_proj=angles, dt=DT, bias_q=bias_q, bias_k=bias_k, 
            conjugate=False, inplace=False, # NOTE: inplace is incompatible with self.nheads != self.num_bc_heads
            rotate_pairwise=rotate_pairwise)

        nxt_v_state = x
        nxt_k_state = B

        if self.is_mimo:
            xpj = rearrange(self.mimo_x, "h r p -> r h p", p=self.headdim).contiguous()
            zpj = rearrange(self.mimo_z, "h r p -> r h p", p=self.headdim).contiguous()
            outpj = rearrange(self.mimo_o, "h r p -> r h p", p=self.headdim).contiguous()
        else:
            xpj = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)
            zpj = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=z.device, dtype=z.dtype)
            outpj = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)

        if self.is_outproj_norm:
            batch = x.shape[0]
            y = torch.empty(batch, self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)
            mamba3_step_fn(
                ssm_state,
                k_state,
                v_state,
                A,
                B,
                C,
                self.D,
                x,
                DT,
                trap,
                xpj,
                outproj=None,
                state_out=None, # can be not in place if pass in state_out
                out=y,
                z=None,
                zproj=None,
                tile_D=64,
                num_warps=4,
            )
            y = self._postprocess(y, outpj, z, zpj, self.headdim)
        else:
            y = torch.empty_like(x)
            mamba3_step_fn(
                ssm_state,
                k_state,
                v_state,
                A,
                B,
                C,
                self.D,
                x,
                DT,
                trap,
                xpj,
                outproj=outpj,
                state_out=None, # can be not in place if pass in state_out
                out=y,
                z=z,
                zproj=zpj,
                tile_D=64,
                num_warps=4,
            )

        # out_proj
        out = rearrange(y, "b h p -> b (h p)")
        out = self.out_proj(out.to(x.dtype))

        angle_state.copy_(nxt_angle_state)
        # Uncomment the following if mamba3_step_fn is not in place:
        # state_out = torch.empty_like(ssm_state)
        # ssm_state.copy_(state_out) 
        k_state.copy_(nxt_k_state)
        v_state.copy_(nxt_v_state)

        return out, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state

    def _step_cpu(self, u, angle_state, ssm_state, k_state, v_state):
        """Pure-PyTorch SISO step for CPU/MPS. Mirrors `step` semantics
        (in-place state updates, identical return signature) without relying
        on the CuteDSL/Triton kernels.

        Only the SISO path is implemented; MIMO falls through to an explicit
        NotImplementedError because the reference math diverges materially.
        """
        if self.is_mimo:
            raise NotImplementedError("Mamba3 CPU/MPS step only supports SISO (is_mimo=False).")
        if self.is_outproj_norm:
            raise NotImplementedError("Mamba3 CPU/MPS step does not yet support is_outproj_norm=True.")

        zxBCdt = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        A = torch.clamp(-F.softplus(dd_A.to(torch.float32)), max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap)

        B = rearrange(B, "b (r g s) -> b r g s", g=self.num_bc_heads, r=self.mimo_rank)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.num_bc_heads, r=self.mimo_rank)
        B = rms_norm_ref(B, self.B_norm.weight, None, eps=self.B_norm.eps)
        C = rms_norm_ref(C, self.C_norm.weight, None, eps=self.C_norm.eps)
        B = B.expand(-1, -1, self.nheads, -1)
        C = C.expand(-1, -1, self.nheads, -1)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
        angles = angles.unsqueeze(-2).expand(-1, self.nheads, -1)

        bias_q = rearrange(self.C_bias, "h r n -> r h n")
        bias_k = rearrange(self.B_bias, "h r n -> r h n")
        rotate_pairwise = not self.is_mimo
        C, B, nxt_angle_state = _apply_rotary_qk_inference_cpu(
            q=C, k=B, angle_state=angle_state, angle_proj=angles,
            dt=DT, bias_q=bias_q, bias_k=bias_k,
            rotate_pairwise=rotate_pairwise,
        )

        nxt_v_state = x
        nxt_k_state = B

        B_curr = B.squeeze(1).to(torch.float32)
        C_curr = C.squeeze(1).to(torch.float32)
        B_prev = k_state.squeeze(1).to(torch.float32)
        x_prev = v_state.to(torch.float32)
        x_f = x.to(torch.float32)
        z_f = z.to(torch.float32)
        DT_f = DT.to(torch.float32)
        trap_f = trap.to(torch.float32)

        alpha = torch.exp(A * DT_f)
        beta = (1.0 - trap_f) * DT_f * alpha
        gamma = trap_f * DT_f

        prev_Bx = torch.einsum("bhn, bhp -> bhpn", B_prev, x_prev)
        curr_Bx = torch.einsum("bhn, bhp -> bhpn", B_curr, x_f)
        new_ssm = (
            alpha[..., None, None] * ssm_state.to(torch.float32)
            + beta[..., None, None] * prev_Bx
            + gamma[..., None, None] * curr_Bx
        )

        y = torch.einsum("bhpn, bhn -> bhp", new_ssm, C_curr)
        y = y + self.D[None, :, None].to(torch.float32) * x_f
        y = y * F.silu(z_f)

        ssm_state.copy_(new_ssm.to(ssm_state.dtype))
        angle_state.copy_(nxt_angle_state.to(angle_state.dtype))
        k_state.copy_(nxt_k_state.to(k_state.dtype))
        v_state.copy_(nxt_v_state.to(v_state.dtype))

        out = self.out_proj(rearrange(y, "b h p -> b (h p)").to(self.in_proj.weight.dtype))
        return out, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, inplace_state=None, **kwargs):
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype

        # RoPE State
        angle_dt_state = torch.zeros(
            (batch_size, self.nheads, self.num_rope_angles),
            device=device,
            dtype=torch.float32,
        )

        # Mamba-3 Combined Kernel States
        # SSM State
        ssm_state = torch.zeros(
            (batch_size, self.nheads, self.headdim, self.d_state),
            device=device,
            dtype=torch.float32,
        )

        # K (=B) State
        if self.is_mimo:
            k_state = torch.zeros(
                (batch_size, self.mimo_rank, self.nheads, self.d_state),
                device=device,
                dtype=dtype,
            )
        else:
            k_state = torch.zeros(
                (batch_size, 1, self.nheads, self.d_state),
                device=device,
                dtype=dtype,
            )

        # V (=x) State
        v_state = torch.zeros(
            (batch_size, self.nheads, self.headdim),
            device=device,
            dtype=dtype,
        )

        return (angle_dt_state, ssm_state, k_state, v_state)
    
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype

        if self.layer_idx not in inference_params.key_value_memory_dict:
            angle_dt_state = torch.zeros(
                (batch_size, self.nheads, self.num_rope_angles),
                device=device,
                dtype=torch.float32,
            )
            ssm_state = torch.zeros(
                (batch_size, self.nheads, self.headdim, self.d_state),
                device=device,
                dtype=torch.float32,
            )
            if self.is_mimo:
                k_state = torch.zeros(
                    (batch_size, self.mimo_rank, self.nheads, self.d_state),
                    device=device,
                    dtype=dtype,
                )
            else:
                k_state = torch.zeros(
                    (batch_size, 1, self.nheads, self.d_state),
                    device=device,
                    dtype=dtype,
                )
            v_state = torch.zeros(
                (batch_size, self.nheads, self.headdim),
                device=device,
                dtype=dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (angle_dt_state, ssm_state, k_state, v_state)
        else:
            angle_dt_state, ssm_state, k_state, v_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                angle_dt_state.zero_()
                ssm_state.zero_()
                k_state.zero_()
                v_state.zero_()
        return angle_dt_state, ssm_state, k_state, v_state
