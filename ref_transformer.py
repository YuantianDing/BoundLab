"""Standard zonotope abstract transformers.

Each function: (layer, Zonotope, PropState, ...) -> Zonotope

Implements DeepT (Bonaert et al., 2021) abstract transformers for:
- Affine: Dense, TokenDense, LayerNorm, QKV
- Element-wise: ReLU, Tanh, Exp, Reciprocal
- Bilinear: DotProductAttention, ContextProduct
- Softmax: composed via exp -> sum -> reciprocal
- Structural: HeadSplit, HeadConcat, ResidualAdd, SelectToken
- Multi-head wrappers
"""

import torch
from differential_verifier.domain import Zonotope, PropState, zono_bounds, pad_and_cat, DTYPE
from differential_verifier.layers import (
    Dense, ReLU, Tanh, TokenDense, Exp, Reciprocal,
    DotProductAttention, SoftmaxStable, ContextProduct,
    ResidualAdd, SaveSkip, QKVProjection, PrepContextInput,
    LayerNormNoVar, HeadSplit, HeadConcat,
    MultiHeadDotProductAttention, MultiHeadSoftmax,
    MultiHeadPrepContext, MultiHeadContextProduct, SelectToken,
    Network,
)


# ===== Affine (exact) =====

def propagate_dense(layer: Dense, Z: Zonotope, _ps: PropState) -> Zonotope:
    return Zonotope(layer.W @ Z.G, layer.W @ Z.c + layer.b, Z.influence)


def propagate_token_dense(layer: TokenDense, Z: Zonotope, _ps: PropState) -> Zonotope:
    N, d_in, d_out = layer.N, layer.W.shape[1], layer.W.shape[0]
    n_gen = Z.G.shape[1]
    G_new = torch.empty(N * d_out, n_gen, dtype=DTYPE)
    c_new = torch.empty(N * d_out, dtype=DTYPE)
    for i in range(N):
        src = slice(i * d_in, (i + 1) * d_in)
        dst = slice(i * d_out, (i + 1) * d_out)
        G_new[dst] = layer.W @ Z.G[src]
        c_new[dst] = layer.W @ Z.c[src] + layer.b
    return Zonotope(G_new, c_new, Z.influence)


def propagate_layernorm_novar(layer: LayerNormNoVar, Z: Zonotope, _ps: PropState) -> Zonotope:
    N, E = layer.N, layer.E
    M = layer.get_weight_matrix()
    n_gen = Z.G.shape[1]
    G_out = torch.empty(N * E, n_gen, dtype=DTYPE)
    c_out = torch.empty(N * E, dtype=DTYPE)
    for i in range(N):
        src = slice(i * E, (i + 1) * E)
        G_out[src] = M @ Z.G[src]
        c_out[src] = M @ Z.c[src] + layer.beta
    return Zonotope(G_out, c_out, Z.influence)


def propagate_qkv_projection(layer: QKVProjection, Z: Zonotope, _ps: PropState,
                              value_stack: list) -> Zonotope:
    N, E = layer.N, layer.W_Q.shape[1]
    d_k, d_v = layer.W_Q.shape[0], layer.W_V.shape[0]
    n_gen = Z.G.shape[1]
    G_Q = torch.empty(N * d_k, n_gen, dtype=DTYPE)
    c_Q = torch.empty(N * d_k, dtype=DTYPE)
    G_K = torch.empty(N * d_k, n_gen, dtype=DTYPE)
    c_K = torch.empty(N * d_k, dtype=DTYPE)
    G_V = torch.empty(N * d_v, n_gen, dtype=DTYPE)
    c_V = torch.empty(N * d_v, dtype=DTYPE)
    for i in range(N):
        src = slice(i * E, (i + 1) * E)
        dq = slice(i * d_k, (i + 1) * d_k)
        dv = slice(i * d_v, (i + 1) * d_v)
        G_Q[dq] = layer.W_Q @ Z.G[src]
        c_Q[dq] = layer.W_Q @ Z.c[src] + layer.b_Q
        G_K[dq] = layer.W_K @ Z.G[src]
        c_K[dq] = layer.W_K @ Z.c[src] + layer.b_K
        G_V[dv] = layer.W_V @ Z.G[src]
        c_V[dv] = layer.W_V @ Z.c[src] + layer.b_V
    value_stack.append(Zonotope(G_V, c_V, None))
    return Zonotope(torch.cat([G_Q, G_K], dim=0), torch.cat([c_Q, c_K]), Z.influence)


# ===== Element-wise nonlinear =====

def propagate_relu(layer: ReLU, Z: Zonotope, _ps: PropState, *,
                   bounds=None, use_new_heuristic=True) -> Zonotope:
    """Minimal-area ReLU relaxation (Singh et al., 2018)."""
    row_count = Z.G.shape[0]
    if bounds is None:
        bounds = zono_bounds(Z)
    lower, upper = bounds[:, 0], bounds[:, 1]
    denom = upper - lower
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
    alpha = torch.clamp(upper / denom, 0.0, 1.0)
    lam = torch.where(upper <= 0, torch.zeros_like(alpha),
          torch.where(lower >= 0, torch.ones_like(alpha), alpha))
    crossing = (lower < 0.0) & (upper > 0.0)
    gamma = 0.5 * torch.maximum(
        torch.maximum(-lam * lower, torch.zeros_like(lower)),
        (1.0 - lam) * upper)
    c_hat = lam * Z.c + crossing * gamma
    num_crossing = int(crossing.sum().item())
    G_hat = torch.zeros(row_count, Z.G.shape[1] + num_crossing, dtype=DTYPE)
    G_hat[:, :Z.G.shape[1]] = Z.G
    if num_crossing > 0:
        crossing_idx = torch.where(crossing)[0]
        for local_j, global_i in enumerate(crossing_idx):
            G_hat[global_i, Z.G.shape[1] + local_j] = 1.0
    G_hat[:, :Z.G.shape[1]] *= lam.unsqueeze(1)
    if num_crossing > 0:
        for local_j, global_i in enumerate(crossing_idx):
            G_hat[global_i, Z.G.shape[1] + local_j] *= abs(gamma[global_i].item())
    return Zonotope(G_hat, c_hat, Z.influence)


def propagate_tanh(layer: Tanh, Z: Zonotope, _ps: PropState, *,
                   bounds=None) -> Zonotope:
    """Minimal-area Tanh relaxation (DeepT, Section 4.4).

    y = lambda*x + mu + beta*eps_new
    lambda = min(sech^2(l), sech^2(u)) = min(1-tanh^2(l), 1-tanh^2(u))
    mu = 0.5*(tanh(u) + tanh(l) - lambda*(u + l))
    beta = 0.5*(tanh(u) - tanh(l) - lambda*(u - l))
    """
    n = Z.G.shape[0]
    if bounds is None:
        bounds = zono_bounds(Z)
    l, u = bounds[:, 0], bounds[:, 1]
    degen = torch.abs(u - l) < 1e-12

    tl = torch.tanh(l)
    tu = torch.tanh(u)
    lam = torch.minimum(1 - tl**2, 1 - tu**2)
    mu = 0.5 * (tu + tl - lam * (u + l))
    beta = 0.5 * (tu - tl - lam * (u - l))

    lam = torch.where(degen, 1 - torch.tanh(l)**2, lam)
    mu = torch.where(degen, torch.zeros_like(mu), mu)
    beta = torch.where(degen, torch.zeros_like(beta), beta)
    beta = torch.abs(beta)

    active = ~degen & (beta > 1e-15)
    num_new = int(active.sum().item())
    G_new = torch.zeros(n, Z.G.shape[1] + num_new, dtype=DTYPE)
    G_new[:, :Z.G.shape[1]] = lam.unsqueeze(1) * Z.G
    c_new = lam * Z.c + mu
    if num_new > 0:
        active_idx = torch.where(active)[0]
        for j, i in enumerate(active_idx):
            G_new[i, Z.G.shape[1] + j] = beta[i]
    return Zonotope(G_new, c_new, Z.influence)


def propagate_exp(layer: Exp, Z: Zonotope, _ps: PropState, *,
                  bounds=None) -> Zonotope:
    """Minimal-area Exp relaxation (DeepT, Section 4.5)."""
    n = Z.G.shape[0]
    if bounds is None:
        bounds = zono_bounds(Z)
    l, u = bounds[:, 0], bounds[:, 1]
    degen = torch.abs(u - l) < 1e-12
    el = torch.exp(torch.clamp(l, -30, 30))
    eu = torch.exp(torch.clamp(u, -30, 30))
    secant_slope = torch.where(degen, el, (eu - el) / (u - l + 1e-30))
    t_crit = torch.log(torch.clamp(secant_slope, min=1e-30))
    t_crit2 = l + 1.0 - 0.01
    t_opt = torch.minimum(t_crit, t_crit2)
    lam = torch.exp(torch.clamp(t_opt, -30, 30))
    et = torch.exp(torch.clamp(t_opt, -30, 30))
    mu = 0.5 * (et - lam * t_opt + eu - lam * u)
    beta = 0.5 * (lam * t_opt - et + eu - lam * u)
    lam = torch.where(degen, torch.zeros_like(lam), lam)
    mu = torch.where(degen, el, mu)
    beta = torch.where(degen, torch.zeros_like(beta), beta)
    beta = torch.abs(beta)
    active = ~degen & (beta > 1e-15)
    num_new = int(active.sum().item())
    G_new = torch.zeros(n, Z.G.shape[1] + num_new, dtype=DTYPE)
    G_new[:, :Z.G.shape[1]] = lam.unsqueeze(1) * Z.G
    c_new = lam * Z.c + mu
    if num_new > 0:
        active_idx = torch.where(active)[0]
        for j, i in enumerate(active_idx):
            G_new[i, Z.G.shape[1] + j] = beta[i]
    return Zonotope(G_new, c_new, Z.influence)


def propagate_reciprocal(layer: Reciprocal, Z: Zonotope, _ps: PropState, *,
                         bounds=None) -> Zonotope:
    """Minimal-area Reciprocal relaxation (DeepT, Section 4.6)."""
    n = Z.G.shape[0]
    if bounds is None:
        bounds = zono_bounds(Z)
    l, u = bounds[:, 0].clone(), bounds[:, 1].clone()
    l = torch.clamp(l, min=1e-9)
    u = torch.clamp(u, min=l + 1e-12)
    degen = torch.abs(u - l) < 1e-12
    t_crit = torch.sqrt(u * l)
    t_crit2 = 0.5 * u + 0.01
    t_opt = torch.minimum(t_crit, t_crit2)
    lam = -1.0 / (t_opt ** 2)
    val_at_t = 1.0 / t_opt
    val_at_l = 1.0 / l
    mu = 0.5 * (val_at_t - lam * t_opt + val_at_l - lam * l)
    beta = 0.5 * (lam * t_opt - val_at_t + val_at_l - lam * l)
    lam = torch.where(degen, torch.zeros_like(lam), lam)
    mu = torch.where(degen, 1.0 / l, mu)
    beta = torch.where(degen, torch.zeros_like(beta), beta)
    beta = torch.abs(beta)
    active = ~degen & (beta > 1e-15)
    num_new = int(active.sum().item())
    G_new = torch.zeros(n, Z.G.shape[1] + num_new, dtype=DTYPE)
    G_new[:, :Z.G.shape[1]] = lam.unsqueeze(1) * Z.G
    c_new = lam * Z.c + mu
    if num_new > 0:
        active_idx = torch.where(active)[0]
        for j, i in enumerate(active_idx):
            G_new[i, Z.G.shape[1] + j] = beta[i]
    return Zonotope(G_new, c_new, Z.influence)


# ===== Bilinear =====

def propagate_dot_product_attention(layer: DotProductAttention, Z: Zonotope,
                                     _ps: PropState) -> Zonotope:
    """DeepT-Precise dot product (Section 4.8, Eq. 6)."""
    N, d_k, scale = layer.N, layer.d_k, layer.scale
    n_gen = Z.G.shape[1]
    Q_G = Z.G[:N*d_k].view(N, d_k, n_gen)
    Q_c = Z.c[:N*d_k].view(N, d_k)
    K_G = Z.G[N*d_k:].view(N, d_k, n_gen)
    K_c = Z.c[N*d_k:].view(N, d_k)
    c_out = torch.zeros(N * N, dtype=DTYPE)
    affine_G, quad_bounds = [], []
    for i in range(N):
        for j in range(N):
            qi_c, qi_G = Q_c[i], Q_G[i]
            kj_c, kj_G = K_c[j], K_G[j]
            c_ij = (qi_c @ kj_c) / scale
            G_ij = (qi_c @ kj_G + kj_c @ qi_G) / scale
            cross = (qi_G.T @ kj_G) / scale
            diag = torch.diag(cross)
            c_ij = c_ij + 0.5 * diag.sum()
            bound = 0.5 * torch.abs(diag).sum()
            off_diag = cross.clone(); off_diag.fill_diagonal_(0.0)
            bound = bound + torch.abs(off_diag).sum()
            c_out[i * N + j] = c_ij
            affine_G.append(G_ij)
            quad_bounds.append(bound.item())
    n_active = sum(1 for b in quad_bounds if b > 1e-15)
    G_new = torch.zeros(N * N, n_gen + n_active, dtype=DTYPE)
    col_offset = n_gen
    for idx in range(N * N):
        G_new[idx, :n_gen] = affine_G[idx]
        if quad_bounds[idx] > 1e-15:
            G_new[idx, col_offset] = quad_bounds[idx]
            col_offset += 1
    return Zonotope(G_new, c_out, None)


def propagate_context_product(layer: ContextProduct, Z: Zonotope,
                               _ps: PropState) -> Zonotope:
    """DeepT-Precise context product (A @ V)."""
    N, d_v = layer.N, layer.d_v
    n_gen = Z.G.shape[1]
    A_G = Z.G[:N*N].view(N, N, n_gen)
    A_c = Z.c[:N*N].view(N, N)
    V_G = Z.G[N*N:].view(N, d_v, n_gen)
    V_c = Z.c[N*N:].view(N, d_v)
    c_out = torch.zeros(N * d_v, dtype=DTYPE)
    affine_G, quad_bounds = [], []
    for i in range(N):
        for d in range(d_v):
            a_c, a_G_i = A_c[i], A_G[i]
            v_c, v_G_d = V_c[:, d], V_G[:, d, :]
            c_ij = a_c @ v_c
            G_ij = a_c @ v_G_d + v_c @ a_G_i
            cross = a_G_i.T @ v_G_d
            diag = torch.diag(cross)
            c_ij = c_ij + 0.5 * diag.sum()
            bound = 0.5 * torch.abs(diag).sum()
            off_diag = cross.clone(); off_diag.fill_diagonal_(0.0)
            bound = bound + torch.abs(off_diag).sum()
            c_out[i * d_v + d] = c_ij
            affine_G.append(G_ij)
            quad_bounds.append(bound.item())
    n_active = sum(1 for b in quad_bounds if b > 1e-15)
    G_new = torch.zeros(N * d_v, n_gen + n_active, dtype=DTYPE)
    col_offset = n_gen
    for idx in range(N * d_v):
        G_new[idx, :n_gen] = affine_G[idx]
        if quad_bounds[idx] > 1e-15:
            G_new[idx, col_offset] = quad_bounds[idx]
            col_offset += 1
    return Zonotope(G_new, c_out, None)


# ===== Softmax (composed) =====

def propagate_softmax_stable(layer: SoftmaxStable, Z: Zonotope,
                              _ps: PropState) -> Zonotope:
    """DeepT softmax: sigma_ij = 1 / sum_k exp(s_ik - s_ij) (Section 5.2)."""
    N = layer.N
    n_gen_in = Z.G.shape[1]
    exp_layer, recip_layer = Exp(), Reciprocal()
    ps_dummy = PropState(first=True)
    element_affine_G, element_c, element_new_gens = [], [], []
    for i in range(N):
        rs = slice(i * N, (i + 1) * N)
        G_row, c_row = Z.G[rs], Z.c[rs]
        for j in range(N):
            G_diff = G_row - G_row[j:j+1, :]
            c_diff = c_row - c_row[j]
            Z_exp = propagate_exp(exp_layer, Zonotope(G_diff, c_diff, None), ps_dummy)
            c_sum = Z_exp.c.sum()
            G_sum = Z_exp.G.sum(dim=0, keepdim=True)
            Z_recip = propagate_reciprocal(recip_layer, Zonotope(G_sum, c_sum.unsqueeze(0), None), ps_dummy)
            g_full = Z_recip.G[0]
            element_affine_G.append(g_full[:n_gen_in])
            element_c.append(Z_recip.c[0])
            element_new_gens.append(g_full[n_gen_in:])
    total_new = sum(len(g) for g in element_new_gens)
    G_out = torch.zeros(N * N, n_gen_in + total_new, dtype=DTYPE)
    c_out = torch.zeros(N * N, dtype=DTYPE)
    col_offset = n_gen_in
    for idx in range(N * N):
        G_out[idx, :n_gen_in] = element_affine_G[idx]
        c_out[idx] = element_c[idx]
        n_new = len(element_new_gens[idx])
        if n_new > 0:
            G_out[idx, col_offset:col_offset + n_new] = element_new_gens[idx]
            col_offset += n_new
    return Zonotope(G_out, c_out, None)


# ===== Structural =====

def propagate_prep_context_input(layer: PrepContextInput, Z_attn: Zonotope,
                                  _ps: PropState, value_stack: list) -> Zonotope:
    Z_V = value_stack.pop()
    n1, n2 = Z_attn.G.shape[1], Z_V.G.shape[1]
    n_gen = max(n1, n2)
    G_attn = torch.zeros(Z_attn.G.shape[0], n_gen, dtype=DTYPE); G_attn[:, :n1] = Z_attn.G
    G_V = torch.zeros(Z_V.G.shape[0], n_gen, dtype=DTYPE); G_V[:, :n2] = Z_V.G
    return Zonotope(torch.cat([G_attn, G_V], dim=0), torch.cat([Z_attn.c, Z_V.c]), None)


def propagate_head_split(layer: HeadSplit, Z: Zonotope, _ps: PropState,
                          value_stack: list) -> Zonotope:
    perm = layer.get_QK_permutation()
    Z_out = Zonotope(Z.G[perm], Z.c[perm], Z.influence)
    if value_stack:
        Z_V = value_stack.pop()
        perm_v = layer.get_V_permutation()
        value_stack.append(Zonotope(Z_V.G[perm_v], Z_V.c[perm_v], Z_V.influence))
    return Z_out


def propagate_head_concat(layer: HeadConcat, Z: Zonotope, _ps: PropState) -> Zonotope:
    perm = layer.get_permutation()
    return Zonotope(Z.G[perm], Z.c[perm], Z.influence)


def propagate_select_token(layer: SelectToken, Z: Zonotope, _ps: PropState) -> Zonotope:
    N, E = layer.N, layer.E
    if layer.mode == "mean":
        G_out = torch.zeros(E, Z.G.shape[1], dtype=DTYPE)
        c_out = torch.zeros(E, dtype=DTYPE)
        for i in range(N):
            G_out += Z.G[i*E:(i+1)*E]; c_out += Z.c[i*E:(i+1)*E]
        return Zonotope(G_out / N, c_out / N, Z.influence)
    idx = layer.token_idx
    src = slice(idx * E, (idx + 1) * E)
    return Zonotope(Z.G[src].clone(), Z.c[src].clone(), Z.influence)


# ===== Multi-head wrappers =====

def propagate_multihead_dot_product(layer: MultiHeadDotProductAttention,
                                     Z: Zonotope, _ps: PropState) -> Zonotope:
    N, H, d, scale = layer.N, layer.H, layer.d_head, layer.scale
    head_in = 2 * N * d
    results_G, results_c = [], []
    for h in range(H):
        src = slice(h * head_in, (h + 1) * head_in)
        Z_h = Zonotope(Z.G[src], Z.c[src], Z.influence)
        Z_h_out = propagate_dot_product_attention(DotProductAttention(N, d, scale), Z_h, _ps)
        results_G.append(Z_h_out.G); results_c.append(Z_h_out.c)
    return pad_and_cat(results_G, results_c)


def propagate_multihead_softmax(layer: MultiHeadSoftmax, Z: Zonotope,
                                 _ps: PropState) -> Zonotope:
    N, H = layer.N, layer.H
    results_G, results_c = [], []
    for h in range(H):
        src = slice(h * N * N, (h + 1) * N * N)
        Z_h = Zonotope(Z.G[src], Z.c[src], Z.influence)
        Z_h_out = propagate_softmax_stable(SoftmaxStable(N), Z_h, _ps)
        results_G.append(Z_h_out.G); results_c.append(Z_h_out.c)
    return pad_and_cat(results_G, results_c)


def propagate_multihead_prep_context(layer: MultiHeadPrepContext,
                                      Z_attn: Zonotope, _ps: PropState,
                                      value_stack: list) -> Zonotope:
    N, H, d_v = layer.N, layer.H, layer.d_v
    Z_V = value_stack.pop()
    n1, n2 = Z_attn.G.shape[1], Z_V.G.shape[1]
    n_gen = max(n1, n2)
    G_a = torch.zeros(Z_attn.G.shape[0], n_gen, dtype=DTYPE); G_a[:, :n1] = Z_attn.G
    G_v = torch.zeros(Z_V.G.shape[0], n_gen, dtype=DTYPE); G_v[:, :n2] = Z_V.G
    block_a, block_v = N * N, N * d_v
    total_out = H * (block_a + block_v)
    G_out = torch.zeros(total_out, n_gen, dtype=DTYPE)
    c_out = torch.zeros(total_out, dtype=DTYPE)
    for h in range(H):
        out_off = h * (block_a + block_v)
        G_out[out_off:out_off+block_a] = G_a[h*block_a:(h+1)*block_a]
        c_out[out_off:out_off+block_a] = Z_attn.c[h*block_a:(h+1)*block_a]
        G_out[out_off+block_a:out_off+block_a+block_v] = G_v[h*block_v:(h+1)*block_v]
        c_out[out_off+block_a:out_off+block_a+block_v] = Z_V.c[h*block_v:(h+1)*block_v]
    return Zonotope(G_out, c_out, None)


def propagate_multihead_context_product(layer: MultiHeadContextProduct,
                                         Z: Zonotope, _ps: PropState) -> Zonotope:
    N, H, d_v = layer.N, layer.H, layer.d_v
    block = N * N + N * d_v
    results_G, results_c = [], []
    for h in range(H):
        src = slice(h * block, (h + 1) * block)
        Z_h = Zonotope(Z.G[src], Z.c[src], Z.influence)
        Z_h_out = propagate_context_product(ContextProduct(N, d_v), Z_h, _ps)
        results_G.append(Z_h_out.G); results_c.append(Z_h_out.c)
    return pad_and_cat(results_G, results_c)


# ===== Full network propagation =====

def propagate_network(net: Network, Z: Zonotope, ps: PropState) -> Zonotope:
    """Propagate a zonotope through a full network."""
    skip_stack, value_stack = [], []
    for layer in net.layers:
        if isinstance(layer, Dense):
            Z = propagate_dense(layer, Z, ps)
        elif isinstance(layer, TokenDense):
            Z = propagate_token_dense(layer, Z, ps)
        elif isinstance(layer, ReLU):
            Z = propagate_relu(layer, Z, ps)
        elif isinstance(layer, Tanh):
            Z = propagate_tanh(layer, Z, ps)
        elif isinstance(layer, Exp):
            Z = propagate_exp(layer, Z, ps)
        elif isinstance(layer, Reciprocal):
            Z = propagate_reciprocal(layer, Z, ps)
        elif isinstance(layer, DotProductAttention):
            Z = propagate_dot_product_attention(layer, Z, ps)
        elif isinstance(layer, SoftmaxStable):
            Z = propagate_softmax_stable(layer, Z, ps)
        elif isinstance(layer, ContextProduct):
            Z = propagate_context_product(layer, Z, ps)
        elif isinstance(layer, QKVProjection):
            Z = propagate_qkv_projection(layer, Z, ps, value_stack)
        elif isinstance(layer, PrepContextInput):
            Z = propagate_prep_context_input(layer, Z, ps, value_stack)
        elif isinstance(layer, LayerNormNoVar):
            Z = propagate_layernorm_novar(layer, Z, ps)
        elif isinstance(layer, HeadSplit):
            Z = propagate_head_split(layer, Z, ps, value_stack)
        elif isinstance(layer, HeadConcat):
            Z = propagate_head_concat(layer, Z, ps)
        elif isinstance(layer, MultiHeadDotProductAttention):
            Z = propagate_multihead_dot_product(layer, Z, ps)
        elif isinstance(layer, MultiHeadSoftmax):
            Z = propagate_multihead_softmax(layer, Z, ps)
        elif isinstance(layer, MultiHeadPrepContext):
            Z = propagate_multihead_prep_context(layer, Z, ps, value_stack)
        elif isinstance(layer, MultiHeadContextProduct):
            Z = propagate_multihead_context_product(layer, Z, ps)
        elif isinstance(layer, SelectToken):
            Z = propagate_select_token(layer, Z, ps)
        elif isinstance(layer, SaveSkip):
            skip_stack.append(Z.copy())
        elif isinstance(layer, ResidualAdd):
            Z_skip = skip_stack.pop()
            n1, n2 = Z.G.shape[1], Z_skip.G.shape[1]
            if n1 < n2:
                G_pad = torch.zeros(Z.G.shape[0], n2, dtype=DTYPE)
                G_pad[:, :n1] = Z.G
                Z = Zonotope(G_pad, Z.c, Z.influence)
            elif n2 < n1:
                G_pad = torch.zeros(Z_skip.G.shape[0], n1, dtype=DTYPE)
                G_pad[:, :n2] = Z_skip.G
                Z_skip = Zonotope(G_pad, Z_skip.c, Z_skip.influence)
            Z = Zonotope(Z.G + Z_skip.G, Z.c + Z_skip.c, Z.influence)
        else:
            raise ValueError(f"Unsupported layer: {type(layer)}")
    return Z
