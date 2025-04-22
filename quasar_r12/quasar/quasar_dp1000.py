import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -----------------------------
# Piecewise Linear Function (PLF)
# -----------------------------

class PLF(nn.Module):
    def __init__(self, breakpoints, init_slope=0., eps=1e-4):
        super().__init__()
        self.register_buffer('x', torch.tensor(breakpoints, dtype=torch.float32))
        M = len(breakpoints) - 1
        self.theta = nn.Parameter(torch.zeros(M))
        self.s0 = nn.Parameter(torch.tensor(init_slope))
        self.eps = eps

    def forward(self, t):
        deltas = F.softplus(self.theta) + self.eps
        slopes = torch.cumsum(torch.cat([self.s0.unsqueeze(0), deltas]), dim=0)
        idx = torch.bucketize(t, self.x) - 1
        idx = torch.clamp(idx, 0, len(self.x) - 2)
        h = t - self.x[idx]
        f_xi = torch.cumsum(torch.cat([torch.zeros(1, device=t.device), slopes[:-1] * torch.diff(self.x)]), dim=0)
        return f_xi[idx] + slopes[idx] * h

# -----------------------------
# Routing Cost Module (RCost)
# -----------------------------

class RCost(nn.Module):
    def __init__(self, A_mask, breakpoints, init_slope=0., eps=1e-4):
        super().__init__()
        self.register_buffer('A', A_mask.float())
        self.plf = PLF(breakpoints, init_slope, eps)
        self.preference = nn.Parameter(torch.zeros_like(self.A))

    def forward(self, tail_wind):
        plf_cost = self.plf(tail_wind)
        return plf_cost + self.preference * self.A

# -----------------------------
# Soft Bellman-Ford Iterations
# -----------------------------

def soft_bellman_fw(E_src, E_dst, edge_costs, source, N, max_iter=500, tol=1e-3, eps=1e-10):
    device = edge_costs.device
    V = torch.full((N,), float('inf'), device=device)
    V[source] = 0.0
    for _ in range(max_iter):
        V_prev = V.clone()
        w = torch.exp(-(V[E_src] + edge_costs))
        S = torch.zeros_like(V).scatter_add_(0, E_dst, w)
        V = -torch.log(S + eps)
        V[source] = 0.0
        if torch.max(torch.abs(V - V_prev)) < tol:
            break
    return V


def soft_bellman_bw(E_src, E_dst, edge_costs, goal, N, max_iter=500, tol=1e-3, eps=1e-10):
    device = edge_costs.device
    V = torch.full((N,), float('inf'), device=device)
    V[goal] = 0.0
    for _ in range(max_iter):
        V_prev = V.clone()
        w = torch.exp(-(V[E_dst] + edge_costs))
        S = torch.zeros_like(V).scatter_add_(0, E_src, w)
        V = -torch.log(S + eps)
        V[goal] = 0.0
        if torch.max(torch.abs(V - V_prev)) < tol:
            break
    return V

# Optionally compile for speed
try:
    soft_bellman_fw = torch.compile(soft_bellman_fw)
    soft_bellman_bw = torch.compile(soft_bellman_bw)
except Exception:
    pass

# -----------------------------
# Model Fitting (Gradient Descent)
# -----------------------------

def fit_routing_model(
    A_mask,               # Tensor[N, N] adjacency mask (0/1 float)
    breakpoints,          # list or array-like of PLF breakpoints
    tail_wind,            # Tensor[N, N] input feature
    E_src, E_dst,         # LongTensor edge indices (E,)
    empirical_freq,       # Tensor[E,] empirical edge frequencies
    source, goal,         # int node indices
    max_vi_steps=500,
    vi_tol=1e-3,
    max_steps=1000,
    lr=1e-2,
    use_checkpoint=False,
    device=None
):
    # Device setup
    if device is None:
        device = tail_wind.device
    A_mask = A_mask.to(device)
    tail_wind = tail_wind.to(device)
    E_src = E_src.to(device)
    E_dst = E_dst.to(device)
    empirical_freq = empirical_freq.to(device)

    N = A_mask.shape[0]
    # Initialize cost model
    model = RCost(A_mask, breakpoints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(max_steps):
        optimizer.zero_grad()
        cost_mat = model(tail_wind)
        edge_costs = cost_mat[E_src, E_dst]

        # Optionally re-compute with no_grad to save memory, then enable grad only on final V
        if use_checkpoint:
            V_fw = checkpoint(soft_bellman_fw, E_src, E_dst, edge_costs, source, N, max_vi_steps, vi_tol)
        else:
            V_fw = soft_bellman_fw(E_src, E_dst, edge_costs, source, N, max_vi_steps, vi_tol)
        V_bw = soft_bellman_bw(E_src, E_dst, edge_costs, goal, N, max_vi_steps, vi_tol)

        log_w = -(V_fw[E_src] + edge_costs + V_bw[E_dst])
        log_w = log_w - V_bw[source]
        P_edge = torch.exp(log_w)

        # Mean squared error loss
        loss = F.mse_loss(P_edge, empirical_freq)
        loss.backward()
        optimizer.step()

        print(f"Step {step:4d}, Loss = {loss.item():.6f}")
        if loss.item() < vi_tol:
            print(f"Converged at step {step} with loss {loss.item():.6f}")
            break

    return model


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Construct a simple directed graph with 4 nodes and 5 edges
    # Node IDs: 0,1,2,3
    N = 4
    # Adjacency mask (N x N)
    A_mask = torch.tensor([
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ], dtype=torch.float32)

    # Tail-wind feature for each possible edge (random for demo)
    torch.manual_seed(0)
    tail_wind = torch.randn(N, N)

    # Extract edge indices (COO style)
    E_src, E_dst = A_mask.nonzero(as_tuple=True)

    # Suppose we have 3 demonstration paths
    demos = [
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 2, 3]
    ]
    # Compute empirical frequencies per edge
    # Count unique edge traversals per path
    counts = {}
    for path in demos:
        seen = set()
        for u, v in zip(path[:-1], path[1:]):
            seen.add((u, v))
        for e in seen:
            counts[e] = counts.get(e, 0) + 1
    empirical_freq = torch.zeros(E_src.size(0), dtype=torch.float32)
    for idx, (i, j) in enumerate(zip(E_src.tolist(), E_dst.tolist())):
        empirical_freq[idx] = counts.get((i, j), 0) / len(demos)

    # Define PLF breakpoints
    breakpoints = [ -1.0, 0.0, 1.0, 2.0 ]

    # Source and goal nodes
    source, goal = 0, 3

    # Fit the model
    model = fit_routing_model(
        A_mask=A_mask,
        breakpoints=breakpoints,
        tail_wind=tail_wind,
        E_src=E_src,
        E_dst=E_dst,
        empirical_freq=empirical_freq,
        source=source,
        goal=goal,
        max_vi_steps=200,
        vi_tol=1e-4,
        max_steps=200,
        lr=0.05,
        use_checkpoint=False
    )

    # After fitting, we can inspect learned preferences
    learned_pref = model.preference.detach()
    print("Learned per-edge preferences:")
    print(learned_pref)
