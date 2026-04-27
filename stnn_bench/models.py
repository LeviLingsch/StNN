from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizedModel(nn.Module):
    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


class StNNLayer(nn.Module):
    """Single structured StNN transformation layer mapping R^n -> R^n."""

    def __init__(
        self,
        state_dim: int,
        p: int = 16,
        post_hidden: int = 64,
        dropout: float = 0.05,
        use_input_layernorm: bool = False,
        use_output_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.n = state_dim
        self.r = 2 * self.n
        self.p = p

        self.input_norm = nn.LayerNorm(self.n) if use_input_layernorm else nn.Identity()
        self.output_norm = nn.LayerNorm(self.n) if use_output_layernorm else nn.Identity()

        self.D_hat = nn.Parameter(torch.randn(self.r))
        self.D_grave = nn.Parameter(torch.randn(self.n))
        self.D_check = nn.Parameter(torch.randn(self.n))
        self.F_n = nn.Parameter(torch.randn(self.n, self.n) / (self.n ** 0.5))

        self.b1 = nn.Parameter(torch.zeros(self.p, self.r))
        self.b2 = nn.Parameter(torch.zeros(self.p, self.n))
        self.b3 = nn.Parameter(torch.zeros(self.p, self.n))
        self.b4 = nn.Parameter(torch.zeros(self.n))

        self.dropout = nn.Dropout(dropout)
        self.post = nn.Sequential(
            nn.Linear(self.n, post_hidden),
            nn.SiLU(),
            nn.Linear(post_hidden, self.n),
        )

        self.register_buffer("I_n", torch.eye(self.n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)

        # Build H_r = [[I, I], [diag(D_grave), -diag(D_grave)]].
        D_g = torch.diag(self.D_grave)
        Hr_top = torch.cat([self.I_n, self.I_n], dim=1)
        Hr_bottom = torch.cat([D_g, -D_g], dim=1)
        Hr = torch.cat([Hr_top, Hr_bottom], dim=0)

        # Build block F and interleave with permutation.
        Fn_block = torch.block_diag(self.F_n, self.F_n)
        idx_even = torch.arange(0, self.r, 2, device=x.device)
        idx_odd = torch.arange(1, self.r, 2, device=x.device)
        idx = torch.cat([idx_even, idx_odd], dim=0)
        P_r = torch.eye(self.r, device=x.device)[idx]
        F_r = P_r.T @ Fn_block @ Hr

        J = torch.cat([self.I_n, torch.zeros(self.n, self.n, device=x.device)], dim=0)

        w1 = F_r @ J  # (2n, n)
        h1 = torch.tanh(x @ w1.T)  # (B, 2n)
        h1 = h1.unsqueeze(1) + self.b1.unsqueeze(0)  # (B, p, 2n)

        positive_d_hat = F.softplus(self.D_hat) + 1e-4
        w2 = J.T @ F_r @ torch.diag(positive_d_hat)  # (n, 2n)
        h2 = torch.einsum("bpr,nr->bpn", h1, w2)
        h2 = F.gelu(h2 + self.b2.unsqueeze(0))

        h3 = torch.flip(h2, dims=[-1])
        h3 = F.silu(h3 + self.b3.unsqueeze(0))
        h3 = self.dropout(h3)

        out = (h3 * self.D_check.view(1, 1, self.n)).sum(dim=1) + self.b4
        out = out + self.post(self.output_norm(out))
        return out

    def regularization_loss(self) -> torch.Tensor:
        eye = torch.eye(self.n, device=self.F_n.device, dtype=self.F_n.dtype)
        return (self.F_n @ self.F_n.T - eye).pow(2).mean()


class ImprovedStNN(RegularizedModel):
    """Structured model with stacked StNN layers and residual updates."""

    def __init__(
        self,
        state_dim: int,
        dt: float,
        p: int = 16,
        post_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.05,
        residual_update: bool = True,
        use_input_layernorm: bool = False,
        use_output_layernorm: bool = False,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.n = state_dim
        self.num_layers = max(1, int(num_layers))
        self.dt = dt
        self.residual_update = residual_update

        self.layers = nn.ModuleList(
            [
                StNNLayer(
                    state_dim=self.n,
                    p=p,
                    post_hidden=post_hidden,
                    dropout=dropout,
                    use_input_layernorm=use_input_layernorm,
                    use_output_layernorm=use_output_layernorm,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.log_residual_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale_init))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_base = x
        out = x
        for layer in self.layers:
            out = layer(out)

        if self.residual_update:
            residual_scale = torch.exp(self.log_residual_scale)
            out = x_base + residual_scale * self.dt * out

        return out

    def regularization_loss(self) -> torch.Tensor:
        reg_losses = [layer.regularization_loss() for layer in self.layers]
        return torch.stack(reg_losses).mean()


class MLPBaseline(RegularizedModel):
    def __init__(
        self,
        state_dim: int,
        dt: float,
        hidden_dims: Iterable[int] = (128, 128, 128),
        dropout: float = 0.05,
        residual_update: bool = True,
        use_input_layernorm: bool = False,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.residual_update = residual_update
        self.input_norm = nn.LayerNorm(state_dim) if use_input_layernorm else nn.Identity()
        self.log_residual_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale_init))))

        dims: List[int] = [state_dim, *list(hidden_dims), state_dim]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.net(self.input_norm(x))
        if self.residual_update:
            residual_scale = torch.exp(self.log_residual_scale)
            return x + residual_scale * self.dt * delta
        return delta


class MLPVectorField(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if activation == "tanh":
            activation_layer: nn.Module = nn.Tanh()
        elif activation == "silu":
            activation_layer = nn.SiLU()
        else:
            raise ValueError(f"Unsupported vector field activation: {activation}")

        layers: List[nn.Module] = [nn.Linear(state_dim, hidden_dim), activation_layer]
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_layer)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StNNVectorField(nn.Module):
    """Structured vector field used inside NeuralODE integration steps."""

    def __init__(
        self,
        state_dim: int,
        p: int = 16,
        post_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.05,
        use_input_layernorm: bool = False,
        use_output_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StNNLayer(
                    state_dim=state_dim,
                    p=p,
                    post_hidden=post_hidden,
                    dropout=dropout,
                    use_input_layernorm=use_input_layernorm,
                    use_output_layernorm=use_output_layernorm,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def regularization_loss(self) -> torch.Tensor:
        reg_losses = [layer.regularization_loss() for layer in self.layers]
        return torch.stack(reg_losses).mean()


class NeuralODEBaseline(RegularizedModel):
    """Fixed-step RK4 one-step integrator for fair one-step prediction."""

    def __init__(
        self,
        state_dim: int,
        dt: float,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        field_type: str = "mlp",
        field_activation: str = "tanh",
        stnn_p: int = 16,
        stnn_post_hidden: int = 64,
        stnn_num_layers: int = 1,
        stnn_use_input_layernorm: bool = False,
        stnn_use_output_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.field_type = field_type

        if field_type == "mlp":
            self.func = MLPVectorField(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                activation=field_activation,
            )
        elif field_type == "stnn":
            self.func = StNNVectorField(
                state_dim=state_dim,
                p=stnn_p,
                post_hidden=stnn_post_hidden,
                num_layers=stnn_num_layers,
                dropout=dropout,
                use_input_layernorm=stnn_use_input_layernorm,
                use_output_layernorm=stnn_use_output_layernorm,
            )
        else:
            raise ValueError(f"Unknown neural_ode field_type: {field_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        k1 = self.func(x)
        k2 = self.func(x + 0.5 * dt * k1)
        k3 = self.func(x + 0.5 * dt * k2)
        k4 = self.func(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def regularization_loss(self) -> torch.Tensor:
        if hasattr(self.func, "regularization_loss"):
            return self.func.regularization_loss()
        return super().regularization_loss()


def build_model(model_cfg: Dict[str, object], state_dim: int, dt: float) -> RegularizedModel:
    model_type = str(model_cfg.get("type", "")).strip().lower()

    if model_type == "stnn":
        stnn_cfg = dict(model_cfg.get("stnn", {}))
        return ImprovedStNN(
            state_dim=state_dim,
            dt=dt,
            p=int(stnn_cfg.get("p", 16)),
            post_hidden=int(stnn_cfg.get("post_hidden", 64)),
            num_layers=int(stnn_cfg.get("num_layers", 1)),
            dropout=float(stnn_cfg.get("dropout", 0.05)),
            residual_update=bool(stnn_cfg.get("residual_update", True)),
            use_input_layernorm=bool(stnn_cfg.get("use_input_layernorm", False)),
            use_output_layernorm=bool(stnn_cfg.get("use_output_layernorm", False)),
            residual_scale_init=float(stnn_cfg.get("residual_scale_init", 1.0)),
        )

    if model_type == "mlp":
        mlp_cfg = dict(model_cfg.get("mlp", {}))
        hidden_dims = mlp_cfg.get("hidden_dims", [128, 128, 128])
        return MLPBaseline(
            state_dim=state_dim,
            dt=dt,
            hidden_dims=hidden_dims,
            dropout=float(mlp_cfg.get("dropout", 0.05)),
            residual_update=bool(mlp_cfg.get("residual_update", True)),
            use_input_layernorm=bool(mlp_cfg.get("use_input_layernorm", False)),
            residual_scale_init=float(mlp_cfg.get("residual_scale_init", 1.0)),
        )

    if model_type in {"neural_ode", "node"}:
        node_cfg = dict(model_cfg.get("neural_ode", {}))
        field_type = str(node_cfg.get("field_type", "mlp")).strip().lower()
        return NeuralODEBaseline(
            state_dim=state_dim,
            dt=dt,
            hidden_dim=int(node_cfg.get("hidden_dim", 128)),
            num_layers=int(node_cfg.get("num_layers", 2)),
            dropout=float(node_cfg.get("dropout", 0.0)),
            field_type=field_type,
            field_activation=str(node_cfg.get("field_activation", "tanh")).strip().lower(),
            stnn_p=int(node_cfg.get("stnn_p", 16)),
            stnn_post_hidden=int(node_cfg.get("stnn_post_hidden", 64)),
            stnn_num_layers=int(node_cfg.get("stnn_num_layers", 1)),
            stnn_use_input_layernorm=bool(node_cfg.get("stnn_use_input_layernorm", False)),
            stnn_use_output_layernorm=bool(node_cfg.get("stnn_use_output_layernorm", False)),
        )

    raise ValueError(f"Unknown model type: {model_type}")
