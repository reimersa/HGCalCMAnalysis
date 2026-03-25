import torch # type: ignore
import torch.nn as nn # type: ignore


class PerChannelDNN(nn.Module):

    def __init__(self, input_dim: int, nodes_per_layer: list[int], dropout_rate: float = 0.0, tag: str = ""):
        super().__init__()

        self.input_dim       = input_dim   # C
        self.nodes_per_layer = nodes_per_layer
        self.dropout_rate    = dropout_rate
        self.tag             = tag

        # model name (for saving paths etc.)
        hidden_part  = "-".join(str(n) for n in self.nodes_per_layer)
        dropout_part = f"dr{self.dropout_rate:g}"
        str_tag      = f"{self.tag}_" if self.tag else ""
        self.model_string = (
            f"{str_tag}perchannel_F{self.input_dim}"
            f"__{hidden_part}__{dropout_part}"
        )

        layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for h in self.nodes_per_layer:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            # layers.append(nn.LeakyReLU(negative_slope=0.1))
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, F]
        returns: [N]
        """
        # x: [N, F]
        if x.ndim != 2:
            raise ValueError(f"Expected x [N,F], got {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected F={self.input_dim}, got {x.shape[1]}")

        return self.mlp(x).squeeze(-1)  # [N]


    def get_model_string(self) -> str:
        return self.model_string

    def override_model_string(self, new_model_string: str):
        self.model_string = new_model_string



class AllChannelsDNN(nn.Module):
    """
    Flattened multi-output MLP.

    Input:
      x_cf: [B, C, F]   (event-level features should already be broadcast into F if you want them)
    Output:
      y   : [B, C]
    """

    def __init__(
        self,
        num_channels: int,                 # C
        per_channel_dim: int,              # F
        nodes_per_layer: list[int],
        dropout_rate: float = 0.0,
        tag: str = "",
    ):
        super().__init__()

        self.num_channels = int(num_channels)
        self.per_channel_dim = int(per_channel_dim)
        self.nodes_per_layer = nodes_per_layer
        self.dropout_rate = float(dropout_rate)
        self.tag = tag

        # model name (for saving paths etc.)
        hidden_part  = "-".join(str(n) for n in self.nodes_per_layer)
        dropout_part = f"dr{self.dropout_rate:g}"
        str_tag      = f"{self.tag}_" if self.tag else ""
        self.model_string = (
            f"{str_tag}allchannels_inC{self.num_channels}_F{self.per_channel_dim}"
            f"__{hidden_part}__{dropout_part}"
        )

        self.flatten = nn.Flatten(start_dim=1)
        d_in = self.num_channels * self.per_channel_dim  # C*F

        layers: list[nn.Module] = []
        prev_dim = d_in
        for h in self.nodes_per_layer:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = h

        # final head: C outputs
        layers.append(nn.Linear(prev_dim, self.num_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cf: torch.Tensor) -> torch.Tensor:
        """
        x_cf: [B, C, F]
        returns: [B, C]
        """
        if x_cf is None:
            raise ValueError("x_cf must not be None")

        if x_cf.ndim != 3:
            raise ValueError(f"Expected x_cf [B,C,F], got {tuple(x_cf.shape)}")

        B, C, F = x_cf.shape
        if C != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got {C}")
        if F != self.per_channel_dim:
            raise ValueError(f"Expected F={self.per_channel_dim}, got {F}")

        # Flatten: [B, C, F] -> [B, C*F]
        x = self.flatten(x_cf)

        # MLP: [B, C*F] -> [B, C]
        y = self.mlp(x)
        return y

    def get_model_string(self) -> str:
        return self.model_string

    def override_model_string(self, new_model_string: str):
        self.model_string = new_model_string





def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x:    [..., C, H] (or similar)
    mask: [..., C] boolean
    returns: mean over dim with mask (keeps other dims)
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    m = mask.unsqueeze(-1).to(x.dtype)              # [..., C, 1]
    s = (x * m).sum(dim=dim)                        # [..., H]
    n = m.sum(dim=dim).clamp_min(1.0)               # [..., 1]
    return s / n


class DeepSetsDNN(nn.Module):
    """
    Event-level DeepSets with per-channel outputs.

    Inputs:
      x_evt:  [B, Fevt]          (same for all channels in an event)
      x_ch:   [B, C, Fch]        (varies per channel)
      mask:   [B, C] (bool)      (which channels are present / supervised)

    Output:
      yhat:   [B, C]
    """

    def __init__(self, evt_dim: int, ch_dim: int, phi_nodes: list[int], psi_nodes: list[int], dropout_rate: float = 0.0, tag: str = ""):
        super().__init__()

        self.evt_dim = int(evt_dim)
        self.ch_dim  = int(ch_dim)
        self.phi_nodes = list(phi_nodes)
        self.psi_nodes = list(psi_nodes)
        self.dropout_rate = float(dropout_rate)
        self.tag = tag

        # model name (for saving paths etc.)
        phi_part = "-".join(str(n) for n in self.phi_nodes)
        psi_part = "-".join(str(n) for n in self.psi_nodes)
        dropout_part = f"dr{self.dropout_rate:g}"
        str_tag = f"{self.tag}_" if self.tag else ""
        self.model_string = f"{str_tag}deepsets_evt{self.evt_dim}_ch{self.ch_dim}__phi{phi_part}__psi{psi_part}__{dropout_part}"

        # phi: per-channel encoder sees [x_ch, x_evt] (evt broadcast inside forward)
        phi_layers: list[nn.Module] = []
        prev = self.ch_dim + self.evt_dim
        for h in self.phi_nodes:
            phi_layers.append(nn.Linear(prev, h))
            phi_layers.append(nn.ReLU())
            if self.dropout_rate > 0.0:
                phi_layers.append(nn.Dropout(self.dropout_rate))
            prev = h
        self.phi = nn.Sequential(*phi_layers)  # output H = last phi_nodes

        H = self.phi_nodes[-1] if len(self.phi_nodes) else (self.ch_dim + self.evt_dim)

        # psi: per-channel decoder sees [h_c, g, x_evt]
        psi_layers: list[nn.Module] = []
        prev = H + H + self.evt_dim
        for h in self.psi_nodes:
            psi_layers.append(nn.Linear(prev, h))
            psi_layers.append(nn.ReLU())
            if self.dropout_rate > 0.0:
                psi_layers.append(nn.Dropout(self.dropout_rate))
            prev = h
        psi_layers.append(nn.Linear(prev, 1))
        self.psi = nn.Sequential(*psi_layers)

    def forward(self, x_evt: torch.Tensor, x_ch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x_evt: [B, Fevt]
        x_ch:  [B, C, Fch]
        mask:  [B, C] (bool)
        returns: [B, C]
        """
        if x_evt.ndim != 2:
            raise ValueError(f"Expected x_evt [B,Fevt], got {tuple(x_evt.shape)}")
        if x_ch.ndim != 3:
            raise ValueError(f"Expected x_ch [B,C,Fch], got {tuple(x_ch.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B,C], got {tuple(mask.shape)}")

        B, C, Fch = x_ch.shape
        if x_evt.shape[0] != B:
            raise ValueError("x_evt and x_ch batch dims differ")
        if mask.shape[0] != B or mask.shape[1] != C:
            raise ValueError("mask shape must match [B,C]")
        if x_evt.shape[1] != self.evt_dim:
            raise ValueError(f"Expected Fevt={self.evt_dim}, got {x_evt.shape[1]}")
        if Fch != self.ch_dim:
            raise ValueError(f"Expected Fch={self.ch_dim}, got {Fch}")

        # ---- DeepSets core ----
        x_evt_bc = x_evt[:, None, :].expand(B, C, self.evt_dim)             # [B,C,Fevt]

        # phi input: [B,C,Fch+Fevt]
        phi_in = torch.cat([x_ch, x_evt_bc], dim=-1)
        h = self.phi(phi_in)                                                # [B,C,H]

        # pool to get event context g: [B,H]
        g = masked_mean(h, mask=mask, dim=1)                                # [B,H]
        g_bc = g[:, None, :].expand(B, C, g.shape[1])                       # [B,C,H]

        # psi input: [B,C, H + H + Fevt]
        psi_in = torch.cat([h, g_bc, x_evt_bc], dim=-1)
        yhat = self.psi(psi_in).squeeze(-1)                                 # [B,C]
        return yhat

    def get_model_string(self) -> str:
        return self.model_string

    def override_model_string(self, new_model_string: str):
        self.model_string = new_model_string