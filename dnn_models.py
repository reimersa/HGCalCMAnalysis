import torch # type: ignore
import torch.nn as nn # type: ignore


def _build_model_string(prefix: str, input_dim: int, nodes_per_layer: list[int], dropout_rate: float, tag: str) -> str:
    hidden_part = "-".join(str(n) for n in nodes_per_layer)
    dropout_part = f"dr{dropout_rate:g}"
    str_tag = f"{tag}_" if tag else ""
    return f"{str_tag}{prefix}_F{input_dim}__{hidden_part}__{dropout_part}"


def _build_hidden_layers(input_dim: int, nodes_per_layer: list[int], dropout_rate: float) -> tuple[list[nn.Module], int]:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h in nodes_per_layer:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        prev_dim = h
    return layers, prev_dim


class PerChannelDNN(nn.Module):

    def __init__(self, input_dim: int, nodes_per_layer: list[int], dropout_rate: float = 0.0, tag: str = ""):
        super().__init__()

        self.input_dim       = input_dim   # C
        self.nodes_per_layer = nodes_per_layer
        self.dropout_rate    = dropout_rate
        self.tag             = tag

        # model name (for saving paths etc.)
        self.model_string = _build_model_string(
            prefix="perchannel",
            input_dim=self.input_dim,
            nodes_per_layer=self.nodes_per_layer,
            dropout_rate=self.dropout_rate,
            tag=self.tag,
        )

        layers, prev_dim = _build_hidden_layers(
            input_dim=self.input_dim,
            nodes_per_layer=self.nodes_per_layer,
            dropout_rate=self.dropout_rate,
        )
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


def build_per_channel_model(
    input_dim: int,
    nodes_per_layer: list[int],
    dropout_rate: float = 0.0,
    tag: str = "",
) -> nn.Module:
    return PerChannelDNN(
        input_dim=input_dim,
        nodes_per_layer=nodes_per_layer,
        dropout_rate=dropout_rate,
        tag=tag,
    )
