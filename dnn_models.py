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