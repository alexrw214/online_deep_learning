from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class WeightedWaypointLoss(nn.Module):
    def __init__(self, weight_lat: float = 1.0, weight_lon: float = 1.0, reduction: str = 'mean'):
        """
        Custom weighted MSE loss for waypoint regression with separate control over
        lateral (y) and longitudinal (x) components.

        Args:
            weight_lat (float): Weight for the lateral (y-axis) component.
            weight_lon (float): Weight for the longitudinal (x-axis) component.
            reduction (str): Specifies the reduction to apply: 'mean' | 'sum'.
        """
        super().__init__()
        self.weight_lat = weight_lat
        self.weight_lon = weight_lon
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted waypoints (B, n_waypoints, 2)
            target (torch.Tensor): Ground truth waypoints (B, n_waypoints, 2)
            mask (torch.Tensor): Valid waypoints mask (B, n_waypoints)
        Returns:
            torch.Tensor: The weighted MSE loss
        """
        # Compute squared error separately for longitudinal (x) and lateral (y)
        error = (pred - target) ** 2  # shape: (B, n_waypoints, 2)
        lon_loss = error[..., 0] * self.weight_lon  # x-axis
        lat_loss = error[..., 1] * self.weight_lat  # y-axis
        loss = lon_loss + lat_loss  # shape: (B, n_waypoints)

        if mask is not None:
            mask = mask.float()
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / mask.sum()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()

        return loss


class MLPPlanner(nn.Module):
    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
            transform_pipeline: nn.Module = None,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()
        # handle transform_pipeline
        self.transform_pipeline = transform_pipeline
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.input_dim = n_track * 2 * 2
        self.output_dim = n_waypoints * 2
        self.hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.mlp.apply(self._init_weights)
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
            self,
            track_left: torch.Tensor,
            track_right: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate the left and right track points
        x = torch.cat([track_left, track_right], dim=-1)
        # Flatten the input tensor
        x = x.view(x.size(0), -1)

        # Pass through the MLP
        x = self.mlp(x)
        # Reshape the output to (b, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)
        return x


import torch
import torch.nn as nn


class TransformerPlanner(nn.Module):
    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
            d_model: int = 128,
            nhead: int = 8,
            num_layers: int = 4,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input projection: from 4D (track_left + track_right) to d_model
        self.input_proj = nn.Linear(8, d_model)

        # Transformer decoder layers (cross-attention from queries to track features)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Learned query embeddings (1 per waypoint)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Output projection to 2D coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def add_positional_encoding(self, x):
        """
        Adds sinusoidal positional encoding to track inputs.

        Args:
            x: Tensor of shape (B, N, D=4) representing concatenated track points

        Returns:
            Tensor of shape (B, N, D + 4) with positional encoding appended
        """
        B, N, D = x.shape
        device = x.device

        # Positional indices
        pos = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)

        # Frequencies: base 10000
        freqs = torch.pow(10000, -torch.arange(0, 4, 2, dtype=torch.float32, device=device) / 4)
        angles = pos * freqs  # (N, 2)

        # Sin/Cos pair
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (N, 4)
        pe = pe.unsqueeze(0).expand(B, -1, -1)  # (B, N, 4)

        # Concatenate to input
        return torch.cat([x, pe], dim=-1)  # (B, N, 8)

    def forward(
            self,
            track_left: torch.Tensor,  # shape (B, n_track, 2)
            track_right: torch.Tensor,  # shape (B, n_track, 2)
            **kwargs,
    ) -> torch.Tensor:
        B = track_left.shape[0]

        # Combine track sides: (B, n_track, 4)
        track_input = torch.cat([track_left, track_right], dim=-1)
        track_input = self.add_positional_encoding(track_input)  # → (B, n_track, 8)

        # Project to model dimension: (B, n_track, d_model)
        memory = self.input_proj(track_input)

        # Prepare queries: (B, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Cross-attend queries to encoded memory
        decoded = self.decoder(tgt=queries, memory=memory)  # (B, n_waypoints, d_model)

        # Project to output waypoints
        waypoints = self.output_proj(decoded)  # (B, n_waypoints, 2)
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
            self,
            n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
