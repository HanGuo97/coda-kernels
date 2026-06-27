import torch


def preprocess_vector(
    x: torch.Tensor,
    permute: bool,
) -> torch.Tensor:
    if x.dim() == 1:
        x = torch.unsqueeze(x, dim=0)
    if x.dim() != 2:
        raise ValueError("Input must be 2D")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    if permute:
        # Apply permutation from (L, *) -> (*, L) for selected tensors
        x = torch.permute(
            x,
            dims=(1, 0),
        )

    return x


def preprocess_tensor(
    x: torch.Tensor,
    permute: bool,
    transpose: bool = False,
) -> torch.Tensor:
    if x.dim() == 2:
        x = torch.unsqueeze(x, dim=0)
    if x.dim() != 3:
        raise ValueError("Input must be 3D")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    if transpose:
        # e.g., (K, N) -> (N, K) or (L, K, N) -> (L, N, K)
        x = x.mT

    if permute:
        # Apply permutation from (L, *, *) -> (*, *, L) for selected tensors
        x = torch.permute(
            x,
            dims=(1, 2, 0),
        )

    return x
