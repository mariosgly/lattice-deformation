import torch

def bspline_weights_torch(t: torch.Tensor) -> torch.Tensor:
    """
    Compute the 4 cubic B-spline basis weights for fractional coords t in [0,1].
    t: (B, N) => B = batch_size, N = number_of_vertices
    returns (B, N, 4)
    """
    # Ensure input is clamped to [0,1]
    t = torch.clamp(t, 0.0, 1.0)
    
    t2 = t * t
    t3 = t2 * t
    
    w0 = (1 - 3*t + 3*t2 - t3) / 6.0
    w1 = (4 - 6*t2 + 3*t3) / 6.0
    w2 = (1 + 3*t + 3*t2 - 3*t3) / 6.0
    w3 = t3 / 6.0
    
    return torch.stack([w0, w1, w2, w3], dim=-1)  # (B, N, 4)

def build_corner_table_torch(corner_offsets: torch.Tensor) -> torch.Tensor:
    """
    Build a (B, 4, 4, 4, 3) array for the offsets used by each (dU,dV,dW) in [-1..2].
    corner_offsets: (B, 2, 2, 2, 3) where B is batch size
    returns: (B, 4, 4, 4, 3) corner table
    """
    B = corner_offsets.shape[0]
    device = corner_offsets.device
    corner_table = torch.zeros((B, 4, 4, 4, 3), device=device)
    
    for dW in range(-1, 3):  # -1..2
        for dV in range(-1, 3):
            for dU in range(-1, 3):
                # clamp to 0..1
                w_idx = max(0, min(1, dW))
                v_idx = max(0, min(1, dV))
                u_idx = max(0, min(1, dU))
                corner_table[:, dW+1, dV+1, dU+1] = corner_offsets[:, u_idx, v_idx, w_idx]
    
    return corner_table

def bspline_2x2x2_deform_torch(
    vertices: torch.Tensor,           # (B, N, 3)
    corner_offsets: torch.Tensor,     # (B, 2, 2, 2, 3)
    box_min: torch.Tensor,            # (3,) or (B, 3)
    box_max: torch.Tensor             # (3,) or (B, 3)
) -> torch.Tensor:
    """
    Batched 2×2×2 B-spline deformation. Returns (B, N, 3).
    corner_offsets[b, i, j, k] = offset (3D) for corner (i,j,k) in {0,1}
    """
    B, N, _ = vertices.shape
    device = vertices.device
    
    # Build corner table once
    corner_table = build_corner_table_torch(corner_offsets)  # (B, 4, 4, 4, 3)
    
    # Ensure box_min, box_max can broadcast to (B,1,3)
    if box_min.dim() == 1:
        box_min = box_min.view(1, 1, 3)
        box_max = box_max.view(1, 1, 3)
    else:
        box_min = box_min.view(B, 1, 3)
        box_max = box_max.view(B, 1, 3)
    
    # Normalize points to [0,1] in each axis
    box_size = box_max - box_min
    normed = (vertices - box_min) / box_size
    normed = torch.clamp(normed, 0.0, 1.0)
    
    # Get integer and fractional parts
    int_part = torch.floor(normed).long()  # (B, N, 3)
    frac = normed - int_part.float()       # (B, N, 3)
    
    # Compute B-spline weights for each axis
    Wu = bspline_weights_torch(frac[..., 0])  # (B, N, 4)
    Wv = bspline_weights_torch(frac[..., 1])  # (B, N, 4)
    Ww = bspline_weights_torch(frac[..., 2])  # (B, N, 4)
    
    # Accumulate deformation
    delta = torch.zeros_like(vertices)  # (B, N, 3)
    
    # For all 64 combinations (4x4x4)
    for dW in range(4):
        w_weight = Ww[..., dW].unsqueeze(-1)  # (B, N, 1)
        for dV in range(4):
            v_weight = Wv[..., dV].unsqueeze(-1)  # (B, N, 1)
            wv_weight = w_weight * v_weight       # (B, N, 1)
            for dU in range(4):
                u_weight = Wu[..., dU].unsqueeze(-1)  # (B, N, 1)
                weight = wv_weight * u_weight         # (B, N, 1)
                
                # Get offset from corner table (B, 4, 4, 4, 3) -> (B, 1, 3)
                offset = corner_table[:, dW, dV, dU].unsqueeze(1)  # (B, 1, 3)
                
                # Add weighted offset to all vertices
                delta += weight * offset
    
    # Add offset to original points
    deformed = vertices + delta
    return deformed