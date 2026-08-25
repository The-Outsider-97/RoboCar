
"""
Euclidean Distance Transform for 2D occupancy maps (robotics-friendly).

Features
--------
- Computes true Euclidean distance-to-nearest-obstacle for a 2D binary map.
- Accepts your occupancy grid dict (width/height/resolution/origin/grid).
- Fast O(N) 2-pass Felzenszwalb & Huttenlocher algorithm (per axis).
- Returns distances in **meters** (not cells).
- Optional obstacle inflation helper for safety margins.
- Uses NumPy if available; otherwise falls back to pure Python.

Conventions
-----------
We treat cells as:
  * occupied if grid value >= occupied_threshold (default 50)
  * unknown (-1) treated as obstacles by default (configurable)

Reference
---------
Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions" (2004).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

try:
    import numpy as _np
except Exception:
    _np = None  # pure-Python fallback


def _zeros(shape: Tuple[int, int], fill: float = 0.0):
    h, w = shape
    if _np is not None:
        return _np.full((h, w), fill, dtype=float) if fill != 0.0 else _np.zeros((h, w), dtype=float)
    return [[fill for _ in range(w)] for _ in range(h)]


def _edt_1d(f: Sequence[float]) -> List[float]:
    """
    1D squared Euclidean distance transform for an array `f` where:
      f[i] = 0 at obstacle positions, +inf elsewhere.

    Returns a list of squared distances.
    (Felzenszwalb-Huttenlocher 1D algorithm)
    """
    n = len(f)
    v = [0] * n                 # locations of parabolas in lower envelope
    z = [0.0] * (n + 1)         # locations of boundaries between parabolas
    d = [0.0] * n               # output
    INF = float('inf')

    k = 0
    v[0] = 0
    z[0] = -INF
    z[1] = +INF

    def _val(i):  # safe accessor
        return float(f[i])

    # forward pass: compute lower envelope
    for q in range(1, n):
        s = ((_val(q) + q*q) - (_val(v[k]) + v[k]*v[k])) / (2.0*q - 2.0*v[k])
        while s <= z[k]:
            k -= 1
            s = ((_val(q) + q*q) - (_val(v[k]) + v[k]*v[k])) / (2.0*q - 2.0*v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = +INF

    # backward pass: compute distances
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        d[q] = dq*dq + _val(v[k])
    return d


def edt2d_from_mask(obstacle_mask: Sequence[Sequence[bool]], resolution_m: float = 1.0):
    """
    Compute Euclidean distances (meters) to nearest True cell in `obstacle_mask`.

    Parameters
    ----------
    obstacle_mask : 2D bool array-like (H x W). True marks obstacles.
    resolution_m  : meters per cell.

    Returns
    -------
    distances_m : 2D float array (H x W) of distances in meters.
    """
    if _np is not None:
        obs = _np.asarray(obstacle_mask, dtype=bool)
        H, W = obs.shape
        # No obstacles → +inf
        if not obs.any():
            return _np.full((H, W), _np.inf, dtype=float)
        INF = 1e30
        f = _np.where(obs, 0.0, INF).astype(float)
        # columns
        for x in range(W):
            f[:, x] = _np.asarray(_edt_1d(f[:, x]))
        # rows
        for y in range(H):
            f[y, :] = _np.asarray(_edt_1d(f[y, :]))
        dist = _np.sqrt(f) * float(resolution_m)
        dist[_np.isinf(dist)] = _np.inf
        return dist
    else:
        # Pure Python
        H = len(obstacle_mask); W = len(obstacle_mask[0]) if H else 0
        has_obs = any(any(row) for row in obstacle_mask)
        if not has_obs:
            return [[float('inf') for _ in range(W)] for _ in range(H)]
        INF = 1e30
        f = _zeros((H, W), fill=INF)
        for y in range(H):
            for x in range(W):
                if obstacle_mask[y][x]:
                    f[y][x] = 0.0
        # columns
        for x in range(W):
            col = [f[y][x] for y in range(H)]
            col_dt = _edt_1d(col)
            for y in range(H):
                f[y][x] = col_dt[y]
        # rows
        for y in range(H):
            row_dt = _edt_1d(f[y])
            f[y] = row_dt
        out = _zeros((H, W))
        for y in range(H):
            for x in range(W):
                out[y][x] = (f[y][x] ** 0.5) * float(resolution_m)
        return out


def mask_from_occupancy(occ: Union[Dict[str, Any], Any],
                        occupied_threshold: int = 50,
                        treat_unknown_as_obstacle: bool = True):
    """
    Convert an occupancy grid (dict or object) into a boolean obstacle mask.

    Parameters
    ----------
    occ : dict with keys 'width','height','resolution','grid' (row-major), optional 'origin_x','origin_y'.
          Or an object with attributes of the same names.
    occupied_threshold : cells >= this value are obstacles (default 50).
    treat_unknown_as_obstacle : if True, -1 cells are obstacles; else treated as free.

    Returns
    -------
    mask : 2D bool array (H x W), True where obstacle.
    meta : (width, height, resolution, origin_x, origin_y)
    """
    if isinstance(occ, dict):
        W = int(occ["width"]); H = int(occ["height"]); res = float(occ["resolution"])
        ox = float(occ.get("origin_x", 0.0)); oy = float(occ.get("origin_y", 0.0))
        grid = occ["grid"]
    else:
        W = int(getattr(occ, "width"))
        H = int(getattr(occ, "height"))
        res = float(getattr(occ, "resolution"))
        ox = float(getattr(occ, "origin_x", 0.0))
        oy = float(getattr(occ, "origin_y", 0.0))
        grid = getattr(occ, "grid")

    if _np is not None:
        arr = _np.asarray(grid, dtype=float).reshape(H, W)
        if treat_unknown_as_obstacle:
            mask = (arr >= occupied_threshold) | (arr < 0)
        else:
            mask = (arr >= occupied_threshold)
        return mask, (W, H, res, ox, oy)
    else:
        mask = [[False for _ in range(W)] for _ in range(H)]
        i = 0
        for y in range(H):
            for x in range(W):
                v = float(grid[i]); i += 1
                if v < 0 and treat_unknown_as_obstacle:
                    mask[y][x] = True
                elif v >= occupied_threshold:
                    mask[y][x] = True
        return mask, (W, H, res, ox, oy)


def distance_map_from_occupancy(occ: Union[Dict[str, Any], Any],
                                occupied_threshold: int = 50,
                                treat_unknown_as_obstacle: bool = True):
    """
    Compute Euclidean distance map (meters) directly from an occupancy grid.

    Returns
    -------
    distances_m : 2D float array (H x W), distances in meters (np.ndarray or list of lists).
    meta        : (width, height, resolution, origin_x, origin_y)
    """
    mask, meta = mask_from_occupancy(occ, occupied_threshold, treat_unknown_as_obstacle)
    W, H, res, ox, oy = meta
    dist_m = edt2d_from_mask(mask, resolution_m=res)
    return dist_m, meta


def inflate_obstacles(distance_map_m, inflation_radius_m: float):
    """
    Create an inflated obstacle mask by thresholding the distance map.

    Parameters
    ----------
    distance_map_m : 2D float array (H x W) in meters (output of EDT).
    inflation_radius_m : cells with distance < radius become obstacles (True).

    Returns
    -------
    mask : 2D bool array (H x W) where True indicates inflated obstacles.
    """
    r = float(inflation_radius_m)
    if r <= 0.0:
        if _np is not None:
            return _np.zeros_like(distance_map_m, dtype=bool)
        # shape preserving zeros
        H = len(distance_map_m); W = len(distance_map_m[0]) if H else 0
        return [[False for _ in range(W)] for _ in range(H)]

    if _np is not None:
        dm = _np.asarray(distance_map_m, dtype=float)
        return (dm < r)
    else:
        H = len(distance_map_m); W = len(distance_map_m[0]) if H else 0
        mask = [[False for _ in range(W)] for _ in range(H)]
        for y in range(H):
            row = distance_map_m[y]
            for x in range(W):
                mask[y][x] = (float(row[x]) < r)
        return mask
