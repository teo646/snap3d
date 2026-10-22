"""Orbit camera + GL matrices, shared by the interactive viewers and the export stage.

The orbit turns about an explicit `up` axis (PosedImageSet.mean_up_vector,
viewers/_common.capture_up_vector) rather than world +Y, since a reconstruction's world frame
has no up of its own.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_UP", "OrbitCamera", "horizontal_basis", "orbit_angles", "look_at", "perspective"]

DEFAULT_UP = np.array([0.0, 1.0, 0.0])

# Seeds for the horizontal reference axis, tried in order; 0.9 means "not nearly parallel".
_BASIS_SEEDS = (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
_BASIS_MAX_ALIGNMENT = 0.9


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(right, forward): an orthonormal pair spanning the plane perpendicular to `up`."""
    seed = next(s for s in _BASIS_SEEDS if abs(float(s @ up)) < _BASIS_MAX_ALIGNMENT)
    forward = seed - float(seed @ up) * up
    forward /= np.linalg.norm(forward)
    return np.cross(up, forward), forward


class OrbitCamera:
    """Spherical orbit around `origin` about the `up` axis."""

    def __init__(self, origin: np.ndarray, radius: float, up: np.ndarray = DEFAULT_UP):
        self.origin = origin.astype(np.float64)
        self.radius = float(radius)
        self.azimuth = 45.0
        self.elevation = 20.0
        self.up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        self._right, self._forward = horizontal_basis(self.up)

    @property
    def position(self) -> np.ndarray:
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        d = np.cos(el) * (np.sin(az) * self._right + np.cos(az) * self._forward) + np.sin(el) * self.up
        return self.origin + self.radius * d

    @property
    def view_up(self) -> np.ndarray:
        """The up vector the view is drawn with: `up`, or `-up` once the camera has gone over the pole."""
        return self.up if np.cos(np.radians(self.elevation)) >= 0.0 else -self.up

    def rotate(self, dx_deg: float, dy_deg: float) -> None:
        """Orbit by a drag of (`dx_deg`, `dy_deg`) with no limit, over the poles. Elevation is
        wrapped to (-180, 180] and never left on the pole."""
        flipped = np.cos(np.radians(self.elevation)) < 0.0
        self.azimuth += dx_deg if flipped else -dx_deg
        elevation = self.elevation + dy_deg
        elevation = (elevation + 180.0) % 360.0 - 180.0
        if abs(abs(elevation) - 90.0) < 1e-3:  # look_at degenerates along `up`
            elevation += 2e-3
        self.elevation = elevation
        self.azimuth = (self.azimuth + 180.0) % 360.0 - 180.0

    @property
    def forward_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, up) in the current view direction, for WASD panning."""
        view_dir = self.origin - self.position
        view_dir /= np.linalg.norm(view_dir) + 1e-10
        right = np.cross(view_dir, self.view_up)
        right /= np.linalg.norm(right) + 1e-10
        up = np.cross(right, view_dir)
        return right, up

    def view_matrix(self) -> np.ndarray:
        """The GL view matrix looking at `origin` from `position`, `up` upright."""
        return look_at(self.position.astype(np.float32), self.origin.astype(np.float32), self.view_up.astype(np.float32))

    def roll(self, angle_deg: float) -> None:
        """Tilt `up` by rotating it about the current view axis, keeping position fixed. `_right`
        and `_forward` rotate by the same amount."""
        view_dir = self.origin - self.position
        view_dir /= np.linalg.norm(view_dir) + 1e-10
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        def rotated(v: np.ndarray) -> np.ndarray:
            return v * cos_t + np.cross(view_dir, v) * sin_t + view_dir * (view_dir @ v) * (1.0 - cos_t)

        self.up = rotated(self.up)
        self._right = rotated(self._right)
        self._forward = rotated(self._forward)


def orbit_angles(direction: np.ndarray, up: np.ndarray) -> tuple[float, float]:
    """(azimuth, elevation) in degrees such that an OrbitCamera with `up` looks from `direction`."""
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    d = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    right, forward = horizontal_basis(up)
    elevation = np.degrees(np.arcsin(np.clip(d @ up, -1.0, 1.0)))
    horizontal = d - (d @ up) * up
    norm = np.linalg.norm(horizontal)
    if norm < 1e-9:  # straight down the up axis; azimuth is arbitrary
        return 0.0, float(elevation)
    horizontal /= norm
    return float(np.degrees(np.arctan2(horizontal @ right, horizontal @ forward))), float(elevation)


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-8
    u = np.cross(r, f)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, :3] = r
    M[0, 3] = -(r @ eye)
    M[1, :3] = u
    M[1, 3] = -(u @ eye)
    M[2, :3] = -f
    M[2, 3] = f @ eye
    M[3, 3] = 1.0
    return M


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = 2.0 * far * near / (near - far)
    M[3, 2] = -1.0
    return M
