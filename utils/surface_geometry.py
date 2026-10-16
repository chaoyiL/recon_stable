"""Shared material-grid frames and local normal/slope conversions."""
from __future__ import annotations
from typing import Protocol
import cv2
import numpy as np


class ParametricSurface(Protocol):
    parameter_bounds: tuple[float, float, float, float]

    def evaluate(self, parameters: np.ndarray) -> np.ndarray: ...
    def first_derivatives(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def surface_grid_curve_lengths(xyz: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    """Return representative (width, height) in mm along grid columns and rows.

    Sum 3D edge lengths on each coordinate curve, estimating missing segments
    from that curve's mean visible step. Medians summarize the curve families;
    they do not imply that a curved patch has an isometric planar unfolding.
    """
    xyz = np.asarray(xyz, np.float64)
    valid = np.asarray(valid_mask, bool)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError('XYZ and validity must describe the same surface grid')
    valid = valid & np.isfinite(xyz).all(axis=-1)
    points = np.where(valid[..., None], xyz, 0)
    lengths = []
    for axis in (0, 1):
        distances = np.linalg.norm(np.diff(points, axis=axis), axis=-1)
        supported = (valid[:-1] & valid[1:] if axis == 0
                     else valid[:, :-1] & valid[:, 1:])
        counts = supported.sum(axis=axis)
        totals = np.where(supported, distances, 0).sum(axis=axis)
        curves = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        curves *= valid.shape[axis]-1
        measurable = curves > 1e-12
        if not measurable.any():
            raise ValueError('Surface grid has no measurable coordinate curves')
        lengths.append(float(np.median(curves[measurable])))
    return lengths[1], lengths[0]


def compute_surface_geometry(
    xyz: np.ndarray,valid_mask: np.ndarray,
) -> dict[str,np.ndarray]:
    points=np.asarray(xyz,np.float64)
    valid=np.asarray(valid_mask,dtype=np.bool_)&np.all(np.isfinite(points),axis=-1)
    if points.ndim!=3 or points.shape[-1]!=3 or min(points.shape[:2])<3:
        raise ValueError("xyz 必须是至少 3x3 的规则曲面点阵")
    kernel=np.ones((3,3),np.uint8)
    interior=cv2.erode(
        valid.astype(np.uint8),kernel,iterations=1,
        borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    xu=np.zeros_like(points); xv=np.zeros_like(points)
    xuu=np.zeros_like(points); xvv=np.zeros_like(points); xuv=np.zeros_like(points)
    xu[1:-1,1:-1]=(points[1:-1,2:]-points[1:-1,:-2])/2
    xv[1:-1,1:-1]=(points[2:,1:-1]-points[:-2,1:-1])/2
    xuu[1:-1,1:-1]=(points[1:-1,2:]-2*points[1:-1,1:-1]
                     +points[1:-1,:-2])
    xvv[1:-1,1:-1]=(points[2:,1:-1]-2*points[1:-1,1:-1]
                     +points[:-2,1:-1])
    xuv[1:-1,1:-1]=(points[2:,2:]-points[2:,:-2]
                     -points[:-2,2:]+points[:-2,:-2])/4
    xu_norm=np.linalg.norm(xu,axis=-1)
    cross=np.cross(xu,xv); cross_norm=np.linalg.norm(cross,axis=-1)
    nonsingular=interior&(xu_norm>1e-9)&(cross_norm>1e-12)
    e1=xu/np.maximum(xu_norm[...,None],1e-12)
    normal=cross/np.maximum(cross_norm[...,None],1e-12)
    e2=np.cross(normal,e1)
    reverse=np.sum(e2*xv,axis=-1)<0
    e2=np.where(reverse[...,None],-e2,e2)
    normal=np.where(reverse[...,None],-normal,normal)
    g=np.empty((*points.shape[:2],2,2),np.float64)
    g[...,0,0]=np.sum(xu*xu,axis=-1)
    g[...,0,1]=g[...,1,0]=np.sum(xu*xv,axis=-1)
    g[...,1,1]=np.sum(xv*xv,axis=-1)
    det=g[...,0,0]*g[...,1,1]-g[...,0,1]**2
    nonsingular &= det>1e-14
    g_inverse=np.zeros_like(g)
    safe=np.maximum(det,1e-14)
    g_inverse[...,0,0]=g[...,1,1]/safe
    g_inverse[...,1,1]=g[...,0,0]/safe
    g_inverse[...,0,1]=g_inverse[...,1,0]=-g[...,0,1]/safe
    second=np.empty_like(g)
    second[...,0,0]=np.sum(normal*xuu,axis=-1)
    second[...,0,1]=second[...,1,0]=np.sum(normal*xuv,axis=-1)
    second[...,1,1]=np.sum(normal*xvv,axis=-1)
    jacobian=np.stack([xu,xv],axis=-1)
    frame=np.stack([e1,e2],axis=-1)
    # K = E^T J G^-1
    k_matrix=np.einsum(
        "...ca,...cb,...bd->...ad",frame,jacobian,g_inverse,optimize=True)
    return {
        "valid":nonsingular,"interior":interior,"boundary":valid&~interior,
        "normal":normal,"frame":frame,"jacobian":jacobian,"metric":g,
        "metric_inverse":g_inverse,"second":second,"k":k_matrix,
        "area":np.sqrt(np.maximum(det,0)),
    }



def normals_to_local_slopes(normals, reference_normals, tangent_frames, minimum_dot=0.1):
    normals = np.asarray(normals, np.float64)
    reference = np.asarray(reference_normals, np.float64)
    frames = np.asarray(tangent_frames, np.float64)
    dot = np.sum(normals * reference, axis=-1)
    valid = (np.isfinite(normals).all(axis=-1) & np.isfinite(reference).all(axis=-1)
             & np.isfinite(frames).all(axis=(-1, -2)) & (dot > minimum_dot))
    slopes = -np.einsum("...ca,...c->...a", frames, normals) / np.maximum(dot[..., None], 1e-15)
    valid &= np.isfinite(slopes).all(axis=-1)
    return np.where(valid[..., None], slopes, 0), valid


def local_slopes_to_normals(slopes, reference_normals, tangent_frames):
    slopes = np.asarray(slopes, np.float64)
    normals = np.asarray(reference_normals, np.float64) - np.einsum(
        "...ca,...a->...c", np.asarray(tangent_frames, np.float64), slopes)
    lengths = np.linalg.norm(normals, axis=-1)
    valid = np.isfinite(normals).all(axis=-1) & (lengths > 1e-12)
    return np.where(valid[..., None], normals / np.maximum(lengths[..., None], 1e-15), 0), valid
