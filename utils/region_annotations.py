"""Image-space region annotations, independent of camera and contact geometry."""
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import cv2
import numpy as np
import yaml


BOUNDARY_SAMPLE_COUNT = 128
EDITOR_VERTEX_COUNT = 16


@dataclass
class RegionAnnotation:
    image: str
    image_size: tuple[int, int]
    status: str = 'draft'
    center_px: tuple[float, float] = (0., 0.)
    shape: str = 'ellipse'
    semi_axes_px: tuple[float, float] = (40., 30.)
    angle_degrees: float = 0.
    contour_px: list | None = None
    semantics: str = 'region'

    def __post_init__(self):
        if self.status not in {'draft', 'accepted', 'skip'} or self.shape not in {'circle', 'ellipse', 'polygon'}:
            raise ValueError('Invalid annotation status or shape')
        if (len(self.image_size) != 2 or min(self.image_size) <= 0
                or len(self.center_px) != 2 or not np.isfinite(self.center_px).all()
                or not np.isfinite(self.angle_degrees) or len(self.semi_axes_px) != 2
                or not np.isfinite(self.semi_axes_px).all() or min(self.semi_axes_px) <= 0):
            raise ValueError('Invalid image size or region geometry')
        if self.shape == 'circle' and self.semi_axes_px[0] != self.semi_axes_px[1]:
            raise ValueError('A circle needs two equal semi_axes_px radii')
        if self.shape == 'polygon':
            contour = np.asarray(self.contour_px, np.float64)
            if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3 or not np.isfinite(contour).all():
                raise ValueError('A polygon needs at least three finite image-space vertices')


def load_regions(path, image_root=None):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('version') != 2:
        raise ValueError('Annotations must use version 2 and explicit region semantics')
    root = (path.parent / value['image_root']).resolve()
    target_root = root if image_root is None else Path(image_root).resolve()
    regions = {}
    for raw in value['regions']:
        annotation = RegionAnnotation(**raw)
        full = (root/annotation.image).resolve()
        key = Path(os.path.relpath(full, target_root)).as_posix()
        if key in regions:
            raise ValueError(f'Duplicate annotation key: {key}')
        annotation.image = key
        regions[key] = annotation
    return regions


def _yaml_native(value):
    """Keep annotation YAML portable when edits produce NumPy coordinates."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _yaml_native(value.tolist())
    if isinstance(value, dict):
        return {key: _yaml_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_native(item) for item in value]
    return value


def save_regions(path, regions, *, image_root):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = regions.values() if isinstance(regions, dict) else regions
    value = dict(version=2, image_root=os.path.relpath(Path(image_root).resolve(), path.parent),
                 regions=[asdict(row) for row in rows])
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(yaml.safe_dump(_yaml_native(value), allow_unicode=True, sort_keys=False), encoding='utf-8')
    temporary.replace(path)
    return path


def region_boundary_points(region: RegionAnnotation, count=BOUNDARY_SAMPLE_COUNT) -> np.ndarray:
    if count < 3:
        raise ValueError('At least three boundary points are required')
    if region.shape in {'circle', 'ellipse'}:
        angles = np.arange(count)*2*np.pi/count
        local = np.column_stack((np.cos(angles), np.sin(angles)))*region.semi_axes_px
        theta = 0. if region.shape == 'circle' else np.deg2rad(region.angle_degrees)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return local @ rotation.T+region.center_px
    polygon = np.asarray(region.contour_px, np.float64)
    edges = np.roll(polygon, -1, axis=0)-polygon
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError('Polygon has repeated adjacent vertices')
    cumulative = np.r_[0, np.cumsum(lengths)]
    samples = np.arange(count)*cumulative[-1]/count
    ids = np.minimum(np.searchsorted(cumulative, samples, side='right')-1, len(polygon)-1)
    return polygon[ids]+edges[ids]*((samples-cumulative[ids])/lengths[ids])[:, None]


def region_masks(region, image_shape, *, outer_margin_px):
    if tuple(region.image_size) != tuple(image_shape[:2][::-1]):
        raise ValueError('Annotation image_size differs from source image')
    contour = region_boundary_points(region, 512)
    mask = np.zeros(image_shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 1)
    if not cv2.pointPolygonTest(contour.astype(np.float32), tuple(map(float, region.center_px)), False) > 0:
        raise ValueError('Representative contact center must be inside boundary')
    def kernel(radius):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    excluded = cv2.dilate(mask, kernel(outer_margin_px))
    return dict(contact=mask.astype(bool), exclusion=excluded.astype(bool))
