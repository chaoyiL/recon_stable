"""Frozen reference arc-length coordinates for arbitrary parametric patches."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree
from utils.surface_geometry import ParametricSurface
from utils.surface_projection import CameraModel
from utils.surface_sampling import bilinear_sample

COORDINATE_KIND = 'reference_arclength'
FRAME_CONVENTION = 'st_grid_columns_rows'


def measure_coordinate_curve_lengths(surface: ParametricSurface, parameter_bounds, integration_shape) -> dict[str, np.ndarray]:
    amin, amax, bmin, bmax = parameter_bounds
    nb, na = integration_shape
    if not np.isfinite(parameter_bounds).all() or amax <= amin or bmax <= bmin:
        raise ValueError('Invalid parameter bounds')
    a, b = np.linspace(amin, amax, na), np.linspace(bmin, bmax, nb)
    aa, bb = np.meshgrid(a, b)
    da, db = surface.first_derivatives(np.column_stack((aa.ravel(), bb.ravel())))
    da, db = np.asarray(da, np.float64).reshape(nb, na, 3), np.asarray(db, np.float64).reshape(nb, na, 3)
    speed_a, speed_b = np.linalg.norm(da, axis=-1), np.linalg.norm(db, axis=-1)
    area = np.linalg.norm(np.cross(da, db), axis=-1)
    if (not np.isfinite(da).all() or not np.isfinite(db).all()
            or np.any(speed_a <= 1e-12) or np.any(speed_b <= 1e-12)
            or np.any(area <= 1e-12 * speed_a * speed_b)):
        raise ValueError('Degenerate surface metric or coordinate speed')
    la = cumulative_trapezoid(speed_a, x=a, axis=1, initial=0)
    lb = cumulative_trapezoid(speed_b, x=b, axis=0, initial=0)
    if np.any(la[:, -1] <= 1e-12) or np.any(lb[-1, :] <= 1e-12):
        raise ValueError('Degenerate coordinate curve length')
    return dict(a_axis=a, b_axis=b, arc_lengths_mm=np.stack((la, lb), axis=-1),
                row_lengths_mm=la[:, -1], column_lengths_mm=lb[-1, :])


@dataclass(frozen=True)
class ArcLengthChart:
    chart_id: str
    parameter_bounds: tuple[float, float, float, float]
    a_axis: np.ndarray
    b_axis: np.ndarray
    st_table: np.ndarray
    arc_lengths_mm: np.ndarray
    row_lengths_mm: np.ndarray
    column_lengths_mm: np.ndarray
    coordinate_kind: str = COORDINATE_KIND
    _tree: object = field(default=None, init=False, repr=False, compare=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.chart_id or self.coordinate_kind != COORDINATE_KIND:
            raise ValueError('Invalid coordinate chart identity')
        bounds = np.asarray(self.parameter_bounds, np.float64)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[[1, 3]] <= bounds[[0, 2]]):
            raise ValueError('Invalid coordinate chart bounds')
        object.__setattr__(self, 'parameter_bounds', tuple(bounds))
        for name in ('a_axis', 'b_axis', 'st_table', 'arc_lengths_mm',
                     'row_lengths_mm', 'column_lengths_mm'):
            value = np.array(getattr(self, name), np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite chart {name}')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        na, nb = len(self.a_axis), len(self.b_axis)
        if min(na, nb) < 2 or self.st_table.shape != (nb, na, 2) or self.arc_lengths_mm.shape != (nb, na, 2):
            raise ValueError('Invalid chart table shape')
        if (self.row_lengths_mm.shape != (nb,) or self.column_lengths_mm.shape != (na,)
                or np.any(self.row_lengths_mm <= 0) or np.any(self.column_lengths_mm <= 0)):
            raise ValueError('Invalid chart lengths')
        if (not np.allclose(self.a_axis, np.linspace(bounds[0], bounds[1], na), rtol=0, atol=1e-12)
                or not np.allclose(self.b_axis, np.linspace(bounds[2], bounds[3], nb), rtol=0, atol=1e-12)):
            raise ValueError('Chart native axes must be uniform and match bounds')
        arcs = self.arc_lengths_mm
        expected = np.stack((arcs[..., 0]/self.row_lengths_mm[:, None],
                             arcs[..., 1]/self.column_lengths_mm[None, :]), axis=-1)
        if (not np.allclose(arcs[:, -1, 0], self.row_lengths_mm, atol=1e-12, rtol=1e-12)
                or not np.allclose(arcs[-1, :, 1], self.column_lengths_mm, atol=1e-12, rtol=1e-12)
                or not np.allclose(expected, self.st_table, atol=1e-12, rtol=1e-12)):
            raise ValueError('Coordinate chart lengths and st table differ')
        st = self.st_table
        if (not np.allclose(st[:, 0, 0], 0, atol=1e-12, rtol=0)
                or not np.allclose(st[:, -1, 0], 1, atol=1e-12, rtol=0)
                or not np.allclose(st[0, :, 1], 0, atol=1e-12, rtol=0)
                or not np.allclose(st[-1, :, 1], 1, atol=1e-12, rtol=0)):
            raise ValueError('Coordinate chart boundaries are not the unit square')
        p, q, r, z = st[:-1, :-1], st[:-1, 1:], st[1:, :-1], st[1:, 1:]
        for du in (q-p, z-r):
            for dv in (r-p, z-q):
                det = (du[..., 0]*dv[..., 1]-du[..., 1]*dv[..., 0])*(na-1)*(nb-1)
                if np.any(det <= 1e-8):
                    raise ValueError('Coordinate chart is not invertible: cell Jacobian')

    def _evaluate_normalized(self, normalized, with_jacobian=False):
        nb, na = self.st_table.shape[:2]
        sample = bilinear_sample(self.st_table, np.asarray(normalized)*[na-1, nb-1],
                                 allow_boundary=True, with_jacobian=with_jacobian)
        jac = sample.jacobian
        if jac is not None:
            jac = jac * np.array([na-1, nb-1])
        return sample.values, sample.valid, jac

    def to_st(self, parameters):
        lo = np.array(self.parameter_bounds)[[0, 2]]
        span = np.array(self.parameter_bounds)[[1, 3]] - lo
        values, valid, _ = self._evaluate_normalized((np.asarray(parameters)-lo)/span)
        return values, valid

    def jacobian(self, parameters):
        lo = np.array(self.parameter_bounds)[[0, 2]]
        span = np.array(self.parameter_bounds)[[1, 3]] - lo
        _, valid, jac = self._evaluate_normalized((np.asarray(parameters)-lo)/span, True)
        return jac / span, valid

    def to_parameters(self, st):
        st = np.asarray(st, np.float64)
        if st.shape[-1:] != (2,):
            raise ValueError('st must end in two coordinates')
        shape = st.shape
        target = st.reshape(-1, 2)
        valid = np.isfinite(target).all(1) & (target >= -1e-12).all(1) & (target <= 1+1e-12).all(1)
        safe = np.where(valid[:, None], np.clip(target, 0, 1), 0)
        if self._tree is None:
            object.__setattr__(self, '_tree', cKDTree(self.st_table.reshape(-1, 2)))
        index = self._tree.query(safe)[1]
        nb, na = self.st_table.shape[:2]
        xy = np.column_stack((index % na / (na-1), index // na / (nb-1)))
        for _ in range(16):
            value, ok, jac = self._evaluate_normalized(xy, True)
            residual = value-safe
            active = valid & ok & (np.max(np.abs(residual), axis=1) > 1e-8)
            if not active.any():
                break
            ids = np.flatnonzero(active)
            step = np.linalg.solve(jac[ids], residual[ids, :, None])[..., 0]
            old_error = np.max(np.abs(residual[ids]), axis=1)
            accepted = np.zeros(len(ids), bool)
            for backtrack in range(8):
                candidate = xy[ids]-step*(0.5**backtrack)
                inside = (candidate >= -1e-12).all(1) & (candidate <= 1+1e-12).all(1)
                candidate = np.clip(candidate, 0, 1)
                value, ok, _ = self._evaluate_normalized(candidate)
                improve = inside & ok & (np.max(np.abs(value-safe[ids]), axis=1) < old_error) & ~accepted
                xy[ids[improve]] = candidate[improve]
                accepted |= improve
                if accepted.all():
                    break
            valid[ids[~accepted]] = False
        value, ok, _ = self._evaluate_normalized(xy)
        valid &= ok & (np.max(np.abs(value-safe), axis=1) <= 1e-8)
        lo = np.array(self.parameter_bounds)[[0, 2]]
        span = np.array(self.parameter_bounds)[[1, 3]]-lo
        parameters = lo+xy*span
        return np.where(valid[:, None], parameters, np.nan).reshape(shape), valid.reshape(shape[:-1])


def build_arclength_chart(surface: ParametricSurface, *, initial_size=129, maximum_size=1025, tolerance=1e-5) -> ArcLengthChart:
    if initial_size < 3 or maximum_size < 2*initial_size-1 or tolerance <= 0:
        raise ValueError('Invalid chart integration settings')
    previous = None
    size = initial_size
    while size <= maximum_size:
        lengths = measure_coordinate_curve_lengths(surface, surface.parameter_bounds, (size, size))
        arcs = lengths['arc_lengths_mm']
        st = np.stack((arcs[..., 0]/lengths['row_lengths_mm'][:, None],
                       arcs[..., 1]/lengths['column_lengths_mm'][None, :]), axis=-1)
        if previous is not None:
            old_st, old = previous
            error_st = np.max(np.abs(st[::2, ::2]-old_st))
            error_lengths = max(np.max(np.abs(lengths[key][::2]/old[key]-1))
                                for key in ('row_lengths_mm', 'column_lengths_mm'))
            if error_st <= tolerance and error_lengths <= tolerance:
                return ArcLengthChart(str(uuid4()), tuple(surface.parameter_bounds), st_table=st, **lengths)
        previous = st, lengths
        size = 2*size-1
    raise ValueError('Reference arc-length integration did not converge')


@dataclass(frozen=True)
class SurfaceGrid:
    parameters: np.ndarray
    st: np.ndarray
    xyz: np.ndarray
    coordinate_valid: np.ndarray


def sample_surface_grid(surface: ParametricSurface, chart: ArcLengthChart, rows: int, columns: int) -> SurfaceGrid:
    if min(rows, columns) < 2:
        raise ValueError('Surface grids require at least 2x2 points')
    if not np.allclose(surface.parameter_bounds, chart.parameter_bounds, rtol=0, atol=1e-12):
        raise ValueError('Surface and reference parameter bounds differ')
    key = chart.chart_id, rows, columns
    if key not in chart._cache:
        ss, tt = np.meshgrid(np.linspace(0, 1, columns), np.linspace(0, 1, rows))
        st = np.stack((ss, tt), axis=-1)
        parameters, valid = chart.to_parameters(st)
        if not valid.all():
            raise ValueError('Regular st grid inversion failed')
        for a in (st, parameters, valid):
            a.setflags(write=False)
        chart._cache[key] = parameters, st, valid
    parameters, st, valid = chart._cache[key]
    xyz = np.asarray(surface.evaluate(parameters.reshape(-1, 2)), np.float64).reshape(rows, columns, 3)
    return SurfaceGrid(parameters, st, xyz, valid & np.isfinite(xyz).all(axis=-1))


def build_material_sampling_grid(
    surface: ParametricSurface,
    chart: ArcLengthChart,
    shape: tuple[int, int],
    template_xy_mm: np.ndarray,
    camera: CameraModel,
) -> dict[str, np.ndarray]:
    """Build frozen-grid NPZ arrays from this reference chart, never an older grid.

    shape is (rows, columns). material_coordinates are native (a, b) in mm;
    st is the regular dimensionless reference arc-length grid. Both are float64.
    Camera/template arrays bind the grid to the reconstruction geometry.
    """
    grid = sample_surface_grid(surface, chart, *shape)
    if not grid.coordinate_valid.all():
        raise ValueError('Reference material sampling grid must have finite geometry everywhere')
    return dict(
        format_version=np.asarray(1, np.int32), grid_kind=np.asarray('frozen_material_grid'),
        material_coordinates=grid.parameters, st=grid.st,
        coordinate_kind=np.asarray(chart.coordinate_kind), coordinate_chart_id=np.asarray(chart.chart_id),
        frame_convention=np.asarray(FRAME_CONVENTION), residual_domain=np.asarray('material_grid'),
        color_space=np.asarray('linear_rgb'), template_xy_mm=np.asarray(template_xy_mm, np.float64),
        camera_matrix=camera.matrix, dist_coeffs=camera.distortion,
        image_shape=np.asarray(camera.image_shape, np.int32),
    )


def save_arclength_chart(path: str | Path, chart: ArcLengthChart) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {key: getattr(chart, key) for key in ('chart_id', 'coordinate_kind', 'parameter_bounds',
              'a_axis', 'b_axis', 'st_table', 'arc_lengths_mm', 'row_lengths_mm', 'column_lengths_mm')}
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, format_version=np.int32(1), **values)
    temporary.replace(path)
    return path


def load_arclength_chart(path: str | Path) -> ArcLengthChart:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        if int(data['format_version']) != 1:
            raise ValueError('Unsupported coordinate chart version')
        values = {key: data[key] for key in ('parameter_bounds', 'a_axis', 'b_axis', 'st_table',
                  'arc_lengths_mm', 'row_lengths_mm', 'column_lengths_mm')}
        return ArcLengthChart(str(data['chart_id']), coordinate_kind=str(data['coordinate_kind']), **values)
