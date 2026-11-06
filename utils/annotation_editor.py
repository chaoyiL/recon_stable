"""Callback-driven, three-panel region editor. No calibration imports."""
from copy import deepcopy
from dataclasses import asdict
from typing import Callable
import cv2
import numpy as np
from utils.region_annotations import (BOUNDARY_SAMPLE_COUNT, EDITOR_VERTEX_COUNT,
                                      RegionAnnotation, region_boundary_points)
from utils.annotation_window import AnnotationWindow, compose_annotation_panels


# Repeated surface fits may differ by insignificant floating-point noise.
_CONTOUR_TOLERANCE_PX = 1e-7


def _same_state(left, right):
    for name, value in left.items():
        other = right[name]
        if (name == 'contour_px' and left['shape'] == right['shape'] == 'sphere_contact'
                and value is not None and other is not None):
            a, b = np.asarray(value, np.float64), np.asarray(other, np.float64)
            if a.shape != b.shape or not np.allclose(a, b, rtol=0., atol=_CONTOUR_TOLERANCE_PX):
                return False
        elif not np.array_equal(value, other):
            return False
    return True


def _error_message(exc):
    return getattr(exc, 'display_message', str(exc))


class _RegionEditor:
    """Edit one annotation while keeping derived geometry and cached fits consistent."""

    def __init__(self, region, window, *, panel_callback, recompute_callback, save_callback,
                 editable_panel, panel_widths, panel_height, update_geometry, depth_range):
        self.region = region
        self.window = window
        self.panel_callback = panel_callback
        self.recompute_callback = recompute_callback
        self.save_callback = save_callback
        self.editable_panel = editable_panel
        self.panel_widths = panel_widths
        self.panel_height = panel_height
        self.update_geometry = update_geometry
        self.depth_range = depth_range
        self.panel_left = sum(panel_widths[:editable_panel])

        if region.shape == 'sphere_contact':
            if update_geometry is None or depth_range is None:
                raise ValueError('sphere_contact editing needs update_geometry and indentation_range_mm')
            self.status = 'Sphere contact | drag: move | wheel: depth | c center | z undo'
        elif region.shape == 'circle':
            self.status = 'Circle | drag inside: move | drag edge: radius | wheel: radius only | c center | z undo'
        else:
            self.status = 'Drag: move/resize | p polygon | wheel scale | a/d rotate | c center | z undo'

        self.cached = None
        self.dirty = True
        self.geometry_valid = False
        self.history = []
        self.mode = None
        self.point = None
        self.vertex = None
        self.scale = 1.
        self.origin = (self.panel_left, 0)
        self.center_mode = False
        self.drag_before = None
        self.drag_recorded = False

        before = self._snapshot()
        try:
            self._refresh_geometry()
            self.geometry_valid = True
        except (ValueError, RuntimeError) as exc:
            self._restore(before)
            region.status = 'draft'
            self.status = f'Invalid geometry: {_error_message(exc)}'

    def _snapshot(self):
        return asdict(self.region)

    def _restore(self, values):
        # Save callbacks retain the original annotation object.
        for name, value in values.items():
            setattr(self.region, name, deepcopy(value))

    def _refresh_geometry(self):
        if self.update_geometry is not None:
            self.update_geometry(self.region)
        self.region.__post_init__()
        # A repeated polygon vertex cannot be resampled for display or saving.
        region_boundary_points(self.region, 3)

    def _stop_drag(self):
        if self.drag_recorded and _same_state(self.drag_before, self._snapshot()):
            self.history.pop()
        self.mode = self.point = self.vertex = self.drag_before = None
        self.drag_recorded = False

    def _record_change(self, before, *, dragging=False):
        was_valid = self.geometry_valid
        try:
            self._refresh_geometry()
        except (ValueError, RuntimeError) as exc:
            self._restore(before)
            self.geometry_valid = was_valid
            self.status = f'Invalid geometry: {_error_message(exc)}'
            self.dirty = True
            return False
        self.geometry_valid = True
        if _same_state(before, self._snapshot()):
            if not was_valid:
                self.cached = None
                self.status = 'Geometry restored. g: recompute | Enter: accept'
                self.dirty = True
            return True
        # Keep one snapshot for the whole drag, independent of mouse-event count.
        if not dragging or not self.drag_recorded:
            self.history.append(self.drag_before if dragging else before)
            if dragging:
                self.drag_recorded = True
        self.dirty, self.cached = True, None
        self.status = 'Edited. z: undo | g: recompute | Enter: accept'
        return True

    def _mouse_down(self, pixel):
        region = self.region
        self._stop_drag()
        if self.center_mode:
            before = self._snapshot()
            region.center_px = tuple(pixel)
            self.center_mode = False
            self._record_change(before)
            return
        if region.shape == 'sphere_contact':
            self.mode = 'move'
        elif region.shape == 'polygon':
            contour = np.asarray(region.contour_px)
            distances = np.linalg.norm(contour-pixel, axis=1)*self.scale
            self.vertex = int(np.argmin(distances)) if distances.min() < 14 else None
            self.mode = 'vertex' if self.vertex is not None else 'move'
        else:
            boundary = region_boundary_points(region, BOUNDARY_SAMPLE_COUNT)
            distance = np.min(np.linalg.norm(boundary-pixel, axis=1))*self.scale
            self.mode = 'resize' if distance < 14 else 'move'
        self.point = pixel
        self.drag_before = self._snapshot()

    def _mouse_move(self, pixel):
        region = self.region
        delta = pixel-self.point
        if not np.any(delta):
            return
        before = self._snapshot()
        if self.mode == 'vertex':
            region.contour_px[self.vertex] = pixel.tolist()
        elif self.mode == 'resize':
            vector = pixel-region.center_px
            if region.shape == 'circle':
                radius = max(float(np.linalg.norm(vector)), 2.)
                region.semi_axes_px = (radius, radius)
            else:
                theta = np.deg2rad(region.angle_degrees)
                rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
                region.semi_axes_px = tuple(np.maximum(np.abs(rotation @ vector), 2.))
        else:
            region.center_px = tuple(np.asarray(region.center_px)+delta)
            if region.shape == 'polygon':
                region.contour_px = (np.asarray(region.contour_px)+delta).tolist()
        if self._record_change(before, dragging=True):
            self.point = pixel

    def _mouse_wheel(self, flags):
        region = self.region
        self._stop_drag()
        before = self._snapshot()
        if region.shape == 'sphere_contact':
            lower, upper = self.depth_range
            step = upper*.01*(1 if flags > 0 else -1)
            region.indentation_depth_mm = float(np.clip(region.indentation_depth_mm+step, lower, upper))
        else:
            factor = 1.05 if flags > 0 else 1/1.05
            region.semi_axes_px = tuple(np.asarray(region.semi_axes_px)*factor)
            if region.shape == 'polygon':
                region.contour_px = (region.center_px+(np.asarray(region.contour_px)-region.center_px)*factor).tolist()
        self._record_change(before)

    def _mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP:
            self._stop_drag()
            return
        if not self.panel_left <= x < self.panel_left+self.panel_widths[self.editable_panel]:
            return
        pixel = (np.array([x, y], float)-self.origin)/self.scale
        if np.any(pixel < 0) or np.any(pixel >= self.region.image_size):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down(pixel)
        elif event == cv2.EVENT_MOUSEMOVE and self.mode:
            self._mouse_move(pixel)
        elif event == cv2.EVENT_MOUSEWHEEL and flags:
            self._mouse_wheel(flags)

    def _undo(self):
        self._stop_drag()
        if self.center_mode:
            self.center_mode = False
            self.status = 'Center selection cancelled. z: undo previous edit'
        elif self.history:
            before = self._snapshot()
            was_valid = self.geometry_valid
            self._restore(self.history[-1])
            try:
                self._refresh_geometry()
            except (ValueError, RuntimeError) as exc:
                self._restore(before)
                self.geometry_valid = was_valid
                self.status = f'Cannot undo: {_error_message(exc)}'
            else:
                self.history.pop()
                self.geometry_valid, self.cached = True, None
                self.status = 'Last edit undone. z: undo again | g: recompute'
        else:
            self.status = 'No edits to undo in this image'
        self.dirty = True

    def _recompute(self, accept):
        self._stop_drag()
        if not self.geometry_valid:
            before = self._snapshot()
            try:
                self._refresh_geometry()
            except (ValueError, RuntimeError) as exc:
                self._restore(before)
                self.status = f'Invalid geometry: {_error_message(exc)}'
                self.dirty = True
                return False
            self.geometry_valid = True
        before = self._snapshot()
        print(f'{self.region.image}: recomputing calibration...', flush=True)
        result = None
        error = None
        try:
            result = self.recompute_callback(self.region)
        except (ValueError, RuntimeError) as exc:
            error = exc
        try:
            # The recomputation may have replaced the underlying surface.
            self._refresh_geometry()
        except (ValueError, RuntimeError) as exc:
            self._restore(before)
            self.region.status = 'draft'
            self.geometry_valid, self.cached = False, None
            self.status = f'Invalid geometry after recompute: {_error_message(exc)}'
            if error is not None:
                self.status = f'{_error_message(error)}\n{self.status}'
        else:
            self.geometry_valid = True
            changed = not _same_state(before, self._snapshot())
            if changed:
                self.cached = None
                self.status = 'Surface changed the boundary. g: recompute with updated geometry'
                if error is not None:
                    self.status = f'{_error_message(error)}\n{self.status}'
            elif error is not None:
                self.cached = getattr(error, 'preview', None)
                self.status = _error_message(error)
            else:
                self.cached = result
                self.status = 'Final geometry accepted. Enter saves and continues.'
            if error is not None:
                self.region.status = 'draft'
                self.save_callback()
            elif not changed and accept:
                self.region.status = 'accepted'
                self.save_callback()
                print(f'{self.region.image}: {self.status}', flush=True)
                return True
        print(f'{self.region.image}: {self.status}', flush=True)
        self.dirty = True
        return False

    def _convert_to_polygon(self):
        self._stop_drag()
        before = self._snapshot()
        self.region.contour_px = region_boundary_points(self.region, EDITOR_VERTEX_COUNT).tolist()
        self.region.shape = 'polygon'
        self._record_change(before)

    def _rotate(self, angle):
        self._stop_drag()
        before = self._snapshot()
        self.region.angle_degrees += angle
        if self.region.shape == 'polygon':
            theta = np.deg2rad(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            self.region.contour_px = ((np.asarray(self.region.contour_px)-self.region.center_px)
                                      @ rotation.T+self.region.center_px).tolist()
        self._record_change(before)

    def run(self):
        self.window.set_mouse_callback(self._mouse)
        while True:
            if self.dirty:
                panels = self.panel_callback(self.region, self.cached, self.status)
                display = compose_annotation_panels(panels, self.panel_widths, self.panel_height, self.status)
                self.scale = display.scales[self.editable_panel]
                self.origin = display.origins[self.editable_panel]
                self.window.show_image(display.image, display.labels)
                self.dirty = False
            key = self.window.wait_key(30)
            if key == 27 or self.window.is_closed:
                self.region.status = 'draft'
                self.save_callback()
                return False
            if key == ord('x'):
                self.region.status = 'skip'
                self.save_callback()
                return True
            if key == ord('z'):
                self._undo()
            elif key in (ord('g'), 10, 13):
                if self._recompute(key in (10, 13)):
                    return True
            elif key == ord('p') and self.region.shape not in {'circle', 'sphere_contact'}:
                self._convert_to_polygon()
            elif key == ord('c'):
                self._stop_drag()
                self.center_mode = True
                self.status = 'Click center in the annotation panel'
                self.dirty = True
            elif key in (ord('a'), ord('d')) and self.region.shape not in {'circle', 'sphere_contact'}:
                self._rotate(-3 if key == ord('a') else 3)


def edit_regions(regions, *, panel_callback, recompute_callback, save_callback,
                 editable_panel: int = 0, panel_widths: tuple[int, int, int] = (640, 640, 640),
                 panel_height: int = 480,
                 update_geometry: Callable[[RegionAnnotation], None] | None = None,
                 indentation_range_mm: tuple[float, float] | None = None):
    """Edit original-pixel geometry. Return False after saving an Escape draft.

    panel_callback receives the annotation, cached result and status; it returns three
    AnnotationPanel objects containing display-only BGR8 canvases. Linear RGB source
    images must be directly quantized before display. editable_panel selects the image
    receiving mouse edits (0, 1 or 2) and must use original image coordinates.
    panel_widths and panel_height control display space, preserving aspect ratios.

    update_geometry updates derived geometry in place after an edit, undo, initial
    load, and recomputation. It may raise ValueError or RuntimeError for invalid
    geometry. A sphere_contact requires this callback and a positive, finite depth
    range (mm); each wheel step changes indentation by 1% of its upper bound.
    If updating fails, draft and skip can retain a previous derived contour so the
    center and depth remain editable, but such geometry cannot be accepted.
    recompute_callback may raise ValueError or RuntimeError, optionally with a
    display-only preview and a multiline display_message.
    """
    if editable_panel not in (0, 1, 2):
        raise ValueError('editable_panel must be 0, 1 or 2')
    depth_range = None
    if indentation_range_mm is not None:
        try:
            lower, upper = map(float, indentation_range_mm)
        except (TypeError, ValueError):
            raise ValueError('indentation_range_mm needs two finite positive bounds') from None
        if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
            raise ValueError('indentation_range_mm needs two increasing, finite positive bounds')
        depth_range = (lower, upper)

    window = AnnotationWindow('Contact boundary | z undo, g recompute, Enter accept, x skip, Esc draft')
    try:
        for region in regions:
            editor = _RegionEditor(region, window, panel_callback=panel_callback,
                recompute_callback=recompute_callback, save_callback=save_callback,
                editable_panel=editable_panel, panel_widths=panel_widths, panel_height=panel_height,
                update_geometry=update_geometry, depth_range=depth_range)
            if not editor.run():
                return False
    finally:
        window.close()
    return True
