"""Contact-boundary adapter for the generic image-region editor."""
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from utils.region_annotations import (BOUNDARY_SAMPLE_COUNT, RegionAnnotation, load_regions, save_regions,
                                      region_boundary_points)
from utils.color_difference import signed_difference_preview
from utils.image_io import linear_rgb_to_bgr8
from utils.annotation_editor import edit_regions
from utils.annotation_window import AnnotationPanel
from utils.surface_sampling import rasterize_surface_grid
from utils.surface_projection import projected_circle_radius_px

ContactAnnotation = RegionAnnotation
ANNOTATION_PANEL_WIDTHS = (640, 960, 280)
ANNOTATION_PANEL_HEIGHT = 720


def _maximum_circle_radius_px(frame, sphere_radius_mm, center_px):
    grid = frame.systems.observation
    depth = grid.xyz[..., 2]
    valid = grid.valid_mask & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError('A valid positive surface depth is required to limit the annotation circle')
    return projected_circle_radius_px(center_px, sphere_radius_mm, float(depth[valid].max()), frame.camera)


def _color_image_preview(frame, record):
    if 'color_residual_grid' not in record:
        return np.zeros(frame.linear_rgb.shape, np.uint8)
    fields = record['color_residual_grid']
    if 'sample_valid_mask' in record:
        fields = np.dstack((fields, record['sample_valid_mask']))
    projected = rasterize_surface_grid(fields, frame.systems.observation.uv,
                                      record['photometric_valid_mask'], frame.linear_rgb.shape[:2])
    valid = projected.valid & frame.surface_mask & ~frame.marker_image_mask
    color = signed_difference_preview(projected.values[..., :3], valid)
    if 'sample_valid_mask' in record:
        sample = (projected.values[..., 3] >= .5) & valid
        contours, _ = cv2.findContours(sample.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color, contours, -1, (0, 255, 255), 1)
    return color


def calibration_panels(frame, annotation, record=None, *, color_image=None,
                       maximum_circle_radius_px=None):
    record = {} if record is None else record
    display_frame = record.get('frame', frame)
    original = linear_rgb_to_bgr8(frame.linear_rgb)
    contours, _ = cv2.findContours(frame.surface_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original, contours, -1, (255, 200, 0), 1)
    original[frame.marker_image_mask] = original[frame.marker_image_mask]//2+np.array([0, 0, 100], np.uint8)
    grid = display_frame.systems.observation
    stage = 'Reference' if display_frame.geometry_stage == 'preview_reference' else 'Final'
    metrics = record.get('metrics', {})
    color = (_color_image_preview(display_frame, record) if color_image is None else color_image).copy()
    if annotation is not None:
        boundary = np.rint(region_boundary_points(annotation, BOUNDARY_SAMPLE_COUNT)).astype(np.int32)
        center = tuple(np.rint(annotation.center_px).astype(int))
        for image in (original, color):
            cv2.polylines(image, [boundary], True, (0, 255, 0), 2)
            cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            if annotation.shape == 'polygon':
                for point in annotation.contour_px:
                    cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (0, 255, 255), -1)
    if 'color_residual_grid' in record:
        photo = record['photometric_valid_mask']
        color_lines = [f'{stage} UV colors | valid points: {int(photo.sum())}']
        if stage == 'Reference':
            color_lines.append('Approximate geometry; g recomputes final UV')
        if 'sample_valid_mask' in record:
            sample = record['sample_valid_mask']
            color_lines.append(f'Paired samples: {int(sample.sum())}')
    else:
        color_lines = ['Color preview unavailable', 'g: recompute; see status below']
    color_lines.append('Edit boundary here | wheel: radius | c: center | z: undo')
    if maximum_circle_radius_px is not None:
        color_lines.append(f'Circle diameter: {2*annotation.semi_axes_px[0]:.2f} px | '
                           f'maximum: {2*maximum_circle_radius_px:.2f} px')
    if 'normal_gt_grid' in record:
        normal = np.rint(np.clip(.5+.5*record['normal_gt_grid'][..., ::-1], 0, 1)*255).astype(np.uint8)
        normal[~record['sample_valid_mask']] = 0
        normal_lines = [f"Sphere normals | samples: {metrics['sample_count']}",
                        'Camera XYZ normals | RGB = (nx, ny, nz)',
                        f"Boundary RMS (sigma): {metrics['sphere_boundary_rms']:.3f}",
                        f"Reprojection: {metrics.get('reprojection_rms_px', 0):.3f}px",
                        f"{record['sphere_side']} root {record['sphere_root_sign']:+d}"]
    else:
        z = grid.xyz[..., 2]
        valid = grid.valid_mask
        normal = np.zeros((*valid.shape, 3), np.uint8)
        normal_lines = [f'{stage} overall depth | camera Z (mm)', 'Sphere normals not available']
        if valid.any():
            values = np.clip((z-z[valid].min())/max(float(np.ptp(z[valid])), 1e-6), 0, 1)
            normal = cv2.applyColorMap(np.rint(values*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            normal[~valid] = 0
            normal_lines.append(f'Z: {z[valid].min():.3f} .. {z[valid].max():.3f} mm')
        if stage == 'Reference':
            normal_lines.append('preview_reference | g: recompute')
        else:
            normal_lines.append(f"Reprojection: {metrics.get('reprojection_rms_px', 0):.3f}px")
    errors = record.get('preview_errors', [])
    if errors:
        codes = ', '.join(item.split(':', 1)[0] for item in errors)
        color_lines.insert(0, f'REJECTED: {codes}')
        normal_lines.insert(0, f'REJECTED: {codes}')
    elif 'normal_gt_grid' in record:
        normal_lines.insert(0, 'CHECKS PASSED | Enter: accept')

    return (AnnotationPanel(original), AnnotationPanel(color, tuple(color_lines)),
            AnnotationPanel(normal, tuple(normal_lines), header=True))


def annotate_images(settings, *, edit=False):
    # Import orchestration only in this adapter, never from the generic editor.
    from calibrate.utils.norm_pipeline import (FrameCalibrationError, compute_frame_colors,
        load_manifest, load_frame_calibration_data, finalize_image)
    settings.require_sphere()
    manifest = load_manifest(settings)
    root = Path(manifest['image_root'])
    regions = load_regions(settings.annotations_file, image_root=root)
    reference = load_frame_calibration_data(settings.work_dir/'reference_surface.npz')
    for entry in manifest['images']:
        if entry['status'] == 'skip':
            continue
        key = entry['key']
        frame = load_frame_calibration_data(settings.work_dir/'prepared'/f"{entry['index']:06d}.npz")
        annotation = regions.get(key)
        if annotation is not None and annotation.status in {'accepted', 'skip'} and not edit:
            continue
        if annotation is None:
            yy, xx = np.nonzero(frame.surface_mask)
            center = (float(np.median(xx)), float(np.median(yy)))
            annotation = RegionAnnotation(key, frame.linear_rgb.shape[1::-1], center_px=center,
                                          shape='circle', semi_axes_px=(40., 40.),
                                          semantics='contact_boundary')
        elif annotation.shape != 'circle':
            # Preserve the old contact area as an initial guess for manual review.
            if annotation.shape == 'ellipse':
                radius = float(np.sqrt(np.prod(annotation.semi_axes_px)))
            else:
                area = cv2.contourArea(np.asarray(annotation.contour_px, np.float32))
                radius = max(float(np.sqrt(area/np.pi)), 2.)
            annotation = replace(annotation, shape='circle', semi_axes_px=(radius, radius),
                                 angle_degrees=0., contour_px=None)
        regions[key] = annotation
        annotation.status = 'draft'
        initial_preview = dict(frame=frame)
        try:
            initial_preview.update(compute_frame_colors(frame, reference, settings))
        except FrameCalibrationError as error:
            initial_preview['preview_errors'] = [str(error)]
        preview_record, color_image = None, None
        limit_frame = frame

        def maximum_radius(region):
            return _maximum_circle_radius_px(limit_frame, settings.sphere_radius_mm, region.center_px)

        def recompute(region):
            nonlocal limit_frame
            try:
                record = finalize_image(frame, reference, region, settings, diagnostic_preview=True)
            except FrameCalibrationError as error:
                if error.preview is not None:
                    limit_frame = error.preview['frame']
                raise
            limit_frame = record['frame']
            return record

        def panels(region, record, _status):
            nonlocal preview_record, color_image
            record = initial_preview if record is None else record
            if record is not preview_record:
                # Geometry and colors change on recompute, not while dragging the annotation.
                color_image = _color_image_preview(record.get('frame', frame), record)
                preview_record = record
            return calibration_panels(frame, region, record, color_image=color_image,
                                      maximum_circle_radius_px=maximum_radius(region))

        completed = edit_regions([annotation],
            editable_panel=1, panel_widths=ANNOTATION_PANEL_WIDTHS, panel_height=ANNOTATION_PANEL_HEIGHT,
            panel_callback=panels,
            recompute_callback=recompute, maximum_circle_radius=maximum_radius,
            save_callback=lambda: save_regions(settings.annotations_file, regions, image_root=root))
        if not completed:
            return False
    save_regions(settings.annotations_file, regions, image_root=root)
    return True
