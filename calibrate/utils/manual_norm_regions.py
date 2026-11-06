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
from utils.surface_projection import invert_surface_projection
from utils.surface_geometry import compute_surface_geometry
from utils.sphere_contact import SphereSurfaceContact

ContactAnnotation = RegionAnnotation
ANNOTATION_PANEL_WIDTHS = (640, 960, 280)
ANNOTATION_PANEL_HEIGHT = 720


def _initial_indentation_depth(frame, annotation, radius_mm):
    """Use old image boundaries only as an initial depth guess for manual review."""
    boundary = invert_surface_projection(region_boundary_points(annotation, 8),
        frame.surface, frame.camera, frame.systems.geometry)
    center = invert_surface_projection(np.asarray(annotation.center_px)[None],
        frame.surface, frame.camera, frame.systems.geometry)
    if not center.valid[0] or not boundary.valid.any():
        return .1*radius_mm
    rho = min(float(np.median(np.linalg.norm(boundary.xyz[boundary.valid]-center.xyz[0], axis=1))), radius_mm)
    return max(radius_mm*1e-4, radius_mm-np.sqrt(max(radius_mm**2-rho**2, 0.)))


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
                       maximum_indentation_mm=None, maximum_boundary_px=None):
    record = {} if record is None else record
    display_frame = record.get('frame', frame)
    original = linear_rgb_to_bgr8(frame.linear_rgb)
    contours, _ = cv2.findContours(frame.surface_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original, contours, -1, (255, 200, 0), 1)
    original[frame.marker_image_mask] = original[frame.marker_image_mask]//2+np.array([0, 0, 100], np.uint8)
    grid = display_frame.systems.observation
    stage = 'Final' if display_frame.geometry_stage == 'final' else 'Initial'
    metrics = record.get('metrics', {})
    color = (_color_image_preview(display_frame, record) if color_image is None else color_image).copy()
    if annotation is not None:
        boundary = np.rint(region_boundary_points(annotation, BOUNDARY_SAMPLE_COUNT)).astype(np.int32)
        center = tuple(np.rint(annotation.center_px).astype(int))
        for image in (original, color):
            cv2.polylines(image, [boundary], True, (0, 255, 0), 2)
            cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            if maximum_boundary_px is not None:
                maximum = np.rint(maximum_boundary_px).astype(np.int32)
                cv2.polylines(image, [maximum], True, (0, 160, 255), 1)
            if annotation.shape == 'polygon':
                for point in annotation.contour_px:
                    cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (0, 255, 255), -1)
    if 'color_residual_grid' in record:
        photo = record['photometric_valid_mask']
        color_lines = [f'{stage} UV colors | valid points: {int(photo.sum())}']
        if stage == 'Initial':
            color_lines.append('Current overall fit; g excludes contact markers')
        if 'sample_valid_mask' in record:
            sample = record['sample_valid_mask']
            color_lines.append(f'Paired samples: {int(sample.sum())}')
    else:
        color_lines = ['Color preview unavailable', 'g: recompute; see status below']
    color_lines.append('Drag: contact position | wheel: depth | c: center | z: undo')
    if maximum_indentation_mm is not None:
        color_lines.append(f'Indentation: {annotation.indentation_depth_mm:.4f} mm | '
                           f'maximum: {maximum_indentation_mm:.4f} mm')
        color_lines.append('Green: contact | orange: maximum-depth footprint')
    if 'normal_gt_grid' in record:
        normal = np.rint(np.clip(.5+.5*record['normal_gt_grid'][..., ::-1], 0, 1)*255).astype(np.uint8)
        normal[~record['sample_valid_mask']] = 0
        normal_lines = [f"Sphere normals | samples: {metrics['sample_count']}",
                        'Camera XYZ normals | RGB = (nx, ny, nz)',
                        f"Boundary RMS (sigma): {metrics['sphere_boundary_rms']:.3f}",
                        f"Reprojection: {metrics.get('reprojection_rms_px', 0):.3f}px",
                        f"{record['sphere_side']} root {record['sphere_root_sign']:+d}"]
    else:
        if 'principal_curvatures_per_mm' in record:
            valid = record['curvature_valid_mask']
            principal = record['principal_curvatures_per_mm']
        else:
            geometry = compute_surface_geometry(grid.xyz, grid.valid_mask)
            valid, principal = geometry['valid'], geometry['principal_curvatures']
        curvature = np.max(np.abs(principal), axis=-1)
        normal = np.zeros((*valid.shape, 3), np.uint8)
        normal_lines = [f'{stage} overall curvature', 'max(abs(k1), abs(k2)) | 1/mm',
                        'Sphere normals: g recompute']
        if valid.any():
            maximum = max(float(curvature[valid].max()), 1e-12)
            values = np.where(valid, curvature/maximum, 0.)
            normal = cv2.applyColorMap(np.rint(np.clip(values, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            normal[~valid] = 0
            normal_lines.append(f'Max curvature: {maximum:.5f} 1/mm')
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
        load_manifest, load_frame_calibration_data, finalize_image, initial_calibration_surface)
    settings.require_sphere()
    manifest = load_manifest(settings)
    root = Path(manifest['image_root'])
    regions = load_regions(settings.annotations_file, image_root=root)
    reference = load_frame_calibration_data(settings.work_dir/'reference_surface.npz', settings)
    for entry in manifest['images']:
        if entry['status'] == 'skip':
            continue
        key = entry['key']
        frame = load_frame_calibration_data(settings.work_dir/'prepared'/f"{entry['index']:06d}.npz", settings)
        annotation = regions.get(key)
        if (annotation is not None and annotation.status in {'accepted', 'skip'} and not edit
                and (annotation.status == 'skip' or annotation.shape == 'sphere_contact')):
            continue
        frame = initial_calibration_surface(frame, reference)
        if annotation is None:
            grid = frame.systems.geometry
            pixels = grid.uv[grid.valid_mask]
            if not len(pixels):
                raise ValueError(f'{key}: reconstructed surface has no visible points')
            center = pixels[np.argmin(np.linalg.norm(pixels-np.median(pixels, axis=0), axis=1))]
            annotation = RegionAnnotation(key, frame.linear_rgb.shape[1::-1], center_px=tuple(center),
                shape='circle', semi_axes_px=(40., 40.), semantics='contact_boundary')
            depth_mm = .1*settings.sphere_radius_mm
        elif annotation.shape == 'sphere_contact':
            depth_mm = annotation.indentation_depth_mm
        else:
            depth_mm = _initial_indentation_depth(frame, annotation, settings.sphere_radius_mm)
        contour = region_boundary_points(annotation).tolist()
        annotation = replace(annotation, shape='sphere_contact', contour_px=contour,
            angle_degrees=0., indentation_depth_mm=float(np.clip(depth_mm,
                settings.sphere_radius_mm*1e-4, settings.sphere_radius_mm)))
        regions[key] = annotation
        annotation.status = 'draft'
        grid = frame.systems.observation
        geometry = compute_surface_geometry(grid.xyz, grid.valid_mask)
        initial_preview = dict(frame=frame,
            principal_curvatures_per_mm=geometry['principal_curvatures'], curvature_valid_mask=geometry['valid'])
        try:
            initial_preview.update(compute_frame_colors(frame, reference, settings))
        except FrameCalibrationError as error:
            initial_preview['preview_errors'] = [str(error)]
        preview_record, color_image = None, None
        contact_frame = frame
        projector = SphereSurfaceContact(frame.surface, frame.camera, frame.systems.geometry)
        maximum_boundary, maximum_key = None, None

        def update_geometry(region):
            nonlocal maximum_boundary, maximum_key
            contact = projector.compute(region.center_px, region.indentation_depth_mm,
                                         settings.sphere_radius_mm, settings.side)
            region.contour_px = contact.boundary_px.tolist()
            cache_key = tuple(region.center_px), id(contact_frame)
            if cache_key != maximum_key:
                maximum_boundary = None
                try:
                    maximum_boundary = projector.compute(region.center_px, settings.sphere_radius_mm,
                        settings.sphere_radius_mm, settings.side).boundary_px
                except ValueError:
                    # A valid shallow contact may have its maximum-depth outline outside the patch.
                    pass
                maximum_key = cache_key

        def use_surface(record):
            nonlocal contact_frame, projector, maximum_key, initial_preview
            contact_frame = record['frame']
            projector = SphereSurfaceContact(contact_frame.surface, contact_frame.camera,
                                             contact_frame.systems.geometry)
            maximum_key = None
            # Editing invalidates labels, but retains colors from the latest overall fit.
            initial_preview = {name: record[name] for name in (
                'frame', 'color_residual_grid', 'photometric_valid_mask',
                'principal_curvatures_per_mm', 'curvature_valid_mask') if name in record}

        def recompute(region):
            try:
                record = finalize_image(frame, reference, region, settings, diagnostic_preview=True)
            except FrameCalibrationError as error:
                if error.preview is not None:
                    use_surface(error.preview)
                raise
            use_surface(record)
            return record

        def panels(region, record, _status):
            nonlocal preview_record, color_image
            record = initial_preview if record is None else record
            if record is not preview_record:
                color_image = _color_image_preview(record.get('frame', frame), record)
                preview_record = record
            return calibration_panels(frame, region, record, color_image=color_image,
                maximum_indentation_mm=settings.sphere_radius_mm, maximum_boundary_px=maximum_boundary)

        completed = edit_regions([annotation],
            editable_panel=1, panel_widths=ANNOTATION_PANEL_WIDTHS, panel_height=ANNOTATION_PANEL_HEIGHT,
            panel_callback=panels, recompute_callback=recompute, update_geometry=update_geometry,
            indentation_range_mm=(settings.sphere_radius_mm*1e-4, settings.sphere_radius_mm),
            save_callback=lambda: save_regions(settings.annotations_file, regions, image_root=root))
        if not completed:
            return False
    save_regions(settings.annotations_file, regions, image_root=root)
    return True
