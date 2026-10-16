"""Callback-driven, three-panel OpenCV region editor. No calibration imports."""
from copy import deepcopy
from dataclasses import asdict
from textwrap import wrap
import cv2
import numpy as np
from utils.region_annotations import BOUNDARY_SAMPLE_COUNT, EDITOR_VERTEX_COUNT, region_boundary_points


def _window_closed(window: str) -> bool:
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error as error:
        # Qt may delete its GUI receiver when the last native window closes.
        if error.code == cv2.Error.StsNullPtr:
            return True
        raise


def edit_regions(regions, *, panel_callback, recompute_callback, save_callback,
                 editable_panel: int = 0):
    """Edit original-pixel geometry. Return False after saving an Escape draft.

    Callbacks receive the annotation. panel_callback returns three BGR images;
    editable_panel selects the image receiving mouse edits (0, 1 or 2). It must
    use the original image dimensions and pixel coordinates, before display scaling.
    recompute_callback raises ValueError/RuntimeError on invalid geometry; errors
    may carry a display-only preview and a multiline display_message.
    """
    if editable_panel not in (0, 1, 2):
        raise ValueError('editable_panel must be 0, 1 or 2')
    panel_left = 640*editable_panel
    window = 'Contact boundary | z undo, g recompute, Enter accept, x skip, Esc draft'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for region in regions:
            if region.shape == 'circle':
                status = 'Circle | drag inside: move | drag edge: radius | wheel scale | c center | z undo'
            else:
                status = 'Drag: move/resize | p polygon | wheel scale | a/d rotate | c center | z undo'
            cached = None
            dirty = True
            history = []
            state = dict(mode=None, point=None, vertex=None, scale=1., center_mode=False,
                         drag_before=None, drag_recorded=False)

            def stop_drag():
                if state['drag_recorded'] and state['drag_before'] == asdict(region):
                    history.pop()  # Returning to the starting geometry is a no-op.
                state.update(mode=None, point=None, vertex=None, drag_before=None, drag_recorded=False)

            def record_change(before, *, dragging=False):
                nonlocal dirty, cached, status
                if before == asdict(region):
                    return
                # Keep one snapshot for the whole drag, independent of mouse-event count.
                if not dragging or not state['drag_recorded']:
                    history.append(before)
                    if dragging:
                        state['drag_recorded'] = True
                dirty, cached = True, None
                status = 'Edited. z: undo | g: recompute | Enter: accept'

            def mouse(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONUP:
                    stop_drag()
                    return
                if not panel_left <= x < panel_left+640:
                    return
                pixel = np.array([x-panel_left, y], float)/state['scale']
                if np.any(pixel < 0) or np.any(pixel >= region.image_size):
                    return
                if event == cv2.EVENT_LBUTTONDOWN:
                    stop_drag()
                    if state['center_mode']:
                        before = asdict(region)
                        region.center_px = tuple(pixel)
                        state['center_mode'] = False
                        record_change(before)
                        return
                    elif region.shape == 'polygon':
                        contour = np.asarray(region.contour_px)
                        distances = np.linalg.norm(contour-pixel, axis=1)*state['scale']
                        state['vertex'] = int(np.argmin(distances)) if distances.min() < 14 else None
                        state['mode'] = 'vertex' if state['vertex'] is not None else 'move'
                    else:
                        boundary = region_boundary_points(region, BOUNDARY_SAMPLE_COUNT)
                        distance = np.min(np.linalg.norm(boundary-pixel, axis=1))*state['scale']
                        state['mode'] = 'resize' if distance < 14 else 'move'
                    state['point'] = pixel
                    state['drag_before'] = asdict(region)
                elif event == cv2.EVENT_MOUSEMOVE and state['mode']:
                    delta = pixel-state['point']
                    if not np.any(delta):
                        return
                    if state['mode'] == 'vertex':
                        region.contour_px[state['vertex']] = pixel.tolist()
                    elif state['mode'] == 'resize':
                        vector = pixel-region.center_px
                        if region.shape == 'circle':
                            radius = max(float(np.linalg.norm(vector)), 2.)
                            region.semi_axes_px = (radius, radius)
                        else:
                            theta = np.deg2rad(region.angle_degrees)
                            local = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) @ vector
                            region.semi_axes_px = tuple(np.maximum(np.abs(local), 2.))
                    else:
                        region.center_px = tuple(np.asarray(region.center_px)+delta)
                        if region.shape == 'polygon':
                            region.contour_px = (np.asarray(region.contour_px)+delta).tolist()
                    state['point'] = pixel
                    record_change(state['drag_before'], dragging=True)
                elif event == cv2.EVENT_MOUSEWHEEL:
                    stop_drag()
                    before = asdict(region)
                    factor = 1.05 if flags > 0 else 1/1.05
                    region.semi_axes_px = tuple(np.asarray(region.semi_axes_px)*factor)
                    if region.shape == 'polygon':
                        region.contour_px = (region.center_px+(np.asarray(region.contour_px)-region.center_px)*factor).tolist()
                    record_change(before)

            cv2.setMouseCallback(window, mouse)
            while True:
                if dirty:
                    panels = panel_callback(region, cached, status)
                    canvas = []
                    for i, panel in enumerate(panels):
                        h, w = panel.shape[:2]
                        scale = min(640/w, 480/h)
                        if i == editable_panel:
                            state['scale'] = scale
                        target = np.zeros((560, 640, 3), np.uint8)
                        resized = cv2.resize(panel, (round(w*scale), round(h*scale)))
                        target[:resized.shape[0], :resized.shape[1]] = resized
                        canvas.append(target)
                    display = np.concatenate(canvas, axis=1)
                    lines = [line for paragraph in status.splitlines()
                             for line in (wrap(paragraph, width=180) or [''])]
                    for i, line in enumerate(lines[:4]):
                        cv2.putText(display, line, (8, 495+18*i), cv2.FONT_HERSHEY_SIMPLEX,
                                    .5, (255, 255, 255), 1)
                    cv2.imshow(window, display)
                    dirty = False
                key = cv2.waitKey(30) & 0xff
                if key == 27 or _window_closed(window):
                    region.status = 'draft'
                    save_callback()
                    return False
                if key == ord('x'):
                    region.status = 'skip'
                    save_callback()
                    break
                if key == ord('z'):
                    stop_drag()
                    if state['center_mode']:
                        state['center_mode'] = False
                        status = 'Center selection cancelled. z: undo previous edit'
                    elif history:
                        before = history.pop()
                        # Restore in place: save callbacks retain the original annotation object.
                        for name, value in before.items():
                            setattr(region, name, deepcopy(value))
                        cached = None
                        status = 'Last edit undone. z: undo again | g: recompute'
                    else:
                        status = 'No edits to undo in this image'
                    dirty = True
                if key in (ord('g'), 10, 13):
                    print(f'{region.image}: recomputing calibration...', flush=True)
                    try:
                        cached = recompute_callback(region)
                        status = 'Final geometry accepted. Enter saves and continues.'
                        print(f'{region.image}: {status}', flush=True)
                        if key in (10, 13):
                            region.status = 'accepted'
                            save_callback()
                            break
                    except (ValueError, RuntimeError) as exc:
                        region.status = 'draft'
                        status = getattr(exc, 'display_message', str(exc))
                        cached = getattr(exc, 'preview', None)
                        print(f'{region.image}: {status}', flush=True)
                        save_callback()
                    dirty = True
                if key == ord('p') and region.shape != 'circle':
                    stop_drag()
                    before = asdict(region)
                    region.contour_px = region_boundary_points(region, EDITOR_VERTEX_COUNT).tolist()
                    region.shape = 'polygon'
                    record_change(before)
                if key == ord('c'):
                    stop_drag()
                    state['center_mode'] = True
                    status = 'Click center in the annotation panel'
                    dirty = True
                if key in (ord('a'), ord('d')) and region.shape != 'circle':
                    stop_drag()
                    before = asdict(region)
                    angle = -3 if key == ord('a') else 3
                    region.angle_degrees += angle
                    if region.shape == 'polygon':
                        theta = np.deg2rad(angle)
                        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                        region.contour_px = ((np.asarray(region.contour_px)-region.center_px) @ rotation.T+region.center_px).tolist()
                    record_change(before)
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass  # A user may already have closed the native window.
    return True
