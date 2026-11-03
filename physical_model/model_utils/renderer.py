"""Public analytic renderer dispatch and result serialization."""
import json
import numpy as np
from PIL import Image
from .geometry import LIGHT_GEOMETRY_MODEL
from .renderer_cpu import camera_ground_truth, camera_rays, render_cpu
from .observation import IMAGE_ENCODING, OBSERVATION_MODEL_ID, PREVIEW_ENCODING
from .surface_coordinates import bending_radius, center_uv, center_material_mm, COORDINATE_MODEL


def render(config, device=None):
    """Render with the configured analytic device.

    CUDA is the default for configs without an explicit device. Importing this
    module remains safe in CPU-only environments because PyTorch is loaded only
    when the CUDA path is selected.
    """
    selected = device or config.get('render', {}).get('device', 'cuda:0')
    if str(selected).lower() == 'cpu':
        return render_cpu(config)
    if bending_radius(config) is not None:
        from .curve_renderer_gpu import render_curve_host
        return render_curve_host(config, str(selected))
    from .renderer_gpu import render_host
    return render_host(config, str(selected))


def encode_linear_rgb(sensor: np.ndarray, config: dict) -> np.ndarray:
    """Return unclipped, unencoded exposure-scaled camera-linear RGB."""
    sensor = np.asarray(sensor)
    if sensor.dtype.kind != 'f':
        sensor = sensor.astype(np.float32)
    response = np.asarray(config['camera']['response_rgb_3x3'], dtype=sensor.dtype)
    exposure = sensor.dtype.type(config['render']['exposure_scale'])
    return sensor @ response.T * exposure


def encode_display(sensor, config):
    """Make a clipped linear uint8 preview; calibration uses encode_linear_rgb."""
    rgb = encode_linear_rgb(sensor, config)
    return np.uint8(np.rint(255 * np.clip(rgb, 0, 1)))


def save_result(out, sensor, config, backend, gt=None, radiance=None, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    camera_linear_rgb = encode_linear_rgb(sensor, config)
    Image.fromarray(np.uint8(np.rint(255 * np.clip(camera_linear_rgb, 0, 1)))).save(out / 'image.png')
    np.save(out / 'sensor_irradiance_rgb_w_m2.npy', sensor.astype(np.float32))
    np.save(out / 'camera_linear_rgb.npy', camera_linear_rgb)
    if gt is not None:
        np.savez_compressed(out / 'geometry_gt.npz', **gt)
    if radiance is not None:
        np.save(out / 'radiance_rgb_w_m2_sr.npy', radiance.astype(np.float32))
    meta = {'backend': backend, **(extra or {}),
            'sensor_array_order': 'height,width,RGB',
            'light_geometry_model': LIGHT_GEOMETRY_MODEL,
            'sensor_unit': 'W/m^2 per independent RGB band',
            'observation_model_id': OBSERVATION_MODEL_ID,
            'image_encoding': IMAGE_ENCODING,
            'preview_image_encoding': PREVIEW_ENCODING,
            'config': config}
    meta.setdefault('coordinate_model_id', COORDINATE_MODEL)
    meta.setdefault('center_uv', center_uv(config).tolist())
    meta.setdefault('center_material_mm', center_material_mm(config).tolist())
    if bending_radius(config) is not None:
        from .curve_geometry import GEOMETRY_MODEL, LIGHT_MODEL
        meta['geometry_model_id'] = GEOMETRY_MODEL
        meta['light_geometry_model'] = LIGHT_MODEL
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
