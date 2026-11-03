"""Camera transfer anchors and protection against mixed calibration domains."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from utils.color_difference import (
    bgr8_to_linear_rgb,
    linear_rgb_to_display_bgr,
)


class CameraTransfer(unittest.TestCase):
    def test_bt709_anchors_and_bgr_order(self):
        # BT.709 reference values, including the 8-bit codes around E'=0.081.
        bgr = np.array([[[0, 10, 20], [21, 64, 128], [150, 192, 255]]], np.uint8)
        expected = np.array([[[.0174291939, .00871459695, 0.],
                              [.2614815069, .07864259298, .01824613623],
                              [1., .5678697100, .3522947245]]], np.float32)
        actual = bgr8_to_linear_rgb(bgr)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)

    def test_rejects_non_bgr8(self):
        for invalid in (np.zeros((2, 3), np.uint8), np.zeros((2, 3, 4), np.uint8),
                        np.zeros((2, 3, 3), np.float32)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    bgr8_to_linear_rgb(invalid)

    def test_srgb_display_anchors_and_neutral_gray(self):
        levels = np.array([0.0, 0.0031308, 0.18, 1.0], np.float32)
        linear_rgb = np.repeat(levels[:, None, None], 3, axis=2)

        actual = linear_rgb_to_display_bgr(linear_rgb)

        expected = np.array([0, 10, 118, 255], np.uint8)
        np.testing.assert_array_equal(actual[:, 0, 0], expected)
        np.testing.assert_array_equal(actual[:, 0, 1], expected)
        np.testing.assert_array_equal(actual[:, 0, 2], expected)

    def test_srgb_display_converts_rgb_to_bgr_and_clips(self):
        linear_rgb = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
              [1.2, -0.1, 0.0]]],
            np.float32,
        )

        actual = linear_rgb_to_display_bgr(linear_rgb)

        np.testing.assert_array_equal(
            actual,
            np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255]]],
                     np.uint8),
        )

    def test_srgb_display_rejects_bad_shape_and_non_float(self):
        for invalid in (
            np.zeros((2, 3), np.float32),
            np.zeros((2, 3, 4), np.float32),
            np.zeros((2, 3, 3), np.uint8),
        ):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    linear_rgb_to_display_bgr(invalid)

    def test_camera_input_rejects_old_lut_but_accepts_bt709(self):
        from reconstruction.local_recon import load_reconstruction_arrays

        values = dict(xyz=np.ones((2, 2, 3)), uv=np.zeros((2, 2, 2)),
                      observed_bgr=np.full((3, 3, 3), 128, np.uint8),
                      reference_bgr=np.full((3, 3, 3), 64, np.uint8))
        for metadata in ({}, {'rgb_transfer_function': 'srgb'}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, '重新生成颜色样本'):
                    load_reconstruction_arrays(values, metadata)
        _, colors, geometry, photo, _ = load_reconstruction_arrays(
            values, {'rgb_transfer_function': 'bt709'})
        np.testing.assert_allclose(colors, .18283891395, atol=1e-7)
        self.assertTrue(geometry.all() and photo.all())
        with self.assertRaisesRegex(ValueError, '输入图像'):
            load_reconstruction_arrays(dict(values, rgb_transfer_function='srgb'))

    def test_precomputed_colors_require_matching_transfer(self):
        from reconstruction.local_recon import load_reconstruction_arrays

        values = dict(xyz=np.ones((2, 2, 3)), color_residual_grid=np.zeros((2, 2, 3)))
        expected = {'rgb_transfer_function': 'bt709'}
        for source in (values, dict(values, rgb_transfer_function='srgb')):
            with self.assertRaisesRegex(ValueError, 'rgb_transfer_function'):
                load_reconstruction_arrays(source, expected)
        load_reconstruction_arrays(dict(values, **expected), expected)
        # External, already-linear data has no assumed camera encoding.
        load_reconstruction_arrays(values, {})

    def test_grouped_old_samples_cannot_be_refitted_as_bt709(self):
        from calibrate.norm_pipeline import CalibrationSettings, _fit_and_validate_samples

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'samples.npz'
            settings = CalibrationSettings(root, path, root/'lut.npz', root/'regions.yaml')
            for transfer in (None, 'srgb'):
                declarations = {} if transfer is None else {'rgb_transfer_function': transfer}
                np.savez(path, format_version=1, color_residual_samples=np.zeros((2, 3)),
                         slope_samples=np.zeros((2, 2)), image_index=np.array([0, 1]), **declarations)
                with self.assertRaisesRegex(ValueError, '重新生成颜色样本'):
                    _fit_and_validate_samples(path, settings)
                self.assertFalse(settings.output.exists())

    def test_visualization_rejects_mixed_curves_and_allows_geometry_cache(self):
        from calibrate.visualize_norm import _check_declarations
        from utils.normal_mapping import NormalCalibration

        metadata = dict(color_space='linear_rgb', rgb_transfer_function='bt709',
                        coordinate_kind='reference_arclength')
        lut = NormalCalibration(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2)),
                                -np.ones(3), np.ones(3), 1., metadata=metadata)
        for transfer in (None, 'srgb'):
            source = {'color_space': 'linear_rgb'}
            if transfer is not None:
                source['rgb_transfer_function'] = transfer
            with self.assertRaisesRegex(ValueError, 'rgb_transfer_function'):
                _check_declarations(source, lut, ('color_space',))
        _check_declarations(metadata, lut, ('color_space',))
        _check_declarations({'coordinate_kind': 'reference_arclength'}, lut, ('coordinate_kind',))


if __name__ == '__main__':
    unittest.main()
