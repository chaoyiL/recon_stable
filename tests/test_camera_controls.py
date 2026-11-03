"""Camera configuration and requested OpenCV controls."""

import unittest
from unittest.mock import patch

import cv2

from utils.camera import open_camera
from utils.config import ConfigError, parse_camera_config


class FakeVideoCapture:
    def __init__(self, rejected=(), ineffective=(), initial=None):
        self.rejected = set(rejected)
        self.ineffective = set(ineffective)
        self.properties = dict(initial or {})
        self.calls = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        if prop in self.rejected:
            return False
        if prop not in self.ineffective:
            self.properties[prop] = value
        return True

    def get(self, prop):
        return self.properties.get(prop, 0.)

    def release(self):
        self.released = True


class CameraConfigTests(unittest.TestCase):
    def setUp(self):
        self.legacy_config = {
            "device": 0,
            "exposure": 20,
            "white_balance_temperature": 6500,
            "width": 640,
            "height": 480,
        }

    def test_legacy_config_and_optional_controls(self):
        legacy = parse_camera_config(self.legacy_config)
        self.assertIsNone(legacy.fourcc)
        self.assertIsNone(legacy.backlight_compensation)

        configured = parse_camera_config({
            **self.legacy_config,
            "fourcc": "YUYV",
            "backlight_compensation": 0,
        })
        self.assertEqual(configured.fourcc, "YUYV")
        self.assertEqual(configured.backlight_compensation, 0.)

    def test_rejects_invalid_optional_controls(self):
        for fourcc in ("abc", "12345", "YÜYV", 1234):
            with self.subTest(fourcc=fourcc), self.assertRaises(ConfigError):
                parse_camera_config({**self.legacy_config, "fourcc": fourcc})
        for backlight in (True, -0.1, float("nan"), float("inf"), "0"):
            with self.subTest(backlight=backlight), self.assertRaises(ConfigError):
                parse_camera_config({
                    **self.legacy_config,
                    "backlight_compensation": backlight,
                })


class CameraControlTests(unittest.TestCase):
    def test_applies_requested_format_backlight_and_manual_exposure(self):
        camera = FakeVideoCapture()
        with patch("utils.camera.cv2.VideoCapture", return_value=camera):
            result = open_camera(
                0, 20, 6500, 640, 480,
                fourcc="YUYV", backlight_compensation=0,
            )

        self.assertIs(result, camera)
        expected_fourcc = cv2.VideoWriter_fourcc(*"YUYV")
        self.assertEqual(camera.get(cv2.CAP_PROP_FOURCC), expected_fourcc)
        properties = [prop for prop, _ in camera.calls]
        self.assertLess(
            properties.index(cv2.CAP_PROP_FOURCC),
            properties.index(cv2.CAP_PROP_FRAME_WIDTH),
        )
        self.assertEqual(camera.get(cv2.CAP_PROP_BACKLIGHT), 0)
        self.assertEqual(camera.get(cv2.CAP_PROP_AUTO_EXPOSURE), 1)
        self.assertEqual(camera.get(cv2.CAP_PROP_EXPOSURE), 20)

    def test_failed_or_ineffective_controls_release_camera(self):
        cases = (
            (FakeVideoCapture(rejected=(cv2.CAP_PROP_FOURCC,)), "FOURCC"),
            (
                FakeVideoCapture(
                    ineffective=(cv2.CAP_PROP_BACKLIGHT,),
                    initial={cv2.CAP_PROP_BACKLIGHT: 2},
                ),
                "背光补偿",
            ),
        )
        for camera, message in cases:
            with self.subTest(message=message), patch(
                "utils.camera.cv2.VideoCapture", return_value=camera
            ), self.assertRaisesRegex(RuntimeError, message):
                open_camera(
                    0, 20, 6500, 640, 480,
                    fourcc="YUYV" if message == "FOURCC" else None,
                    backlight_compensation=0 if message == "背光补偿" else None,
                )
            self.assertTrue(camera.released)

    def test_rejects_manual_exposure_mode_that_does_not_read_back(self):
        camera = FakeVideoCapture(
            ineffective=(cv2.CAP_PROP_AUTO_EXPOSURE,),
            initial={cv2.CAP_PROP_AUTO_EXPOSURE: 3},
        )
        with patch("utils.camera.cv2.VideoCapture", return_value=camera):
            with self.assertRaisesRegex(RuntimeError, "手动曝光模式"):
                open_camera(0, 20, 6500, None, None)
        self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
