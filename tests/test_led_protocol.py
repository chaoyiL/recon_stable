"""Layout and wire-contract checks independent of GUI and serial hardware."""

from itertools import product
import unittest

from hardware.led_protocol import (
    MAX_LED_COUNT, LedLayout, LedSettings, default_colors, encode_frame,
)


class LedProtocolTests(unittest.TestCase):
    def test_wire_bytes_include_dark_prefix_count_and_scaled_rgb(self):
        layout = LedLayout(top=1, right=0, bottom=0, left=0, skip=2)
        frame = encode_frame(layout, [(200, 100, 50, 50)], brightness=80)
        self.assertEqual(frame, bytes.fromhex("AA 55 03 00 00 00 00 00 00 00 50 28 14 6F"))

    def test_output_calibration_preserves_rgb_order_brightness_and_dark_prefix(self):
        layout = LedLayout(top=1, right=0, bottom=0, left=0, skip=2)
        frame = encode_frame(
            layout, [(200, 150, 50, 50)], brightness=80,
            rgb_scale=(0.5, 0.8, 0.25),
        )
        self.assertEqual(
            frame,
            bytes.fromhex("AA 55 03 00 00 00 00 00 00 00 28 30 05 1E"),
        )

    def test_led_settings_separate_layout_brightness_and_rgb_scale(self):
        settings = LedSettings.from_mapping({
            "top": 2,
            "right": 1,
            "brightness": 65,
            "rgb_scale": [1, 0.8, 0.25],
        })
        self.assertEqual(settings.layout, LedLayout(top=2, right=1))
        self.assertEqual(settings.brightness, 65)
        self.assertEqual(settings.rgb_scale, (1.0, 0.8, 0.25))

        defaults = LedSettings.from_mapping({})
        self.assertEqual(defaults.layout, LedLayout())
        self.assertEqual(defaults.brightness, 100)
        self.assertEqual(defaults.rgb_scale, (1.0, 1.0, 1.0))
        zero_red = LedSettings.from_mapping({"rgb_scale": [0, 1, 1]})
        self.assertEqual(zero_red.rgb_scale, (0.0, 1.0, 1.0))

    def test_led_settings_and_layout_reject_unknown_or_invalid_fields(self):
        with self.assertRaises(ValueError):
            LedLayout.from_mapping({"rgb_scale": [1, 1, 1]})
        with self.assertRaises(ValueError):
            LedSettings.from_mapping({"unexpected": 1})

        for brightness in (True, 100.0, 101, -1):
            with self.subTest(brightness=brightness), self.assertRaises(ValueError):
                LedSettings.from_mapping({"brightness": brightness})

        invalid_scales = (
            (True, 1, 1), (1, 1), (0, 1, 1, 1),
            (-0.01, 1, 1), (1.01, 1, 1), (float("nan"), 1, 1),
            (1, float("inf"), 1),
        )
        layout = LedLayout(top=1, right=0, bottom=0, left=0)
        for rgb_scale in invalid_scales:
            with self.subTest(rgb_scale=rgb_scale), self.assertRaises(ValueError):
                LedSettings.from_mapping({"rgb_scale": rgb_scale})
            with self.subTest(encoder_rgb_scale=rgb_scale), self.assertRaises(ValueError):
                encode_frame(layout, [(255, 0, 0, 100)], rgb_scale=rgb_scale)

    def test_default_layout_keeps_original_wire_order_and_pattern(self):
        layout = LedLayout()
        colors = default_colors(layout)
        self.assertEqual(layout.active_count, 26)
        self.assertEqual(layout.total_count, 29)
        self.assertEqual(colors[0], (248, 0, 83, 100))
        self.assertEqual(colors[8], (248, 83, 0, 100))
        self.assertEqual(colors[9:13], [(0, 255, 0, 100)] * 4)
        self.assertEqual(colors[13], (0, 255, 0, 100))
        self.assertEqual(colors[21], (0, 0, 255, 100))
        self.assertEqual(colors[22:], [(0, 0, 255, 100)] * 4)
        positions = layout.grid_positions()
        self.assertEqual([positions[i] for i in (0, 8, 9, 12, 13, 21, 22, 25)],
                         [(0, 1), (0, 9), (1, 10), (4, 10), (5, 9), (5, 1), (4, 0), (1, 0)])

    def test_empty_single_and_unequal_edges_have_unique_cells(self):
        for counts in product(range(4), repeat=4):
            if sum(counts) == 0:
                continue
            with self.subTest(counts=counts):
                layout = LedLayout(*counts)
                positions = layout.grid_positions()
                self.assertEqual(len(positions), sum(counts))
                self.assertEqual(len(set(positions)), sum(counts))
                self.assertEqual(len(default_colors(layout)), sum(counts))
        layout = LedLayout(top=2, right=0, bottom=5, left=1)
        self.assertEqual(layout.grid_positions()[:2], [(0, 1), (0, 5)])

    def test_maximum_total_includes_dark_prefix(self):
        layout = LedLayout(top=1, right=0, bottom=0, left=0, skip=MAX_LED_COUNT - 1)
        frame = encode_frame(layout, [(255, 0, 0, 100)])
        self.assertEqual(int.from_bytes(frame[2:4], "little"), MAX_LED_COUNT)
        self.assertEqual(len(frame), 3 * MAX_LED_COUNT + 5)
        self.assertEqual(frame[4:-4], bytes(3 * (MAX_LED_COUNT - 1)))
        with self.assertRaises(ValueError):
            LedLayout(top=1, right=0, bottom=0, left=0, skip=MAX_LED_COUNT)

    def test_invalid_layout_fails_before_gui_or_serial_access(self):
        for values in ({"top": -1}, {"skip": True}, {"right": 1.5},
                       {"top": 0, "right": 0, "bottom": 0, "left": 0}, {"unknown": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LedLayout.from_mapping(values)

    def test_invalid_color_count_and_ranges_are_rejected(self):
        layout = LedLayout(top=1, right=0, bottom=0, left=0)
        for colors in ([], [(256, 0, 0, 100)], [(0, 0, 0, 101)], [(0, 0, 0)], [(True, 0, 0, 100)]):
            with self.subTest(colors=colors), self.assertRaises(ValueError):
                encode_frame(layout, colors)
        with self.assertRaises(ValueError):
            encode_frame(layout, [(255, 255, 255, 100)], brightness=101)

    def test_zero_brightness_blacks_out_every_pixel(self):
        layout = LedLayout(skip=3)
        frame = encode_frame(layout, default_colors(layout), brightness=0)
        self.assertEqual(frame[4:-1], bytes(layout.total_count * 3))


if __name__ == "__main__":
    unittest.main()
