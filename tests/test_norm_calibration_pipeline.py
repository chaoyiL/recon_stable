"""Four acceptance scenario groups for reference-material contact calibration.

Run: python -m unittest discover -s tests -p test_norm_calibration_pipeline.py
"""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import yaml
from utils.surface_coordinates import (ArcLengthChart, build_arclength_chart, sample_surface_grid,
                                       save_arclength_chart, load_arclength_chart)
from utils.surface_sampling import (bilinear_sample, sample_linear_rgb_to_surface,
                                    sample_mask_to_surface, rasterize_surface_grid)
from utils.surface_geometry import compute_surface_geometry, normals_to_local_slopes, local_slopes_to_normals
from utils.surface_projection import (CameraModel, project_points, unproject_rays, invert_surface_projection,
                                      projected_circle_radius_px)
from utils.sphere_geometry import (Sphere, fit_sphere_center, intersect_sphere_rays,
                                   select_sphere_cap, sphere_normals)
from utils.region_annotations import (RegionAnnotation, region_boundary_points, region_masks,
                                      save_regions, load_regions)
from utils.normal_mapping import NormalCalibration, build_normal_lut, find_ambiguous_color_pairs
from utils.annotation_window import AnnotationPanel
from reconstruction.surface_reconstruction.surface import SurfaceModel, BicubicSurfaceAdapter
from reconstruction.surface_reconstruction.markers import find_inner_region, find_inner_roi, build_marker_image_mask
from utils.reconstruction_interface import CoordinateGridSpec, MaterialCoordinateSpecs
from calibrate.norm_pipeline import (CalibrationSettings, FrameCalibrationData, _provider_from_frame,
    _save_frame, _save_json, load_frame_calibration_data, run_samples, fit_and_validate_samples,
    compute_frame_colors, FrameCalibrationError, run_prepare, parse_settings)
from reconstruction.local_recon import reconstruct_local_surface, _load_reconstruction_input, lookup_slopes
from calibrate.analyze_ambiguities import (classify_normal_neighborhood, compare_branch_context,
                                         classify_branch_support, aggregate_branch_classification,
                                         predict_conditioned_normals)


def _normals_at_angles(angles):
    angles = np.radians(angles)
    return np.column_stack((np.sin(angles), np.zeros(len(angles)), np.cos(angles)))


class NormalNeighbourhoods(unittest.TestCase):
    def test_continuous_chain_is_not_split_into_multiple_modes(self):
        normals = _normals_at_angles([0, 8, 16, 24, 32, 40])
        result = classify_normal_neighborhood(normals, np.arange(6))
        self.assertEqual(result['kind'], 'connected_spread')
        self.assertEqual(result['sizes'], [6])

    def test_supported_separate_groups_and_a_sparse_alternative_are_distinct(self):
        result = classify_normal_neighborhood(_normals_at_angles([-2, 0, 2, 38, 40, 42]),
                                             np.array([0, 0, 1, 2, 3, 3]))
        self.assertEqual(result['kind'], 'multiple_supported_components')
        self.assertEqual(result['sizes'], [3, 3])
        self.assertEqual(result['image_counts'], [2, 2])
        self.assertEqual(len(result['separated_components']), 1)
        result = classify_normal_neighborhood(_normals_at_angles([-5, -3, -1, 1, 3, 5, 40]),
                                             np.arange(7))
        self.assertEqual(result['kind'], 'dominant_with_sparse_alternative')
        self.assertEqual(result['sizes'], [6, 1])

    def test_two_disagreeing_points_are_not_enough_to_identify_an_outlier(self):
        result = classify_normal_neighborhood(_normals_at_angles([0, 40]), np.array([0, 1]))
        self.assertEqual(result['kind'], 'sparse')


class BranchContinuation(unittest.TestCase):
    def setUp(self):
        self.colors = np.array([[0., 0, 0], [.005, 0, 0], [.002, 0, 0],
                                [.020, 0, 0], [.022, 0, 0], [.025, 0, 0]])
        self.normals = _normals_at_angles([0, 50, 0, 2, 5, 48])
        self.images = np.array([0, 1, 2, 2, 3, 4])

    def test_both_branches_can_have_context_without_six_inner_samples(self):
        result = compare_branch_context(self.colors, self.normals, self.images,
                                        np.array([0, 1]), np.arange(6))
        self.assertEqual(result[0]['support_indices'], [3, 4])
        self.assertEqual(result[1]['support_indices'], [5])
        self.assertEqual(result[0]['support_image_count'], 2)
        self.assertEqual(result[1]['other_image_count'], 1)

    def test_anchors_and_both_original_neighbourhoods_cannot_support_themselves(self):
        result = compare_branch_context(self.colors, self.normals, self.images,
                                        np.array([0, 1]), np.arange(3))
        self.assertEqual([row['support_count'] for row in result], [0, 0])
        self.assertIsNone(result[0]['minimum_context_normal_angle_degrees'])

    def test_missing_outer_support_is_recorded_without_automatically_rejecting_a_branch(self):
        result = compare_branch_context(self.colors, self.normals, self.images,
                                        np.array([0, 1]), np.arange(5))
        self.assertEqual([row['support_count'] for row in result], [2, 0])
        self.assertAlmostEqual(result[1]['minimum_context_normal_angle_degrees'], 45.)
        self.assertNotIn('rejected', result[1])


class ExpandedBranchClassification(unittest.TestCase):
    def test_two_inner_samples_can_be_classified_using_outer_branches(self):
        colors = np.array([[0., 0, 0], [.005, 0, 0], [.020, 0, 0],
                           [.022, 0, 0], [.025, 0, 0]])
        normals = _normals_at_angles([0, 50, 2, 5, 48])
        branches = compare_branch_context(colors, normals, np.arange(5), np.array([0, 1]), np.arange(5))
        # Both original color balls contain only the two anchors. Outer evidence
        # is 2:1; dropping the sparsely sampled alternative would lose a branch.
        counts = [branch['support_count'] for branch in branches]
        self.assertEqual(classify_branch_support(counts), 2)
        branches = compare_branch_context(colors, normals, np.arange(5), np.array([0, 1]), np.arange(4))
        self.assertEqual(classify_branch_support([branch['support_count'] for branch in branches]), 1)

    def test_dominance_boundary_and_missing_evidence_remain_explicit(self):
        kinds = classify_branch_support([[8, 2], [2, 8], [3, 1], [1, 1], [1, 0], [0, 0]])
        np.testing.assert_array_equal(kinds, [1, 1, 2, 2, 1, 0])
        self.assertEqual(classify_branch_support([8, 2], dominance_fraction=.9), 2)

    def test_a_point_overlapping_one_partner_is_not_hidden_by_dominant_other_pairs(self):
        pairs = np.array([[0, 1], [0, 2], [0, 3], [4, 5]])
        kinds = classify_branch_support([[10, 0], [10, 0], [4, 4], [0, 0]])
        actual = aggregate_branch_classification(pairs, kinds, 7)
        np.testing.assert_array_equal(actual, [2, 1, 1, 2, 0, 0, -1])

    def test_invalid_support_and_dominance_threshold_are_rejected(self):
        for counts in ([1, 2, 3], [np.nan, 1], [-1, 2]):
            with self.assertRaises(ValueError):
                classify_branch_support(counts)
        for threshold in (.5, 1.1, np.nan):
            with self.assertRaises(ValueError):
                classify_branch_support([1, 1], dominance_fraction=threshold)


class ConditionedNormalAverage(unittest.TestCase):
    def test_position_can_select_a_branch_when_colors_are_identical(self):
        colors = np.zeros((2, 3))
        positions = np.array([[0., 0.], [1., 1.]])
        normals = _normals_at_angles([0, 45])
        predicted = predict_conditioned_normals(colors, positions, normals, colors[:1], positions[:1],
                                               position_weight=.2, neighbours=2)
        np.testing.assert_allclose(predicted, normals[:1], atol=1e-12)
        mixed = predict_conditioned_normals(colors, positions, normals, colors[:1], positions[:1],
                                           position_weight=0., neighbours=2)
        self.assertGreater(float(np.linalg.norm(mixed-predicted)), .1)
        np.testing.assert_allclose(np.linalg.norm(mixed, axis=-1), 1.)


class ColorSlopeAmbiguities(unittest.TestCase):
    def test_all_close_pairs_are_checked_even_when_the_nearest_neighbour_agrees(self):
        colors = np.array([[0., 0, 0], [.001, 0, 0], [.009, 0, 0], [.03, 0, 0]])
        slopes = np.array([[0., 0], [0., 0], [1., 0], [0., 0]])
        pairs, distances, angles = find_ambiguous_color_pairs(colors, slopes,
            color_distance_threshold=.01, normal_angle_threshold_degrees=20.)
        self.assertEqual(set(map(tuple, pairs)), {(0, 2), (1, 2)})
        np.testing.assert_allclose(np.sort(distances), [.008, .009])
        np.testing.assert_allclose(angles, 45.)

    def test_color_boundary_is_inclusive_and_identical_slopes_do_not_conflict(self):
        colors = np.array([[0., 0, 0], [.01, 0, 0]])
        pairs, _, _ = find_ambiguous_color_pairs(colors, [[0., 0], [1., 0]],
            color_distance_threshold=.01, normal_angle_threshold_degrees=20.)
        np.testing.assert_array_equal(pairs, [[0, 1]])
        for threshold in (0., 180.):
            pairs, _, _ = find_ambiguous_color_pairs(np.zeros((2, 3)), [[.3, -.2], [.3, -.2]],
                color_distance_threshold=0., normal_angle_threshold_degrees=threshold)
            self.assertEqual(pairs.shape, (0, 2))

    def test_empty_samples_and_invalid_inputs(self):
        pairs, distances, angles = find_ambiguous_color_pairs(np.empty((0, 3)), np.empty((0, 2)))
        self.assertEqual(pairs.shape, (0, 2))
        self.assertEqual(distances.shape, (0,))
        self.assertEqual(angles.shape, (0,))
        with self.assertRaisesRegex(ValueError, 'paired'):
            find_ambiguous_color_pairs(np.zeros((2, 3)), np.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, 'finite'):
            find_ambiguous_color_pairs([[np.nan, 0., 0.]], [[0., 0.]])
        with self.assertRaisesRegex(ValueError, 'threshold'):
            find_ambiguous_color_pairs([[0., 0., 0.]], [[0., 0.]], color_distance_threshold=-1.)


class AnalyticPatch:
    parameter_bounds = (0., 1., 0., 1.)

    def __init__(self, kind='parabola', stretch=1.):
        self.kind, self.stretch = kind, stretch

    def evaluate(self, parameters):
        a, b = np.asarray(parameters).T
        if self.kind == 'cylinder':
            return np.column_stack((np.sin(a), b, 20+np.cos(a)))
        if self.kind == 'coupled':
            return np.column_stack((a+.2*a*b, b+.15*a*a*b, 20+.8*a*a+.3*a*b*b))
        return np.column_stack((self.stretch*a, b, 20+a*a))

    def first_derivatives(self, parameters):
        a, b = np.asarray(parameters).T
        z, o = np.zeros_like(a), np.ones_like(a)
        if self.kind == 'cylinder':
            return np.column_stack((np.cos(a), z, -np.sin(a))), np.column_stack((z, o, z))
        if self.kind == 'coupled':
            return (np.column_stack((1+.2*b, .3*a*b, 1.6*a+.3*b*b)),
                    np.column_stack((.2*a, 1+.15*a*a, .6*a*b)))
        return np.column_stack((o*self.stretch, z, 2*a)), np.column_stack((z, o, z))


class ProjectedCircleLimit(unittest.TestCase):
    def test_projected_radius_uses_the_smaller_focal_scale_and_real_depth(self):
        camera = CameraModel(np.array([[400., 0, 320], [0, 600., 240], [0, 0, 1.]]), np.zeros(5), (480, 640))
        # Moving the disk does not change pinhole magnification, even at the image edge.
        for center in ((320., 240.), (639., 240.)):
            self.assertAlmostEqual(projected_circle_radius_px(center, 3., 100., camera), 12.)
            self.assertAlmostEqual(projected_circle_radius_px(center, 3., 200., camera), 6.)
        with self.assertRaises(ValueError):
            projected_circle_radius_px((320., 240.), 3., 0., camera)

    def test_projected_radius_includes_radial_distortion(self):
        camera = CameraModel(np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]]),
                             np.array([-.2, 0, 0, 0, 0]), (480, 640))
        self.assertAlmostEqual(projected_circle_radius_px((320., 240.), 3., 100., camera),
                               15.*(1-.2*.03**2))

    def test_annotation_limit_uses_maximum_valid_positive_camera_z(self):
        from calibrate.manual_norm_regions import _maximum_circle_radius_px
        camera = CameraModel(np.array([[400., 0, 320], [0, 600., 240], [0, 0, 1.]]), np.zeros(5), (480, 640))
        xyz = np.zeros((2, 3, 3))
        xyz[..., 2] = [[50., 100., 400.], [np.nan, -10., 0.]]
        grid = SimpleNamespace(xyz=xyz, valid_mask=np.array([[True, True, False], [True, True, True]]))
        frame = SimpleNamespace(camera=camera, systems=SimpleNamespace(observation=grid))
        self.assertAlmostEqual(_maximum_circle_radius_px(frame, 3., (320., 240.)), 12.)
        grid.valid_mask[:] = False
        with self.assertRaisesRegex(ValueError, 'positive surface depth'):
            _maximum_circle_radius_px(frame, 3., (320., 240.))


class ArcLengthAndCoordinates(unittest.TestCase):
    def test_lengths_coupling_roundtrip_and_frozen_identity(self):
        chart = build_arclength_chart(AnalyticPatch())
        values, valid = chart.to_st([[.5, .5]])
        self.assertTrue(valid.all())
        self.assertAlmostEqual(values[0, 0], .388045, places=5)
        cylinder = build_arclength_chart(AnalyticPatch('cylinder'))
        points = np.random.default_rng(3).uniform(0, 1, (100, 2))
        np.testing.assert_allclose(cylinder.to_st(points)[0], points, atol=1e-12)
        surface = AnalyticPatch('coupled')
        chart = build_arclength_chart(surface)
        st, valid = chart.to_st(points)
        ab, inverse_valid = chart.to_parameters(st)
        self.assertTrue((valid & inverse_valid).all())
        np.testing.assert_allclose(ab, points, atol=2e-8)
        jac, _ = chart.jacobian(points[:3])
        for axis in (0, 1):
            step = np.zeros(2); step[axis] = 1e-6
            finite_diff = (chart.to_st(points[:3]+step)[0]-chart.to_st(points[:3]-step)[0])/2e-6
            np.testing.assert_allclose(jac[..., axis], finite_diff, atol=1e-7)
        grid = sample_surface_grid(surface, chart, 17, 13)
        moved = sample_surface_grid(AnalyticPatch(stretch=2), chart, 17, 13)
        np.testing.assert_array_equal(grid.parameters, moved.parameters)
        self.assertFalse(np.allclose(grid.xyz, moved.xyz))
        with TemporaryDirectory() as directory:
            loaded = load_arclength_chart(save_arclength_chart(Path(directory)/'chart.npz', chart))
            self.assertEqual(loaded.chart_id, chart.chart_id)
            np.testing.assert_allclose(loaded.to_parameters(st)[0], points, atol=2e-8)
            self.assertFalse(loaded.st_table.flags.writeable)
        query, ok = chart.to_parameters([[0, 0], [1, 1], [-1e-13, .5], [-1e-4, .5], [np.nan, 0]])
        np.testing.assert_array_equal(ok, [True, True, True, False, False])
        np.testing.assert_allclose(query[:2], [[0, 0], [1, 1]], atol=1e-8)
        table = chart.st_table.copy()
        table[table.shape[0]//2, table.shape[1]//2] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'invertible'):
            replace(chart, st_table=table, arc_lengths_mm=np.stack((table[..., 0]*chart.row_lengths_mm[:, None],
                         table[..., 1]*chart.column_lengths_mm[None, :]), axis=-1))

    def test_projection_inverse_and_multiple_solutions(self):
        surface = AnalyticPatch('coupled')
        chart = build_arclength_chart(surface)
        grid = sample_surface_grid(surface, chart, 35, 31)
        camera = CameraModel(np.array([[400, 0, 100], [0, 400, 100], [0, 0, 1.]]), np.zeros(5), (200, 200))
        uv, valid = project_points(grid.xyz, camera)
        seed = dict(parameters=grid.parameters, uv=uv, valid_mask=valid)
        ab = np.array([[.2, .3], [.6, .7], [.8, .2]])
        pixels, _ = project_points(surface.evaluate(ab), camera)
        result = invert_surface_projection(pixels, surface, camera, seed)
        self.assertTrue(result.valid.all())
        np.testing.assert_allclose(result.parameters, ab, atol=1e-7)
        self.assertTrue(np.all(result.sigma_mm > 0))
        class Fold:
            parameter_bounds = (-1., 1., -1., 1.)
            def evaluate(self, ab):
                a, b = ab.T
                return np.column_stack((a*a, b, np.full(len(a), 10.)))
            def first_derivatives(self, ab):
                a, b = ab.T
                return np.column_stack((2*a, a*0, a*0)), np.column_stack((b*0, b*0+1, b*0))
        a, b = np.meshgrid(np.linspace(-1, 1, 31), np.linspace(-1, 1, 25))
        ab = np.stack((a, b), axis=-1)
        xyz = Fold().evaluate(ab.reshape(-1, 2)).reshape(25, 31, 3)
        uv, valid = project_points(xyz, camera)
        pixel, _ = project_points([[.25, .1, 10]], camera)
        result = invert_surface_projection(pixel, Fold(), camera, dict(parameters=ab, uv=uv, valid_mask=valid))
        self.assertFalse(result.valid[0])
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            build_arclength_chart(Fold())


class IndependentColorsAndHoles(unittest.TestCase):
    def test_reconstruction_uses_equal_color_equations_independent_of_confidence(self):
        from scipy.sparse.linalg import lsmr

        row, column = np.indices((13, 15), dtype=float)
        xyz = np.stack((.25*column, .5*row, np.full_like(row, 10.)), axis=-1)
        colors = np.stack((.05*np.sin(column/2)*np.cos(row/3),
                           -.03*np.cos(column/4)*np.sin(row/2), row*0), axis=-1)
        photo = np.ones(row.shape, bool)
        photo[6, 7] = False
        colors[~photo] = np.nan
        trusted = np.zeros(row.shape, bool)
        trusted[1:3, 1:3] = True
        axis = np.linspace(-1, 1, 5)
        r, g, _ = np.meshgrid(axis, axis, axis, indexing='ij')
        calibration = NormalCalibration(np.stack((r, g), axis=-1), np.zeros((5, 5, 5)),
            -np.ones(3), np.ones(3), 1.)
        slopes, variance, _ = lookup_slopes(colors, calibration, photo,
            trusted_no_contact_mask=trusted)
        confidences = [photo.astype(np.float32),
                       np.where(photo, .01+.95*column/column.max(), 0).astype(np.float32),
                       np.zeros(row.shape, np.float32)]
        results, systems = [], []
        for confidence in confidences:
            with patch('reconstruction.local_recon.lookup_slopes', return_value=(slopes, variance, confidence)), \
                    patch('reconstruction.local_recon.lsmr', wraps=lsmr) as solve:
                result = reconstruct_local_surface(xyz, colors, calibration,
                    photometric_valid_mask=photo, trusted_no_contact_mask=trusted,
                    displacement_zero_lambda_per_mm2=.25)
            results.append(result)
            systems.append(solve.call_args.args[:2])
            np.testing.assert_array_equal(result.confidence, confidence)
            self.assertTrue(result.inferred_mask[6, 7])
        self.assertGreater(np.max(np.abs(results[0].displacement)), .001)
        matrix, target = systems[0]
        # A centered derivative on this plane has coefficients 1/(2*0.25)
        # horizontally and 1/(2*0.5) vertically, with no extra row scaling.
        np.testing.assert_allclose(matrix.getrow(0).data, [2.])
        np.testing.assert_allclose(matrix.getrow(1).data, [1.])
        for result, (other_matrix, other_target) in zip(results[1:], systems[1:], strict=True):
            np.testing.assert_array_equal(result.displacement, results[0].displacement)
            np.testing.assert_array_equal(result.xyz_out, results[0].xyz_out)
            self.assertEqual((matrix-other_matrix).nnz, 0)
            np.testing.assert_array_equal(other_target, target)

    def test_zero_slope_prior_sets_unit_confidence_only_for_valid_inner_colors(self):
        calibration = NormalCalibration(
            np.broadcast_to([.4, -.2], (2, 2, 2, 2)).copy(),
            np.full((2, 2, 2), 9.), -np.ones(3), np.ones(3), 1.)
        colors = np.array([[0, 0, 0], [-.02, .01, 0], [0, 0, .04],
                           [.06, 0, 0], [.08, 0, 0], [.12, 0, 0],
                           [np.nan, 0, 0], [.01, 0, 0]], np.float32)
        valid = np.ones(len(colors), bool)
        valid[-1] = False
        raw_slopes, raw_variance, raw_confidence = lookup_slopes(colors, calibration, valid)
        slopes, variance, confidence = lookup_slopes(colors, calibration, valid,
            zero_color_inner_radius=.04, zero_color_outer_radius=.08)
        np.testing.assert_array_equal(slopes[:3], 0)
        np.testing.assert_array_equal(confidence[:3], 1)
        np.testing.assert_allclose(slopes[3], raw_slopes[3]/2, atol=1e-7)
        np.testing.assert_array_equal(slopes[4:6], raw_slopes[4:6])
        np.testing.assert_array_equal(confidence[3:6], raw_confidence[3:6])
        np.testing.assert_array_equal(slopes[6:], 0)
        np.testing.assert_array_equal(confidence[6:], 0)
        np.testing.assert_array_equal(variance, raw_variance)
        np.testing.assert_allclose(raw_confidence[:6], .1, atol=1e-7)

    def test_no_contact_zero_slope_prior_reaches_solver_without_filling_marker_confidence(self):
        row, column = np.indices((13, 15), dtype=float)
        xyz = np.stack((column, row, np.full_like(row, 10.)), axis=-1)
        colors = np.full_like(xyz, .02, dtype=np.float32)
        photo = np.ones(row.shape, bool)
        photo[6, 7] = False
        colors[~photo] = np.nan
        calibration = NormalCalibration(
            np.broadcast_to([.4, -.2], (2, 2, 2, 2)).copy(),
            np.full((2, 2, 2), 9.), -np.ones(3), np.ones(3), 1.)
        result = reconstruct_local_surface(xyz, colors, calibration,
            photometric_valid_mask=photo, zero_color_inner_radius=.04,
            zero_color_outer_radius=.08)
        np.testing.assert_array_equal(result.confidence[photo], 1)
        np.testing.assert_array_equal(result.confidence[~photo], 0)
        np.testing.assert_array_equal(result.slopes, 0)
        np.testing.assert_allclose(result.displacement, 0, atol=1e-7)
        np.testing.assert_allclose(result.xyz_out, xyz, atol=1e-7)
        self.assertTrue(result.inferred_mask[6, 7])
        self.assertFalse(result.boundary_mask[6, 7])

    def test_signed_protection_uses_independent_thresholds_and_mixed_channel_response(self):
        from reconstruction.local_recon import parse_zero_color_protection
        from utils.color_protection import resolve_zero_color_radii, signed_color_protection_factors

        calibration = NormalCalibration(np.broadcast_to([.4, -.2], (2, 2, 2, 2)).copy(),
            np.full((2, 2, 2), 9.), -np.ones(3), np.ones(3), 1.)
        colors = np.array([[0, 0, 0], [.06, 0, 0], [-.02, 0, 0], [-.03, 0, 0],
            [-.055, 0, 0], [-.08, 0, 0], [-.12, 0, 0], [.115, 0, 0], [.15, 0, 0],
            [.10, -.055, 0], [.07, -.08, 0], [.09, -.02, 0], [np.nan, 0, 0], [-.055, 0, 0]], np.float32)
        valid = np.ones(len(colors), bool)
        valid[-1] = False
        kwargs = dict(zero_color_inner_radius=.08, zero_color_outer_radius=.15,
                      zero_color_negative_inner_radius=.03, zero_color_negative_outer_radius=.08)
        slopes, variance, confidence = lookup_slopes(colors, calibration, valid, **kwargs)
        t = (.09-.08)/(.15-.08)
        factors = np.array([0, 0, 0, 0, .5, 1, 1, .5, 1, .5, 1, t*t*(3-2*t), 0, 0])
        np.testing.assert_allclose(slopes, factors[:, None]*[.4, -.2], atol=2e-7)
        np.testing.assert_allclose(confidence, [1, 1, 1, 1, .1, .1, .1, .1, .1, .1, .1, .1, 0, 0], atol=1e-7)
        np.testing.assert_array_equal(variance, lookup_slopes(colors, calibration, valid)[1])
        np.testing.assert_array_equal(variance[12:], 0)
        trusted = np.zeros(len(colors), bool)
        trusted[5] = True
        forced, _, forced_confidence = lookup_slopes(colors, calibration, valid,
            trusted_no_contact_mask=trusted, **kwargs)
        np.testing.assert_array_equal(forced[5], 0)
        self.assertEqual(forced_confidence[5], 1)
        self.assertEqual(parse_zero_color_protection(dict(zero_color_protection=dict(inner_radius=.08, outer_radius=.15))),
                         (.08, .15, .08, .15))
        self.assertEqual(parse_zero_color_protection(dict(zero_color_protection=dict(inner_radius=.08, outer_radius=.15,
            negative_inner_radius=.03, negative_outer_radius=.08))), (.08, .15, .03, .08))
        for negative_pair in ((.08, .03), (-.03, .08), (.01, 0), (np.nan, .08)):
            with self.assertRaisesRegex(ValueError, 'negative'):
                resolve_zero_color_radii(.08, .15, *negative_pair)
        # Disabling one sign does not let its zero-valued channels disable the other sign's protection.
        samples = np.array([[-.02, 0, 0], [.001, 0, 0], [0, 0, 0]], np.float32)
        attenuation, zero = signed_color_protection_factors(samples, 0, 0, .03, .08)
        np.testing.assert_array_equal(attenuation, [0, 1, 0])
        attenuation, zero = signed_color_protection_factors(samples, 0, 0, 0, 0)
        np.testing.assert_array_equal(attenuation, 1)
        self.assertFalse(zero.any())

    def test_asymmetric_negative_protection_reaches_the_local_solver(self):
        rows, columns = np.mgrid[-1:1:13j, -1:1:15j]
        xyz = np.stack((columns, rows, np.full_like(rows, 10.)), axis=-1)
        colors = np.zeros_like(xyz, dtype=np.float32)
        colors[..., 0] = -.03-.04*(1-columns**2)*(1-rows**2)
        calibration = NormalCalibration(np.broadcast_to([.4, -.2], (2, 2, 2, 2)).copy(),
            np.full((2, 2, 2), 9.), -np.ones(3), np.ones(3), 1.)
        legacy = reconstruct_local_surface(xyz, colors, calibration,
            zero_color_inner_radius=.08, zero_color_outer_radius=.15)
        asymmetric = reconstruct_local_surface(xyz, colors, calibration,
            zero_color_inner_radius=.08, zero_color_outer_radius=.15,
            zero_color_negative_inner_radius=.03, zero_color_negative_outer_radius=.08)
        np.testing.assert_array_equal(legacy.displacement, 0)
        self.assertGreater(float(np.max(abs(asymmetric.displacement))), .001)
        self.assertGreater(float(np.max(abs(asymmetric.slopes))), .1)

    def test_asymmetric_protection_matches_jax_jit(self):
        import jax
        import jax.numpy as jnp
        from utils.jax_local_reconstruction import lookup_slopes_jax

        calibration = NormalCalibration(np.broadcast_to([.4, -.2], (2, 2, 2, 2)).copy(),
            np.full((2, 2, 2), 9.), -np.ones(3), np.ones(3), 1.)
        colors = np.array([[0, 0, 0], [.06, -.055, 0], [.10, -.055, 0],
                           [0, -.08, .07], [.115, -.03, 0], [np.nan, 0, 0], [-.06, 0, 0]], np.float32)
        valid = np.ones(len(colors), bool)
        valid[-1] = False
        kwargs = dict(zero_color_inner_radius=.08, zero_color_outer_radius=.15,
                      zero_color_negative_inner_radius=.03, zero_color_negative_outer_radius=.08)
        expected = lookup_slopes(colors, calibration, valid, **kwargs)
        lut, variance = jnp.asarray(calibration.slopes), jnp.asarray(calibration.variances)
        low, high = jnp.asarray(calibration.color_min), jnp.asarray(calibration.color_max)
        compiled = jax.jit(lambda c, m: lookup_slopes_jax(c, lut, variance, low, high,
            calibration.sigma_ref2, m, **kwargs))
        actual = compiled(jnp.asarray(colors), jnp.asarray(valid))
        for cpu, device in zip(expected, actual, strict=True):
            np.testing.assert_allclose(cpu, np.asarray(device), rtol=5e-7, atol=2e-7)

    def test_inference_samples_each_image_and_preserves_geometry(self):
        y, x = np.indices((24, 32), dtype=float)
        reference = np.stack((x/100, y/100, (x+y)/200), axis=-1)
        observed = np.stack(((x-3)/100, (y-2)/100, (x+y-5)/200), axis=-1)
        uv = np.array([[[5.2, 6.7], [12., 15.]], [[3.5, 4.], [22.2, 13.9]]])
        st = np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], float)
        metadata = dict(coordinate_kind='reference_arclength', coordinate_chart_id='test-chart',
                        frame_convention='st_grid_columns_rows', residual_domain='material_grid', color_space='linear_rgb')
        values = dict(xyz=np.ones((2, 2, 3)), uv=uv+[3, 2], reference_uv=uv,
                      st=st, reference_st=st, material_coordinates=st, reference_material_coordinates=st,
                      observed_linear_rgb=observed, reference_linear_rgb=reference,
                      surface_mask=np.ones((24, 32), bool), reference_surface_mask=np.ones((24, 32), bool),
                      marker_image_mask=np.zeros((24, 32), bool), reference_marker_image_mask=np.zeros((24, 32), bool),
                      **metadata)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'input.npz'
            np.savez_compressed(path, **values)
            xyz, colors, geometry, photo, _ = _load_reconstruction_input(path, metadata)
            np.testing.assert_allclose(colors, 0, atol=1e-8)
            self.assertTrue(photo.all() and geometry.all())
            # Missing reference visibility, current marker and saturated reference
            # each remove one photometric sample while retaining all geometry.
            values['reference_surface_mask'][6:8, 5:7] = False
            values['marker_image_mask'][17, 15] = True
            values['reference_linear_rgb'][4, 3, 0] = 1
            np.savez_compressed(path, **values)
            _, _, geometry, photo, _ = _load_reconstruction_input(path, metadata)
            self.assertTrue(geometry.all())
            np.testing.assert_array_equal(photo, [[False, False], [False, True]])

    def test_sampling_masks_and_distinct_uv(self):
        y, x = np.indices((24, 32), dtype=float)
        reference = np.stack((x/100, y/100, (x+y)/200), axis=-1)
        observed = np.stack(((x-3)/100, (y-2)/100, (x+y-5)/200), axis=-1)
        uv = np.array([[[5.2, 6.7], [12., 15.]], [[3.5, 4.], [22.2, 13.9]]])
        valid = np.ones((24, 32), bool)
        ref = sample_linear_rgb_to_surface(reference, uv, valid)
        current = sample_linear_rgb_to_surface(observed, uv+[3, 2], valid)
        np.testing.assert_allclose(ref.values, current.values, atol=1e-15)
        self.assertEqual(ref.values.dtype, np.float64)
        mask = valid.copy(); mask[7, 6] = False
        self.assertFalse(sample_mask_to_surface(mask, uv)[0, 0])
        result = bilinear_sample(reference, [[31, 4], [5, 23], [-.1, 2], [np.nan, 2]])
        self.assertFalse(result.valid.any())
        result = bilinear_sample(reference, [[31, 23]], allow_boundary=True, with_jacobian=True)
        self.assertTrue(result.valid[0])
        np.testing.assert_allclose(result.jacobian[0], [[.01, 0], [0, .01], [.005, .005]], atol=1e-15)
        image = np.full((80, 80, 3), 180, np.uint8)
        cv2.line(image, (36, 40), (44, 40), (0, 0, 0), 1)
        cv2.line(image, (40, 36), (40, 44), (0, 0, 0), 1)
        with patch('reconstruction.surface_reconstruction.markers.DETECT_X_MIN', 0):
            mask = build_marker_image_mask(image, np.array([[40., 40.]]), [True], [[9, 9]])
            self.assertTrue(mask[40, 40] and mask[40, 45])
            self.assertFalse(mask[40, 46])
            fallback = build_marker_image_mask(image, np.array([[60., 60.]]), [False], [[9, 9]])
            self.assertTrue(fallback[60, 60] and fallback[60, 66])

    def test_surface_rasterization_preserves_pixel_coordinates_and_holes(self):
        y, x = np.indices((5, 7), dtype=float)
        uv = np.stack((20+4*x+y, 10+3*y), axis=-1)
        field = uv.copy()
        valid = np.ones((5, 7), bool)
        valid[2, 3] = False
        field[2, 3] = np.nan
        result = rasterize_surface_grid(field, uv, valid, (40, 60))
        for px, py in ((25, 13), (27, 19), (23, 12)):
            self.assertTrue(result.valid[py, px])
            np.testing.assert_allclose(result.values[py, px], [px, py], atol=1e-10)
        self.assertFalse(result.valid[16, 34])  # Projected position of the missing sample.
        self.assertFalse(result.valid[13, 35])  # Four-neighbour support touches the hole.
        self.assertFalse(result.valid[0, 0])
        np.testing.assert_array_equal(result.values[~result.valid], 0)
        degenerate = rasterize_surface_grid(field, np.zeros_like(uv), valid, (40, 60))
        self.assertFalse(degenerate.valid.any())

    def test_marker_hole_remains_displacement_unknown(self):
        y, x = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25), indexing='ij')
        xyz = np.stack((x, y, np.full_like(x, 10)), axis=-1)
        p = -.4*x*(1-x*x)*(1-y*y)**2
        q = -.4*y*(1-y*y)*(1-x*x)**2
        colors = np.stack((p, q, p*0), axis=-1).astype(np.float32)
        axis = np.linspace(-1, 1, 9)
        r, g, b = np.meshgrid(axis, axis, axis, indexing='ij')
        calibration = NormalCalibration(np.stack((r, g), axis=-1), np.zeros((9, 9, 9)),
                                         -np.ones(3), np.ones(3), 1e-4)
        photo = np.ones(x.shape, bool); photo[10:15, 10:15] = False
        colors[~photo] = np.nan
        result = reconstruct_local_surface(xyz, colors, calibration, photometric_valid_mask=photo)
        self.assertFalse(result.boundary_mask[12, 12])
        self.assertTrue(result.valid_mask[12, 12] and result.inferred_mask[12, 12])
        self.assertEqual(result.confidence[12, 12], 0)
        self.assertGreater(result.displacement[12, 12], .01)
        with self.assertRaisesRegex(ValueError, '颜色方程'):
            reconstruct_local_surface(xyz, colors, calibration, photometric_valid_mask=np.zeros_like(photo))

    def test_marker_search_clips_both_x_limits_and_keeps_source_coordinates(self):
        from reconstruction.surface_reconstruction import markers

        upper = markers.DETECT_X_MAX
        for lower in (10, 50):
            with self.subTest(lower=lower), patch.object(markers, 'DETECT_X_MIN', lower):
                image = np.full((80, 640), 220, np.uint8)
                for x, y in ((lower-5, 40), (lower, 20), (lower+10, 40),
                             (upper-5, 40), (upper+5, 40)):
                    cv2.line(image, (x-3, y), (x+3, y), 0, 1)
                    cv2.line(image, (x, y-3), (x, y+3), 0, 1)
                centers, binary, diagnostics = markers.detect_cross_centers(image, (0, 0, 640, 80))
                np.testing.assert_allclose(centers, [[lower+10., 40.], [upper-5., 40.]], atol=1e-8)
                self.assertEqual(binary.shape, (80, upper-lower+1))
                self.assertEqual([item['bbox'][0] for item in diagnostics], [lower+7, upper-8])
                centers, binary, _ = markers.detect_cross_centers(image, (lower+5, 0, 40, 80))
                np.testing.assert_allclose(centers, [[lower+10., 40.]], atol=1e-8)
                self.assertEqual(binary.shape, (80, 40))
                for roi in ((0, 0, lower, 80), (upper+1, 0, 20, 80)):
                    centers, binary, diagnostics = markers.detect_cross_centers(image, roi)
                    self.assertEqual(centers.shape, (0, 2))
                    self.assertEqual(binary.shape, (80, 0))
                    self.assertEqual(diagnostics, [])

    def test_full_contour_mask_survives_detector_x_limits(self):
        from reconstruction.surface_reconstruction import markers

        image = np.full((350, 640), 220, np.uint8)
        polygon = np.array([[110, 40], [550, 40], [550, 290], [400, 290], [400, 250], [110, 250]])
        cv2.polylines(image, [polygon], True, 0, 7)
        for yy in (80, 120, 160, 200):
            for xx in (160, 210, 260, 310):
                cv2.line(image, (xx-3, yy), (xx+3, yy), 0, 1)
                cv2.line(image, (xx, yy-3), (xx, yy+3), 0, 1)
        for lower in (markers.DETECT_X_MIN, 200):
            with patch('reconstruction.surface_reconstruction.markers.DETECT_X_MIN', lower):
                region = find_inner_region(image)
                self.assertEqual(find_inner_roi(image), region.roi)
                self.assertTrue(region.mask[150, 520])
                self.assertTrue(region.mask[150, 150])
                self.assertFalse(region.mask[275, 200])
                self.assertTrue(region.mask[120, 210])
                self.assertGreaterEqual(region.roi[0], lower)
                self.assertLessEqual(region.roi[0]+region.roi[2], markers.DETECT_X_MAX+1)


class SphereAndSigns(unittest.TestCase):
    def test_known_center_both_caps_and_normal_roundtrip(self):
        radius, contact = 10., 3.
        angles = np.arange(128)*2*np.pi/128
        a = np.array([0., 0., 100.])
        e1, e2, n0 = np.eye(3)
        boundary = a+np.column_stack((contact*np.cos(angles), contact*np.sin(angles), angles*0))
        for side in (-1, 1):
            sphere, rms = fit_sphere_center(boundary, np.full(128, .01), radius, (a, e1, e2, n0), side)
            np.testing.assert_allclose(sphere.center, a+side*np.sqrt(radius**2-contact**2)*n0, atol=1e-9)
            self.assertLess(rms, 1e-9)
            _, candidates, valid = intersect_sphere_rays([0, 0, 0], [[0, 0, 1]], sphere)
            normals = -side*sphere_normals(candidates[0], sphere)
            good = valid[0] & ((normals @ n0) > .1)
            root = select_sphere_cap(np.where(good[:, None], candidates[0], np.nan), a, n0, side, .01)
            index = 0 if root == -1 else 1
            self.assertGreater(np.dot(normals[index], n0), .99)
            self.assertLessEqual(side*np.dot(candidates[0, index]-a, n0), .01)
        slopes = np.random.default_rng(8).uniform(-.3, .3, (30, 2))
        normals, valid = local_slopes_to_normals(slopes, n0, np.stack((e1, e2), axis=-1))
        back, ok = normals_to_local_slopes(normals, n0, np.stack((e1, e2), axis=-1))
        self.assertTrue((valid & ok).all())
        np.testing.assert_allclose(back, slopes, atol=1e-15)
        with self.assertRaises(ValueError):
            fit_sphere_center(boundary, np.full(128, .01), 2., (a, e1, e2, n0), 1)
        damaged = boundary.copy(); damaged[:30, 2] += 1
        with self.assertRaises(ValueError):
            fit_sphere_center(damaged, np.full(128, .001), radius, (a, e1, e2, n0), 1)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            select_sphere_cap([[-1, 0, 100], [1, 0, 100]], a, n0, 1, .01)


# The headless fixture contains known coefficients and exact marker observations.
# It exercises cache replay and final contact exclusion with the real surface fitter.
def synthetic_batch(directory):
    work = Path(directory)
    settings = CalibrationSettings(work, work/'normal_samples.npz', work/'normal_lut.npz',
        work/'manual_regions.yaml', sphere_radius_mm=10., sphere_side='positive_normal',
        boundary_model='zero_boundary_displacement',
        lut_size=12)
    camera = CameraModel(np.array([[650, 0, 320], [0, 650, 240], [0, 0, 1.]]), np.zeros(5), (480, 640))
    template = np.array([(a, b) for b in np.linspace(0, 42, 8) for a in np.linspace(0, 15, 6)])
    model = SurfaceModel(template, camera.matrix, camera.distortion)
    points = np.column_stack((template-[7.5, 21], np.full(48, 100.)))
    coefficients = np.linalg.lstsq(model.B, points, rcond=None)[0]
    surface = BicubicSurfaceAdapter(model, coefficients)
    chart = build_arclength_chart(surface)
    chart_file = save_arclength_chart(work/'reference_coordinates.npz', chart)
    marker_pixels, _ = project_points(points, camera)
    image = np.full((480, 640, 3), 150, np.uint8)
    for pixel in marker_pixels:
        x, y = np.rint(pixel).astype(int)
        cv2.line(image, (x-3, y), (x+3, y), (0, 0, 0), 1)
        cv2.line(image, (x, y-3), (x, y+3), (0, 0, 0), 1)
    sizes = np.full((48, 2), 7.)
    marker_mask = build_marker_image_mask(image, marker_pixels, np.ones(48, bool), sizes)
    specs = MaterialCoordinateSpecs(*[CoordinateGridSpec(100, 48)]*3)
    reference = FrameCalibrationData('__reference__', -1, work/'reference.png', image, np.ones((480, 640), bool),
        dict(pixels=marker_pixels, valid=np.ones(48, bool), confidence=np.ones(48)), template, sizes, marker_mask,
        coefficients, surface, camera, chart, chart_file, None, 'final', {'status': 'estimated'}, dict(model.options))
    reference.systems = _provider_from_frame(reference).sample_coefficients(coefficients, specs, reference.surface_mask)
    _save_frame(work/'reference_surface.npz', reference)
    cv2.imwrite(str(reference.source_image), image)
    a = np.array([0., 0., 100.]); n0 = np.array([0., 0., 1.])
    sphere = Sphere(a+np.sqrt(100-9)*n0, 10.)
    yy, xx = np.indices((480, 640), dtype=float)
    pixels = np.stack((xx, yy), axis=-1)
    rays, valid = unproject_rays(pixels, camera)
    _, intersection, roots = intersect_sphere_rays(np.zeros(3), rays, sphere)
    normals = -sphere_normals(intersection[..., 0, :], sphere)
    p = -normals[..., 0]/np.maximum(normals[..., 2], .1)
    q = -normals[..., 1]/np.maximum(normals[..., 2], .1)
    delta = np.stack((.25*p, .25*q, .1*(p+q)), axis=-1)
    base = ((150/255+.055)/1.055)**2.4
    linear = np.clip(base+delta, 0, 1)
    srgb = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    observed = np.rint(np.clip(srgb[..., ::-1], 0, 1)*255).astype(np.uint8)
    contact_pixels = (xx-320)**2+(yy-240)**2 < (650*3/100)**2
    image_current = np.where(contact_pixels[..., None] & ~marker_mask[..., None], observed, image)
    entries, annotations = [], {}
    for index in range(3):
        key = f'folder{index}/same_name.png'
        source = work/key
        source.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(source), image_current)
        prepared = replace(reference, image_key=key, image_index=index, source_image=source,
                           original_bgr=image_current, geometry_stage='preview_reference')
        _save_frame(work/'prepared'/f'{index:06d}.npz', prepared)
        entries.append(dict(index=index, key=key, source_image=str(source), reference_image=str(reference.source_image), status='prepared'))
        annotations[key] = RegionAnnotation(key, (640, 480), status='accepted', center_px=(320., 240.),
            semi_axes_px=(19.5, 19.5), semantics='contact_boundary')
    _save_json(work/'manifest.json', dict(version=1, image_root=str(work), reference_image=str(reference.source_image),
                                        coordinate_chart_id=chart.chart_id, images=entries))
    save_regions(settings.annotations_file, annotations, image_root=work)
    return settings, reference, annotations


class HeadlessPipelineAndReuse(unittest.TestCase):
    def test_cached_marker_mask_refresh_preserves_geometry_and_chart(self):
        with TemporaryDirectory() as directory:
            settings, reference, _ = synthetic_batch(directory)
            old_mask = build_marker_image_mask(reference.original_bgr, reference.tracking['pixels'],
                reference.tracking['valid'], reference.reference_marker_sizes, dilation_px=2)
            path = settings.work_dir/'reference_surface.npz'
            _save_frame(path, replace(reference, marker_image_mask=old_mask))
            loaded = load_frame_calibration_data(path)
            self.assertLess(loaded.marker_image_mask.sum(), old_mask.sum())
            np.testing.assert_array_equal(loaded.marker_image_mask, reference.marker_image_mask)
            np.testing.assert_array_equal(loaded.coefficients, reference.coefficients)
            self.assertEqual(loaded.chart.chart_id, reference.chart.chart_id)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['marker_image_mask'], old_mask)
                np.testing.assert_array_equal(loaded.systems.observation.xyz, data['observation_xyz'])

    def test_circle_boundary_masks_and_yaml_roundtrip(self):
        circle = RegionAnnotation('circle.png', (80, 60), center_px=(40., 30.),
            shape='circle', semi_axes_px=(12., 12.), semantics='contact_boundary')
        boundary = region_boundary_points(circle)
        np.testing.assert_allclose(np.linalg.norm(boundary-circle.center_px, axis=1), 12., atol=1e-12)
        np.testing.assert_allclose(boundary[::32], [[52, 30], [40, 42], [28, 30], [40, 18]])
        masks = region_masks(circle, (60, 80, 3), outer_margin_px=5)
        self.assertTrue(masks['contact'][30, 40])
        self.assertTrue(masks['contact'][30, 51])
        self.assertFalse(masks['contact'][30, 55])
        self.assertTrue(masks['exclusion'][30, 55])
        with TemporaryDirectory() as directory:
            path = Path(directory)/'regions.yaml'
            save_regions(path, [circle], image_root=directory)
            loaded = load_regions(path, image_root=directory)['circle.png']
            self.assertEqual(loaded.shape, 'circle')
            np.testing.assert_array_equal(region_boundary_points(loaded), boundary)
        with self.assertRaisesRegex(ValueError, 'equal'):
            replace(circle, semi_axes_px=(12., 10.))

    def test_calibration_editor_uses_circles_for_new_and_reopened_annotations(self):
        from calibrate.manual_norm_regions import annotate_images
        with TemporaryDirectory() as directory:
            settings, _, regions = synthetic_batch(directory)
            keys = list(regions)
            regions[keys[0]] = replace(regions[keys[0]], semi_axes_px=(12., 27.), angle_degrees=30.)
            regions[keys[1]] = replace(regions[keys[1]], shape='polygon',
                contour_px=[[310., 230.], [330., 230.], [330., 250.], [310., 250.]])
            del regions[keys[2]]
            save_regions(settings.annotations_file, regions, image_root=directory)
            edited = []

            def accept_circle(regions_to_edit, *, save_callback, **callbacks):
                self.assertEqual(callbacks['editable_panel'], 1)
                region = regions_to_edit[0]
                self.assertEqual(region.shape, 'circle')
                self.assertEqual(region.status, 'draft')
                self.assertEqual(region.angle_degrees, 0.)
                self.assertIsNone(region.contour_px)
                edited.append(asdict(region))
                region.status = 'accepted'
                save_callback()
                return True

            with patch('calibrate.manual_norm_regions.edit_regions', side_effect=accept_circle):
                self.assertTrue(annotate_images(settings, edit=True))
            self.assertEqual(len(edited), 3)
            np.testing.assert_allclose(edited[0]['semi_axes_px'], [18., 18.])
            np.testing.assert_allclose(edited[1]['semi_axes_px'], [20/np.sqrt(np.pi)]*2)
            np.testing.assert_array_equal(edited[2]['semi_axes_px'], [40., 40.])
            np.testing.assert_array_equal(edited[0]['center_px'], [320., 240.])
            np.testing.assert_array_equal(edited[1]['center_px'], [320., 240.])
            loaded = load_regions(settings.annotations_file, image_root=directory)
            self.assertTrue(all(row.shape == 'circle' and row.status == 'accepted' for row in loaded.values()))

    def test_calibration_editor_updates_size_limit_from_recomputed_geometry(self):
        from calibrate.manual_norm_regions import annotate_images
        with TemporaryDirectory() as directory:
            settings, reference, _ = synthetic_batch(directory)
            deeper_grid = replace(reference.systems.observation, xyz=reference.systems.observation.xyz*2)
            deeper = replace(reference, systems=replace(reference.systems, observation=deeper_grid))

            def inspect_limit(regions_to_edit, *, maximum_circle_radius, recompute_callback,
                              panel_callback, **callbacks):
                region = regions_to_edit[0]
                initial = maximum_circle_radius(region)
                self.assertAlmostEqual(initial, 65.)
                recompute_callback(region)
                self.assertAlmostEqual(maximum_circle_radius(region), initial/2)
                panels = panel_callback(region, None, 'initial')
                self.assertTrue(any('maximum:' in line for line in panels[1].captions))
                return False

            with patch('calibrate.norm_pipeline.finalize_image', return_value={'frame': deeper}), \
                 patch('calibrate.manual_norm_regions.edit_regions', side_effect=inspect_limit):
                self.assertFalse(annotate_images(settings, edit=True))

    def test_color_preview_matches_original_marker_positions_and_spacing(self):
        from calibrate.manual_norm_regions import calibration_panels
        with TemporaryDirectory() as directory:
            settings, reference, _ = synthetic_batch(directory)
            record = compute_frame_colors(reference, reference, settings)
            middle = calibration_panels(reference, None, record)[1].image
            # Image-space positions must stay fixed even though the surface grid is 100x48.
            for pixel in reference.tracking['pixels'][6:-6]:
                x, y = np.rint(pixel).astype(int)
                np.testing.assert_array_equal(middle[y, x], 0)
            self.assertFalse(middle[100:, :240].any())
            midpoint = reference.tracking['pixels'][26:28].mean(axis=0)
            x, y = np.rint(midpoint).astype(int)
            self.assertTrue(middle[y, x].any())

    def test_color_panel_overlay_retains_source_pixels_without_changing_cached_colors(self):
        from calibrate.manual_norm_regions import calibration_panels
        with TemporaryDirectory() as directory:
            _, reference, _ = synthetic_batch(directory)
            shape = (720, 1280)
            frame = replace(reference, original_bgr=np.zeros((*shape, 3), np.uint8),
                surface_mask=np.ones(shape, bool), marker_image_mask=np.zeros(shape, bool))
            circle = RegionAnnotation('circle.png', (1280, 720), center_px=(800., 400.),
                shape='circle', semi_axes_px=(60., 60.))
            cached_color = np.full((*shape, 3), 17, np.uint8)
            panels = calibration_panels(frame, circle, color_image=cached_color)
            self.assertEqual(panels[1].image.shape, cached_color.shape)
            self.assertTrue(panels[1].captions)
            for panel in panels[:2]:
                np.testing.assert_array_equal(panel.image[400, 860], [0, 255, 0])
                np.testing.assert_array_equal(panel.image[400, 800], [0, 0, 255])
            np.testing.assert_array_equal(panels[1].image[:100], 17)
            self.assertTrue(np.all(cached_color == 17))
            circle.center_px = (800., 480.)
            middle = calibration_panels(frame, circle, color_image=cached_color)[1].image
            np.testing.assert_array_equal(middle[400, 860], [17, 17, 17])
            np.testing.assert_array_equal(middle[480, 860], [0, 255, 0])
            self.assertTrue(np.all(cached_color == 17))

    def _run_annotation_editor(self, region, events, recompute=None, save_callback=None,
                               close_window=False, editable_panel=0, panel_images=None,
                               panel_widths=(640, 640, 640), panel_height=480,
                               maximum_circle_radius=None):
        from utils.annotation_editor import edit_regions
        events = iter(events)
        previews, saved = [], []
        panel = np.zeros((480, 640, 3), np.uint8)
        panel_images = tuple(AnnotationPanel(image) for image in
                             ((panel, panel, panel) if panel_images is None else panel_images))

        def render(current, cached, status):
            previews.append((asdict(current), cached))
            return panel_images

        def save():
            if save_callback is not None:
                save_callback()
            saved.append(asdict(region))

        with patch('utils.annotation_editor.AnnotationWindow') as factory:
            window = factory.return_value
            window.is_closed = False

            def dispatch(delay):
                event = next(events)
                if isinstance(event, int):
                    return event
                mouse = window.set_mouse_callback.call_args.args[0]
                event(mouse, previews)
                window.is_closed = close_window
                return -1

            window.wait_key.side_effect = dispatch
            result = edit_regions([region], panel_callback=render,
                recompute_callback=recompute or (lambda current: object()),
                save_callback=save, editable_panel=editable_panel,
                panel_widths=panel_widths, panel_height=panel_height,
                maximum_circle_radius=maximum_circle_radius)
        return result, saved

    def test_enlarged_calibration_panel_preserves_circle_edit_coordinates(self):
        from calibrate.manual_norm_regions import ANNOTATION_PANEL_WIDTHS, ANNOTATION_PANEL_HEIGHT
        region = RegionAnnotation('circle.png', (640, 480), center_px=(200., 200.),
            shape='circle', semi_axes_px=(40., 40.))
        initial = asdict(region)

        def wheel(mouse, previews):
            mouse(cv2.EVENT_MOUSEWHEEL, 940, 300, 1, None)
            np.testing.assert_allclose(region.semi_axes_px, (42., 42.))
            self.assertEqual(region.center_px, (200., 200.))

        def move_and_resize(mouse, previews):
            self.assertEqual(asdict(region), initial)
            mouse(cv2.EVENT_LBUTTONDOWN, 940, 300, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 970, 315, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 970, 315, 0, None)
            self.assertEqual(region.center_px, (220., 210.))
            mouse(cv2.EVENT_LBUTTONDOWN, 1030, 315, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 1060, 315, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 1060, 315, 0, None)
            np.testing.assert_allclose(region.semi_axes_px, (60., 60.))
            before = asdict(region)
            for x, y in ((200, 300), (1700, 300), (1000, 730)):
                mouse(cv2.EVENT_MOUSEWHEEL, x, y, 1, None)
                mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                mouse(cv2.EVENT_MOUSEMOVE, x+5, y+5, 0, None)
                mouse(cv2.EVENT_LBUTTONUP, x+5, y+5, 0, None)
            self.assertEqual(asdict(region), before)

        result, saved = self._run_annotation_editor(region,
            [wheel, ord('z'), move_and_resize, 13], editable_panel=1,
            panel_widths=ANNOTATION_PANEL_WIDTHS, panel_height=ANNOTATION_PANEL_HEIGHT)
        self.assertTrue(result)
        self.assertEqual(saved[-1]['center_px'], (220., 210.))
        np.testing.assert_allclose(saved[-1]['semi_axes_px'], (60., 60.))

    def test_middle_panel_edits_account_for_offset_scale_and_padding(self):
        region = RegionAnnotation('circle.png', (1280, 720), center_px=(400., 300.),
            shape='circle', semi_axes_px=(80., 80.))
        initial = asdict(region)
        panels = (np.zeros((480, 640, 3), np.uint8),
                  np.zeros((720, 1280, 3), np.uint8),
                  np.zeros((240, 320, 3), np.uint8))

        def click_outside(mouse, previews):
            for x, y in ((200, 150), (1480, 150), (840, 400), (1280, 150)):
                mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                mouse(cv2.EVENT_MOUSEMOVE, x+5, y+5, 0, None)
                mouse(cv2.EVENT_LBUTTONUP, x+5, y+5, 0, None)
                mouse(cv2.EVENT_MOUSEWHEEL, x, y, 1, None)
            self.assertEqual(asdict(region), initial)

        def resize(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 880, 150, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 880, 180, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 0, 999, 0, None)
            self.assertEqual(region.center_px, (400., 300.))
            np.testing.assert_allclose(region.semi_axes_px, [100., 100.])

        def move_and_scale(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 840, 150, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 850, 160, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 850, 160, 0, None)
            self.assertEqual(region.center_px, (420., 320.))
            mouse(cv2.EVENT_MOUSEWHEEL, 850, 160, 1, None)
            np.testing.assert_allclose(region.semi_axes_px, [105., 105.])

        def set_center(mouse, previews):
            self.assertEqual(asdict(region), initial)
            # Neither the other panels nor letterbox padding may consume center selection.
            mouse(cv2.EVENT_LBUTTONDOWN, 200, 150, 0, None)
            mouse(cv2.EVENT_LBUTTONDOWN, 840, 400, 0, None)
            mouse(cv2.EVENT_LBUTTONDOWN, 1480, 150, 0, None)
            self.assertEqual(asdict(region), initial)
            mouse(cv2.EVENT_LBUTTONDOWN, 940, 250, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 940, 250, 0, None)
            self.assertEqual(region.center_px, (600., 500.))
            np.testing.assert_array_equal(region.semi_axes_px, [80., 80.])

        def restored(mouse, previews):
            self.assertEqual(asdict(region), initial)

        result, saved = self._run_annotation_editor(region,
            [click_outside, resize, move_and_scale, ord('z'), ord('z'), ord('z'), restored,
             ord('c'), set_center, ord('z'), restored, 13],
            editable_panel=1, panel_images=panels)
        self.assertTrue(result)
        self.assertEqual(saved[-1], dict(initial, status='accepted'))

    def test_circle_editor_preserves_radius_constraint_and_undo(self):
        region = RegionAnnotation('circle.png', (640, 480), center_px=(200., 200.),
            shape='circle', semi_axes_px=(40., 40.))
        initial = asdict(region)

        def resize(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 240, 200, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 230, 240, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 230, 240, 0, None)
            self.assertEqual(region.center_px, (200., 200.))
            np.testing.assert_array_equal(region.semi_axes_px, [50., 50.])

        def wheel(mouse, previews):
            mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, 1, None)
            np.testing.assert_allclose(region.semi_axes_px, [52.5, 52.5])

        def unchanged_circle(mouse, previews):
            self.assertEqual(region.shape, 'circle')
            self.assertEqual(region.angle_degrees, 0.)
            self.assertIsNone(region.contour_px)
            np.testing.assert_allclose(region.semi_axes_px, [52.5, 52.5])

        def select_center(mouse, previews):
            self.assertEqual(asdict(region), initial)
            mouse(cv2.EVENT_LBUTTONDOWN, 250, 260, 0, None)
            self.assertEqual(region.center_px, (250., 260.))

        def move(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 250, 260, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 270, 270, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 270, 270, 0, None)
            self.assertEqual(region.center_px, (270., 270.))
            np.testing.assert_array_equal(region.semi_axes_px, [40., 40.])

        def restored(mouse, previews):
            self.assertEqual(asdict(region), initial)

        result, saved = self._run_annotation_editor(region,
            [resize, wheel, ord('p'), ord('a'), ord('d'), unchanged_circle,
             ord('z'), ord('z'), ord('c'), select_center, move,
             ord('z'), ord('z'), restored, 13])
        self.assertTrue(result)
        self.assertEqual(saved[-1], dict(initial, status='accepted'))

    def test_circle_editor_limits_initial_radius_drag_wheel_move_and_undo(self):
        region = RegionAnnotation('circle.png', (640, 480), center_px=(200., 200.),
                                  shape='circle', semi_axes_px=(80., 80.))

        def limited_edits(mouse, previews):
            np.testing.assert_array_equal(previews[-1][0]['semi_axes_px'], (40., 40.))
            mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, 1, None)
            mouse(cv2.EVENT_LBUTTONDOWN, 240, 200, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 320, 200, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 320, 200, 0, None)
            np.testing.assert_array_equal(region.semi_axes_px, (40., 40.))

        def smaller_radius(mouse, previews):
            mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, -1, None)
            np.testing.assert_allclose(region.semi_axes_px, (40./1.05,)*2)

        def move_to_smaller_limit(mouse, previews):
            np.testing.assert_array_equal(region.semi_axes_px, (40., 40.))
            mouse(cv2.EVENT_LBUTTONDOWN, 200, 200, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 300, 200, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 300, 200, 0, None)
            np.testing.assert_array_equal(region.semi_axes_px, (25., 25.))

        def restored(mouse, previews):
            self.assertEqual(region.center_px, (200., 200.))
            np.testing.assert_array_equal(region.semi_axes_px, (40., 40.))

        result, saved = self._run_annotation_editor(region,
            [limited_edits, smaller_radius, ord('z'), move_to_smaller_limit, ord('z'), restored, 13],
            maximum_circle_radius=lambda current: 40. if current.center_px[0] < 250 else 25.)
        self.assertTrue(result)
        np.testing.assert_array_equal(saved[-1]['semi_axes_px'], (40., 40.))

    def test_recomputed_limit_requires_a_new_fit_before_accepting_the_clipped_circle(self):
        region = RegionAnnotation('circle.png', (640, 480), center_px=(200., 200.),
                                  shape='circle', semi_axes_px=(40., 40.))
        maximum = 50.
        fitted_radii = []

        def recompute(current):
            nonlocal maximum
            fitted_radii.append(current.semi_axes_px[0])
            maximum = 30.
            return object()

        def clipped_without_stale_preview(mouse, previews):
            np.testing.assert_array_equal(region.semi_axes_px, (30., 30.))
            self.assertIsNone(previews[-1][1])
            self.assertEqual(region.status, 'draft')

        result, saved = self._run_annotation_editor(region,
            [13, clipped_without_stale_preview, 13], recompute,
            maximum_circle_radius=lambda current: maximum)
        self.assertTrue(result)
        self.assertEqual(fitted_radii, [40., 30.])
        self.assertEqual(len(saved), 1)
        np.testing.assert_array_equal(saved[0]['semi_axes_px'], (30., 30.))

    def test_rejected_recompute_limits_the_radius_before_saving_a_draft(self):
        region = RegionAnnotation('circle.png', (640, 480), center_px=(200., 200.),
                                  shape='circle', semi_axes_px=(40., 40.))
        maximum = 50.

        def recompute(current):
            nonlocal maximum
            maximum = 30.
            raise FrameCalibrationError('sphere_fit_failed', 'invalid fit', preview={'diagnostic': True})

        result, saved = self._run_annotation_editor(region, [ord('g'), 27], recompute,
                                                   maximum_circle_radius=lambda current: maximum)
        self.assertFalse(result)
        self.assertTrue(all(item['status'] == 'draft' and item['semi_axes_px'] == (30., 30.)
                            for item in saved))

    def test_annotation_rejected_preview_remains_visible_and_cannot_be_accepted(self):
        region = RegionAnnotation('image.png', (640, 480), center_px=(260., 240.))
        preview = {'colors': 'diagnostics'}
        error = FrameCalibrationError('background_mismatch', 'median exceeds threshold', preview=preview)
        recompute = Mock(side_effect=error)

        def verify_rejected(mouse, previews):
            self.assertIs(previews[-1][1], preview)
            self.assertEqual(region.status, 'draft')

        def edit(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 260, 240, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 270, 240, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 270, 240, 0, None)

        def verify_invalidated(mouse, previews):
            self.assertIsNone(previews[-1][1])

        result, saved = self._run_annotation_editor(region,
            [ord('g'), verify_rejected, 13, verify_rejected, edit, verify_invalidated, 27], recompute)
        self.assertFalse(result)
        self.assertEqual(recompute.call_count, 2)
        self.assertTrue(all(item['status'] == 'draft' for item in saved))

    def test_background_rejection_keeps_colors_and_sphere_preview_without_export(self):
        from calibrate.manual_norm_regions import annotate_images, calibration_panels
        from calibrate.norm_pipeline import finalize_image, export_normal_samples, _write_final_record
        with TemporaryDirectory() as directory:
            settings, reference, regions = synthetic_batch(directory)
            prepared = load_frame_calibration_data(settings.work_dir/'prepared/000000.npz')
            annotation = regions[prepared.image_key]

            def inspect_initial_editor(regions_to_edit, *, panel_callback, **callbacks):
                panels = panel_callback(regions_to_edit[0], None, 'initial preview')
                # Away from text: actual material-grid colors must already be visible.
                self.assertTrue(panels[1].image[120:400, 240:400].any())
                return False

            with patch('calibrate.manual_norm_regions.edit_regions', side_effect=inspect_initial_editor) as editor:
                self.assertFalse(annotate_images(settings, edit=True))
                editor.assert_called_once()

            initial = dict(frame=prepared, **compute_frame_colors(prepared, reference, settings))
            initial_panels = calibration_panels(prepared, annotation, initial)
            prepared.original_bgr = np.clip(prepared.original_bgr.astype(int)+12, 0, 255).astype(np.uint8)
            with self.assertRaises(FrameCalibrationError) as strict:
                finalize_image(prepared, reference, annotation, settings)
            self.assertEqual(strict.exception.code, 'background_mismatch')
            self.assertIsNone(strict.exception.preview)
            with self.assertRaises(FrameCalibrationError) as diagnostic:
                finalize_image(prepared, reference, annotation, settings, diagnostic_preview=True)
            preview = diagnostic.exception.preview
            self.assertEqual(diagnostic.exception.code, 'background_mismatch')
            self.assertIn('background_mismatch', preview['preview_errors'][0])
            self.assertGreater(preview['metrics']['sample_count'], 0)
            self.assertIn('normal_gt_grid', preview)
            panels = calibration_panels(prepared, annotation, preview)
            self.assertTrue(panels[1].image[150:350, 240:400].any())
            self.assertTrue(panels[2].image.any())
            self.assertTrue(panels[2].captions[0].startswith('REJECTED'))
            self.assertFalse(np.array_equal(panels[2].image, initial_panels[2].image))
            with self.assertRaises(FrameCalibrationError):
                _write_final_record(preview, reference, settings)
            with self.assertRaises(FrameCalibrationError):
                export_normal_samples([preview], settings, settings.input_file)
            self.assertFalse(settings.input_file.exists())
            self.assertFalse((settings.work_dir/'frames').exists())

    def test_sphere_rejection_keeps_final_colors_and_depth_preview(self):
        from calibrate.manual_norm_regions import calibration_panels
        from calibrate.norm_pipeline import finalize_image
        with TemporaryDirectory() as directory:
            settings, reference, regions = synthetic_batch(directory)
            prepared = load_frame_calibration_data(settings.work_dir/'prepared/000000.npz')
            annotation = regions[prepared.image_key]
            # The actual contact circle has radius 3 mm, larger than this sphere.
            with self.assertRaises(FrameCalibrationError) as diagnostic:
                finalize_image(prepared, reference, annotation, replace(settings, sphere_radius_mm=2.),
                               diagnostic_preview=True)
            preview = diagnostic.exception.preview
            self.assertEqual(diagnostic.exception.code, 'sphere_fit_failed')
            self.assertEqual(preview['frame'].geometry_stage, 'final')
            self.assertNotIn('normal_gt_grid', preview)
            panels = calibration_panels(prepared, annotation, preview)
            self.assertTrue(panels[1].image[120:400, 240:400].any())
            self.assertTrue(panels[2].image.any())

    def test_annotation_window_close_saves_current_draft(self):
        with TemporaryDirectory() as directory:
            region = RegionAnnotation('image.png', (640, 480), center_px=(260., 240.))
            path = Path(directory)/'regions.yaml'
            recompute = Mock()

            def drag(mouse, previews):
                mouse(cv2.EVENT_LBUTTONDOWN, 260, 240, 0, None)
                mouse(cv2.EVENT_MOUSEMOVE, 271, 245, 0, None)
                mouse(cv2.EVENT_LBUTTONUP, 271, 245, 0, None)

            result, saved = self._run_annotation_editor(region, [drag], recompute,
                save_callback=lambda: save_regions(path, [region], image_root=directory),
                close_window=True)
            self.assertFalse(result)
            self.assertEqual(len(saved), 1)
            loaded = load_regions(path, image_root=directory)['image.png']
            self.assertEqual(loaded.status, 'draft')
            self.assertEqual(loaded.center_px, [271., 245.])
            recompute.assert_not_called()

    def test_annotation_window_scroll_preserves_view_and_canvas_coordinates(self):
        from matplotlib.backend_bases import MouseEvent, KeyEvent, CloseEvent
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from utils.annotation_window import (AnnotationWindow, compose_annotation_panels,
                                             render_annotation_preview, FONT_SIZE_PX)

        canvas = FigureCanvasAgg(Figure(figsize=(12, 4)))
        manager = Mock(canvas=canvas)
        screen = manager.window.screen().availableGeometry()
        screen.width.return_value, screen.height.return_value = 1920, 1080
        with patch('utils.annotation_window._create_figure_manager', return_value=manager):
            window = AnnotationWindow('test-window')
        mouse = Mock()
        window.set_mouse_callback(mouse)
        source = np.full((480, 640, 3), (20, 40, 60), np.uint8)
        display = compose_annotation_panels(
            (AnnotationPanel(source), AnnotationPanel(source, ('Edit boundary here | wheel: radius',)),
             AnnotationPanel(source, ('Reference overall depth | camera Z (mm)',), header=True)),
            (640, 960, 280), 720, 'z: undo | g: recompute | Enter: accept')
        image = display.image
        window.show_image(image, display.labels)
        canvas.draw()
        axes = canvas.figure.axes[0]
        limits = axes.get_xlim(), axes.get_ylim()
        for size in ((12, 4), (6, 7)):
            canvas.figure.set_size_inches(*size)
            canvas.callbacks.process('resize_event', Mock())
            canvas.draw()
            # The window rasterizes separate font glyphs at its current pixel resolution.
            scale = axes.transData.get_matrix()[0, 0]
            renderer = canvas.get_renderer()
            for text in axes.texts:
                self.assertAlmostEqual(text.get_fontsize()*canvas.figure.dpi/72, FONT_SIZE_PX*scale)
                box = text.get_window_extent(renderer)
                self.assertLessEqual(box.x1, axes.bbox.x1+1)
            for step in (1, -1):
                x, y = axes.transData.transform((840, 150))
                MouseEvent('scroll_event', canvas, x, y, step=step)._process()
                mouse.assert_called_with(cv2.EVENT_MOUSEWHEEL, 840, 150, step, None)
                self.assertEqual((axes.get_xlim(), axes.get_ylim()), limits)
                np.testing.assert_array_equal(axes.images[0].get_array(), image[..., ::-1])
        preview = render_annotation_preview(display)
        self.assertEqual(preview.shape, image.shape)
        self.assertFalse(image[720:].any())
        self.assertTrue(preview[720:].any())
        np.testing.assert_array_equal(source, np.full_like(source, (20, 40, 60)))
        self.assertFalse(axes.get_navigate())
        MouseEvent('button_release_event', canvas, -10, -10, button=1)._process()
        mouse.assert_called_with(cv2.EVENT_LBUTTONUP, -1, -1, 0, None)
        KeyEvent('key_press_event', canvas, key='z')._process()
        self.assertEqual(window.wait_key(30), ord('z'))
        CloseEvent('close_event', canvas)._process()
        self.assertTrue(window.is_closed)
        window.close()

    def test_annotation_yaml_roundtrip_numpy_coordinates(self):
        ellipse = RegionAnnotation('ellipse.png', (np.int64(640), np.int32(480)),
            status='accepted', center_px=tuple(np.array([271., 240.])),
            semi_axes_px=(np.float64(40.), np.float32(30.5)), angle_degrees=np.float64(3.))
        polygon = RegionAnnotation('polygon.png', (640, 480), status='skip',
            center_px=(np.float64(25.), np.float64(25.)), shape='polygon',
            contour_px=np.array([[10., 10.], [40., 10.], [40., 40.], [10., 40.]]))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'regions.yaml'
            save_regions(path, [ellipse, polygon], image_root=directory)
            text = path.read_text(encoding='utf-8')
            content = yaml.safe_load(text)
            self.assertNotIn('!!python', text)
            raw = content['regions'][0]
            self.assertIs(type(raw['image_size'][0]), int)
            self.assertIs(type(raw['center_px'][0]), float)
            self.assertIs(type(raw['semi_axes_px'][1]), float)
            loaded = load_regions(path, image_root=directory)
            self.assertEqual(loaded['ellipse.png'].status, 'accepted')
            np.testing.assert_array_equal(loaded['ellipse.png'].center_px, ellipse.center_px)
            np.testing.assert_array_equal(loaded['ellipse.png'].semi_axes_px, ellipse.semi_axes_px)
            self.assertEqual(loaded['polygon.png'].status, 'skip')
            np.testing.assert_array_equal(loaded['polygon.png'].contour_px, polygon.contour_px)

    def test_annotation_background_failure_saves_draft_and_can_undo(self):
        region = RegionAnnotation('image.png', (640, 480), center_px=(260., 240.),
                                  semantics='contact_boundary')
        recompute = Mock(side_effect=FrameCalibrationError('background_mismatch',
            'fraction=0.83787, median=0.04442954063415527'))

        def drag(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 260, 240, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 271, 240, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 271, 240, 0, None)
            mouse(cv2.EVENT_MOUSEWHEEL, 271, 240, 1, None)

        with TemporaryDirectory() as directory:
            path = Path(directory)/'regions.yaml'

            def verify_saved_draft(mouse, previews):
                loaded = load_regions(path, image_root=directory)['image.png']
                self.assertEqual(loaded.status, 'draft')
                self.assertEqual(loaded.center_px, [271., 240.])
                np.testing.assert_array_equal(loaded.semi_axes_px, region.semi_axes_px)
                self.assertIsNone(previews[-1][1])

            result, saved = self._run_annotation_editor(region,
                [drag, ord('g'), verify_saved_draft, ord('z'), ord('g'),
                 verify_saved_draft, ord('z'), 27], recompute,
                save_callback=lambda: save_regions(path, [region], image_root=directory))
            self.assertFalse(result)
            self.assertEqual(recompute.call_count, 2)
            self.assertEqual(len(saved), 3)
            loaded = load_regions(path, image_root=directory)['image.png']
            self.assertEqual(loaded.status, 'draft')
            self.assertEqual(loaded.center_px, [260., 240.])
            self.assertEqual(loaded.semi_axes_px, [40., 30.])

    def test_annotation_undo_groups_drag_ignores_clicks_and_clears_preview(self):
        region = RegionAnnotation('image.png', (640, 480), center_px=(100., 100.))
        initial = asdict(region)
        computed = object()
        recompute = Mock(return_value=computed)

        def drag(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 110, 105, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 120, 115, 0, None)
            # A release outside the panel must still end this gesture.
            mouse(cv2.EVENT_LBUTTONUP, 800, 115, 0, None)
            self.assertEqual(region.center_px, (120., 115.))

        def no_op_gestures(mouse, previews):
            self.assertIs(previews[-1][1], computed)
            mouse(cv2.EVENT_LBUTTONDOWN, 120, 115, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 120, 115, 0, None)
            # A drag that returns to its starting point must not consume undo.
            mouse(cv2.EVENT_LBUTTONDOWN, 120, 115, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 125, 115, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 120, 115, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 120, 115, 0, None)

        def assert_restored(mouse, previews):
            self.assertEqual(asdict(region), initial)
            self.assertIsNone(previews[-1][1])
            mouse(cv2.EVENT_MOUSEMOVE, 200, 200, 0, None)
            self.assertEqual(asdict(region), initial)

        result, saved = self._run_annotation_editor(region,
            [drag, ord('g'), no_op_gestures, ord('z'), assert_restored,
             ord('z'), assert_restored, 27], recompute)
        self.assertFalse(result)
        self.assertEqual(saved[-1], initial)
        recompute.assert_called_once_with(region)

    def test_annotation_undo_restores_shape_vertices_rotation_scale_and_center(self):
        region = RegionAnnotation('image.png', (640, 480), center_px=(200., 200.))
        initial = asdict(region)
        before_edits = []

        def checkpoint(mouse, previews):
            before_edits.append(asdict(region))

        def wheel(mouse, previews):
            mouse(cv2.EVENT_MOUSEWHEEL, 200, 200, 1, None)
            self.assertGreater(region.semi_axes_px[0], initial['semi_axes_px'][0])

        def resize(mouse, previews):
            x, y = np.rint(region_boundary_points(region)[16]).astype(int)
            mouse(cv2.EVENT_LBUTTONDOWN, int(x), int(y), 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, int(x+6), int(y+5), 0, None)
            mouse(cv2.EVENT_LBUTTONUP, int(x+6), int(y+5), 0, None)
            self.assertNotEqual(region.semi_axes_px, before_edits[-1]['semi_axes_px'])

        def drag_vertex(mouse, previews):
            x, y = np.rint(region.contour_px[0]).astype(int)
            mouse(cv2.EVENT_LBUTTONDOWN, int(x), int(y), 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, int(x+5), int(y+8), 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, int(x+8), int(y+12), 0, None)
            mouse(cv2.EVENT_LBUTTONUP, int(x+8), int(y+12), 0, None)
            self.assertNotEqual(region.contour_px, before_edits[-1]['contour_px'])

        def move_polygon(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 200, 200, 0, None)
            mouse(cv2.EVENT_MOUSEMOVE, 210, 207, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 210, 207, 0, None)
            self.assertEqual(region.center_px, (210., 207.))

        def select_center(mouse, previews):
            mouse(cv2.EVENT_LBUTTONDOWN, 215, 210, 0, None)
            mouse(cv2.EVENT_LBUTTONUP, 215, 210, 0, None)
            self.assertEqual(region.center_px, (215., 210.))

        def assert_undone(mouse, previews):
            self.assertEqual(asdict(region), before_edits.pop())
            self.assertIsNone(previews[-1][1])

        events = [checkpoint, wheel, checkpoint, resize, checkpoint, ord('d'),
                  checkpoint, ord('p'), checkpoint, drag_vertex,
                  checkpoint, move_polygon, checkpoint, ord('c'), select_center,
                  checkpoint, ord('a'), ord('g'), ord('c'), ord('z')]
        for _ in range(8):
            events.extend([ord('z'), assert_undone])
        events.extend([ord('z'), 13])
        result, saved = self._run_annotation_editor(region, events)
        self.assertTrue(result)
        self.assertFalse(before_edits)
        self.assertEqual(saved[-1], dict(initial, status='accepted'))

    def test_prepare_detection_tracking_without_fitting_and_replay(self):
        from reconstruction.get_surface import build_reconstruction_provider, save_material_coordinates
        with TemporaryDirectory() as directory:
            settings, reference, regions = synthetic_batch(directory)
            root = Path(directory)
            for image_path in [reference.source_image, *[root/key for key in regions]]:
                image = cv2.imread(str(image_path))
                cv2.rectangle(image, (255, 87), (385, 393), (0, 0, 0), 5)
                cv2.imwrite(str(image_path), image)
            np.savez(root/'camera.npz', camera_matrix=reference.camera.matrix, dist_coeffs=reference.camera.distortion)
            np.savetxt(root/'template.csv', np.column_stack((np.arange(48), reference.template_xy_mm)),
                       delimiter=',', header='id,x_mm,y_mm', comments='')
            (root/'provider.json').write_text(json.dumps(dict(image_size=[640, 480],
                camera_file='camera.npz', template_file='template.csv')))
            config = dict(get_surface=dict(overall_reconstruction=dict(provider='marker_bicubic', config_file='provider.json')),
                material_coordinates={key: dict(rows=50, columns=24) for key in ('geometry', 'lightfield', 'observation')},
                normal_calibration=dict(input_mode='images', reference_image='reference.png', images=['folder*/*.png'], work_dir='new_batch'))
            config_path = root/'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            parsed = parse_settings(config_path)
            self.assertIsNone(parsed.sphere_radius_mm)
            with self.assertRaisesRegex(ValueError, 'zero_boundary_displacement'):
                replace(parsed, sphere_radius_mm=10., sphere_side='positive_normal').require_sphere()
            manifest = run_prepare(config_path, parsed)
            self.assertEqual([row['status'] for row in manifest['images']], ['prepared']*3)
            prepared = load_frame_calibration_data(parsed.work_dir/'prepared/000000.npz')
            self.assertEqual(prepared.geometry_stage, 'preview_reference')
            provider = build_reconstruction_provider(config_path)
            provider.bind_coordinate_chart(prepared.chart)
            image = cv2.imread(str(reference.source_image))
            with patch.object(provider.session.model, 'fit', side_effect=AssertionError('preview must not fit')):
                provider.track_frame(image)
                provider.session.shape_failed_frames = 2
                provider.track_frame(prepared.original_bgr)
                self.assertEqual(provider.session.shape_failed_frames, 2)
            positions = provider.session.tracker.positions.copy()
            previous_gray = provider.session.tracker.previous_gray.copy()
            specs = MaterialCoordinateSpecs(*[CoordinateGridSpec(50, 24)]*3)
            systems = provider.fit_tracked_frame(specs, prior_coefficients=prepared.coefficients)
            provider.fit_tracked_frame(specs, prior_coefficients=prepared.coefficients)
            np.testing.assert_array_equal(provider.session.tracker.positions, positions)
            np.testing.assert_array_equal(provider.session.tracker.previous_gray, previous_gray)
            path = save_material_coordinates(root/'ordinary.npz', systems,
                provider.last_frame_result['inner_region'].mask, prepared.original_bgr, provider)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(str(data['coordinate_chart_id']), prepared.chart.chart_id)
                self.assertEqual(data['surface_coefficients'].shape, (16, 3))
                self.assertTrue((root/str(data['coordinate_chart_file'])).exists())

    def test_samples_fit_cache_provider_and_metadata(self):
        from reconstruction.pipeline import load_sampling_grid
        with TemporaryDirectory() as directory:
            settings, reference, regions = synthetic_batch(directory)
            loaded = load_frame_calibration_data(settings.work_dir/'prepared/000000.npz')
            np.testing.assert_array_equal(loaded.coefficients, reference.coefficients)
            # A non-bicubic analytic surface uses the same chart/grid primitives.
            analytic_surface = AnalyticPatch('cylinder')
            grid = sample_surface_grid(analytic_surface, build_arclength_chart(analytic_surface), 10, 12)
            self.assertEqual(grid.xyz.shape, (10, 12, 3))
            provider = _provider_from_frame(loaded)
            provider.reset()
            self.assertEqual(provider.coordinate_chart.chart_id, reference.chart.chart_id)
            systems = provider.sample_coefficients(reference.coefficients, MaterialCoordinateSpecs(*[CoordinateGridSpec(100, 48)]*3), loaded.surface_mask)
            np.testing.assert_array_equal(systems.observation.coordinates, reference.systems.observation.coordinates)
            records = run_samples(settings)
            self.assertEqual(len(records), 3)
            for record in records:
                self.assertEqual(record['frame'].geometry_stage, 'final')
                self.assertGreater(record['metrics']['sample_count'], 0)
                self.assertLess(record['frame'].marker_fit_valid.sum(), 48)
                self.assertEqual(record['frame'].tracking['valid'].sum(), 48)
                frame = record['frame']
                contact = region_masks(regions[frame.image_key], frame.original_bgr.shape,
                                       outer_margin_px=20)['contact']
                np.testing.assert_array_equal(record['contact_mask'],
                    sample_mask_to_surface(contact, frame.systems.observation.uv))
            grid_path = settings.work_dir/'lut_material_grid.npz'
            grid_path.write_bytes(b'obsolete grid must not be read')
            with patch('cv2.imread', side_effect=AssertionError('fit must use cached reference geometry')):
                report = fit_and_validate_samples(settings.input_file, settings)
            self.assertEqual(report['status'], 'passed')
            lut = NormalCalibration.load(settings.output)
            self.assertEqual(lut.metadata['validation_status'], 'passed')
            self.assertEqual(lut.metadata['coordinate_chart_id'], reference.chart.chart_id)
            shape = reference.systems.observation.xyz.shape[:2]
            frozen = load_sampling_grid(grid_path, lut, reference.surface, reference.camera, shape)
            direct = load_sampling_grid(reference.chart_file, lut, reference.surface, reference.camera, shape)
            np.testing.assert_array_equal(frozen.parameters, direct.parameters)
            np.testing.assert_array_equal(frozen.st, direct.st)
            self.assertEqual(frozen.chart_id, lut.metadata['coordinate_chart_id'])
            frame_path = settings.work_dir/'frames/000000.npz'
            xyz, colors, geometry, photo, trusted = _load_reconstruction_input(frame_path, lut.metadata)
            self.assertTrue(np.any(geometry & ~photo))
            with self.assertRaisesRegex(ValueError, 'declaration mismatch'):
                _load_reconstruction_input(frame_path, dict(lut.metadata, coordinate_chart_id='different'))
            invalid_old = replace(lut, metadata={})
            with self.assertRaisesRegex(ValueError, 'declaration mismatch'):
                _load_reconstruction_input(frame_path, invalid_old.metadata)
            with np.load(settings.input_file, allow_pickle=False) as data:
                keys = data['image_paths']
                self.assertEqual(len(set(keys)), 3)
                self.assertTrue(all('/same_name.png' in key for key in keys))
            regions_back = load_regions(settings.annotations_file, image_root=settings.work_dir)
            self.assertEqual(set(regions_back), set(regions))
            self.assertTrue((settings.work_dir/'verification/report.csv').is_file())
            self.assertTrue((settings.work_dir/'verification/000000.png').is_file())
            visualization = settings.work_dir/'visualization'
            summary = json.loads((visualization/'summary.json').read_text())
            self.assertEqual(summary['coordinate_chart_id'], reference.chart.chart_id)
            self.assertEqual(len(summary['images']), 3)
            self.assertEqual(summary['final_LUT']['count'], lut.metadata['total_sample_count'])
            self.assertEqual(len(list((visualization/'color_difference').glob('*.png'))), 3)
            self.assertEqual(len(list((visualization/'pq_mapping').glob('*.png'))), 3)
            self.assertIsNotNone(cv2.imread(str(visualization/'calibration_summary.png')))
            self.assertIsNotNone(cv2.imread(str(visualization/'lut_slices.png')))
            self.assertIsNotNone(cv2.imread(str(visualization/'color_pq_mapping.png')))
            self.assertIsNotNone(cv2.imread(str(visualization/'pq_ambiguities.png')))
            self.assertFalse((visualization/'pq_ambiguities.svg').exists())
            with np.load(visualization/'ambiguities.npz', allow_pickle=False) as data:
                expected = np.zeros(lut.metadata['total_sample_count'], bool)
                expected[data['pairs'].ravel()] = True
                np.testing.assert_array_equal(data['ambiguous_mask'], expected)
                self.assertEqual(summary['ambiguities']['ambiguous_sample_count'], int(expected.sum()))
            from calibrate.visualize_norm import generate_calibration_visualizations
            before = (visualization/'summary.json').read_bytes()
            with np.load(settings.input_file, allow_pickle=False) as data:
                wrong_samples = {key: data[key] for key in data.files}
            wrong_samples['coordinate_chart_id'] = np.asarray('old-chart')
            wrong_path = settings.work_dir/'wrong_samples.npz'
            np.savez_compressed(wrong_path, **wrong_samples)
            with self.assertRaisesRegex(ValueError, 'coordinate_chart_id mismatch'):
                generate_calibration_visualizations(settings.output, wrong_path, settings.work_dir)
            self.assertEqual((visualization/'summary.json').read_bytes(), before)

    def test_validation_isolation_failed_publish_and_external_mode(self):
        with TemporaryDirectory() as directory:
            settings, reference, regions = synthetic_batch(directory)
            run_samples(settings)
            fit_and_validate_samples(settings.input_file, settings)
            old_output = settings.output.read_bytes()
            grid_path = settings.work_dir/'lut_material_grid.npz'
            old_grid = grid_path.read_bytes()
            with patch('calibrate.norm_pipeline.build_normal_lut', side_effect=ValueError('No valid LUT nodes')):
                with self.assertRaisesRegex(ValueError, 'LUT'):
                    fit_and_validate_samples(settings.input_file, settings)
            self.assertEqual(settings.output.read_bytes(), old_output)
            self.assertEqual(grid_path.read_bytes(), old_grid)
            failed = json.loads((settings.work_dir/'verification/validation.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            with np.load(settings.input_file, allow_pickle=False) as data:
                values = {key: data[key] for key in data.files}
            # Seed 42 deterministically selects image 2 out of three sorted image keys.
            active = np.arange(3); np.random.default_rng(42).shuffle(active)
            held_out = values['image_index'] == active[0]
            train_min = np.minimum(values['color_residual_samples'][~held_out].min(0), 0)
            train_max = np.maximum(values['color_residual_samples'][~held_out].max(0), 0)
            values['color_residual_samples'][held_out] = [1., 1., 1.]
            np.savez_compressed(settings.input_file, **values)
            with self.assertRaisesRegex(ValueError, 'validation failed'):
                fit_and_validate_samples(settings.input_file, settings)
            self.assertEqual(settings.output.read_bytes(), old_output)
            self.assertEqual(grid_path.read_bytes(), old_grid)
            train = NormalCalibration.load(settings.output.with_name('normal_lut_train.npz'))
            np.testing.assert_array_equal(train.color_min, train_min)
            np.testing.assert_array_equal(train.color_max, train_max)
            report = json.loads((settings.work_dir/'verification/validation.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['out_of_range_fraction'], 1.)
            external = settings.work_dir/'external.npz'
            np.savez_compressed(external, color_residual_samples=values['color_residual_samples'][~held_out],
                                slope_samples=values['slope_samples'][~held_out])
            report = fit_and_validate_samples(external, settings)
            self.assertEqual(report['status'], 'unvalidated_external_samples')
            self.assertEqual(NormalCalibration.load(settings.output).metadata['validation_status'], 'unvalidated_external_samples')
            self.assertEqual(grid_path.read_bytes(), old_grid)

    def test_refresh_sampling_grid_uses_this_reference_and_keeps_the_LUT(self):
        from calibrate.norm_pipeline import update_material_sampling_grid
        from reconstruction.pipeline import load_sampling_grid
        with TemporaryDirectory() as directory:
            settings, reference, _ = synthetic_batch(directory)
            metadata = dict(coordinate_kind='reference_arclength', coordinate_chart_id=reference.chart.chart_id,
                frame_convention='st_grid_columns_rows', residual_domain='material_grid', color_space='linear_rgb',
                reference_image=str(reference.source_image), validation_status='passed')
            lut = NormalCalibration(np.zeros((3, 3, 3, 2)), np.zeros((3, 3, 3)),
                                    -np.ones(3), np.ones(3), 1., metadata=metadata)
            lut.save(settings.output)
            lut_before = settings.output.read_bytes()
            chart_before = reference.chart_file.read_bytes()
            path = settings.work_dir/'lut_material_grid.npz'
            path.write_bytes(b'old grid')
            self.assertEqual(update_material_sampling_grid(settings), path)
            shape = reference.systems.observation.xyz.shape[:2]
            grid = load_sampling_grid(path, lut, reference.surface, reference.camera, shape)
            expected = sample_surface_grid(reference.surface, reference.chart, *shape)
            np.testing.assert_array_equal(grid.parameters, expected.parameters)
            self.assertEqual(settings.output.read_bytes(), lut_before)
            self.assertEqual(reference.chart_file.read_bytes(), chart_before)
            wrong = replace(lut, metadata=dict(metadata, coordinate_chart_id='old-chart'))
            wrong.save(settings.output)
            grid_before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'coordinate_chart_id mismatch'):
                update_material_sampling_grid(settings)
            self.assertEqual(path.read_bytes(), grid_before)

    def test_staged_LUT_write_failure_preserves_published_grid_and_LUT(self):
        from calibrate.norm_pipeline import _publish_image_calibration
        with TemporaryDirectory() as directory:
            settings, reference, _ = synthetic_batch(directory)
            metadata = dict(coordinate_kind='reference_arclength', coordinate_chart_id=reference.chart.chart_id,
                frame_convention='st_grid_columns_rows', residual_domain='material_grid', color_space='linear_rgb',
                reference_image=str(reference.source_image), validation_status='passed')
            lut = NormalCalibration(np.zeros((3, 3, 3, 2)), np.zeros((3, 3, 3)),
                                    -np.ones(3), np.ones(3), 1., metadata=metadata)
            grid_path = settings.work_dir/'lut_material_grid.npz'
            grid_path.write_bytes(b'published grid')
            settings.output.write_bytes(b'published LUT')
            with patch.object(NormalCalibration, 'save', side_effect=OSError('disk write failed')):
                with self.assertRaisesRegex(OSError, 'disk write failed'):
                    _publish_image_calibration(settings, lut, 10)
            self.assertEqual(grid_path.read_bytes(), b'published grid')
            self.assertEqual(settings.output.read_bytes(), b'published LUT')
            self.assertFalse(list(settings.work_dir.glob('.normal-*')))


class NormalCalibrationVisualization(unittest.TestCase):
    def test_failed_diagnostic_lut_updates_plots_without_publishing(self):
        from calibrate.visualize_norm import generate_calibration_visualizations, _validation_label
        with TemporaryDirectory() as directory:
            settings, _, _ = synthetic_batch(directory)
            run_samples(settings)
            with patch('calibrate.visualize_norm.generate_calibration_visualizations',
                       return_value=settings.work_dir/'visualization'):
                fit_and_validate_samples(settings.input_file, settings)
            published = settings.output.read_bytes()
            calibration = NormalCalibration.load(settings.output)
            validation = dict(calibration.metadata['validation'], status='failed')
            diagnostic = replace(calibration, metadata=dict(calibration.metadata,
                                 validation_status='failed', validation=validation))
            path = settings.work_dir/'diagnostic.npz'
            diagnostic.save(path)
            train = NormalCalibration.load(settings.output.with_name(settings.output.stem+'_train.npz'))
            train.save(path.with_name('diagnostic_train.npz'))

            output = generate_calibration_visualizations(path, settings.input_file, settings.work_dir)

            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['validation_status'], 'failed')
            self.assertEqual(summary['validation']['status'], 'failed')
            self.assertEqual(summary['calibration_file'], str(path))
            self.assertEqual(summary['original_valid_node_count'], int(calibration.original_valid.sum()))
            self.assertEqual(summary['retained_sample_count'], int(calibration.sample_counts.sum()))
            self.assertIn('FAILED', _validation_label(diagnostic))
            self.assertIn('not published', _validation_label(diagnostic))
            self.assertEqual(len(list(output.rglob('*.png'))), 10)
            self.assertEqual(settings.output.read_bytes(), published)

    def _metadata(self):
        return dict(coordinate_kind='reference_arclength', coordinate_chart_id='this-chart',
                    frame_convention='st_grid_columns_rows', residual_domain='material_grid',
                    color_space='linear_rgb', reference_image='reference.png', validation_status='passed',
                    total_sample_count=2, validation=dict(training_images=['train.png'],
                    validation_images=['held.png'], thresholds=dict(rms_degrees=25.)))

    def _lut(self, metadata, p):
        slopes = np.zeros((3, 3, 3, 2))
        slopes[..., 0] = p
        return NormalCalibration(slopes, np.zeros((3, 3, 3)), -np.ones(3), np.ones(3), 1., metadata=metadata)

    def _samples(self):
        return dict(color_residual_samples=np.zeros((2, 3)), slope_samples=np.zeros((2, 2)),
                    reference_normal_samples=np.tile([0., 0., 1.], (2, 1)),
                    tangent_frame_samples=np.tile(np.eye(3)[:, :2], (2, 1, 1)),
                    image_index=np.array([0, 1]), image_paths=np.array(['train.png', 'held.png']))

    def test_raw_prediction_is_visible_outside_the_truth_comparison_mask(self):
        from calibrate.visualize_norm import _predict
        samples = self._samples()
        truth = np.tile([1., 0.], (2, 1))
        predicted, angles, valid = _predict(self._lut(self._metadata(), 1.), samples['color_residual_samples'],
            truth, samples['reference_normal_samples'], samples['tangent_frame_samples'], np.array([True, False]))
        # A zero color still receives the LUT's raw value; reconstruction protection is absent.
        np.testing.assert_array_equal(predicted, truth)
        np.testing.assert_array_equal(valid, [True, False])
        self.assertAlmostEqual(angles[0], 0., places=5)
        self.assertTrue(np.isnan(angles[1]))

    def test_held_out_plot_queries_training_LUT_instead_of_the_final_LUT(self):
        from calibrate.visualize_norm import _validation_predictions
        metadata = self._metadata()
        final = self._lut(metadata, 10.)
        train_metadata = dict(metadata, validation_status='training_only',
                              training_images=['train.png'], validation_images=['held.png'])
        with TemporaryDirectory() as directory:
            path = Path(directory)/'current.npz'
            final.save(path)
            self._lut(train_metadata, 0.).save(path.with_name('current_train.npz'))
            prediction, angles, mask = _validation_predictions(path, final, self._samples())
            np.testing.assert_array_equal(mask, [False, True])
            np.testing.assert_array_equal(prediction[mask], 0.)
            np.testing.assert_array_equal(angles[mask], 0.)

    def test_frame_selection_excludes_stale_batches(self):
        from calibrate.visualize_norm import DECLARATIONS, _frame_paths
        metadata = self._metadata()
        declarations = {key: metadata[key] for key in DECLARATIONS}
        samples = self._samples()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'frames').mkdir()
            for index, key in enumerate(samples['image_paths']):
                np.savez(root/'frames'/f'{index:06d}.npz', **declarations, image_key=key, geometry_stage='final')
            old = dict(declarations, coordinate_chart_id='old-chart')
            np.savez(root/'frames/000009.npz', **old, image_key='train.png', geometry_stage='final')
            paths = _frame_paths(root, samples, self._lut(metadata, 0.))
            self.assertEqual(set(paths), {'train.png', 'held.png'})
            self.assertEqual(paths['train.png'].name, '000000.npz')

    def test_visualization_directory_rejects_unrelated_files(self):
        from calibrate.visualize_norm import _check_output_directory
        with TemporaryDirectory() as directory:
            output = Path(directory)/'visualization'
            output.mkdir()
            summary = dict(report_kind='normal_calibration_visualization', images=[])
            (output/'summary.json').write_text(json.dumps(summary))
            note = output/'notes.txt'
            note.write_text('keep this user file')
            with self.assertRaisesRegex(ValueError, 'unrelated files'):
                _check_output_directory(output, ())
            self.assertEqual(note.read_text(), 'keep this user file')

    def test_plot_failure_does_not_change_a_passed_validation_report(self):
        with TemporaryDirectory() as directory:
            work = Path(directory)
            settings = CalibrationSettings(work, work/'samples.npz', work/'lut.npz', work/'regions.yaml')
            report_path = work/'verification/validation.json'
            _save_json(report_path, dict(status='passed'))
            with patch('calibrate.norm_pipeline._fit_and_validate_samples', return_value={'status': 'passed'}), \
                 patch('calibrate.visualize_norm.generate_calibration_visualizations', side_effect=ValueError('plot data')):
                with self.assertRaisesRegex(RuntimeError, '已保存.*可视化生成失败'):
                    fit_and_validate_samples(settings.input_file, settings)
            self.assertEqual(json.loads(report_path.read_text()), {'status': 'passed'})


class SingleImageReconstruction(unittest.TestCase):
    def _configuration(self, root, reference):
        for source in (reference.source_image, root/'folder0/same_name.png'):
            image = cv2.imread(str(source))
            cv2.rectangle(image, (255, 87), (385, 393), (0, 0, 0), 5)
            cv2.imwrite(str(source), image)
        np.savez(root/'camera.npz', camera_matrix=reference.camera.matrix, dist_coeffs=reference.camera.distortion)
        np.savetxt(root/'template.csv', np.column_stack((np.arange(48), reference.template_xy_mm)),
                   delimiter=',', header='id,x_mm,y_mm', comments='')
        (root/'provider.json').write_text(json.dumps(dict(image_size=[640, 480],
            camera_file='camera.npz', template_file='template.csv')))
        axis = np.linspace(-1, 1, 3)
        r, g, _ = np.meshgrid(axis, axis, axis, indexing='ij')
        metadata = dict(coordinate_kind='reference_arclength', coordinate_chart_id=reference.chart.chart_id,
                        frame_convention='st_grid_columns_rows', residual_domain='material_grid', color_space='linear_rgb')
        calibration = NormalCalibration(np.stack((r, g), axis=-1), np.zeros((3, 3, 3)),
                                        -np.ones(3), np.ones(3), 1., metadata=metadata)
        calibration.save(root/'runtime_lut.npz')
        config = dict(get_surface=dict(overall_reconstruction=dict(provider='marker_bicubic', config_file='provider.json')),
            material_coordinates={name: dict(rows=50, columns=24) for name in ('geometry', 'lightfield', 'observation')},
            reconstruction=dict(reference_image='reference.png', material_grid_file='reference_coordinates.npz',
                                output_dir='results', match_surface_aspect=False),
            local_reconstruction=dict(calibration_file='runtime_lut.npz',
                                      zero_color_protection=dict(inner_radius=.01, outer_radius=.02)))
        path = root/'runtime.yaml'
        path.write_text(yaml.safe_dump(config))
        return path, calibration

    def test_single_image_pipeline_matches_offline_solver_and_reference_is_zero(self):
        from reconstruction.pipeline import reconstruct_image
        from reconstruction.local_recon import (local_main, reconstruct_from_input,
            load_local_reconstruction_settings, save_reconstruction_result)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference, _ = synthetic_batch(directory)
            config, _ = self._configuration(root, reference)
            lut_before = (root/'runtime_lut.npz').read_bytes()
            grid_before = reference.chart_file.read_bytes()
            with patch('reconstruction.pipeline.reconstruct_from_input', wraps=reconstruct_from_input) as solve:
                result, output = reconstruct_image(root/'folder0/same_name.png', config)
            self.assertEqual(output, root/'results/pc/same_name_local_result.npz')
            save_reconstruction_result(result, output)
            self.assertTrue(output.is_file())
            self.assertIsNotNone(cv2.imread(str(root/'results/same_name_local_result.png')))
            self.assertFalse(output.with_suffix('.png').exists())
            self.assertEqual(result.xyz_out.shape, (50, 24, 3))
            self.assertTrue(np.max(np.abs(result.displacement)) > .001)
            data = solve.call_args.args[0]
            self.assertTrue(data['geometry_valid_mask'].all())
            self.assertFalse(result.photometric_valid_mask.all())
            np.testing.assert_array_equal(data['material_coordinates'], data['reference_material_coordinates'])
            source, offline = root/'paired_input.npz', root/'offline.npz'
            np.savez_compressed(source, **data)
            settings = load_local_reconstruction_settings(config, input_override=str(source))
            self.assertEqual(settings.output_file, root/'results/pc/paired_input_local_reconstruction.npz')
            with patch('sys.argv', ['local-recon', '--config', str(config), '--input', str(source), '--output', str(offline)]):
                local_main()
            with np.load(offline, allow_pickle=False) as replay:
                np.testing.assert_array_equal(replay['displacement'], result.displacement)
                np.testing.assert_array_equal(replay['xyz_out'], result.xyz_out)
            self.assertIsNotNone(cv2.imread(str(offline.with_suffix('.png'))))
            zero, _ = reconstruct_image(reference.source_image, config)
            np.testing.assert_array_equal(zero.displacement, 0)
            self.assertEqual((root/'runtime_lut.npz').read_bytes(), lut_before)
            self.assertEqual(reference.chart_file.read_bytes(), grid_before)
            with self.assertRaisesRegex(ValueError, '预览图不能覆盖'):
                reconstruct_image(reference.source_image, config, preview_output=str(reference.source_image))

    def test_frozen_sampling_grid_reuses_identity_and_rejects_mismatches(self):
        from reconstruction.pipeline import load_sampling_grid
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference, _ = synthetic_batch(directory)
            _, calibration = self._configuration(root, reference)
            full = load_sampling_grid(reference.chart_file, calibration, reference.surface, reference.camera, (50, 24))
            values = dict(format_version=1, grid_kind='frozen_material_grid',
                          material_coordinates=full.parameters, st=full.st,
                          template_xy_mm=reference.template_xy_mm, camera_matrix=reference.camera.matrix,
                          dist_coeffs=reference.camera.distortion, image_shape=reference.camera.image_shape,
                          **calibration.metadata)
            path = root/'grid.npz'
            np.savez_compressed(path, **values)
            frozen = load_sampling_grid(path, calibration, reference.surface, reference.camera, (50, 24))
            np.testing.assert_array_equal(frozen.parameters, full.parameters)
            self.assertEqual(frozen.chart_id, full.chart_id)
            resized = load_sampling_grid(path, calibration, reference.surface, reference.camera, (25, 12))
            direct = load_sampling_grid(reference.chart_file, calibration, reference.surface, reference.camera, (25, 12))
            self.assertEqual(resized.parameters.shape, (25, 12, 2))
            self.assertEqual(resized.chart_id, frozen.chart_id)
            np.testing.assert_allclose(resized.parameters, direct.parameters, atol=1e-10)
            wrong = replace(calibration, metadata=dict(calibration.metadata, coordinate_chart_id='another-reference'))
            for file in (path, reference.chart_file):
                with self.assertRaisesRegex(ValueError, 'coordinate_chart_id mismatch'):
                    load_sampling_grid(file, wrong, reference.surface, reference.camera, (50, 24))
            values['camera_matrix'] = reference.camera.matrix.copy()
            values['camera_matrix'][0, 0] += 1
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, 'camera_matrix'):
                load_sampling_grid(path, calibration, reference.surface, reference.camera, (50, 24))

    def test_solver_grid_matches_physical_region_using_resampled_frozen_coordinates(self):
        from reconstruction.pipeline import reconstruct_image
        from utils.surface_geometry import surface_grid_curve_lengths
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference, _ = synthetic_batch(directory)
            config, calibration = self._configuration(root, reference)
            original = reference.systems.observation
            values = dict(format_version=1, grid_kind='frozen_material_grid',
                          material_coordinates=original.coordinates, st=original.st,
                          template_xy_mm=reference.template_xy_mm, camera_matrix=reference.camera.matrix,
                          dist_coeffs=reference.camera.distortion, image_shape=reference.camera.image_shape,
                          **calibration.metadata)
            np.savez_compressed(root/'frozen_grid.npz', **values)
            raw = yaml.safe_load(config.read_text())
            raw['reconstruction'].update(material_grid_file='frozen_grid.npz', match_surface_aspect=True)
            config.write_text(yaml.safe_dump(raw))
            lut_before = (root/'runtime_lut.npz').read_bytes()
            grid_before = (root/'frozen_grid.npz').read_bytes()
            result, _ = reconstruct_image(root/'folder0/same_name.png', config)
            rows, columns = result.displacement.shape
            baseline = result.xyz_out.astype(float)-result.displacement_vectors.astype(float)
            width, height = surface_grid_curve_lengths(baseline, result.valid_mask)
            self.assertEqual(rows, 50)
            self.assertEqual(columns, 1+round((rows-1)*width/height))
            self.assertLess(abs((width/(columns-1))/(height/(rows-1))-1), .04)
            self.assertEqual(result.xyz_out.shape, (rows, columns, 3))
            self.assertTrue(result.photometric_valid_mask.any())
            self.assertEqual((root/'runtime_lut.npz').read_bytes(), lut_before)
            self.assertEqual((root/'frozen_grid.npz').read_bytes(), grid_before)

    def test_coupled_curved_grid_resampling_preserves_the_original_chart(self):
        from reconstruction.pipeline import load_sampling_grid
        from utils.surface_geometry import surface_grid_curve_lengths
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference, _ = synthetic_batch(directory)
            _, calibration = self._configuration(root, reference)
            model = reference.surface.model
            normalized = reference.template_xy_mm/[15, 42]
            points = AnalyticPatch('coupled').evaluate(normalized)
            coefficients = np.linalg.lstsq(model.B, points, rcond=None)[0]
            surface = BicubicSurfaceAdapter(model, coefficients)
            chart = build_arclength_chart(surface)
            chart_path = save_arclength_chart(root/'curved_chart.npz', chart)
            calibration = replace(calibration, metadata=dict(calibration.metadata, coordinate_chart_id=chart.chart_id))
            original = sample_surface_grid(surface, chart, 129, 101)
            np.savez_compressed(root/'curved_grid.npz', format_version=1, grid_kind='frozen_material_grid',
                material_coordinates=original.parameters, st=original.st, template_xy_mm=model.xy,
                camera_matrix=reference.camera.matrix, dist_coeffs=reference.camera.distortion,
                image_shape=reference.camera.image_shape, **calibration.metadata)
            resized = load_sampling_grid(root/'curved_grid.npz', calibration, surface, reference.camera, (87, 35))
            direct = load_sampling_grid(chart_path, calibration, surface, reference.camera, (87, 35))
            self.assertEqual(resized.chart_id, chart.chart_id)
            np.testing.assert_allclose(resized.parameters, direct.parameters, rtol=0, atol=.002)
            recovered_st, valid = chart.to_st(resized.parameters)
            self.assertTrue(valid.all())
            np.testing.assert_allclose(recovered_st, resized.st, rtol=0, atol=2e-5)
            # A quarter-cylinder must use the 3D arc, even if it is tilted toward the camera.
            a, b = np.meshgrid(np.linspace(0, np.pi/2, 101), np.linspace(0, 40, 129))
            xyz = np.stack((10*np.sin(a), b, 10*np.cos(a)), axis=-1)
            width, height = surface_grid_curve_lengths(xyz, np.ones(a.shape, bool))
            self.assertAlmostEqual(width, 100*20*np.sin(np.pi/400), places=10)
            self.assertAlmostEqual(height, 40., places=10)

    def test_geometry_export_keeps_separate_observation_and_geometry_masks(self):
        with TemporaryDirectory() as directory:
            _, reference, _ = synthetic_batch(directory)
            provider = _provider_from_frame(reference)
            specs = MaterialCoordinateSpecs(CoordinateGridSpec(13, 7), CoordinateGridSpec(17, 9), CoordinateGridSpec(29, 15))
            systems = provider.sample_coefficients(reference.coefficients, specs, reference.surface_mask)
            path = Path(directory)/'different_grids.npz'
            _save_frame(path, replace(reference, systems=systems))
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data['geometry_valid_mask'].shape, (29, 15))
                self.assertEqual(data['geometry_grid_valid_mask'].shape, (13, 7))
            replay = load_frame_calibration_data(path)
            self.assertEqual(replay.systems.geometry.valid_mask.shape, (13, 7))
            self.assertEqual(replay.systems.observation.valid_mask.shape, (29, 15))


class ReconstructionVisualization(unittest.TestCase):
    def test_coarse_surface_does_not_bridge_missing_geometry(self):
        from reconstruction.viewer import _surface_quads
        row, column = np.mgrid[:205, :205]
        xyz = np.stack((column, row, np.zeros_like(row)), axis=-1).astype(float)
        valid = np.ones(row.shape, bool)
        valid[80, :] = False
        quads, _ = _surface_quads(xyz, np.zeros_like(row, dtype=float), valid)
        self.assertGreater(len(quads), 0)
        spans_hole = (quads[..., 1].min(1) <= 80) & (quads[..., 1].max(1) >= 80)
        self.assertFalse(spans_hole.any())

    def test_two_panel_preview_preserves_geometry_mask_and_inferred_points(self):
        from io import BytesIO
        from reconstruction.viewer import create_reconstruction_figure
        row, column = np.mgrid[:8, :6]
        xyz = np.stack((column, row, np.zeros_like(row)), axis=-1).astype(float)
        valid = np.ones(row.shape, bool)
        valid[0, 0] = False
        displacement = np.zeros(row.shape)
        displacement[0, 0] = np.nan
        figure = create_reconstruction_figure(xyz, displacement, valid)
        # Two data panels plus the displacement colorbar, with no confidence panel.
        self.assertEqual(len(figure.axes), 3)
        self.assertFalse(any('confidence' in axis.get_title().lower() for axis in figure.axes))
        heatmap = figure.axes[1].images[0]
        self.assertTrue(heatmap.get_array().mask[0, 0])
        self.assertFalse(heatmap.get_array().mask[3, 3])
        self.assertLess(heatmap.norm.vmin, 0)
        self.assertGreater(heatmap.norm.vmax, 0)
        stream = BytesIO()
        figure.savefig(stream, format='png')
        self.assertTrue(stream.getvalue().startswith(b'\x89PNG'))
        with self.assertRaisesRegex(ValueError, 'must match'):
            create_reconstruction_figure(xyz, displacement[:-1], valid)
        with self.assertRaisesRegex(ValueError, 'no valid geometry'):
            create_reconstruction_figure(xyz, displacement, np.zeros_like(valid))

    def test_view_cli_saves_existing_result_without_config_or_lut(self):
        from reconstruction.local_recon import local_main
        from reconstruction.viewer import view_reconstruction
        with TemporaryDirectory() as directory:
            source = Path(directory)/'result.npz'
            image = Path(directory)/'preview.png'
            row, column = np.mgrid[:8, :6]
            np.savez_compressed(source,
                xyz_out=np.stack((column, row, np.zeros_like(row)), axis=-1),
                displacement=np.zeros(row.shape), valid_mask=np.ones(row.shape, bool))
            original = source.read_bytes()
            with patch('sys.argv', ['local-recon', '--view', str(source), '--preview-output', str(image),
                                    '--config', str(Path(directory)/'missing.yaml')]), \
                 patch('reconstruction.local_recon.load_local_reconstruction_settings', side_effect=AssertionError('must not load config')):
                local_main()
            self.assertEqual(source.read_bytes(), original)
            self.assertIsNotNone(cv2.imread(str(image)))
            with self.assertRaisesRegex(ValueError, 'distinct'):
                view_reconstruction(source, image_output=source, show=False)
            np.savez_compressed(source, xyz=np.zeros((8, 6, 3)))
            with self.assertRaisesRegex(ValueError, 'missing'):
                view_reconstruction(source, show=False)


if __name__ == '__main__':
    unittest.main()
