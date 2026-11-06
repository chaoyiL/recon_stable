"""Diagnose color/normal ambiguities and evaluate remedies on held-out images.

Outputs are PNG, JSON, CSV and NPZ. Calibration inputs are never modified.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from calibrate.utils.visualize_norm import _check_declarations, _frame_paths, _load_samples
from utils.normal_mapping import NormalCalibration, build_normal_lut, find_ambiguous_color_pairs, lookup_slopes_raw
from utils.image_io import cached_linear_rgb
from utils.region_annotations import load_regions
from utils.surface_geometry import local_slopes_to_normals


MODE_NAMES = ('sparse', 'connected_spread', 'dominant_with_sparse_alternative',
              'multiple_supported_components', 'fragmented_low_support')
BRANCH_NAMES = ('no_outer_support', 'main_support', 'multiple_branches')


def _normals(slopes):
    return local_slopes_to_normals(slopes, np.array([0., 0., 1.]), np.eye(3)[:, :2])[0]


def _angles(first, second):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1),
                                np.sum(first*second, axis=-1)))


def _distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return ({'count': 0} if not len(values) else
            dict(count=len(values), median=float(np.median(values)),
                 p95=float(np.percentile(values, 95)), maximum=float(values.max())))


def classify_normal_neighborhood(normals: np.ndarray, image_index: np.ndarray, *,
                                 link_angle_degrees: float = 10., separation_degrees: float = 20.,
                                 minimum_samples: int = 6) -> dict:
    """Classify one color ball using connected components of unit local normals.

    Edges join normals at <= link_angle_degrees. A connected spread can span a
    large angle through intermediate samples; it is not called a disjoint mode.
    Separate components need >=3 samples each and mean separation > the supplied
    threshold. Counts concern overlapping color balls, not independent modes.
    """
    normals, image_index = np.asarray(normals, float), np.asarray(image_index)
    count = len(normals)
    labels = (fcluster(linkage(pdist(normals), method='single'),
                       2*np.sin(np.radians(link_angle_degrees)/2), criterion='distance')-1
              if count > 1 else np.zeros(count, int))
    groups = [np.flatnonzero(labels == key) for key in np.unique(labels)]
    groups.sort(key=lambda ids: -len(ids))
    means, image_counts = [], []
    for key, ids in enumerate(groups):
        labels[ids] = key
        mean = normals[ids].sum(axis=0)
        means.append(mean/np.linalg.norm(mean))
        image_counts.append(len(np.unique(image_index[ids])))
    sizes = [len(ids) for ids in groups]
    supported = [i for i, size in enumerate(sizes) if size >= 3]
    separated = [(i, j) for offset, i in enumerate(supported) for j in supported[offset+1:]
                 if _angles(means[i], means[j]) > separation_degrees]
    if count < minimum_samples:
        kind = 'sparse'
    elif separated:
        kind = 'multiple_supported_components'
    elif len(groups) == 1:
        kind = 'connected_spread'
    elif sizes[0]/count >= .8:
        kind = 'dominant_with_sparse_alternative'
    else:
        kind = 'fragmented_low_support'
    return dict(kind=kind, labels=labels, sizes=sizes, image_counts=image_counts,
                mean_normals=np.asarray(means), separated_components=separated)


def _neighborhoods(colors, normals, groups, bad, color_threshold, angle_threshold):
    tree = cKDTree(colors)
    classes = np.full(len(colors), -1, np.int8)
    counts = Counter()
    examples = {}
    sensitivity = {str(limit): Counter() for limit in (7.5, 10., 15.)}
    for index in np.flatnonzero(bad):
        ids = np.asarray(tree.query_ball_point(colors[index], color_threshold, return_sorted=True))
        for limit in (7.5, 10., 15.):
            info = classify_normal_neighborhood(normals[ids], groups[ids],
                link_angle_degrees=limit, separation_degrees=angle_threshold)
            sensitivity[str(limit)][info['kind']] += 1
            if limit != 10.:
                continue
            kind = info['kind']
            classes[index] = MODE_NAMES.index(kind)
            counts[kind] += 1
            # Prefer two well-supported components over a large one plus singletons.
            score = (min(info['sizes'][:2]), len(ids)) if kind == 'multiple_supported_components' else (len(ids),)
            if kind not in examples or score > examples[kind]['score']:
                examples[kind] = dict(score=score, index=int(index), ids=ids.tolist(),
                    color=colors[index].tolist(), labels=info['labels'].tolist(), sizes=info['sizes'],
                    image_counts=info['image_counts'], mean_normals=info['mean_normals'].tolist(),
                    separated_components=info['separated_components'])
    for example in examples.values():
        del example['score']
    return classes, dict(counts), {key: dict(value) for key, value in sensitivity.items()}, examples


def compare_branch_context(
    colors: np.ndarray, normals: np.ndarray, image_index: np.ndarray,
    pair: np.ndarray, context_indices: np.ndarray, *, inner_radius: float = .01,
    normal_tolerance_degrees: float = 10.,
) -> list[dict]:
    """Measure outside support for both anchors, regardless of inner sample count.

    The caller supplies a larger color ball. Exclude BOTH original color balls,
    including the anchors, before comparing context normals with each anchor.
    Support is branch-continuation evidence, not a label-correctness decision.
    """
    colors, normals = np.asarray(colors, float), np.asarray(normals, float)
    pair, ids = np.asarray(pair, int), np.asarray(context_indices, int)
    distances = np.linalg.norm(colors[ids, None]-colors[pair][None], axis=-1)
    outside = (distances > inner_radius).all(axis=-1)
    ids, distances = ids[outside], distances[outside]
    angles = _angles(normals[ids, None], normals[pair][None])
    result = []
    for branch, anchor in enumerate(pair):
        selected = angles[:, branch] <= normal_tolerance_degrees
        support = ids[selected]
        result.append(dict(sample_index=int(anchor), support_indices=support.tolist(),
            support_count=len(support), support_image_count=len(np.unique(image_index[support])),
            other_image_count=len(np.unique(image_index[support][image_index[support] != image_index[anchor]])),
            nearest_support_color_distance=float(distances[selected, branch].min()) if selected.any() else None,
            minimum_context_normal_angle_degrees=float(angles[:, branch].min()) if len(ids) else None))
    return result


def classify_branch_support(support_counts: np.ndarray, *, dominance_fraction: float = .8) -> np.ndarray:
    """Classify (..., 2) outer support counts, with no minimum sample count.

    Codes index BRANCH_NAMES. Dominance concerns observed support density, not
    correctness: even a weakly supported alternative is retained for review.
    """
    counts = np.asarray(support_counts, float)
    if counts.ndim < 1 or counts.shape[-1] != 2 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Expected finite nonnegative support counts with shape (..., 2)')
    if not .5 < dominance_fraction <= 1:
        raise ValueError('Dominance fraction must be in (0.5, 1]')
    total = counts.sum(axis=-1)
    dominant = counts.max(axis=-1) >= dominance_fraction*total
    return np.where(total == 0, 0, np.where(dominant, 1, 2)).astype(np.int8)


def aggregate_branch_classification(pairs: np.ndarray, pair_classes: np.ndarray,
                                    sample_count: int) -> np.ndarray:
    """Assign each endpoint its strongest evidence; a supported overlap wins.

    -1 means no conflicting partner. A point with both dominant and overlapping
    relationships is counted once as multiple_branches, never twice as modes.
    """
    result = np.full(sample_count, -1, np.int8)
    np.maximum.at(result, np.asarray(pairs, int).ravel(), np.repeat(pair_classes, 2))
    return result


def _branch_class_counts(classes):
    return {name: int(np.sum(classes == index)) for index, name in enumerate(BRANCH_NAMES)}


def _expanded_classification(samples, pairs, support, images, radii, inner_counts,
                             original_classes, output, dominance_fraction):
    """Classify all pairs, then deduplicate endpoints; record scale sensitivity."""
    sample_count = len(samples['image_index'])
    pair_classes = classify_branch_support(support, dominance_fraction=dominance_fraction)
    point_classes = np.stack([aggregate_branch_classification(pairs, row, sample_count)
                              for row in pair_classes])
    primary = 1  # Three times the original color threshold.
    selected = point_classes[primary] >= 0
    stable = (point_classes == point_classes[primary]).all(axis=0)
    stable_pair = (pair_classes == pair_classes[primary]).all(axis=0)
    image_classes = classify_branch_support(images, dominance_fraction=dominance_fraction)
    # A point can overlap one partner while dominating a different partner.
    has_main = np.zeros(sample_count, bool)
    has_main[pairs[pair_classes[primary] == 1].ravel()] = True
    mixed = has_main & (point_classes[primary] == 2)
    overlap_degree, main_degree = np.zeros(sample_count, int), np.zeros(sample_count, int)
    for kind, degree in ((1, main_degree), (2, overlap_degree)):
        np.add.at(degree, pairs[pair_classes[primary] == kind].ravel(), 1)
    dominant_degree, alternative_degree = np.zeros(sample_count, int), np.zeros(sample_count, int)
    main_pairs = np.flatnonzero(pair_classes[primary] == 1)
    winners = support[primary, main_pairs].argmax(axis=-1)
    np.add.at(dominant_degree, pairs[main_pairs, winners], 1)
    np.add.at(alternative_degree, pairs[main_pairs, 1-winners], 1)
    roles = np.full(sample_count, 'not_ambiguous', dtype='<U32')
    roles[selected] = 'no_outer_support'
    main_only = point_classes[primary] == 1
    roles[main_only & (dominant_degree > 0)] = 'dominant_branch'
    roles[main_only & (alternative_degree > 0)] = 'weak_alternative'
    roles[main_only & (dominant_degree > 0) & (alternative_degree > 0)] = 'partner_dependent_dominance'
    roles[point_classes[primary] == 2] = 'overlapping_branch'
    both_multiple_images = (images[primary] >= 2).all(axis=-1)
    cross_image_overlap = np.zeros(sample_count, bool)
    cross_image_overlap[pairs[(pair_classes[primary] == 2) & both_multiple_images].ravel()] = True
    with (output/'branch_classification_pairs.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_A', 'sample_B', 'classification', 'support_A', 'support_B',
            'support_images_A', 'support_images_B', 'dominant_support_fraction',
            'dominant_sample', 'stable_all_radii', 'image_count_classification',
            *[f'classification_radius_{radius:g}' for radius in radii]])
        for index, pair in enumerate(pairs):
            sizes = support[primary, index]
            total = sizes.sum()
            kind = pair_classes[primary, index]
            writer.writerow([*pair, BRANCH_NAMES[kind], *sizes, *images[primary, index],
                float(sizes.max()/total) if total else None,
                pair[sizes.argmax()] if kind == 1 else None, bool(stable_pair[index]),
                BRANCH_NAMES[image_classes[primary, index]],
                *[BRANCH_NAMES[value] for value in pair_classes[:, index]]])
    with (output/'branch_classification_points.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_index', 'image', 'delta_R', 'delta_G', 'delta_B', 'p', 'q',
            'classification', 'role', 'inner_sample_count', 'original_neighborhood_kind',
            'overlap_partners', 'main_support_partners', 'dominant_over_partners',
            'weaker_than_partners', 'mixed_pair_classifications', 'stable_all_radii',
            'overlap_supported_by_multiple_images_on_both_branches',
            *[f'classification_radius_{radius:g}' for radius in radii]])
        for index in np.flatnonzero(selected):
            writer.writerow([index, samples['image_paths'][samples['image_index'][index]],
                *samples['color_residual_samples'][index], *samples['slope_samples'][index],
                BRANCH_NAMES[point_classes[primary, index]], roles[index], inner_counts[index],
                MODE_NAMES[original_classes[index]], overlap_degree[index], main_degree[index],
                dominant_degree[index], alternative_degree[index], bool(mixed[index]), bool(stable[index]),
                bool(cross_image_overlap[index]), *[BRANCH_NAMES[value] for value in point_classes[:, index]]])
    sensitivity = []
    for limit in (.7, .8, .9):
        kinds = classify_branch_support(support[primary], dominance_fraction=limit)
        sensitivity.append(dict(dominance_fraction=limit, pairs=_branch_class_counts(kinds),
            points=_branch_class_counts(aggregate_branch_classification(pairs, kinds, sample_count))))
    old_to_new = {name: _branch_class_counts(point_classes[primary, original_classes == index])
                  for index, name in enumerate(MODE_NAMES)}
    return dict(primary_outer_radius=radii[primary], dominance_fraction=dominance_fraction,
        pair_counts=_branch_class_counts(pair_classes[primary]),
        point_counts=_branch_class_counts(point_classes[primary]),
        stable_all_radii_point_counts=_branch_class_counts(point_classes[primary, stable]),
        stable_larger_two_radii_point_counts=_branch_class_counts(
            point_classes[primary, point_classes[primary] == point_classes[2]]),
        stable_all_radii_pair_counts=_branch_class_counts(pair_classes[primary, stable_pair]),
        mixed_pair_classification_point_count=int(mixed.sum()),
        main_support_point_roles=dict(Counter(roles[main_only].tolist())),
        multiple_branches_with_both_branches_in_multiple_images_point_count=int(cross_image_overlap.sum()),
        pair_classification_disagrees_when_counting_images=int(np.sum(
            pair_classes[primary] != image_classes[primary])),
        radius_sensitivity=[dict(outer_radius=radius, pairs=_branch_class_counts(pair_classes[index]),
            points=_branch_class_counts(point_classes[index])) for index, radius in enumerate(radii)],
        dominance_fraction_sensitivity=sensitivity, original_to_expanded=old_to_new,
        method='For each conflicting pair, compare counts in disjoint anchor-normal cones outside BOTH '
               'original color balls. At the primary radius, >= dominance_fraction of the combined '
               'outer support means main_support; otherwise both supported cones mean multiple_branches. '
               'Zero total means no_outer_support. No minimum inner or outer sample count is imposed. '
               'A point is multiple_branches if ANY partner demonstrates overlap; otherwise main_support '
               'if any partner has dominant support. Counts are unique samples, not independent regions.',
        limitation='Classification describes observed continuation density, not correctness or independent '
                   'physical sheets. Counts may reflect per-image sampling density. Image-count agreement '
                   'and radius stability are reported separately; no samples are rejected.'), point_classes[primary]


def _continuation_analysis(samples, pairs, near_pairs, color_threshold, angle_threshold, output,
                           original_classes, dominance_fraction):
    colors = samples['color_residual_samples'].astype(float)
    normals, groups = _normals(samples['slope_samples']), samples['image_index']
    tree = cKDTree(colors)
    counts = 1+np.bincount(near_pairs.ravel(), minlength=len(colors))
    scopes = dict(all=np.ones(len(pairs), bool),
        either_endpoint_fewer_than6=(counts[pairs] < 6).any(axis=-1),
        both_endpoints_only2=(counts[pairs] == 2).all(axis=-1))
    radii = [2*color_threshold, 3*color_threshold, 4*color_threshold]
    summaries, examples, support_by_radius, images_by_radius = [], {}, [], []
    with (output/'branch_continuation.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_A', 'sample_B', 'outer_color_radius', 'inner_count_A', 'inner_count_B',
                         'support_A', 'support_B', 'support_images_A', 'support_images_B',
                         'other_images_A', 'other_images_B', 'min_context_angle_A', 'min_context_angle_B'])
        for radius in radii:
            evidence = []
            for index, pair in enumerate(pairs):
                center = colors[pair].mean(axis=0)
                ids = np.asarray(tree.query_ball_point(center, radius, return_sorted=True), int)
                branches = compare_branch_context(colors, normals, groups, pair, ids,
                    inner_radius=color_threshold, normal_tolerance_degrees=angle_threshold/2)
                sizes = [branch['support_count'] for branch in branches]
                image_counts = [branch['support_image_count'] for branch in branches]
                other_images = [branch['other_image_count'] for branch in branches]
                evidence.append((sizes, image_counts, other_images))
                writer.writerow([*pair, radius, *counts[pair], *sizes, *image_counts, *other_images,
                    *[branch['minimum_context_normal_angle_degrees'] for branch in branches]])
                if radius != radii[1] or not scopes['both_endpoints_only2'][index]:
                    continue
                kind = 'both' if min(sizes) > 0 else ('one' if max(sizes) > 0 else 'neither')
                score = (min(sizes), min(image_counts), sum(sizes)) if kind == 'both' else (max(sizes),)
                if kind not in examples or score > examples[kind]['score']:
                    examples[kind] = dict(pair=pair.tolist(), context_indices=ids.tolist(), branches=branches,
                                         outer_radius=radius, score=score)
            sizes = np.asarray([item[0] for item in evidence]).reshape(-1, 2)
            images = np.asarray([item[1] for item in evidence]).reshape(-1, 2)
            other_images = np.asarray([item[2] for item in evidence]).reshape(-1, 2)
            support_by_radius.append(sizes)
            images_by_radius.append(images)
            for scope, selected in scopes.items():
                summaries.append(dict(scope=scope, outer_radius=radius, pair_count=int(selected.sum()),
                    both_have_context=int(np.sum(selected & (sizes > 0).all(axis=-1))),
                    one_has_context=int(np.sum(selected & (sizes > 0).any(axis=-1) & (sizes == 0).any(axis=-1))),
                    neither_has_context=int(np.sum(selected & (sizes == 0).all(axis=-1))),
                    both_have_at_least3=int(np.sum(selected & (sizes >= 3).all(axis=-1))),
                    both_have_at_least2_context_images=int(np.sum(selected & (images >= 2).all(axis=-1))),
                    both_have_other_image_support=int(np.sum(selected & (other_images > 0).all(axis=-1)))))
    for example in examples.values():
        del example['score']
    classification, point_classes = _expanded_classification(samples, pairs, np.asarray(support_by_radius),
        np.asarray(images_by_radius), radii, counts, original_classes, output, dominance_fraction)
    return dict(inner_radius=color_threshold, outer_radii=radii,
        normal_tolerance_degrees=angle_threshold/2,
        method='Every conflicting pair is examined, without a minimum inner-neighbour count. '
               'Query a ball around the pair color midpoint; exclude both original color balls. '
               'Measure context support separately within each anchor-normal cone.',
        interpretation='Counts describe continuation evidence, never automatic rejection. '
                       'Anchor normals select support cones: this is not an unbiased prediction test. '
                       'No outer support can mean sparse sampling or rapid branch variation, not a wrong label.',
        summaries=summaries, examples=examples, expanded_classification=classification), point_classes


def _continuation_figure(samples, report, output):
    examples = [(key, report['examples'][key]) for key in ('both', 'one') if key in report['examples']]
    if not examples:
        return
    colors, slopes = samples['color_residual_samples'], samples['slope_samples']
    figure = _figure((13, 5*len(examples)))
    for row, (kind, example) in enumerate(examples):
        pair, ids = np.asarray(example['pair']), np.asarray(example['context_indices'])
        outside = (np.linalg.norm(colors[ids, None].astype(float)-colors[pair][None].astype(float), axis=-1)
                   > report['inner_radius']).all(axis=-1)
        ids = ids[outside]
        color_axis = figure.add_subplot(len(examples), 2, 2*row+1, projection='3d')
        slope_axis = figure.add_subplot(len(examples), 2, 2*row+2)
        color_axis.scatter(*colors[ids].T, color='#bbbbbb', s=9, alpha=.3)
        slope_axis.scatter(*slopes[ids].T, color='#bbbbbb', s=10, alpha=.3)
        for branch, color, name in zip(example['branches'], ('#2476b6', '#e87519'), ('A', 'B'), strict=True):
            support = np.asarray(branch['support_indices'], int)
            anchor = branch['sample_index']
            color_axis.scatter(*colors[support].T, color=color, s=20)
            color_axis.scatter(*colors[anchor], color=color, marker='*', edgecolors='black', s=150)
            slope_axis.scatter(*slopes[support].T, color=color, s=25,
                label=f'{name}: {len(support)} points / {branch["support_image_count"]} context images')
            slope_axis.scatter(*slopes[anchor], color=color, marker='*', edgecolors='black', s=180)
            slope_axis.annotate(name, slopes[anchor], xytext=(6, -10), textcoords='offset points')
        color_axis.set(xlabel='delta R', ylabel='delta G', zlabel='delta B',
                       title=f'Pair {pair.tolist()} | original neighborhoods: 2 + 2')
        slope_axis.set(xlabel='p', ylabel='q', title='Both branches have context' if kind == 'both'
                       else 'Only one branch has context at this radius')
        slope_axis.invert_yaxis()
        slope_axis.legend(fontsize=9)
        slope_axis.grid(alpha=.2)
    figure.suptitle(f'Continuation beyond the original color neighborhood | outer radius '
        f'{report["outer_radii"][1]:g}\nBoth inner {report["inner_radius"]:g} balls excluded; '
        f'normal support <= {report["normal_tolerance_degrees"]:g} deg; stars = excluded anchors\n'
        'Evidence of neighboring branches; no automatic sample deletion', fontsize=11)
    _save(figure, output/'branch_continuation.png')


def _classification_figure(samples, classes, report, output):
    figure = _figure((16, 5))
    colors, slopes = samples['color_residual_samples'], samples['slope_samples']
    axes = [figure.add_subplot(1, 3, index+1) for index in range(3)]
    for kind, name, color in ((-1, 'Other samples', '#bdbdbd'),
                              (1, 'Main support', '#2476b6'),
                              (2, 'Multiple branches', '#d33333'),
                              (0, 'No outer support', '#e8aa20')):
        selected = classes == kind
        if not selected.any():
            continue
        for axis, values in zip(axes[:2], (colors[:, :2], slopes), strict=True):
            axis.scatter(*values[selected].T, s=5 if kind == -1 else 10,
                         alpha=.2 if kind == -1 else .65, color=color,
                         label=f'{name}: {selected.sum()}')
    axes[0].set(xlabel='delta R (linear)', ylabel='delta G (linear)', title='Color projection (delta B omitted)')
    axes[1].set(xlabel='p', ylabel='q', title='Local gradient space')
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal', adjustable='box')
    axes[0].legend(fontsize=8)
    for kind, name, color in ((1, 'Main support', '#2476b6'), (2, 'Multiple branches', '#d33333')):
        rows = report['radius_sensitivity']
        axes[2].plot([row['outer_radius'] for row in rows],
                     [row['points'][BRANCH_NAMES[kind]] for row in rows], 'o-', color=color, label=name)
    axes[2].set(xlabel='Outer color radius', ylabel='Unique ambiguous samples', title='Classification vs expansion radius')
    axes[2].axvline(report['primary_outer_radius'], color='#888888', linestyle='--', linewidth=1)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle(f'Expanded branch classification | outer radius {report["primary_outer_radius"]:g} | '
        f'main support >= {100*report["dominance_fraction"]:g}%\n'
        'Any supported overlap takes priority for a point; no sample deletion', fontsize=12)
    _save(figure, output/'branch_classification.png')


def _spatial_context(work, samples, frame_paths):
    groups = samples['image_index']
    shape = len(groups)
    context = dict(uv=np.empty((shape, 2)), radial_fraction=np.empty(shape),
                   spatial_normal_residual_degrees=np.empty(shape), observed=np.empty((shape, 3)),
                   reference=np.empty((shape, 3)))
    regions = load_regions(work/'manual_regions.yaml')
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    for index, name in enumerate(samples['image_paths']):
        name = str(name)
        group = groups == index
        with np.load(frame_paths[name], allow_pickle=False) as frame:
            selected, valid = frame['sample_valid_mask'], frame['sphere_label_valid']
            if (str(frame['image_key']) != name
                    or not np.array_equal(frame['slope_grid'][selected], samples['slope_samples'][group])
                    or not np.array_equal(frame['color_residual_grid'][selected],
                                          samples['color_residual_samples'][group])):
                raise ValueError(f'Frame/sample mismatch: {name}')
            slopes = frame['slope_grid']
            normals = _normals(slopes)
            mean = np.stack([convolve(np.where(valid, normals[..., j], 0), kernel,
                                     mode='constant') for j in range(3)], axis=-1)
            count = convolve(valid.astype(float), kernel, mode='constant')
            mean /= np.maximum(np.linalg.norm(mean, axis=-1, keepdims=True), 1e-30)
            angle = np.where(count > 0, _angles(normals, mean), np.nan)
            uv = frame['uv'][selected]
            region = regions[str(name)]
            context['uv'][groups == index] = uv
            context['radial_fraction'][groups == index] = np.linalg.norm(
                (uv-region.center_px)/region.semi_axes_px, axis=-1)
            context['spatial_normal_residual_degrees'][groups == index] = angle[selected]
            context['observed'][groups == index] = frame['observed_color_grid'][selected]
            context['reference'][groups == index] = frame['reference_color_grid'][selected]
    return context, regions


def _rate(label, selected, bad, near_degree, bad_degree):
    count = int(selected.sum())
    endpoints = int(near_degree[selected].sum())
    return dict(label=label, sample_count=count, ambiguous_sample_count=int(bad[selected].sum()),
                ambiguous_sample_fraction=float(bad[selected].mean()) if count else None,
                fraction_of_all_ambiguous_samples=float(bad[selected].sum()/max(1, bad.sum())),
                conflicting_neighbour_fraction=(float(bad_degree[selected].sum()/endpoints) if endpoints else None))


def _rates_by_bins(values, bins, bad, near_degree, bad_degree):
    return [_rate(f'{low:g}-{high:g}', (values >= low) & (values < high), bad, near_degree, bad_degree)
            for low, high in zip(bins[:-1], bins[1:])]


def _pattern_statistics(samples, context, bad, pairs, near_pairs):
    colors, slopes, groups = (samples[key] for key in ('color_residual_samples', 'slope_samples', 'image_index'))
    near_degree = np.bincount(near_pairs.ravel(), minlength=len(colors))
    bad_degree = np.bincount(pairs.ravel(), minlength=len(colors))
    tilt = np.degrees(np.arctan(np.linalg.norm(slopes.astype(float), axis=-1)))
    signs = (colors > 0).astype(int) @ np.array([4, 2, 1])
    octants = [_rate(' '.join(channel+('+' if code & bit else '-')
                    for channel, bit in zip('RGB', (4, 2, 1))), signs == code, bad, near_degree, bad_degree)
               for code in range(8)]
    images = [_rate(str(name), groups == index, bad, near_degree, bad_degree)
              for index, name in enumerate(samples['image_paths'])]
    smoothness = context['spatial_normal_residual_degrees']
    return dict(color_signs=octants, images=images,
        color_strength=_rates_by_bins(np.linalg.norm(colors, axis=-1), [0, .05, .1, .2, .3, .4, .6, 2], bad, near_degree, bad_degree),
        tilt=_rates_by_bins(tilt, [0, 10, 20, 30, 40, 50, 60, 70, 90], bad, near_degree, bad_degree),
        radial_fraction=_rates_by_bins(context['radial_fraction'], [0, .25, .5, .75, .9, 1.1], bad, near_degree, bad_degree),
        spatial_smoothness={label: dict(_distribution(smoothness[mask]),
            fraction_over_5_degrees=float(np.mean(smoothness[mask] > 5)) if mask.any() else None)
            for label, mask in (('ambiguous', bad), ('other', ~bad))},
        observed_channels={label: dict(median=np.median(context['observed'][mask], axis=0).tolist(),
            fraction_below_001=np.mean(context['observed'][mask] < .01, axis=0).tolist(),
            fraction_above_095=np.mean(context['observed'][mask] > .95, axis=0).tolist())
            for label, mask in (('ambiguous', bad), ('other', ~bad))}), near_degree, bad_degree, tilt


def _figure(size):
    figure = Figure(figsize=size, layout='constrained')
    FigureCanvasAgg(figure)
    return figure


def _save(figure, path):
    figure.savefig(path, dpi=160, format='png')
    figure.clear()


def _patterns_figure(colors, bad, patterns, output):
    figure = _figure((16, 9))
    axes = figure.subplots(2, 3)
    for axis, (first, second) in zip(axes[0, :2], ((0, 1), (1, 2)), strict=True):
        count, xx, yy = np.histogram2d(colors[:, first], colors[:, second], bins=30)
        flagged, _, _ = np.histogram2d(colors[bad, first], colors[bad, second], bins=(xx, yy))
        rate = np.ma.array(100*flagged/np.maximum(count, 1), mask=count < 15)
        artist = axis.pcolormesh(xx, yy, rate.T, vmin=0, vmax=60, cmap='magma')
        axis.set(xlabel=f'delta {"RGB"[first]} (linear)', ylabel=f'delta {"RGB"[second]} (linear)',
                 title='Ambiguous samples (%) | bins with >=15 samples')
        figure.colorbar(artist, ax=axis, label='%')
    datasets = [(axes[0, 2], 'color_signs', 'Signed RGB regions'),
                (axes[1, 0], 'tilt', 'Sphere-labelled tilt (degrees)'),
                (axes[1, 1], 'images', 'Source image (capture time)'),
                (axes[1, 2], 'radial_fraction', 'Image distance from contact center / radius')]
    for axis, key, title in datasets:
        rows = patterns[key]
        labels = [row['label'][17:23] if key == 'images' else row['label'] for row in rows]
        axis.bar(np.arange(len(rows)), [100*(row['ambiguous_sample_fraction'] or 0) for row in rows], color='#c83434')
        axis.set(xticks=np.arange(len(rows)), xticklabels=labels, ylabel='Ambiguous samples (%)', title=title)
        axis.tick_params(axis='x', rotation=45)
    figure.suptitle('Ambiguity patterns | sample rates, not raw red-point counts\n'
                   'Projected RGB panels average over the third channel; sparse bins are hidden')
    _save(figure, output/'ambiguity_patterns.png')


def _examples_figure(samples, examples, output):
    order = [('multiple_supported_components', 'Separated supported components'),
             ('dominant_with_sparse_alternative', 'Dominant group + sparse alternative'),
             ('connected_spread', 'Connected broad spread'), ('sparse', 'Too few samples to choose a branch')]
    figure = _figure((13, 11))
    axes = figure.subplots(2, 2)
    colors = colormaps['tab10']
    used_images = set()
    report = {}
    for axis, (kind, title) in zip(axes.flat, order, strict=True):
        if kind not in examples:
            axis.set_visible(False)
            continue
        example = examples[kind]
        ids = np.asarray(example['ids'])
        slopes = samples['slope_samples'][ids]
        groups = samples['image_index'][ids]
        components = np.asarray(example['labels'])
        for group in np.unique(groups):
            used_images.add(int(group))
            for component in np.unique(components):
                mask = (groups == group) & (components == component)
                axis.scatter(slopes[mask, 0], slopes[mask, 1], s=55, color=colors(int(group) % 10),
                             marker=('o', '^', 's', 'D')[component % 4])
        center = samples['slope_samples'][example['index']]
        axis.scatter(*center, marker='*', s=170, facecolors='none', edgecolors='black', linewidths=1.2)
        axis.set(xlabel='p', ylabel='q', title=f'{title}\nN={len(ids)} | component sizes={example["sizes"]}\n'
                 f'RGB center={np.round(example["color"], 4)}', aspect='equal')
        axis.invert_yaxis()
        axis.grid(alpha=.2)
        example['mean_slopes'] = [(-np.asarray(normal[:2])/normal[2]).tolist() for normal in example['mean_normals']]
        for component, mean in enumerate(example['mean_slopes'][:2]):
            axis.scatter(*mean, marker='x', color='black', s=65)
            axis.annotate(f'{chr(65+component)}: n={example["sizes"][component]}', mean,
                          xytext=(5, -12), textcoords='offset points', fontsize=9)
        example['component_mean_separations_degrees'] = [float(_angles(np.asarray(example['mean_normals'][i]),
            np.asarray(example['mean_normals'][j]))) for i, j in example['separated_components']]
        example['cross_component_conflicts'] = []
        for first, second in example['separated_components']:
            a, b = ids[components == first], ids[components == second]
            distances = np.linalg.norm(samples['color_residual_samples'][a, None].astype(float)-
                                       samples['color_residual_samples'][b].astype(float), axis=-1)
            angles = _angles(_normals(samples['slope_samples'][a])[:, None],
                             _normals(samples['slope_samples'][b])[None])
            example['cross_component_conflicts'].append(dict(components=[first, second],
                color_close_pairs=int((distances <= example['color_distance_threshold']).sum()),
                conflicting_pairs=int(((distances <= example['color_distance_threshold']) &
                                       (angles > example['normal_angle_threshold_degrees'])).sum())))
        report[kind] = example
    handles = [Line2D([], [], linestyle='', marker='o', color=colors(group % 10),
                      label=f'{group+1}: {samples["image_paths"][group][17:23]}') for group in sorted(used_images)]
    figure.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=9)
    figure.suptitle('Local color balls: color = source image; marker shape = normal component\n'
                   '10 deg component links; black star = query; black crosses = component means; q increases downward')
    _save(figure, output/'pq_neighborhood_examples.png')
    return report


def _spatial_figure(work, samples, frame_paths, calibration, context, regions, bad, patterns, output):
    figure = _figure((17, 7))
    axes = figure.subplots(2, int(np.ceil(len(samples['image_paths'])/2)), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(samples['image_paths']):
            axis.set_visible(False)
            continue
        name = str(samples['image_paths'][index])
        region = regions[name]
        overall_path = work/'overall'/frame_paths[name].name
        if not overall_path.is_file():
            raise ValueError(f'Missing overall geometry cache for image {name}: {overall_path.name}')
        with np.load(overall_path, allow_pickle=False) as frame:
            if str(frame['image_key']) != name:
                raise ValueError(f'Overall geometry cache refers to a different image: {name}')
            _check_declarations(frame, calibration,
                                ('coordinate_kind', 'coordinate_chart_id', 'frame_convention'))
            axis.imshow(cached_linear_rgb(frame), origin='upper', interpolation='nearest')
        points = context['uv'][(samples['image_index'] == index) & bad]
        axis.scatter(points[:, 0], points[:, 1], color='red', s=4, edgecolors='white', linewidths=.12)
        cx, cy = region.center_px
        radius = region.semi_axes_px[0]
        axis.add_patch(Circle((cx, cy), radius, fill=False, color='#20df80', linewidth=.8))
        axis.set(xlim=(cx-1.3*radius, cx+1.3*radius), ylim=(cy+1.3*radius, cy-1.3*radius),
                 title=f'{index+1}: {name[17:23]} | {patterns["images"][index]["ambiguous_sample_fraction"]:.1%}')
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle('Ambiguous samples on original images | red points; green = original contact disk')
    _save(figure, output/'spatial_ambiguities.png')


def _weighted_normals(distances, ids, normals):
    weights = 1/np.maximum(distances, 1e-8)**2
    predicted = np.sum(weights[..., None]*normals[ids], axis=1)
    return predicted/np.maximum(np.linalg.norm(predicted, axis=-1, keepdims=True), 1e-30)


def predict_conditioned_normals(
    train_colors: np.ndarray, train_st: np.ndarray, train_normals: np.ndarray,
    colors: np.ndarray, st: np.ndarray, *, position_weight: float, neighbours: int,
) -> np.ndarray:
    """Experimental inverse-distance normal average in [delta RGB, weight*s,t].

    st is dimensionless frozen-chart position, not a physical distance. Unit
    normals are averaged before renormalization; no slope-component averaging.
    """
    features = np.column_stack((train_colors, position_weight*train_st))
    query = np.column_stack((colors, position_weight*st))
    distances, ids = cKDTree(features).query(query, k=min(neighbours, len(features)))
    if distances.ndim == 1:
        distances, ids = distances[:, None], ids[:, None]
    return _weighted_normals(distances, ids, train_normals)


def _prediction_score(prediction, truth, groups, names):
    angles = _angles(prediction, truth)
    return dict(rms_degrees=float(np.sqrt(np.mean(angles**2))), median_degrees=float(np.median(angles)),
                p95_degrees=float(np.percentile(angles, 95)), fraction_over_20_degrees=float(np.mean(angles > 20)),
                images=[dict(image=str(names[group]), rms_degrees=float(np.sqrt(np.mean(angles[groups == group]**2))))
                        for group in np.unique(groups)])


def _remedy_experiments(work, samples, st, color_threshold, angle_threshold, output):
    colors, slopes, groups, names = (samples[key] for key in
        ('color_residual_samples', 'slope_samples', 'image_index', 'image_paths'))
    normals = _normals(slopes)
    model = NormalCalibration.load(work/'normal_lut_train.npz')
    train = np.isin(names[groups], model.metadata['training_images'])
    held = np.isin(names[groups], model.metadata['validation_images'])
    if not (np.all(train | held) and not np.any(train & held)):
        raise ValueError('Training LUT must declare a disjoint, complete image split')
    weights, ks = (0., .025, .05, .1, .2, .4, .8), (8, 24, 48)
    squared = {(weight, k): 0. for weight in weights for k in ks}
    for group in np.unique(groups[train]):
        fit, query = train & (groups != group), train & (groups == group)
        for weight in weights:
            features = np.column_stack((colors, weight*st))
            distances, ids = cKDTree(features[fit]).query(features[query], k=min(max(ks), int(fit.sum())))
            for k in ks:
                prediction = _weighted_normals(distances[:, :k], ids[:, :k], normals[fit])
                squared[(weight, k)] += float(np.sum(_angles(prediction, normals[query])**2))
    cv = [dict(position_weight=weight, neighbours=k, rms_degrees=float(np.sqrt(value/train.sum())))
          for (weight, k), value in sorted(squared.items(), key=lambda item: item[1])]
    selected = {'color_only_average': next(row for row in cv if row['position_weight'] == 0),
                'position_conditioned_average': next(row for row in cv if row['position_weight'] > 0)}
    predictions = {'current_LUT': _normals(lookup_slopes_raw(colors[held], model)[0])}
    for name, parameters in selected.items():
        predictions[name] = predict_conditioned_normals(colors[train], st[train], normals[train],
            colors[held], st[held], position_weight=parameters['position_weight'], neighbours=parameters['neighbours'])
    # Test indiscriminate removal as a separate, fixed intervention. Only training
    # labels define removal; held-out colors and labels never enter this mask.
    conflict, _, _ = find_ambiguous_color_pairs(colors[train], slopes[train],
        color_distance_threshold=color_threshold, normal_angle_threshold_degrees=angle_threshold)
    keep = np.ones(int(train.sum()), bool)
    keep[conflict.ravel()] = False
    filtered = build_normal_lut(colors[train][keep], slopes[train][keep], size=model.size)
    predictions['LUT_drop_all_ambiguous_training_samples'] = _normals(lookup_slopes_raw(colors[held], filtered)[0])
    scores = {name: _prediction_score(prediction, normals[held], groups[held], names)
              for name, prediction in predictions.items()}
    np.savez_compressed(output/'remedy_predictions.npz', sample_indices=np.flatnonzero(held),
        truth_normals=normals[held], image_index=groups[held], image_paths=names, **predictions)
    parameters = selected['position_conditioned_average']
    np.savez_compressed(output/'experimental_position_model.npz',
        model_kind='experimental_color_position_normal_average', colors=colors[train],
        st=st[train], normals=normals[train], image_index=groups[train], image_paths=names,
        position_weight=parameters['position_weight'], neighbours=parameters['neighbours'],
        coordinate_chart_id=model.metadata['coordinate_chart_id'])
    return dict(training_images=model.metadata['training_images'], validation_images=model.metadata['validation_images'],
        selection='Eight training images only: leave one training image out; minimize pooled angular RMS. '
                  'The two original held-out images are evaluated after selection.',
        normal_average='Inverse squared feature distance; renormalize weighted local normals',
        position_metric='[delta R, delta G, delta B, weight*s, weight*t]; st is dimensionless chart position',
        cross_validation=cv, selected=selected, held_out=scores,
        removed_training_samples=int((~keep).sum()), remaining_training_samples=int(keep.sum()),
        filtered_LUT_options=dict(size=model.size, maximum_rms_angle_degrees=5.),
        status='Exploratory diagnostics only; production LUT and reconstruction policy are unchanged.')


def _remedy_figure(experiments, output):
    figure = _figure((11, 5))
    axis = figure.subplots()
    rows = experiments['held_out']
    labels = ['Current LUT', 'Color-only normal average', 'Color + position', 'Drop ambiguous training samples']
    axis.bar(np.arange(len(rows)), [row['rms_degrees'] for row in rows.values()],
             color=['#777777', '#5e8cba', '#39a06b', '#c83434'])
    axis.set(xticks=np.arange(len(rows)), xticklabels=labels, ylabel='Held-out normal RMS (degrees)',
             title='Fixed held-out images | model parameters selected using training images only')
    axis.tick_params(axis='x', rotation=12)
    for i, row in enumerate(rows.values()):
        axis.text(i, row['rms_degrees']+.2, f'{row["rms_degrees"]:.2f}', ha='center')
    _save(figure, output/'remedy_validation.png')


def _write_point_csv(output, samples, context, bad, near_degree, bad_degree, tilt, classes, expanded_classes):
    with (output/'samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_index', 'image', 'delta_R', 'delta_G', 'delta_B', 'p', 'q', 'tilt_degrees',
                         'pixel_x', 'pixel_y', 'radius_fraction', 'ambiguous', 'near_colour_neighbours',
                         'conflicting_neighbours', 'spatial_normal_residual_degrees', 'neighbourhood_kind',
                         'expanded_branch_classification'])
        for i, (color, slope) in enumerate(zip(samples['color_residual_samples'], samples['slope_samples'], strict=True)):
            writer.writerow([i, samples['image_paths'][samples['image_index'][i]], *color, *slope, tilt[i],
                *context['uv'][i], context['radial_fraction'][i], bool(bad[i]), near_degree[i], bad_degree[i],
                context['spatial_normal_residual_degrees'][i], MODE_NAMES[classes[i]] if bad[i] else 'not_ambiguous',
                BRANCH_NAMES[expanded_classes[i]] if bad[i] else 'not_ambiguous'])


def analyze_ambiguities(work_dir: str | Path, output_dir: str | Path | None = None, *,
                        color_distance_threshold: float = .01,
                        normal_angle_threshold_degrees: float = 20.,
                        dominance_fraction: float = .8) -> Path:
    classify_branch_support(np.empty((0, 2)), dominance_fraction=dominance_fraction)
    work = Path(work_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else work/'ambiguity_analysis'
    if output == work or output in work.parents:
        raise ValueError('Analysis output must not contain calibration inputs')
    if output.exists():
        prior = output/'analysis.json'
        if not prior.is_file() or json.loads(prior.read_text()).get('report_kind') != 'normal_ambiguity_analysis':
            raise ValueError('Analysis output already contains unrelated files')
        generated = {'analysis.json', 'ambiguity_patterns.png', 'pq_neighborhood_examples.png',
                     'spatial_ambiguities.png', 'remedy_validation.png', 'samples.csv',
                     'remedy_predictions.npz', 'experimental_position_model.npz',
                     'branch_continuation.csv', 'branch_continuation.png', 'branch_classification.png',
                     'branch_classification_pairs.csv', 'branch_classification_points.csv'}
        if any(path.name not in generated or not path.is_file() for path in output.iterdir()):
            raise ValueError('Analysis output contains unrelated files')
    sources = [work/'normal_samples.npz', work/'normal_lut.npz', work/'normal_lut_train.npz',
               work/'manual_regions.yaml', *sorted((work/'frames').glob('*.npz')),
               *sorted((work/'overall').glob('*.npz'))]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    model = NormalCalibration.load(work/'normal_lut.npz')
    samples = _load_samples(work/'normal_samples.npz', model)
    frame_paths = _frame_paths(work, samples, model)
    with np.load(work/'normal_samples.npz', allow_pickle=False) as data:
        st = data['st_samples'].astype(float)
    if st.shape != (len(samples['image_index']), 2) or not np.isfinite(st).all():
        raise ValueError('Expected finite Nx2 frozen-chart st positions')
    colors, slopes, groups = (samples[key] for key in ('color_residual_samples', 'slope_samples', 'image_index'))
    pairs, distances, angles = find_ambiguous_color_pairs(colors, slopes,
        color_distance_threshold=color_distance_threshold, normal_angle_threshold_degrees=normal_angle_threshold_degrees)
    bad = np.zeros(len(colors), bool)
    bad[pairs.ravel()] = True
    near_pairs = cKDTree(colors.astype(float)).query_pairs(color_distance_threshold, output_type='ndarray')
    context, regions = _spatial_context(work, samples, frame_paths)
    patterns, near_degree, bad_degree, tilt = _pattern_statistics(samples, context, bad, pairs, near_pairs)
    classes, counts, sensitivity, examples = _neighborhoods(colors, _normals(slopes), groups, bad,
                                                           color_distance_threshold, normal_angle_threshold_degrees)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ambiguity-analysis-', dir=output.parent) as directory:
        staging = Path(directory)/'report'
        staging.mkdir()
        _patterns_figure(colors, bad, patterns, staging)
        _spatial_figure(work, samples, frame_paths, model, context, regions, bad, patterns, staging)
        for example in examples.values():
            example.update(color_distance_threshold=color_distance_threshold,
                           normal_angle_threshold_degrees=normal_angle_threshold_degrees)
        examples = _examples_figure(samples, examples, staging)
        continuation, expanded_classes = _continuation_analysis(samples, pairs, near_pairs, color_distance_threshold,
            normal_angle_threshold_degrees, staging, classes, dominance_fraction)
        _continuation_figure(samples, continuation, staging)
        _classification_figure(samples, expanded_classes, continuation['expanded_classification'], staging)
        experiments = _remedy_experiments(work, samples, st, color_distance_threshold,
                                         normal_angle_threshold_degrees, staging)
        _remedy_figure(experiments, staging)
        _write_point_csv(staging, samples, context, bad, near_degree, bad_degree, tilt, classes, expanded_classes)
        report = dict(report_kind='normal_ambiguity_analysis', source_hashes=hashes,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            definitions=dict(color_distance_threshold=color_distance_threshold,
                normal_angle_threshold_degrees=normal_angle_threshold_degrees,
                pairs='All unordered pairs, within and across images; both endpoints marked',
                local_neighbourhood='All samples within RGB L2 radius of each ambiguous center; balls overlap.',
                modes='Normal graph edges <=10 deg; separated means >20 deg (or supplied angle threshold). '
                      'Each supported component has >=3 samples. Sparse means fewer than 6 samples only '
                      'inside the original color ball; it never stops the outer continuation analysis.',
                spatial_smoothness='Angle to normalized sum of valid 8-neighbour local normals on the material grid.',
                limitation='Sphere labels are model-derived, not independent physical ground truth. '
                           'No repeated-capture noise estimate; close colors need not be physically identical.'),
            sample_count=len(colors), ambiguous_sample_count=int(bad.sum()), near_pair_count=len(near_pairs),
            near_cross_image_pair_count=int(np.sum(groups[near_pairs[:, 0]] != groups[near_pairs[:, 1]])),
            ambiguous_pair_count=len(pairs), cross_image_pair_count=int(np.sum(groups[pairs[:, 0]] != groups[pairs[:, 1]])),
            conflicting_pair_angles_degrees=_distribution(angles), patterns=patterns,
            neighbourhood_classification=counts, classification_link_angle_sensitivity=sensitivity,
            branch_continuation=continuation,
            examples=examples, remedy_experiments=experiments)
        for path, digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Calibration input changed during analysis: {path}')
        (staging/'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        if output.exists():
            output.replace(Path(directory)/'previous')
        try:
            staging.replace(output)
        except OSError:
            previous = Path(directory)/'previous'
            if previous.exists():
                previous.replace(output)
            raise
    return output
