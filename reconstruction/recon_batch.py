"""Reconstruct one image with all five normal models into one batch per output kind."""
from __future__ import annotations

import argparse
from pathlib import Path

from reconstruction.local_recon import (
    LocalReconstructionResult, default_reconstruction_output, save_reconstruction_result,
)
from reconstruction.pipeline import DEFAULT_CONFIG_PATH, reconstruct_image
from reconstruction.recon import PREVIEW_DIRECTORY, add_reconstruction_arguments
from utils.calibration_store import METHODS, CalibrationArtifacts, lut_directory, resolve_artifacts
from utils.config import load_config
from utils.normal_models import load_normal_model


def load_batch_calibrations(config_path: Path) -> dict[str, CalibrationArtifacts]:
    """Select the newest saved model/grid bundle per method, including candidates."""
    root = lut_directory(config_path)
    calibrations = {}
    for method in METHODS:
        paths = sorted((root / method).glob('*/normal_lut.npz'), reverse=True)
        model_path = next((path for path in paths
                           if (path.parent / 'lut_material_grid.npz').is_file()), None)
        if model_path is None:
            raise FileNotFoundError(f'{method} 需要完整的模型和材料网格：{root / method}')
        artifacts = resolve_artifacts(config_path, model_override=model_path)
        if artifacts.method != method:
            raise ValueError(f'{method} 版本目录中的模型声明为 {artifacts.method}：{model_path}')
        model = load_normal_model(artifacts.model_file)
        actual_method = getattr(model, 'method', 'table')
        if actual_method != method:
            raise ValueError(f'{method} 版本目录中的模型类型为 {actual_method}：{model_path}')
        status = model.metadata.get('validation_status', 'unknown')
        print(f'{method}: {model_path.parent.name}，validation_status={status}', flush=True)
        calibrations[method] = artifacts
    return calibrations


def _reference_path(config_path: Path, raw: dict, reference_image: str | Path | None) -> Path:
    reference = Path(reference_image if reference_image is not None else raw['reference_image']).expanduser()
    if reference_image is None and not reference.is_absolute():
        reference = config_path.parent / reference
    return reference.resolve()


def _validate_outputs(paths: tuple[Path, ...], protected: set[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('重建结果和预览图片需要使用不同的输出路径')
    for path, absolute in zip(paths, resolved, strict=True):
        if absolute in protected:
            raise ValueError(f'请为批量重建选择与输入、模型及网格分开的输出路径：{path}')
        if path.is_dir():
            raise IsADirectoryError(path)
        if path.parent.exists() and not path.parent.is_dir():
            raise NotADirectoryError(path.parent)


def reconstruct_with_calibration(
    image_path: str | Path,
    config_path: str | Path,
    artifacts: CalibrationArtifacts,
    *,
    output: Path,
    preview_output: Path,
    reference_image: str | Path | None = None,
    show: bool = False,
    lsmr_atol: float | None = None,
    lsmr_btol: float | None = None,
    lsmr_max_iterations: int | None = None,
) -> tuple[LocalReconstructionResult, Path]:
    """Reconstruct and save one image with one model/grid pair and explicit outputs."""
    image_path = Path(image_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    output = Path(output).expanduser()
    preview_output = Path(preview_output).expanduser()
    if artifacts.grid_file is None:
        raise ValueError('所选标定模型缺少配套材料网格；请提供 --material-grid')
    raw = load_config(config_path).get('reconstruction')
    if not isinstance(raw, dict):
        raise ValueError('config.yaml 缺少 reconstruction 配置段')
    protected = {
        image_path, config_path, _reference_path(config_path, raw, reference_image),
        artifacts.model_file.resolve(), artifacts.grid_file.resolve(),
    }
    _validate_outputs((output, preview_output), protected)
    result, result_output = reconstruct_image(
        image_path, config_path, reference_image=reference_image,
        material_grid_file=str(artifacts.grid_file), calibration_file=str(artifacts.model_file),
        output=str(output), lsmr_atol=lsmr_atol, lsmr_btol=lsmr_btol,
        lsmr_max_iterations=lsmr_max_iterations,
    )
    save_reconstruction_result(result, result_output, preview_output=preview_output, show=show)
    return result, result_output


def reconstruct_batch(
    image_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    reference_image: str | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
    lsmr_atol: float | None = None,
    lsmr_btol: float | None = None,
    lsmr_max_iterations: int | None = None,
) -> dict[str, Path]:
    """Save one NPZ and one PNG per method, using the same pipeline as recon.

    Defaults use reconstruction.output_dir/pc/<image>_batch and the existing
    preview root's <image>_batch. An explicit output_dir replaces the NPZ batch
    directory. Existing files of the same name are replaced, matching recon.
    """
    image_path = Path(image_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    raw = config.get('reconstruction')
    if not isinstance(raw, dict):
        raise ValueError('config.yaml 缺少 reconstruction 配置段')
    batch_name = f'{image_path.stem}_batch'
    pc_directory = (Path(output_dir).expanduser().resolve() if output_dir is not None else
                    default_reconstruction_output(config_path, raw.get('output_dir'), batch_name))
    pics_directory = PREVIEW_DIRECTORY / batch_name
    for directory in (pc_directory, pics_directory):
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(directory)
    calibrations = load_batch_calibrations(config_path)
    protected = {image_path, _reference_path(config_path, raw, reference_image), config_path}
    for artifacts in calibrations.values():
        protected.add(artifacts.model_file.resolve())
        if artifacts.grid_file is not None:
            protected.add(artifacts.grid_file.resolve())
    outputs = {method: pc_directory / f'{method}.npz' for method in METHODS}
    previews = {method: pics_directory / f'{method}.png' for method in METHODS}
    _validate_outputs((*outputs.values(), *previews.values()), protected)

    for method, artifacts in calibrations.items():
        print(f'{image_path.name} / {method}: reconstruct', flush=True)
        reconstruct_with_calibration(
            image_path, config_path, artifacts, output=outputs[method],
            preview_output=previews[method], reference_image=reference_image, show=show,
            lsmr_atol=lsmr_atol, lsmr_btol=lsmr_btol, lsmr_max_iterations=lsmr_max_iterations,
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description='单张图片使用五种 LUT 分别重建并保存为同一批次')
    add_reconstruction_arguments(parser)
    parser.add_argument('--output', help='批次 NPZ 目录；默认 reconstruction.output_dir/pc/<图片名>_batch')
    args = parser.parse_args()
    reconstruct_batch(
        args.image, args.config, reference_image=args.reference_image, output_dir=args.output,
        show=args.show, lsmr_atol=args.lsmr_atol, lsmr_btol=args.lsmr_btol,
        lsmr_max_iterations=args.lsmr_max_iterations,
    )


if __name__ == '__main__':
    main()
