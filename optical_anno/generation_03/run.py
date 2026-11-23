"""按每张图的压入前曲面估计底色，再用球面求交器标注。"""

from optical_anno.generation_02.run import main as run_sphere_annotation
from optical_anno.generation_03.background import fit_material_background
from optical_anno.generation_03.evidence import sphere_evidence


def main() -> None:
    run_sphere_annotation(material_background_factory=fit_material_background,
                          evidence_factory=sphere_evidence,
                          generation='generation_03')


if __name__ == '__main__':
    main()
