from pathlib import Path

import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


def hex_to_rgba(hex_color: str) -> list[int]:
    value = hex_color.strip().lstrip("#")

    if len(value) == 6:
        value += "FF"

    if len(value) != 8:
        raise ValueError("Use #RRGGBB or #RRGGBBAA.")

    return [
        int(value[index:index + 2], 16)
        for index in range(0, 8, 2)
    ]


def create_powerpoint_glb(
    input_stl: str,
    output_glb: str,
    color: str = "#0080FF",
) -> None:
    mesh = trimesh.load_mesh(input_stl, process=True)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("The STL does not contain a valid mesh.")

    rgba = hex_to_rgba(color)

    # Create a real glTF PBR material rather than vertex colors.
    material = PBRMaterial(
        name="PowerPointMaterial",
        baseColorFactor=rgba,
        metallicFactor=0.0,
        roughnessFactor=0.75,
        doubleSided=True,
        alphaMode="OPAQUE",
    )

    mesh.visual = TextureVisuals(material=material)

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="ColoredMesh")

    output_path = Path(output_glb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)

    print(f"Created: {output_path.resolve()}")


if __name__ == "__main__":
    create_powerpoint_glb(
        input_stl="36.stl",
        output_glb="model_powerpoint.glb",
        color="#0080FF",
    )
