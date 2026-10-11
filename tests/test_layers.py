"""The five-layer rule, checked statically: imports only point down, and the bottom two layers
never import torch (or anything else that would make `pip install laterank` heavy)."""

import ast
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "laterank"

# Top to bottom. A module may import from its own layer or any layer below it.
LAYERS = ["api", "orchestration", "components", "storage", "kernels"]
# Third-party packages the numpy/scipy-only layers may use.
LIGHT_DEPS = {"numpy", "scipy", "Stemmer", "tqdm"}
LIGHT_LAYERS = {"storage", "kernels"}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against the module's own package
                base = path.relative_to(PKG.parent).with_suffix("").parts[: -node.level]
                names.append(".".join([*base, node.module] if node.module else base))
            else:
                names.append(node.module)
    return names


def _modules():
    for layer in LAYERS:
        for path in sorted((PKG / layer).rglob("*.py")) if (PKG / layer).is_dir() else []:
            yield pytest.param(layer, path, id=str(path.relative_to(PKG)))


@pytest.mark.parametrize(("layer", "path"), list(_modules()))
def test_imports_point_down(layer, path):
    allowed = set(LAYERS[LAYERS.index(layer) :])
    for name in _imports(path):
        parts = name.split(".")
        if parts[0] == "laterank" and len(parts) > 1 and parts[1] in LAYERS:
            assert parts[1] in allowed, f"{path.name}: layer '{layer}' imports up into '{name}'"


@pytest.mark.parametrize(("layer", "path"), [p for p in _modules() if p.values[0] in LIGHT_LAYERS])
def test_bottom_layers_stay_light(layer, path):
    stdlib = sys.stdlib_module_names
    for name in _imports(path):
        top, *rest = name.split(".")
        if top == "laterank":  # not `laterank` itself: its __init__ will import the api layer
            assert rest and rest[0] in LIGHT_LAYERS | {"errors", "_version"}, (
                f"{path.name}: '{layer}' may only import kernels/storage/errors/_version, "
                f"but imports '{name}'"
            )
            continue
        assert top in stdlib or top in LIGHT_DEPS, (
            f"{path.name}: '{layer}' may only use numpy/scipy/stdlib, but imports '{name}'"
        )


# Leaf modules at the package root that every layer may import: they import only the stdlib.
LEAF_MODULES = ["errors.py", "_version.py"]


@pytest.mark.parametrize("name", LEAF_MODULES)
def test_leaf_modules_import_only_the_stdlib(name):
    for imported in _imports(PKG / name):
        top = imported.split(".")[0]
        assert top in sys.stdlib_module_names, f"{name} must stay stdlib-only, imports '{imported}'"
