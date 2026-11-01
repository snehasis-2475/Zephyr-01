from pathlib import Path

path = Path("D:/ML/Zephyr-01/engines/version2_triposr/utils.py")
code = path.read_text(encoding="utf-8")

top_target = (
    "import importlib\n"
    "import math\n"
    "from collections import defaultdict\n"
    "from dataclasses import dataclass\n"
    "from typing import Any, Callable, Dict, List, Optional, Tuple, Union"
)

top_replacement = (
    "import importlib\n"
    "import math\n"
    "import os\n"
    "import sys\n"
    "from collections import defaultdict\n"
    "from dataclasses import dataclass\n"
    "from typing import Any, Callable, Dict, List, Optional, Tuple, Union\n\n"
    "# Ensure the engine directory is on sys.path so flat imports resolve cleanly\n"
    "CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))\n"
    "if CURRENT_DIR not in sys.path:\n"
    "    sys.path.insert(0, CURRENT_DIR)\n\n"
    "FLAT_MODULE_MAP = {\n"
    '    "tsr.models.tokenizers.image": "image",\n'
    '    "tsr.models.tokenizers.triplane": "triplane",\n'
    '    "tsr.models.transformer.transformer_1d": "transformer_1d",\n'
    '    "tsr.models.network_utils": "network_utils",\n'
    '    "tsr.models.nerf_renderer": "nerf_renderer",\n'
    '    "tsr.models.isosurface": "isosurface",\n'
    "}"
)

old_find_class = (
    "def find_class(cls_string):\n"
    '    module_string = ".".join(cls_string.split(".")[:-1])\n'
    '    cls_name = cls_string.split(".")[-1]\n'
    "    module = importlib.import_module(module_string, package=None)\n"
    "    cls = getattr(module, cls_name)\n"
    "    return cls"
)

new_find_class = (
    "def find_class(cls_string: str):\n"
    '    module_string = ".".join(cls_string.split(".")[:-1])\n'
    '    cls_name = cls_string.split(".")[-1]\n\n'
    "    if module_string in FLAT_MODULE_MAP:\n"
    "        module_string = FLAT_MODULE_MAP[module_string]\n"
    '    elif module_string.startswith("tsr."):\n'
    '        module_string = module_string.split(".")[-1]\n\n'
    "    module = importlib.import_module(module_string, package=None)\n"
    "    cls = getattr(module, cls_name)\n"
    "    return cls"
)

assert top_target in code, "top_target not found in utils.py"
assert old_find_class in code, "old_find_class not found in utils.py"

code = code.replace(top_target, top_replacement).replace(old_find_class, new_find_class)
path.write_text(code, encoding="utf-8")
print("utils.py successfully patched!")
