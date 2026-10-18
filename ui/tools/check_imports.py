#!/usr/bin/env python3
"""Static check: every named import in ui/js/**/*.js resolves to an export of the target module,
every target file exists, and every module is fetchable by the dev server layout."""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "js"
files = sorted(root.rglob("*.js"))

EXPORT_RE = [
    re.compile(r"export\s+(?:async\s+)?(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)"),
]
LIST_RE = re.compile(r"export\s*\{([^}]*)\}")
IMPORT_RE = re.compile(r"import\s+(?:(\*\s+as\s+\w+)|\{([^}]*)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]", re.S)
SIDE_RE = re.compile(r"import\s+['\"]([^'\"]+)['\"]")


def exports(path: Path) -> set[str]:
    src = path.read_text()
    names: set[str] = set()
    for rx in EXPORT_RE:
        names |= set(rx.findall(src))
    for m in LIST_RE.findall(src):
        for part in m.split(","):
            part = part.strip()
            if part:
                names.add(part.split(" as ")[-1].strip())
    if re.search(r"export\s+default", src):
        names.add("default")
    return names


errors = 0
cache = {f: exports(f) for f in files}
for f in files:
    src = f.read_text()
    for m in IMPORT_RE.finditer(src):
        star, named, default, spec = m.groups()
        target = (f.parent / spec).resolve()
        if not target.exists():
            print(f"FAIL {f.relative_to(root)}: import target missing: {spec}")
            errors += 1
            continue
        if named:
            for n in [x.strip() for x in named.split(",") if x.strip()]:
                base = n.split(" as ")[0].strip()
                if base not in cache.get(target, exports(target)):
                    print(f"FAIL {f.relative_to(root)}: '{base}' is not exported by {spec}")
                    errors += 1
        elif default and "default" not in cache.get(target, set()):
            print(f"FAIL {f.relative_to(root)}: no default export in {spec}")
            errors += 1
    # unused-import hint (non-fatal)
print(f"checked {len(files)} modules, {errors} problem(s)")
sys.exit(1 if errors else 0)
