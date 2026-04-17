#!/usr/bin/env python3
"""Audit public markdown docs for broken local links and stale import examples."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from chromecrispr_repo import repo_root, utc_timestamp, write_json, write_text

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FROM_IMPORT_RE = re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+([^\n#]+)", re.MULTILINE)
PLAIN_IMPORT_RE = re.compile(r"^\s*import\s+([\w\.]+)", re.MULTILINE)
REPO_IMPORT_PREFIXES = ("src", "models", "docs", "workflow", "scripts")


def resolve_module_path(root: Path, module_name: str) -> Path | None:
    parts = module_name.split(".")
    base = root.joinpath(*parts)
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    return None


def exported(source: str, symbol: str) -> bool:
    token = symbol.strip()
    if not token or token == "*":
        return True
    return re.search(rf"\b{re.escape(token)}\b", source) is not None


def audit_docs(doc_paths: List[str]) -> dict:
    root = repo_root()
    issues = []

    for doc in doc_paths:
        path = root / doc
        if not path.exists():
            issues.append({"code": "missing_doc", "doc": doc, "message": f"Missing audit target {doc}."})
            continue
        content = path.read_text(encoding="utf-8")

        for raw_link in LINK_RE.findall(content):
            link = raw_link.split("#", 1)[0].strip()
            if not link or link.startswith(("http://", "https://", "mailto:")):
                continue
            if (path.parent / link).exists():
                continue
            issues.append({
                "code": "broken_local_link",
                "doc": doc,
                "message": f"Broken local link `{raw_link}` in {doc}.",
            })

        for module_name, names in FROM_IMPORT_RE.findall(content):
            if not module_name.startswith(REPO_IMPORT_PREFIXES):
                continue
            module_path = resolve_module_path(root, module_name)
            if module_path is None:
                issues.append({
                    "code": "missing_module",
                    "doc": doc,
                    "message": f"Import references missing module `{module_name}` in {doc}.",
                })
                continue
            source = module_path.read_text(encoding="utf-8")
            for name in [item.strip() for item in names.split(",")]:
                if not exported(source, name):
                    issues.append({
                        "code": "missing_symbol",
                        "doc": doc,
                        "message": f"Import references `{name}` from `{module_name}`, but the symbol is not exported in {module_path.relative_to(root)}.",
                    })

        for module_name in PLAIN_IMPORT_RE.findall(content):
            if not module_name.startswith(REPO_IMPORT_PREFIXES):
                continue
            if resolve_module_path(root, module_name) is None:
                issues.append({
                    "code": "missing_module",
                    "doc": doc,
                    "message": f"Import references missing module `{module_name}` in {doc}.",
                })

    return {
        "generated_at": utc_timestamp(),
        "docs_audited": doc_paths,
        "audit_passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# ChromeCRISPR Public Examples Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"- Docs audited: `{len(report['docs_audited'])}`",
        f"- Audit passed: `{str(report['audit_passed']).lower()}`",
        f"- Issue count: `{report['issue_count']}`",
        "",
    ]
    if report["issues"]:
        lines.extend(["## Findings", ""])
        for issue in report["issues"]:
            lines.append(f"- `{issue['code']}`: {issue['message']}")
    else:
        lines.extend(["## Status", "", "- Public markdown links and import examples resolved successfully."])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--docs", nargs="+", required=True)
    args = parser.parse_args()

    report = audit_docs(args.docs)
    write_json(Path(args.json_out), report)
    write_text(Path(args.md_out), render_markdown(report))


if __name__ == "__main__":
    main()
