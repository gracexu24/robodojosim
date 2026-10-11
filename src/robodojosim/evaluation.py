from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationSpec:
    xpolicylab: Path
    checkpoint: Path
    output: Path
    episodes: int = 10
    seed: int = 0
    policy_gpu: int = 0
    environment_gpu: int = 0
    policy_env: str = "RoboDojoLeRobotTrain"
    eval_env: str = "RoboDojo"
    gui: bool = False

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if self.policy_gpu < 0 or self.environment_gpu < 0:
            raise ValueError("GPU indices must be non-negative")


def build_evaluation_command(spec: EvaluationSpec) -> list[str]:
    script = spec.xpolicylab.resolve() / "policy" / "bottle_lerobot" / "eval.sh"
    return [
        "bash",
        str(script),
        "RoboDojo",
        "put_bottles_into_dustbin",
        str(spec.checkpoint.resolve()),
        "arx_x5",
        "ee",
        str(spec.seed),
        str(spec.policy_gpu),
        str(spec.environment_gpu),
        spec.policy_env,
        spec.eval_env,
    ]


def run_evaluation(spec: EvaluationSpec, *, dry_run: bool = False) -> dict[str, Any]:
    command = build_evaluation_command(spec)
    script = Path(command[1])
    if not dry_run and not script.is_file():
        raise FileNotFoundError(f"learned-policy adapter is not installed: {script}")
    if not spec.checkpoint.resolve().is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {spec.checkpoint.resolve()}")
    environment = os.environ.copy()
    environment.update(
        {
            "EVAL_NUM": str(spec.episodes),
            "ROBODOJOSIM_EVAL_OUTPUT": str(spec.output.resolve()),
            "ROBODOJO_HEADLESS": "0" if spec.gui else "1",
        }
    )
    result: dict[str, Any] = {
        "command": command,
        "episodes": spec.episodes,
        "checkpoint": str(spec.checkpoint.resolve()),
        "output": str(spec.output.resolve()),
        "gui": spec.gui,
        "status": "planned" if dry_run else "running",
    }
    if dry_run:
        return result
    spec.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, env=environment, check=True)
    result["status"] = "complete"
    result["report"] = summarize_evaluation(spec.output)
    return result


def summarize_evaluation(output: str | Path) -> dict[str, Any]:
    output = Path(output).resolve()
    records = []
    for path in sorted(output.glob("episode_*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation record {path}: {exc}") from exc
    if not records:
        raise ValueError(f"no episode_*.json evaluation records found in {output}")
    successes = sum(bool(item.get("success")) for item in records)
    steps = [int(item.get("steps", 0)) for item in records]
    report = {
        "format_version": 1,
        "episodes": len(records),
        "successful_episodes": successes,
        "failed_episodes": len(records) - successes,
        "success_rate": successes / len(records),
        "mean_steps": sum(steps) / len(steps),
        "records": records,
    }
    _write_json(output / "metrics.json", report)
    _write_csv(output / "episodes.csv", records)
    (output / "summary.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# RoboDojo learned-policy evaluation",
        "",
        f"- Episodes: {report['episodes']}",
        f"- Successful: {report['successful_episodes']}",
        f"- Failed: {report['failed_episodes']}",
        f"- Success rate: {report['success_rate']:.1%}",
        f"- Mean control steps: {report['mean_steps']:.1f}",
        "",
        "| Episode | Layout seed | Success | Steps | Reason |",
        "|---:|---:|:---:|---:|---|",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record.get('episode_id', '')} | {record.get('layout_seed', '')} | "
            f"{'yes' if record.get('success') else 'no'} | {record.get('steps', '')} | "
            f"{record.get('termination_reason', '')} |"
        )
    return "\n".join(lines) + "\n"
