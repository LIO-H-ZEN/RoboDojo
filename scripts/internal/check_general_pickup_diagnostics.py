#!/usr/bin/env python3
"""Fast pass/fail check for a captured General Pickup rollout result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_result(path: Path, *, layout_id: int | None, required_lift_m: float) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("details")
    if not isinstance(details, dict) or not details:
        raise ValueError(f"result has no episode details: {path}")

    selected = []
    for detail in details.values():
        if not isinstance(detail, dict):
            raise ValueError(f"invalid episode detail in {path}: {detail!r}")
        if layout_id is None or int(detail.get("layout_id", -1)) == layout_id:
            selected.append(detail)
    if not selected:
        raise ValueError(f"layout {layout_id} is absent from {path}")

    failures = []
    for detail in selected:
        diagnostics = detail.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"layout {detail.get('layout_id')} has no diagnostics")
        lift = float(diagnostics["max_lift_height_m"])
        distance = float(diagnostics["min_tcp_target_distance_m"])
        opening = float(diagnostics["min_gripper_opening_m"])
        signal = diagnostics["failure_signal"]
        closed_step = diagnostics.get("first_closed_step")
        close_distance = diagnostics.get("tcp_target_distance_at_first_close_m")
        print(
            f"layout={detail['layout_id']} success={detail['success']} "
            f"max_lift_m={lift:.6f} min_tcp_distance_m={distance:.6f} "
            f"min_gripper_opening_m={opening:.6f} first_closed_step={closed_step} "
            f"tcp_distance_at_first_close_m={close_distance} signal={signal}"
        )
        if not bool(detail["success"]) or lift < required_lift_m:
            failures.append(
                f"layout {detail['layout_id']} failed: success={detail['success']} "
                f"max_lift_m={lift:.6f} required={required_lift_m:.6f} signal={signal}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--layout-id", type=int)
    parser.add_argument("--required-lift-m", type=float, default=0.1)
    args = parser.parse_args()
    if args.required_lift_m <= 0:
        parser.error("--required-lift-m must be positive")
    failures = check_result(args.result, layout_id=args.layout_id, required_lift_m=args.required_lift_m)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: all selected episodes reached the required lift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
