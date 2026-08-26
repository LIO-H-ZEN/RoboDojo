from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from env.global_configs import ASSETS_PATH, BENCHMARK
from utils.load_file import *


class SeedManager:
    def __init__(self, config: Mapping[str, Any]):
        self.config: Mapping[str, Any] = config
        self.num_envs: int = int(self.config["num_envs"])

        # config fields used for directory layout
        self.task_name: str = str(self.config["task_name"])
        self.config_name: str = str(self.config["config_name"])
        self.layout_config_name: str = str(self.config.get("layout_config_name", self.config_name))
        self.layout_task_name: str = str(self.config.get("layout_task_name", self.task_name))
        self.layout_filter = self.config.get("layout_filter")
        self.layout_overrides = self.config.get("layout_overrides")

        self.st_idx: int
        self.ed_idx: int
        self.type: str

        self._current_batch_seeds: List[int] | None = None

    def init_eval(
        self,
        completed_layout_ids: Iterable[int] | None = None,
        abandoned_layout_ids: Iterable[int] | None = None,
    ):
        self.eval_seed = self.config.get("seed", 0)
        layout_dir = Path(ASSETS_PATH, "Eval_Layout", BENCHMARK, self.layout_config_name, str(self.eval_seed))
        pattern = re.compile(rf"{re.escape(self.layout_task_name)}_\d+\.json")
        matching_files = sorted(
            [p for p in layout_dir.iterdir() if pattern.fullmatch(p.name)],
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        matching_files = [path for path in matching_files if self._layout_allowed(load_json(path))]

        matching_files = [str(p) for p in matching_files]
        self.seed_info = {}
        for idx, file_path in enumerate(matching_files):
            self.seed_info[idx] = {"scene_layout": file_path}

        all_layout_ids = list(range(len(matching_files)))
        excluded = set(int(s) for s in (completed_layout_ids or [])) | set(int(s) for s in (abandoned_layout_ids or []))
        if excluded:
            self.seed_list: List[int] = [s for s in all_layout_ids if s not in excluded]
            print(
                f"[SeedManager] init_eval resume filter: excluded={len(excluded)} "
                f"remaining={len(self.seed_list)}/{len(all_layout_ids)}"
            )
        else:
            self.seed_list = all_layout_ids
        self.st_idx = 0
        self.ed_idx = len(self.seed_list)

        self.type = "eval"
        self.idx = 0
        self._current_batch_seeds = None

    def get_seeds(self, max_count: int | None = None) -> List[int] | None:
        """Return a list of seeds for the next `reset()` call.

        Returns None when enough episodes have been successfully collected.
        """

        if self.idx >= self.ed_idx:
            return None
        if max_count is not None:
            batch_size = min(self.num_envs, max(0, int(max_count)))
            if batch_size == 0:
                return None
            batch = self.seed_list[self.idx : min(self.idx + batch_size, self.ed_idx)]
            self.idx += len(batch)
            self._current_batch_seeds = batch
            return batch
        if self.idx + self.num_envs > self.ed_idx:
            batch = self.seed_list[self.idx : self.ed_idx]
            result = deepcopy(batch)
            for _ in range(self.num_envs - len(result)):
                batch.append(self.seed_list[self.ed_idx - 1])  # pad with last seed if not enough remaining
        else:
            batch = self.seed_list[self.idx : self.idx + self.num_envs]
        self.idx += self.num_envs
        self._current_batch_seeds = batch
        return batch

    def get_seed_scene_info(self, seed: int) -> Dict[str, Any]:
        seed_info = self.seed_info.get(seed)
        if seed_info is None:
            raise ValueError(f"Seed {seed} not found in seed list.")
        file_path = seed_info.get("scene_layout")
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"Scene layout file not found for seed {seed} at expected path {file_path}.")
        data = load_json(file_path)
        self._apply_layout_overrides(data)
        return data

    def _layout_allowed(self, data: Mapping[str, Any]) -> bool:
        if self.layout_filter is None:
            return True
        unknown = set(self.layout_filter) - {"target_position"}
        if unknown:
            raise ValueError(f"Unsupported layout filters: {sorted(unknown)}")
        target_filter = self.layout_filter.get("target_position")
        if target_filter is None:
            return True
        label = str(target_filter.get("label", "target"))
        matches = [
            instance
            for object_type in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
            for instances in data.get(object_type, {}).values()
            for instance in instances
            if instance.get("label") == label
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one layout object labeled {label!r}, got {len(matches)}")
        position = matches[0].get("default_pos")
        if not isinstance(position, Sequence) or len(position) < 2:
            raise ValueError(f"Layout target {label!r} has invalid default_pos: {position}")
        xlim = target_filter["xlim"]
        ylim = target_filter["ylim"]
        return float(xlim[0]) <= float(position[0]) <= float(xlim[1]) and float(ylim[0]) <= float(
            position[1]
        ) <= float(ylim[1])

    def _apply_layout_overrides(self, data: dict[str, Any]) -> None:
        if self.layout_overrides is None:
            return
        for object_type, categories in self.layout_overrides.items():
            if object_type not in data:
                raise ValueError(f"Layout override type is absent: {object_type}")
            for category, fields in categories.items():
                instances = data[object_type].get(category)
                if not instances:
                    raise ValueError(f"Layout override category is absent: {object_type}.{category}")
                for instance in instances:
                    instance.update(deepcopy(dict(fields)))

    def eval_step(self):
        self._current_batch_seeds = None
