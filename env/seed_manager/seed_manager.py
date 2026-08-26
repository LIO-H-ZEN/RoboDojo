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
        self.seed_info = {layout_id: {"scene_layout": file_path} for layout_id, file_path in enumerate(matching_files)}

        all_layout_ids = list(range(len(matching_files)))
        requested_layout_ids = self.config.get("layout_ids")
        if requested_layout_ids is not None:
            if isinstance(requested_layout_ids, (str, bytes)) or not isinstance(requested_layout_ids, Sequence):
                raise ValueError("layout_ids must be a sequence of integers")
            selected_layout_ids = [int(layout_id) for layout_id in requested_layout_ids]
            if not selected_layout_ids:
                raise ValueError("layout_ids must not be empty")
            if len(set(selected_layout_ids)) != len(selected_layout_ids):
                raise ValueError(f"layout_ids must be unique: {selected_layout_ids}")
            invalid_layout_ids = [layout_id for layout_id in selected_layout_ids if layout_id not in self.seed_info]
            if invalid_layout_ids:
                raise ValueError(
                    f"layout_ids out of range: {invalid_layout_ids}; available=0..{len(matching_files) - 1}"
                )
            all_layout_ids = selected_layout_ids
            print(f"[SeedManager] init_eval layout shard: layout_ids={all_layout_ids}")

        excluded = set(int(s) for s in (completed_layout_ids or [])) | set(int(s) for s in (abandoned_layout_ids or []))
        if excluded:
            self.seed_list: List[int] = [s for s in all_layout_ids if s not in excluded]
            print(
                f"[SeedManager] init_eval resume filter: excluded={len(excluded)} "
                f"remaining={len(self.seed_list)}/{len(all_layout_ids)}"
            )
        else:
            self.seed_list = all_layout_ids
        if bool(self.config.get("validate_layout_assets", False)):
            validation_count = min(int(self.config.get("eval_num", len(self.seed_list))), len(self.seed_list))
            self._validate_layout_assets(self.seed_list[:validation_count])
        self.st_idx = 0
        self.ed_idx = len(self.seed_list)

        self.type = "eval"
        self.idx = 0
        self._current_batch_seeds = None

    def _validate_layout_assets(self, layout_ids: Sequence[int]) -> None:
        checked_instances = 0
        for layout_id in layout_ids:
            layout = self.get_seed_scene_info(layout_id)
            for object_type in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"):
                for category, instances in layout.get(object_type, {}).items():
                    for instance in instances:
                        category_idx = int(instance["category_idx"])
                        asset_type = "Clutter" if instance.get("type") == "cluttered" else object_type
                        model_dir = Path(
                            ASSETS_PATH,
                            "Object",
                            BENCHMARK,
                            asset_type,
                            category,
                        )
                        require_object_metadata(model_dir, category_idx)
                        asset_dir = model_dir / f"{category_idx:05d}"
                        usd_path = asset_dir / "object.usdz"
                        if not usd_path.is_file():
                            usd_path = asset_dir / "object.usd"
                        if not usd_path.is_file():
                            raise FileNotFoundError(f"Object USD is missing for layout_id={layout_id}: {asset_dir}")
                        if is_git_lfs_pointer(usd_path):
                            raise ValueError(
                                f"Object USD is an unresolved Git LFS pointer for layout_id={layout_id}: {usd_path}"
                            )
                        checked_instances += 1
        print(f"[SeedManager] validated layout assets: layouts={list(layout_ids)} instances={checked_instances}")

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
        return float(xlim[0]) <= float(position[0]) <= float(xlim[1]) and float(ylim[0]) <= float(position[1]) <= float(
            ylim[1]
        )

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
