from collections.abc import MutableMapping
from typing import Any

from omegaconf import OmegaConf


def apply_layout_shard(
    *,
    env_cfg: Any,
    eval_cfg: MutableMapping[str, Any],
    eval_num: int,
    raw_layout_ids: str,
) -> tuple[list[int], int]:
    layout_id_tokens = [token.strip() for token in raw_layout_ids.split(",")]
    if not layout_id_tokens or any(not token for token in layout_id_tokens):
        raise ValueError(f"Invalid EVAL_LAYOUT_IDS: {raw_layout_ids!r}")
    layout_ids = [int(token) for token in layout_id_tokens]
    if len(set(layout_ids)) != len(layout_ids):
        raise ValueError(f"EVAL_LAYOUT_IDS must contain unique IDs: {layout_ids}")

    shard_eval_num = min(int(eval_num), len(layout_ids))
    eval_cfg["layout_ids"] = layout_ids
    eval_cfg["eval_num"] = shard_eval_num
    OmegaConf.update(env_cfg, "eval_cfg.layout_ids", layout_ids, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.eval_num", shard_eval_num, force_add=True)
    return layout_ids, shard_eval_num
