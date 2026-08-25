# General-Pickup 空间方位抓取扩展

本分支在 RoboDojo 上游基础上，对 `general_pickup` 任务进行了扩展，核心新增：

## 1. 空间方位指令抓取（spatial pickup）

`general_pickup` 支持纯空间方位指令（如"拿起最左侧的物体"），完整链路：

- **Rigid 加载**：从预生成的 Eval Layout JSON 还原桌面刚体为可交互对象
- **目标解析**：指令 → 空间方位词 → 目标物体（如 leftmost = min(x)）
- **成功判定**：目标物体被抬起 ≥10cm 即成功
- **Cheat 验证机制**：无真实策略时，`cheat_lift_step` 在指定步随机抬升一个桌面刚体，端到端验证 prompt 生成、目标解析与成功判定（不强制成功）

详见 `docs/general_pickup_spatial_pickup.md`（Rigid 加载 / Cheat / 判定逻辑详解）。

## 2. 桌面增强

`task/RoboDojo/config/general_pickup.yml` 内置桌面视觉/布局增强：

| 维度 | 配置键 | 说明 |
|------|--------|------|
| 显示屏（图片/视频） | `table_display_playlist` | 桌面当作屏幕循环播放图片/视频 |
| 物品重排 | `table_rearrange_interval` | 周期性重新摆放未抓取的物体 |

详见 `docs/table_visual_config.md`。

## 3. 关键配置

```yaml
# general_pickup.yml
step_lim: 200               # 回合步数上限
cheat_lift_step: 40         # 验证钩子：第 N 步抬升随机刚体（0=关闭）
cheat_lift_z_delta: 0.25    # 抬升高度
table_rearrange_interval: 312   # 物品重排间隔（0=关闭）
```

## 4. 文件索引

| 路径 | 说明 |
|------|------|
| `task/RoboDojo/tasks/general_pickup.py` | 任务逻辑（空间方位指令 + 判定） |
| `task/RoboDojo/config/general_pickup.yml` | 任务配置（物体池/布局/桌面增强） |
| `docs/general_pickup_spatial_pickup.md` | 空间方位抓取链路详解 |
| `docs/table_visual_config.md` | 桌面增强系统文档 |
| `utils/cluttered_generator.py` | 桌面 clutter 生成器 |
| `demopicture/` | 桌面播放素材 |

## 5. 环境

- 基于 RoboDojo（Isaac Sim 5.1 / Isaac Lab 2.3），上游：https://github.com/RoboDojo-Benchmark/RoboDojo
- 子模块：`third_party/IsaacLab`、`third_party/curobo`、`XPolicyLab`（通过 `git submodule update --init` 拉取）
- 本分支仓库为 LIO-H-ZEN 的 fork，详见原始 `README.md`
