# Table 桌面增强系统配置文档

`general_pickup` 任务内置了一套**桌面增强系统**，从四个维度控制桌面的视觉与布局变化：

| 维度 | 说明 | 模式 | 配置键 |
|---|---|---|---|
| 纹理 | 循环切换 MDL 材质（木纹、地毯、瓷砖等） | `texture` | `table_visual.texture` |
| 颜色 | 循环切换桌面颜色（RGB 染色） | `color` | `table_visual.color` |
| 显示屏 | 将桌面当作屏幕，循环播放图片/视频（拉伸铺满） | `display` | `table_visual.display` / `table_display_playlist` |
| 物品重排 | 周期性地重新摆放桌面上未被抓取的物品 | 任意模式 | `table_visual.rearrange` |

**互斥规则**：纹理、颜色、显示屏是**互斥**的视觉模式（`table_visual.mode` 决定），一次只能启用其中一个。
**正交规则**：物品重排（`rearrange`）与视觉模式正交，任何模式下都可以独立配置。

---

## 1. 快速开始

配置文件：`task/RoboDojo/config/general_pickup.yml`

**推荐方式（部署最稳）——扁平配置键**：评估部署链路（`src/eval_client/main.py` → OmegaConf）会把 `config.task_env` 转成 `DictConfig/ListConfig`，普通 `isinstance(x, dict/list)` 会失效导致结构化块被静默跳过。因此**推荐使用扁平键**，代码里已通过 `_plain()` 递归转换兼容 OmegaConf，但扁平键最保险：

```yaml
# 扁平键方式：table_display_playlist（图片/视频自由混排）
step_lim: 500
table_display_playlist:
  - type: image
    value: "demopicture/a.jpg"
    hold_steps: 25          # 每张图片停 N 个控制步（25 步 ≈ 1 秒）
  - type: image
    value: "demopicture/b.jpg"
    hold_steps: 25
  - type: video
    value: "demopicture/movie.mp4"
    steps_per_frame: 1      # 每 N 个控制步推进 1 帧视频（1 = 逐帧流畅播放）
    max_frames: 200         # 每视频最多抽多少帧（长视频自动降采样，保证鲁棒）
table_rearrange_interval: 312   # 物品重排（0=关闭）
table_rearrange_rotate_deg: 20
table_rearrange_margin: 0.02
```

**结构化方式（`table_visual` 块）**同样支持，行为一致：

```yaml
table_visual:
  mode: "display"        # texture | color | display | none
  rearrange:
    interval: 312
    rotate_deg: 20
    margin: 0.02
  display:
    load_on_start: true
    playlist:
      - type: image
        value: "demopicture/a.jpg"
        hold_steps: 25
      - type: video
        value: "demopicture/movie.mp4"
        steps_per_frame: 1
        max_frames: 200
```

> **注意**：`table_visual` 结构化块只在 `isinstance(dict)` 检测通过时才被解析；代码已内置 `_plain()` 处理 OmegaConf 包装，两种写法均可用，但扁平键在部署链路上最稳。

---

## 2. 显示屏播放列表（`display`）

把桌面当作**屏幕**，通过一个有序循环的 **播放列表（playlist）** 自由组合图片和视频：

| 类型 | 必备字段 | 可选字段 | 说明 |
|---|---|---|---|
| `image` | `value`（图片路径） | `hold_steps`（默认 25） | 图片停留 `hold_steps` 个控制步后进入下一项 |
| `video` | `value`（视频路径） | `steps_per_frame`（默认 1）、`max_frames`（默认 200） | 首次运行时用 cv2 自动抽帧成 JPEG（存放于 git 忽略的 `.cache/display_frames/<视频名>/`），之后逐帧播放 |

**播放逻辑**：

- 所有播放项被**统一展平成一条"帧时间线"**：图片项 = 1 帧 × `hold_steps`；视频项 = `抽帧数` × `steps_per_frame`。
- **图片帧和视频帧走完全相同的快速贴图路径**（`Table.update_texture`，复用材质仅替换纹理文件），性能稳定。
- 时间线**无限循环**，直到评估结束。

**鲁棒性保证**：

1. 每视频抽帧数有上限（`max_frames`，默认 200），超长视频被**均匀降采样**，不会产生几千张帧。
2. 抽帧失败（cv2 不可用 / 文件损坏 / 无法打开）会**安全跳过**该视频项，不影响其余播放项。
3. 视频帧缓存复用：同一视频二次运行时直接读取 `.cache/display_frames/<stem>/`，不重复抽帧。
4. 图片路径不存在时该播放项被跳过，不影响整体播放。
5. `reset` 后会自动重新贴上当前帧，保证从头开始就有画面（不会出现"开局是默认桌面材质"）。

**简易示例**（5 张图 × 25 步 + 1 个视频逐帧）：

```yaml
table_display_playlist:
  - type: image
    value: "demopicture/0bf7e7774a93f0bb03286109039b74f.jpg"
    hold_steps: 25
  - type: image
    value: "demopicture/0cb484f1334aa8921b1abc2de50a970.jpg"
    hold_steps: 25
  - type: image
    value: "demopicture/3f3bc20ac5b05f2887a6fe89fd0b072.jpg"
    hold_steps: 25
  - type: image
    value: "demopicture/42e03804aad875b2e9585f113dd34af.jpg"
    hold_steps: 25
  - type: image
    value: "demopicture/5b21576fa9a07ff1a43ca88311a9e9b.jpg"
    hold_steps: 25
  - type: video
    value: "demopicture/demo_video.mp4"   # 640x480 25fps 8s
    steps_per_frame: 1
    max_frames: 200
```

---

## 3. 纹理模式（`mode: "texture"`）

```yaml
table_visual:
  mode: "texture"
  texture:
    source: "local"             # "local" | "remote"
    remote_repo: ""             # 远程 git 仓库 URL 或本地目录（source=remote 时使用）
    change_interval: 50         # 每 N 步切换一次材质（0 = 关闭）
    list:                       # MDL 材质清单（本地路径）
      - "Assets/Material/material_0001/Ceiling_Tiles.mdl"
      - "Assets/Material/material_0004/Carpet_Beige.mdl"
      - "Assets/Material/material_0005/Carpet_Berber_Gray.mdl"
      - "Assets/Material/material_0009/Carpet_Diamond_Olive.mdl"
      - "Assets/Material/material_0122/Mahogany_Planks.mdl"
```

---

## 4. 颜色模式（`mode: "color"`）

```yaml
table_visual:
  mode: "color"
  color:
    change_interval: 40         # 每 N 步切换一次颜色（0 = 关闭）
    list:                       # RGB 数组，每个分量 0.0 ~ 1.0
      - [0.8, 0.2, 0.2]
      - [0.2, 0.8, 0.2]
      - [0.2, 0.2, 0.8]
      - [0.8, 0.8, 0.2]
```

---

## 5. 物品重排（`rearrange`，任意模式可用）

```yaml
table_visual:
  mode: "none"                  # 视觉关闭，只做重排
  rearrange:
    interval: 20                # 每 N 步重排 1 次（0 = 关闭）
    rotate_deg: 20              # 每次重排允许的最大随机旋转角度
    margin: 0.02                # 物品之间的最小间隔（米）
```

---

## 6. 从互联网仓库拉取更多材质

```bash
# 从 git 仓库拉取指定材质
python scripts/fetch_table_materials.py \
    --repo https://github.com/example/material_repo.git \
    --dest Assets/Material/table_materials \
    --names "Wood_Oak,Carpet_Beige,Concrete_004"

# 使用本地已 clone 的目录（--names "*" 拷贝全部）
python scripts/fetch_table_materials.py \
    --repo /path/to/material_repo \
    --dest Assets/Material/table_materials \
    --names "*"
```

---

## 7. 向后兼容

旧的扁平配置键仍然有效（当 `table_visual` 块不存在时自动读取）：

- `table_rearrange_interval` / `table_rearrange_rotate_deg` / `table_rearrange_margin`
- `table_color_change_interval` / `table_color_list`
- `table_material_change_interval` / `table_material_list`
- `table_image_change_interval` / `table_image_list`（简单图片轮播）
- `table_display_playlist`（图片/视频混排播放列表，**推荐**）

---

## 8. 相关 API

| 方法 | 说明 |
|---|---|
| `Table.set_image_texture(path)` | 把图片作为桌面漫反射贴图（`UsdUVTexture`），拉伸铺满 |
| `Table.update_texture(path)` | 快速换帧：复用已建材质仅替换纹理文件（图片/视频逐帧播放用） |
| `Table.set_material(mdl_path)` | 切换到指定 MDL 材质 |
| `Table.set_color(rgb)` | 对当前材质叠加 RGB 染色 |
| `GeneralPickupCommon._plain(value)` | 递归转 OmegaConf `DictConfig/ListConfig` 为纯 dict/list（配置解析兼容） |
| `GeneralPickupCommon._build_display_timeline(disp)` | 从播放列表构建统一"帧时间线"（图片/视频混排） |
| `GeneralPickupCommon._resolve_video_frames(value, max_frames)` | cv2 惰性抽帧（有上限、失败安全跳过、缓存复用） |
| `GeneralPickupCommon._apply_display_timeline(step)` | 按当前控制步推进播放列表 |
| `GeneralPickupCommon._apply_table_image(path, reason)` | 对全部环境应用图片并记录日志 |
| `GeneralPickupCommon._change_table_image() / _change_table_color() / _change_table_material()` | 按配置清单循环切换 |
| `GeneralPickupCommon._rearrange_table(env_ids)` | 对指定环境重排桌面物品 |
| `scripts/fetch_table_materials.py` | 从 git 仓库拉取 MDL 材质 |