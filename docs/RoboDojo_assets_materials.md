# RoboDojo 资产库材质梳理（桌面可用纹理）

> 数据来源：`https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo` 仓库 `Assets/Material/`
> 通过 git-lfs 索引（`git lfs ls-files`）枚举，共 **504 个 material_* 目录** + `Base/` 分类目录。

---

## 1. 资产库整体架构

```
Assets/
├─ Background/      # 背景
├─ Eval_Layout/     # 评估布局
├─ Material/        # ★ 材质（本文重点）
│  ├─ material_0001 ... material_0564   # 504 个平铺编号材质（每个含 .mdl + 贴图）
│  └─ Base/                             # 分类基础材质
│     ├─ Natural/    # 自然材质（户外/地面）
│     ├─ Textiles/   # 织物/皮革/布料
│     └─ Templates/  # 模板
├─ Object/          # 物体资产
├─ Robots/          # 机器人
├─ Room/            # 房间
├─ Sensor/          # 传感器
├─ TableImages/     # 桌面图片
└─ Traj/            # 轨迹
```

每个 `material_NNNN/` 目录标准组成：`<Name>.mdl` + `<Name>_BaseColor.png` + `<Name>_Normal.png` + `<Name>_ORM.png`（少数如 `material_0564` 为多贴图变体）。

---

## 2. 桌面可用纹理（按大类罗列）

> 以下均可直接用于桌面外观（MDL 材质，代码走 `Table.set_material()`）。

### 2.1 木材 / 地板类（约 50 种）
```
Ash  Ash_Planks  Bamboo  Bamboo_Planks  Beadboard  Birch  Birch_Planks
Cherry  Cherry_Planks  Laminate_Oak  Mahogany  Mahogany_Planks  MDF  Oak  Oak_Planks
OSB_Wood  OSB_Wood_Splattered  Parquet_Floor  Plywood  Timber  Timber_Cladding
Veneer_OU_Walnut  Veneer_UX_Walnut_Cherry  Veneer_Z5_Maple  Walnut  Walnut_Planks
Wood_Bark  Wood_Cork  Wood_Grain_Raw_01~05  Wood_Grain_Varnish_01~05
Wood_Tiles_Ash  Wood_Tiles_Ash_Multicolor  Wood_Tiles_Beech  Wood_Tiles_Beech_Multicolor
Wood_Tiles_Fineline  Wood_Tiles_Fineline_Multicolor  Wood_Tiles_Oak_Mountain
Wood_Tiles_Oak_Mountain_Multicolor  Wood_Tiles_Pine  Wood_Tiles_Pine_Multicolor
Wood_Tiles_Poplar  Wood_Tiles_Poplar_Multicolor  Wood_Tiles_Walnut  Wood_Tiles_Walnut_Multicolor
```

### 2.2 地毯类（约 15 种）
```
Carpet_Beige  Carpet_Berber_Gray  Carpet_Berber_Multi  Carpet_Charcoal  Carpet_Cream
Carpet_Diamond_Olive  Carpet_Diamond_Yellow  Carpet_Forest  Carpet_Gray
Carpet_Pattern_01  Carpet_Pattern_Leaf_Squares_Tan  Carpet_Pattern_Loop
Carpet_Pattern_Squares_Multi  Carpet_Woven  Rug_Carpet  Fabric_Carpet_Long_Floor
```

### 2.3 瓷砖 / 陶瓷 / 石材类（约 35 种）
```
Adobe_Brick  Brick_Pavers  Brick_Wall_Brown  Brick_Wall_Red  Facade_Brick_Grey
Facade_Brick_Red_Clinker  Ceramic_Smooth_Fired  Ceramic_Tile_6  Ceramic_Tile_12  Ceramic_Tile_18
Ceramic_Tiles_Glazed_Diamond  Ceramic_Tiles_Glazed_Diamond_Offset
Ceramic_Tiles_Glazed_Mosaic_Shifted  Ceramic_Tiles_Glazed_Paseo  Ceramic_Tiles_Glazed_Penny
Ceramic_Tiles_Glazed_Pinwheel  Ceramic_Tiles_Glazed_Square  Ceramic_Tiles_Glazed_Square_Brick
Ceramic_Tiles_Glazed_Subway  Ceramic_Tiles_Glazed_Versailles
Granite_Dark  Granite_Light  Granite_Polished  Marble  Marble_Smooth  Marble_Tile_12  Marble_Tile_18
Mosaic_Multi_Color_Stone  Porcelain_Smooth  Porcelain_Tile_4  Porcelain_Tile_4_Linen
Porcelain_Tile_6  Porcelain_Tile_6_Linen  Slate  Stone_Mediterranean  Stone_Natural_Black
Stone_Pores_Weathered  Stone_Wall  Terracotta  Terrazzo  Large_Granite_Paving  Retaining_Block
```

### 2.4 织物 / 布料 / 皮革 / 绒面类（约 50 种）
```
ABS_Hard_Leather  ABS_Quilted_Leather  ABS_Soft_Leather  ABS_Woven_Leather
Aniline_Leather  Leather_Black  Leather_Brown  Leather_Pumpkin  Leather_Dots_Circle_01
Leather_Dots_Square_01  Leather_Grain_01~06  Leather_Pattern_01  Leather_Pattern_02
PU_Split_Leather  Pigmented_Smooth_Leather  Pull_Up_Leather  Semi_Aniline_Leather  Suede_Leather
Suede_01~10  Velvet  Cloth_Black  Cloth_Gray
Cotton_Denim  Cotton_Roughly_Woven  Fabric_Cotton_Fine_Woven  Fabric_Denim_01  Fabric_Denim_02
Fabric_Grain_01  Fabric_Pattern_01  Fabric_Pattern_02  Fabric_Polyester_01~04
Fabric_Weave_01  Fabric_Weave_02  Felt_Plain  Linen_Beige  Linen_Black  Linen_Blue  Linen_Brown
Linen_White  Polyester_Herringbone  Polyester_Twill  Tweed_Herringbone  Wool_Melton  Caoutchouc
```

### 2.5 金属类（约 90 种）
```
Aluminum  Aluminum_Anodized  Aluminum_Anodized_Black  Aluminum_Anodized_Blue
Aluminum_Anodized_Charcoal  Aluminum_Anodized_Red  Aluminum_Brushed  Aluminum_Cast
Aluminum_Foil  Aluminum_Hammered  Aluminum_Knurling  Aluminum_Polished  Aluminum_Scratched  Aluminum_Sheet
Aging_Copper  Brushed_Antique_Copper  Copper  Copper_Antique_Brushed  Copper_Antique_Brushed_Patinated
Copper_Brushed  Copper_Foil  Copper_Hammered  Copper_Knurling  Copper_Scratched  Copper_Sheet
Brass  Brass_Antique  Brass_Brushed  Brass_Foil  Brass_Hammered  Brass_Knurling  Brass_Polished  Brass_Scratched  Brass_Sheet
Bronze  Bronze_Antique  Bronze_Brushed  Bronze_Foil  Bronze_Hammered  Bronze_Knurling  Bronze_Polished  Bronze_Scratched  Bronze_Sheet  Bronze_Sheet_Punched
Gold  Gold_Brushed  Gold_Foil  Gold_Hammered  Gold_Knurling  Gold_Scratched  Gold_Sheet
Silver  Silver_Brushed  Silver_Foil  Silver_Hammered  Silver_Knurling  Silver_Scratched  Silver_Sheet
Iron  Iron_Brushed  Iron_Foil  Iron_Hammered  Iron_Knurling  Iron_Scratched  Iron_Sheet  Iron_Pitted_Steel
Nickel  Nickel_Brushed  Nickel_Foil  Nickel_Hammered  Nickel_Knurling  Nickel_Scratched  Nickel_Sheet
Platinum  Platinum_Brushed  Platinum_Foil  Platinum_Hammered  Platinum_Knurling  Platinum_Scratched  Platinum_Sheet
Titanium  Titanium_Brushed  Titanium_Foil  Titanium_Hammered  Titanium_Knurling  Titanium_Scratched  Titanium_Sheet
Tungsten  Tungsten_Brushed  Tungsten_Foil  Tungsten_Hammered  Tungsten_Knurling  Tungsten_Scratched  Tungsten_Sheet
Zinc  Zinc_Brushed  Zinc_Foil  Zinc_Galvanizing  Zinc_Hammered  Zinc_Knurling  Zinc_Scratched  Zinc_Sheet
Steel_Blued  Steel_Carbon  Steel_Cast  Steel_Galvanized  Steel_Painted  Steel_Painted_Cracked  Steel_Stainless
Stainless_Steel  Stainless_Steel_Brushed  Stainless_Steel_Brushed_Punched  Stainless_Steel_Milled  Stainless_Steel_Punched
Blued_Steel_Cold  Cast_Metal_Silver_Vein  Chrome  Chrome_Hammered  Chromium  Chromium_Brushed  Chromium_Foil
Chromium_Knurling  Chromium_Scratched  Chromium_Sheet  CorrugatedMetal  Diamond_Plate_*（Single/Double/Triple/Quadruple/Quintuple_Tear, Spike）
Mercury  Metal_Black_Paint  Metal_Blue_Paint  Metal_Green  Metal_Red_Paint  Metal_White_Paint  Metal_Yellow_Paint
Metal_Cast  Metal_Ceramic_Brakes_Clean/Dark/Golden  Metal_Dark_Dirty  Metal_Door  Metal_Grain_01  Metal_Grain_02
Metal_Mesh_Weave_01~03  Metal_Polished_01  Metal_Polished_02  Metal_Polished_Dirty_01~03  Metal_Seamed_Roof
Mirror  Punched_Circular_Plate  Rim_*（黑/橙/红/黄喷漆, 蓝/红/玫瑰金车漆, 银, 原色金属, 原色铬）  RustedMetal
```

### 2.6 混凝土 / 水泥 / 石膏 / 砂浆类（约 25 种）
```
Concrete_Block  Concrete_Floor_Damage  Concrete_Formed  Concrete_Polished  Concrete_Precast
Concrete_Rough  Concrete_Smooth  Concrete_Wall_Aged  Concrete_Wall_Aged_Scratched  Concrete_Wall_Even
Spongy_Concrete_Weathered  Spongy_Concrete_Weathered_Mossy  Gypsum  Mortar  Plaster  Plaster_Wall
Stucco  Cement（见 Concrete）  Clay_Dark  Clay_Light  Grog_Fired_Clay
Cobblestone_Big_and_Loose  Cobblestone_Medieval  Fieldstone  Small_Cobblestone  Paving_Stones
```

### 2.7 塑料 / 橡胶 / 碳纤 / 车漆类（约 40 种）
```
Carbon_Fiber_ANI_01  Carbon_Fiber_ANI_01_Clearcoat  Carbon_Fiber_NH_02  Carbon_Fiber_NH_02_Clearcoat
Caoutchouc  Rubber_Smooth  Rubber_Textured  Tire  Tire_Rubber_Clean  Tire_Rubber_Dirty
Tire_SUV_01  Tire_SUV_02  Tire_Supercar_01~04  Tire_Truck_Jeep_01~04
Plastic_Grain_01~04  Plastic_Hex_Pattern  Plastic_Pattern_Dot_01  Plastic_Pattern_Line_02
Plastic_Pattern_Lrg_Dot_03  Plastic_Standardized_Surface_Finish
Carpaint_01~10  Carpaint_Candy  Carpaint_Metallic  Carpaint_Metallic_01~10  Carpaint_Solid
Carpaint_Shifting_Flakes  PCB_Copper  PCB_Goldfinger  Solder_Paste
Cardboard  Cardboard_Low_Quality  Paper
```

### 2.8 户外 / 地面 / 自然类（约 30 种）
```
Asphalt  Asphalt_Fine  Roof_Tiles  Shingles_01  Ceiling_Tiles
Dirt  Sand  Gravel  Gravel_River_Rock  Gravel_Track_Ballast  Pea_Gravel  Rough_Gravel
Ground_Aggregate_Exposed  Ground_Hard_Court  Ground_Leaves  Ground_Leaves_Oak
Grass_Countryside  Grass_Cut  Grass_Winter  Leaves  Mulch  Mulch_Brown  Soil_Rocky
Water  Water_Opaque  Sandstone_Brick_Vintage  Adobe_Octagon_Dots
Chalk_Paint  Chalk_Paint_Pebbles
```

### 2.9 涂料 / 漆面 / 杂项（约 20 种）
```
Paint_Gloss  Paint_Gloss_Finish  Paint_Matte  Paint_Matte_Finish  Paint_Satin  Paint_Satin_Finish
Hammer_Paint  Chalk_Paint  Chalk_Paint_Pebbles
Cork  Netting_Thread_01  Reflectors_01  Reflectors_01_Glow_Red/White/Yellow  Reflectors_02  Reflectors_03
Reflectors_CC_Red_01~03  Reflectors_Clearcoat_01~03  Retroreflective_Material  Retroreflective_Tape  Retroreflective_Warnstripes
UI_Screen_01~04  Koenigsegg_*（Logo/Emblem/Text/皮革标）等
```

---

## 3. Base/ 分类材质（补充）

### Base/Natural（11 种）
```
Asphalt  Dirt  FlatPatch_3x  Grass_Countryside  Grass_Cut  Grass_Winter  Leaves  Mulch_Brown  Sand  Soil_Rocky  Water  Water_Opaque
```

### Base/Textiles（11 种）
```
Cloth_Black  Cloth_Gray  Leather_Black  Leather_Brown  Leather_Pumpkin
Linen_Beige  Linen_Black  Linen_Blue  Linen_Brown  Linen_White  Tweed_Herringbone
```

### Base/Templates（2 种，**不是桌面纹理**）
```
GlassUtils        # 玻璃材质模板（供透明/玻璃物体用）
GlassWithVolume   # 带体积的玻璃材质（供液体/玻璃杯等用）
```
> Templates 是**物体专用**的 MDL 模板，用于玻璃/透明物体渲染，**不适合当桌面贴图**（桌面一般用上面 2.1~2.9 的平铺类材质）。

---

## 3.5 桌面是如何加载材质的（机制说明）

桌面的视觉表现由 `env/scene_manager/objects/table.py` 控制，两种加载路径：

### 路径 A：MDL 材质（`Table.set_material(mdl_path)`）— 纹理模式/材质轮播
1. **路径解析**：`mdl_path` 相对项目根解析到真实文件（如 `.cache/robodojo_assets_repo/Assets/Material/material_0122/Mahogany_Planks.mdl`）。
2. **建材质 prim**：在 `/<table>/Looks/Mat_<材质名>` 下用 `create_mdl_material()` 加载 MDL（其内部引用同目录的 `_BaseColor/_Normal/_ORM` 贴图）。
3. **绑定**：`BindMaterialCommand` 把材质绑到**桌面 prim + 其所有 Mesh/GeomSubset 子节点**（`strongerThanDescendants` 强制覆盖）。
4. 之后可叠加 `set_color(rgb)` 染色。

### 路径 B：图片纹理（`Table.set_image_texture(image_path)`）— 显示屏/照片轮播
1. 在 `/<table>/Looks/TableImage_<名>` 建 `UsdPreviewSurface` 材质，`diffuseColor` 连接一个 **`UsdUVTexture`** 采样器指向图片文件（PNG/JPG/EXR）。
2. 同样绑定到桌面 prim + 子 Mesh。
3. 整张图拉伸铺满桌面（全 UV 覆盖）。

### 路径 B 加速：`Table.update_texture(image_path)` — 视频逐帧
- 复用已有的 `UsdUVTexture` 采样器，**只更新 `file` 输入**，不重建材质节点。
- 这是"播放视频"能流畅跑的原因（每帧只有一次 file 替换），由 `general_pickup.py` 的 `_apply_display_timeline` 驱动。

### 调度者（`task/RoboDojo/tasks/general_pickup.py`）
- 启动/`reset` 后立即贴当前帧（保证开局就有画面）。
- 每个控制步 `on_step_interval` 推进：
  - `table_material_change_interval`（MDL 材质轮播，路径 A）
  - `table_display_playlist`（图片/视频统一帧时间线，路径 B）
- 模式互斥：`display` 模式下颜色/材质轮播自动禁用，防止覆盖屏幕。

---

## 4. 推荐"桌面纹理"首选清单

适合直接当桌面（平铺、美观、中性）的 Top 推荐：

| 类别 | 推荐材质 |
|---|---|
| 木桌 | `Mahogany_Planks` `Walnut_Planks` `Oak_Planks` `Cherry_Planks` `Wood_Tiles_Walnut` `Veneer_OU_Walnut` |
| 石桌/瓷砖 | `Marble_Tile_18` `Granite_Polished` `Slate` `Terrazzo` `Ceramic_Tiles_Glazed_Square` `Mosaic_Multi_Color_Stone` |
| 布/皮桌 | `Leather_Black` `Leather_Brown` `Suede_Leather` `Velvet` `Felt_Plain` `Linen_Blue` |
| 地毯（桌面贴图风格） | `Carpet_Woven` `Carpet_Berber_Gray` `Rug_Carpet` |
</｜｜DSML｜｜>

<task_progress>
- [x] 下载资产库结构性盘点（Assets/ 布局、Material 504 目录、Base 分类）
- [x] 零成本法枚举全部 504 个材质名 + Base 材质（git-lfs 索引）
- [x] 按大类整理桌面可用纹理
- [x] 输出到 md 文件 docs/RoboDojo_assets_materials.md
</task_progress>
</write_to_file>