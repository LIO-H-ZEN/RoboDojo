#!/usr/bin/env python3
"""Extract color, shape, and common_name attributes from Rigid object
description.json files and write a curated object_attributes.json.

Usage:
  python3 scripts/internal/extract_object_attributes.py

Output:  scripts/internal/object_attributes.json
"""

import json
import os
from collections import Counter
from pathlib import Path

RIGID_ROOT = Path(".cache/robodojo_assets_repo/Assets/Object/RoboDojo/Rigid")
OUTPUT_FILE = Path("scripts/internal/object_attributes.json")

# ============================================================
# 1.  Colour normalisation map  (raw → canonical)
# ============================================================
COLOUR_MAP = {
    "matte black": "black", "glossy black": "black", "deep black": "black",
    "non-reflective black": "black", "uniform black": "black", "solid black": "black",
    "black": "black", "black trim": "black", "black accents": "black",
    "black roof": "black", "black upper": "black", "black outsole": "black",
    "black sole": "black", "black three stripes": "black", "black sole and toe cap": "black",
    "black hair and eyes": "black", "black eye": "black", "black eyes": "black",
    "matte black body": "black", "dark interior": "black", "dark tinted windows": "black",
    "dark charcoal": "gray", "dark charcoal shadow": "gray", "dark gray": "gray",
    "dark gray accents": "gray", "dark gray base": "gray", "dark gray highlight": "gray",
    "dark gray interior": "gray", "dark gray dial": "gray", "dark gray top": "gray",
    "dark gray metal": "gray", "matte charcoal gray": "gray", "matte gray": "gray",
    "gray": "gray", "gray body": "gray", "gray accents": "gray",
    "gray tape dispenser": "gray", "metallic gray": "gray", "silver-gray plastic": "gray",
    "light gray": "gray", "light gray highlights": "gray", "light gray rim": "gray",
    "light silver": "gray", "light silver highlight": "gray", "medium gray": "gray",
    "metal": "gray", "metal bottle opener": "gray", "metal drafting compass": "gray",
    "metal open-end wrench": "gray", "stainless steel": "gray",
    "stainless steel nail clippers": "gray",
    "off-white": "white", "matte white": "white", "white": "white",
    "bright white face": "white", "white face": "white", "white laces": "white",
    "white toe box": "white", "white upper": "white", "white with burger print": "white",
    "white vegetable peeler": "white", "white ceramic saucepan": "white",
    "white thread spool": "white", "white watering can": "white",
    "white packing tape": "white", "white corkscrew tool": "white",
    "white passenger jet toy": "white", "cream body": "white", "cream": "white",
    "cream-colored bands": "white", "cream teddy bear doll": "white",
    "cream kids bike toy": "white", "speckled ceramic dove": "white",
    "beige bear-hat figurine": "beige", "muted beige": "beige", "light tan": "beige",
    "tan stitching": "beige", "tan lining": "beige",
    "warm brown": "brown", "dark brown leather": "brown", "brown": "brown",
    "brown wing spots": "brown", "brown headphones": "brown", "gum brown sole": "brown",
    "deep reddish-brown": "brown", "wooden": "brown", "wooden owl figurine": "brown",
    "wooden hacksaw frame": "brown", "wooden A-frame sign": "brown",
    "wood-handled brayer roller": "brown", "wooden toy": "brown",
    "cookie bag": "brown", "dark brown liquid": "brown", "dark liquid": "brown",
    "light brown": "brown",
    "vibrant red": "red", "deep red": "red", "glossy red": "red",
    "glossy red exterior": "red", "glossy red headscarf": "red", "bright red": "red",
    "bright red base": "red", "shiny crimson": "red", "glossy maroon": "red",
    "maroon cap": "red", "red label": "red", "red floral pattern": "red",
    "shiny red bell pepper": "red", "red": "red",
    "mint": "green", "mint green": "green", "mint green alarm clock": "green",
    "mint green woven box": "green", "mint green scissors": "green",
    "deep green": "green", "bright green": "green", "deep forest green": "green",
    "olive green undertone": "green", "olive green handle": "green",
    "olive gold": "green", "green stems": "green", "green cactus planter": "green",
    "green vortex binoculars": "green", "green kiwi slice": "green",
    "green orange juice carton": "green", "green-handled wire crimper": "green",
    "green": "green",
    "bright yellow": "yellow", "bright yellow body": "yellow",
    "bright yellow upper": "yellow", "mustard yellow": "yellow",
    "golden yellow": "yellow", "yellow hair on face": "yellow",
    "yellow beak": "yellow", "yellow feet": "yellow", "yellow ice axe": "yellow",
    "paint roller with yellow stripes": "yellow",
    "golden": "yellow", "golden hand bell": "yellow", "golden trophy cup": "yellow",
    "golden waffle cookie": "yellow", "yellow": "yellow",
    "orange beak": "orange", "orange": "orange",
    "blue": "blue", "deep blue": "blue", "glossy royal blue": "blue",
    "subtle blue LED glow": "blue", "blue Logitech G logo": "blue",
    "blue sunglasses": "blue", "blue glue stick": "blue",
    "blue police helicopter toy": "blue", "blue lighter": "blue",
    "light blue": "blue", "light blue stapler": "blue",
    "pink cheek spot": "pink", "pink": "pink",
    "deep purple": "purple", "violet": "purple", "shiny magenta gradient": "purple",
    "lavender": "purple", "lavender plastic shovel": "purple",
    "silver": "silver", "silver accents": "silver", "silver cap": "silver",
    "silver wheels": "silver", "shiny silver": "silver", "shiny silver highlight": "silver",
    "metallic silver": "silver", "metallic silver edge": "silver",
    "metallic silver interior": "silver", "reflective silver rim": "silver",
    "gold accents": "gold", "glossy gold": "gold", "gold": "gold",
    "teal glue gun": "teal", "teal": "teal",
    "floral-dress bunny doll": "multicolor", "blue and pink floral patterns": "multicolor",
    "gray metal head": "gray", "gray cup": "gray",
}

# ============================================================
# 2.  Shape normalisation map
# ============================================================
SHAPE_MAP = {
    "perfect cube": "cube", "cube": "cube", "small cube": "cube",
    "rectangular slab": "rectangular", "rectangular block": "rectangular",
    "rectangular block with rounded edges": "rectangular", "rectangular case": "rectangular",
    "rectangular prism": "rectangular", "rectangular face with rounded corners": "rectangular",
    "rectangular": "rectangular", "rounded corners": "rectangular",
    "curved edges": "rectangular", "flat front with slight curve": "rectangular",
    "sharp right angles": "rectangular", "sharp edges": "rectangular",
    "flat square faces": "rectangular", "six flat faces": "rectangular",
    "six-sided exterior": "rectangular", "blocky": "rectangular",
    "blocky humanoid robot": "rectangular", "boxy cab with open cargo bed": "rectangular",
    "raised roofline": "rectangular", "angular front grille": "rectangular",
    "boxy rectangular body": "rectangular", "flat": "rectangular",
    "flat rectangle": "rectangular", "rounded square block": "rectangular",
    "palm-sized": "rectangular", "action camera scale": "rectangular",
    "smartwatch": "rectangular", "wrist-sized": "rectangular",
    "standard smartwatch": "rectangular", "portable music player": "rectangular",
    "smart remote": "rectangular", "USB-C hub adapter": "rectangular",
    "three-dimensional letter b": "letter-shaped", "rounded uppercase r": "letter-shaped",
    "flattened oval with central hole": "oval", "tapered oval": "oval",
    "round face": "round", "round": "round", "rounded": "round",
    "rounded rim": "round", "rounded top": "round", "rounded base": "round",
    "rounded bottom": "round", "rounded head": "round", "rounded egg body": "round",
    "rounded egg-like body": "round", "rounded teardrop": "round",
    "round case": "round", "circular face": "round", "circular earcups": "round",
    "circular ear cups": "round", "wide shallow dome": "round",
    "irregular sphere": "round", "bulbous body": "round", "bulbous head": "round",
    "bulbous base": "round", "sphere": "round", "egg-shaped": "round",
    "grapes cluster": "round", "clock": "round", "donut": "donut-shaped",
    "torus": "donut-shaped", "doughnut-shaped torus": "donut-shaped",
    "tapered cylinder": "cylindrical", "cylindrical body": "cylindrical",
    "cylindrical disc": "cylindrical", "cylindrical handle": "cylindrical",
    "cylindrical body with slight taper": "cylindrical", "slim cylinder": "cylindrical",
    "elongated cylinder": "cylindrical", "ribbed base": "cylindrical",
    "slim tapered handle": "cylindrical", "internal threaded hole": "cylindrical",
    "spray can": "cylindrical", "paint roller with yellow stripes": "cylindrical",
    "flat cylinder": "cylindrical",
    "wide shallow curve": "bowl", "wide shallow bowl": "bowl",
    "shallow round bowl": "bowl", "shallow scoop": "bowl",
    "concave interior": "bowl", "wide rim": "bowl", "flared rim": "bowl",
    "slightly flared rim": "bowl", "wide open rim": "bowl", "wide open mouth": "bowl",
    "slightly flared opening": "bowl", "wider rim than base": "bowl",
    "rounded deep body": "bowl", "flat footed base": "bowl",
    "slightly raised base ring": "bowl",
    "curved band": "curved", "curved headband": "curved", "curved back": "curved",
    "curved handle": "curved", "curved beak": "curved", "curved sickle blade": "curved",
    "curved rectangular trough": "curved",
    "C-shaped handle": "C-shaped", "two upright ears": "C-shaped",
    "L-shaped prism": "L-shaped", "right-angle extrusion": "L-shaped",
    "two perpendicular rectangular arms": "L-shaped", "l-shaped prism": "L-shaped",
    "rounded l-shape": "L-shaped",
    "low-profile": "slim", "low-profile silhouette": "slim", "slim profile": "slim",
    "slim neck": "slim", "low-top silhouette": "slim",
    "low-slung aerodynamic profile": "slim", "low-slung wedge profile": "slim",
    "fastback rear": "slim",
    "aerodynamic teardrop": "teardrop", "teardrop main body": "teardrop",
    "tapered tip": "pointed", "tapered ends": "pointed", "tapered neck": "pointed",
    "hexagonal prism": "hexagonal", "six-sided exterior": "hexagonal",
    "square base": "square",
    "contoured palm rest": "ergonomic",
    "reinforced toe box": "shoe", "rounded toe box": "shoe", "rounded toe": "shoe",
    "low heel": "shoe", "curved heel counter": "shoe", "buckle closure": "shoe",
    "low-cut ankle": "shoe", "shoe": "shoe",
    "narrow neck with screw cap": "bottle-shaped",
    "contoured hourglass body": "hourglass",
    "hollow body with removable lid": "box-shaped", "structured trapezoid": "trapezoid",
    "frog head with bulging eyes": "irregular", "foldable design": "irregular",
    "integrated handle": "irregular", "short stubby legs": "irregular",
    "blocky lobed": "irregular", "irregular": "irregular",
    "sitting pose": "figurine", "long triangular prism": "triangular",
    "conch shell": "spiral", "origami paper crane": "origami",
    "bead bracelet": "beaded", "drawstring pouch": "pouch",
    "cup": "cup-shaped", "noodle cup": "cup-shaped", "yogurt cup": "cup-shaped",
    "bird-shaped": "irregular", "crown": "crown-shaped",
    "bottle": "bottle-shaped", "flat striking face": "rectangular",
}

# ============================================================
# 3.  Manual overrides for categories where auto-extraction fails
# ============================================================
COLOR_OVERRIDES = {
    "can": "silver",         # spray can
    "chick": "white",        # ceramic dove
    "chocolate_bag": "brown",  # cookie bag
    "chocolate_bar": "brown",
    "donut": "brown",
    "grapes": "purple",
    "handled_paint": "brown",  # wood-handled brayer roller
    "music_player": "silver",
    "noodles": "white",      # noodle cup
    "origami_crane": "white",
    "piler": "green",        # green-handled wire crimper
    "prayer_bead": "multicolor",
    "remote": "black",
    "sachet": "white",       # drawstring pouch
    "seashell": "white",
    "sickle": "gray",
    "tiara": "gold",
    "toy": "multicolor",     # floral-dress bunny doll
    "usb_multip": "black",
    "whisk": "silver",
    "yogurt": "white",
    "camera": "black",
    "egg": "white",
    "goblet": "gold",
    "key": "silver",
    "mahjong": "white",
    "mallet": "brown",
    "number": "white",
    "pen_holder": "black",
    "phone": "black",
    "pliers": "gray",
    "socket": "white",
    "sphere": "gray",
    "stackingblocks": "multicolor",
    "tape_measure": "yellow",
    "test_tube": "white",
    "wine_bottle": "green",
    "wuliangye": "white",
    "bread": "brown",  # bread is brown, not black
}

SHAPE_OVERRIDES = {
    "bell": "bell-shaped",
    "bicycle": "bicycle-shaped",
    "binoculars": "rectangular",
    "bottle_opener": "L-shaped",
    "cactus": "cylindrical",
    "can": "cylindrical",
    "cassette": "rectangular",
    "chick": "bird-shaped",
    "chocolate_bag": "rectangular",
    "clock": "round",
    "corkscrew": "irregular",
    "correction_tape": "rectangular",
    "donut": "donut-shaped",
    "drawing_compass": "irregular",
    "figurine": "figurine",
    "frame": "triangular",
    "glasses": "irregular",
    "glue": "cylindrical",
    "glue_gun": "irregular",
    "grapes": "round",
    "handled_paint": "cylindrical",
    "headphone": "curved",
    "helicopter": "irregular",
    "hoe": "pointed",
    "jewelry_box": "rectangular",
    "juice_carton": "rectangular",
    "kiwi": "round",
    "lighter": "rectangular",
    "music_player": "rectangular",
    "nail_clippers": "irregular",
    "noodles": "cup-shaped",
    "origami_crane": "origami",
    "owl": "figurine",
    "paint_roller": "cylindrical",
    "peeler": "irregular",
    "piler": "irregular",
    "plane": "irregular",
    "pot": "cylindrical",
    "power_bank": "rectangular",
    "prayer_bead": "beaded",
    "remote": "rectangular",
    "sachet": "pouch",
    "saw": "irregular",
    "scan": "rectangular",
    "scissor": "irregular",
    "seashell": "spiral",
    "shovel": "irregular",
    "sickle": "curved",
    "speaker": "rectangular",
    "stapler": "irregular",
    "tape_dispense": "irregular",
    "tape_roll": "cylindrical",
    "teddy_bear": "figurine",
    "thread": "cylindrical",
    "tiara": "crown-shaped",
    "toy": "figurine",
    "trophy": "cup-shaped",
    "usb_multip": "rectangular",
    "waffle": "round",
    "watering_can": "irregular",
    "whisk": "irregular",
    "wrench": "irregular",
    "yogurt": "cup-shaped",
    "camera": "rectangular",
    "egg": "oval",
    "goblet": "cup-shaped",
    "key": "irregular",
    "mahjong": "rectangular",
    "mallet": "cylindrical",
    "number": "rectangular",
    "pen_holder": "cylindrical",
    "phone": "rectangular",
    "pliers": "irregular",
    "socket": "rectangular",
    "sphere": "round",
    "stackingblocks": "rectangular",
    "tape_measure": "rectangular",
    "test_tube": "cylindrical",
    "wine_bottle": "bottle-shaped",
    "wuliangye": "bottle-shaped",
}

# ============================================================
# 4.  Common name overrides
# ============================================================
NAME_OVERRIDES = {
    "b": "letter B block", "d": "letter D block", "j": "letter J block",
    "o": "letter O block", "r": "letter R block", "t": "letter T block",
    "division": "division sign block", "equal": "equal sign block",
    "minus": "minus sign block", "multiplication": "multiplication sign block",
    "plus": "plus sign block", "factory_nut": "factory nut",
    "small_cube": "small cube", "triangular_prism": "triangular prism",
    "bell": "hand bell", "bicycle": "toy bicycle", "binoculars": "binoculars",
    "bottle_opener": "bottle opener", "cactus": "cactus plant",
    "can": "spray can", "cassette": "cassette tape",
    "chick": "ceramic dove", "chocolate_bag": "cookie bag",
    "clock": "alarm clock", "corkscrew": "corkscrew",
    "correction_tape": "correction tape", "donut": "donut",
    "drawing_compass": "drafting compass", "figurine": "figurine",
    "frame": "A-frame sign", "glasses": "sunglasses", "glue": "glue stick",
    "glue_gun": "glue gun", "grapes": "grapes",
    "handled_paint": "brayer roller", "headphone": "headphones",
    "helicopter": "toy helicopter", "hoe": "ice axe",
    "jewelry_box": "woven box", "juice_carton": "juice carton",
    "kiwi": "kiwi slice", "lighter": "lighter",
    "music_player": "music player", "nail_clippers": "nail clippers",
    "noodles": "noodle cup", "origami_crane": "origami crane",
    "owl": "owl figurine", "paint_roller": "paint roller",
    "peeler": "vegetable peeler", "piler": "wire crimper",
    "plane": "toy airplane", "pot": "saucepan",
    "power_bank": "power bank", "prayer_bead": "bead bracelet",
    "remote": "remote control", "sachet": "drawstring pouch",
    "saw": "hacksaw", "scan": "barcode scanner", "scissor": "scissors",
    "seashell": "conch shell", "shovel": "toy shovel", "sickle": "sickle",
    "speaker": "portable speaker", "stapler": "stapler",
    "tape_dispense": "tape dispenser", "tape_roll": "packing tape",
    "teddy_bear": "teddy bear", "thread": "thread spool", "tiara": "tiara",
    "toy": "bunny doll", "trophy": "trophy", "usb_multip": "USB hub",
    "waffle": "waffle cookie", "watering_can": "watering can",
    "whisk": "whisk", "wrench": "wrench", "yogurt": "yogurt cup",
    "bread": "bread loaf", "brick": "brick", "broom": "broom",
    "broom_shovel": "broom and shovel set", "chessman": "chess piece",
    "coin": "coin", "earbuds": "earbuds",
    "electric_toothbrush": "electric toothbrush",
    "game_machine": "game machine", "hammer": "hammer",
    "headset": "headset", "ice_cream": "ice cream",
    "matryoshka_dolls": "matryoshka doll", "mouse": "computer mouse",
    "mug": "mug", "oil_pen": "oil pen", "paper_ball": "paper ball",
    "pepper": "bell pepper", "puppet": "puppet", "toy_car": "toy car",
    "wine_bowl": "wine bowl", "action_camera": "action camera",
    "alarm": "alarm clock", "basket_grasp": "basket",
    "block": "building block", "charger": "charger",
    "chocolate_bar": "chocolate bar", "garage": "toy garage",
    "mallet_stand": "mallet stand", "shoe": "shoe", "watch": "watch",
    "wooden_toy": "wooden toy", "car": "toy car", "cup": "cup",
    "plate": "plate", "bowl": "bowl", "box": "box", "pen": "pen",
    "bottle": "bottle", "cube": "cube", "bag": "bag",
    "keyboard": "keyboard", "phone": "phone", "camera": "camera",
    "egg": "egg", "sphere": "sphere", "socket": "socket",
    "pliers": "pliers", "wine_bottle": "wine bottle",
    "wuliangye": "wuliangye bottle", "tape_measure": "tape measure",
    "test_tube": "test tube", "stackingblocks": "stacking blocks",
    "origami_crane": "origami crane", "headphone": "headphones",
    "goblet": "goblet", "key": "key", "mahjong": "mahjong tile",
    "mallet": "mallet", "number": "number block",
    "pen_holder": "pen holder", "watering_can": "watering can",
}

# ============================================================
# 5.  Helper: parse description string for color
# ============================================================
def parse_description(desc: str):
    desc_lower = desc.lower()
    for phrase in ["mint green", "light blue", "dark blue", "bright red",
                   "deep red", "deep blue", "dark gray", "light gray",
                   "dark brown", "stainless steel", "mint", "golden",
                   "metal", "wooden", "lavender", "teal"]:
        if phrase in desc_lower:
            return phrase
    words = desc_lower.split()
    known = {"red", "green", "blue", "yellow", "black", "white", "gray",
             "grey", "brown", "orange", "purple", "pink", "gold", "silver",
             "beige", "cream", "teal", "metal", "wooden"}
    for w in words:
        if w in known:
            return w
    return "unknown"


def extract_for_category(cat_name: str, variants_dir: Path):
    colors = []
    shapes = []
    names = []

    for variant in sorted(os.listdir(variants_dir)):
        desc_path = variants_dir / variant / "description.json"
        if not desc_path.is_file():
            continue
        data = json.loads(desc_path.read_text())
        caption = data.get("caption", {})

        if caption:
            if caption.get("color"):
                colors.append(caption["color"][0])
            if caption.get("shape"):
                shapes.append(caption["shape"][0])
            if caption.get("name"):
                names.append(caption["name"][0])
        else:
            desc = data.get("description", "")
            if desc:
                colors.append(parse_description(desc))

    raw_color = Counter(colors).most_common(1)[0][0] if colors else "unknown"
    raw_shape = Counter(shapes).most_common(1)[0][0] if shapes else "unknown"
    raw_name = Counter(names).most_common(1)[0][0] if names else ""

    # Normalise
    normalised_color = COLOUR_MAP.get(raw_color.lower(), raw_color.lower())
    normalised_shape = SHAPE_MAP.get(raw_shape.lower(), raw_shape.lower())

    # Manual overrides
    if cat_name in COLOR_OVERRIDES:
        normalised_color = COLOR_OVERRIDES[cat_name]
    if cat_name in SHAPE_OVERRIDES:
        normalised_shape = SHAPE_OVERRIDES[cat_name]

    # Common name
    if cat_name in NAME_OVERRIDES:
        common_name = NAME_OVERRIDES[cat_name]
    elif raw_name and raw_name != "unknown":
        common_name = raw_name.lower()
    else:
        common_name = cat_name.replace("_", " ")

    return {
        "color": normalised_color,
        "shape": normalised_shape,
        "common_name": common_name,
    }


def main():
    results = {}
    for cat_dir in sorted(RIGID_ROOT.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        cat_name = cat_dir.name
        results[cat_name] = extract_for_category(cat_name, cat_dir)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote attributes to {OUTPUT_FILE}")
    print(f"Total categories: {len(results)}")

    colors = Counter(v["color"] for v in results.values())
    shapes = Counter(v["shape"] for v in results.values())
    unknown_colors = [k for k, v in results.items() if v["color"] == "unknown"]
    unknown_shapes = [k for k, v in results.items() if v["shape"] == "unknown"]

    print(f"\n=== COLOR DISTRIBUTION ===")
    for c, n in sorted(colors.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    print(f"\n=== SHAPE DISTRIBUTION ===")
    for s, n in sorted(shapes.items(), key=lambda x: -x[1])[:20]:
        print(f"  {s}: {n}")
    print(f"\n=== UNKNOWN COLORS ({len(unknown_colors)}) ===")
    for c in unknown_colors:
        print(f"  {c}")
    print(f"\n=== UNKNOWN SHAPES ({len(unknown_shapes)}) ===")
    for c in unknown_shapes:
        print(f"  {c}")


if __name__ == "__main__":
    main()
