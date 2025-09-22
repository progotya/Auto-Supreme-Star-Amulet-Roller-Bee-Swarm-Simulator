import importlib
import importlib.util
import subprocess
import sys
import os
import io
import time
import threading
import ctypes
from ctypes import wintypes
import re
import json
import requests

REQUIRED_DEPENDENCIES = {
    "pyautogui": "pyautogui",
    "pytesseract": "pytesseract",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "keyboard": "keyboard",
    "colorama": "colorama",
    "requests": "requests"
}


def ensure_dependencies():
    missing = []
    for module_name, package_name in REQUIRED_DEPENDENCIES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    if not missing:
        return
    print("[INFO] Installing missing dependencies: " + ", ".join(missing))
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except subprocess.CalledProcessError as exc:
        print("[ERROR] Failed to install dependencies:", exc)
        print("Please install them manually and re-run the script.")
        sys.exit(1)


ensure_dependencies()

import pyautogui
import pytesseract
from pytesseract import Output
from PIL import Image
import cv2
import numpy as np
import keyboard
from colorama import Fore, Style

# If needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SCREENSHOT_NAME = "current_amulet.png"
WEBHOOKURL = json.load(open('config.json', 'r'))['automation']['Webhook']
SENDWEBHOOK = json.load(open('config.json', 'r'))['automation']['Send webook logs']

STAT_KEYWORDS = {
    "Red Pollen": ["red", "pollen"],
    "White Pollen": ["white", "pollen"],
    "Blue Pollen": ["blue", "pollen"],
    "Bee Gather Pollen": ["gather", "pollen"],
    "Pollen": ["pollen"],
    "Instant Conversion": ["instant", "conversion"],
    "Convert Rate": ["convert"],
    "Bee Ability Rate": ["ability"],
    "Critical Chance": ["critical", "chance"]
}

BUTTON_DEFAULTS = {
    "double_yes": (0.43, 0.57),
    "double_no": (0.57, 0.57),
}

BUTTON_COLOR_CONFIG = {
    "double_no": {"region": (0.45, 0.52, 0.8, 0.76), "color": "red"},
    "double_yes": {"region": (0.18, 0.52, 0.55, 0.76), "color": "green"}
}


def get_screen_dimensions():
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return pyautogui.size()


def get_cursor_position():
    pos = pyautogui.position()
    return int(pos[0]), int(pos[1])


def move_cursor(x, y):
    if sys.platform == "win32":
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
    else:
        pyautogui.moveTo(int(x), int(y), duration=0)


def click_at(x, y):
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        user32.SetCursorPos(int(x), int(y))
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        pyautogui.click(int(x), int(y))

def capture_screen():
    orig_x, orig_y = get_cursor_position()
    width, height = get_screen_dimensions()
    safe_x = int(width * 0.02)
    safe_y = int(height * 0.02)
    move_cursor(safe_x, safe_y)
    time.sleep(0.05)
    try:
        screenshot = pyautogui.screenshot()
        img_bytes = io.BytesIO()
        screenshot.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        pil_img = Image.open(img_bytes)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    finally:
        move_cursor(orig_x, orig_y)

def find_amulet_region(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 40, 140])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[WARN] No yellow regions detected; using full image")
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        cx, cy = x + cw // 2, y + ch // 2
        dist = ((cx - center[0]) ** 2 + (cy - center[1]) ** 2) ** 0.5
        if area > 50000:
            candidates.append((dist, x, y, cw, ch))

    if not candidates:
        print("[WARN] No candidate big enough, using full image")
        return img

    _, x, y, cw, ch = min(candidates, key=lambda c: c[0])

    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    cw = min(img.shape[1] - x, cw + 2 * pad)
    ch = min(img.shape[0] - y, ch + 2 * pad)

    return img[y:y+ch, x:x+cw]

def preprocess_for_ocr(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return thresh

def extract_amulet_stats(cropped_img):
    processed = preprocess_for_ocr(cropped_img)
    pil_img = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_img)
    text = text.replace("\n\n", "\n").strip()

    print(f"[DEBUG] OCR Output:\n{text}\n{'-'*40}")

    lines = text.split("\n")
    old_stats = []
    new_stats = []

    for line in lines:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 2:
            (old_stats if len(old_stats) <= len(new_stats) else new_stats).append(" ".join(parts))
            continue

        mid = len(parts) // 2
        left = " ".join(parts[:mid])
        right = " ".join(parts[mid:])

        old_stats.append(left)
        new_stats.append(right)

    return old_stats, new_stats


def save_cropped_image(img):
    if os.path.exists(SCREENSHOT_NAME):
        os.remove(SCREENSHOT_NAME)
    cv2.imwrite(SCREENSHOT_NAME, img)
    print(f"[INFO] Cropped amulet image saved as {SCREENSHOT_NAME}")
def parse_value(stat):
    import re
    m = re.search(r"([+-]?\d+(\.\d+)?)", stat)
    return float(m.group(1)) if m else None

def stat_name(stat):
    stat = stat.lower()
    ignore_keywords = ["capacity", "replace", "keep old"]
    for kw in ignore_keywords:
        if kw in stat:
            return None
    sorted_keys = sorted(STAT_KEYWORDS.items(), key=lambda x: -len(x[1]))
    for config_key, keywords in sorted_keys:
        if all(k in stat for k in keywords):
            return config_key
    for config_key, keywords in STAT_KEYWORDS.items():
        for k in keywords:
            if k in stat.split():
                return config_key
    for config_key, keywords in STAT_KEYWORDS.items():
        if any(k in stat for k in keywords):
            return config_key
    return None


def resolve_button_ratio(name, overrides):
    x_ratio, y_ratio = BUTTON_DEFAULTS.get(name, (0.5, 0.5))
    if overrides:
        override = overrides.get(name)
        if isinstance(override, (list, tuple)) and len(override) == 2:
            try:
                ox = float(override[0])
                oy = float(override[1])
            except (TypeError, ValueError):
                pass
            else:
                x_ratio = ox
                y_ratio = oy
    return x_ratio, y_ratio


COLOR_RANGES = {
    "red": [
        (np.array([0, 90, 90]), np.array([12, 255, 255])),
        (np.array([165, 90, 90]), np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([35, 80, 80]), np.array([85, 255, 255])),
    ],
}


def find_colored_button(name, image_bgr, min_area=600):
    config = BUTTON_COLOR_CONFIG.get(name)
    if config is None:
        return None

    region = config.get("region")
    color_key = config.get("color")
    if region is None or color_key not in COLOR_RANGES:
        return None

    height, width = image_bgr.shape[:2]
    left = max(int(region[0] * width), 0)
    top = max(int(region[1] * height), 0)
    right = min(int(region[2] * width), width)
    bottom = min(int(region[3] * height), height)

    if right <= left or bottom <= top:
        return None

    roi = image_bgr[top:bottom, left:right]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = None
    for lower, upper in COLOR_RANGES[color_key]:
        part = cv2.inRange(hsv, lower, upper)
        mask = part if mask is None else cv2.bitwise_or(mask, part)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    center_x = left + x + w // 2
    center_y = top + y + h // 2
    return center_x, center_y


def find_button_by_text(phrases, region=None, conf_threshold=60, data=None, screenshot=None):
    if data is None:
        if screenshot is None:
            screenshot = pyautogui.screenshot()
        data = pytesseract.image_to_data(screenshot, output_type=Output.DICT)
    total = len(data['text'])

    def word_at(i):
        return data['text'][i].strip().lower()

    def conf_at(i):
        try:
            return float(data['conf'][i])
        except (ValueError, TypeError):
            return -1.0

    def center_from_range(start, end):
        x_min = min(data['left'][idx] for idx in range(start, end))
        y_min = min(data['top'][idx] for idx in range(start, end))
        x_max = max(data['left'][idx] + data['width'][idx] for idx in range(start, end))
        y_max = max(data['top'][idx] + data['height'][idx] for idx in range(start, end))
        return (x_min + x_max) // 2, (y_min + y_max) // 2

    for phrase in phrases:
        tokens = [tok for tok in phrase.lower().split() if tok]
        if not tokens:
            continue
        for i in range(total):
            if word_at(i) != tokens[0] or conf_at(i) < conf_threshold:
                continue
            matched = True
            last_index = i + 1
            for offset in range(1, len(tokens)):
                j = i + offset
                if j >= total or word_at(j) != tokens[offset] or conf_at(j) < conf_threshold:
                    matched = False
                    break
                last_index = j + 1
            if not matched:
                continue
            cx, cy = center_from_range(i, last_index)
            if region:
                left, top, right, bottom = region
                if not (left <= cx <= right and top <= cy <= bottom):
                    continue
            return cx, cy

    return None


def click_double_passive_option(roll_double, overrides, attempt):
    _ = attempt
    screenshot = pyautogui.screenshot()
    image_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    key = "double_yes" if roll_double else "double_no"

    button_pos = find_colored_button(key, image_bgr)

    if not button_pos:
        region = (int(width * 0.3), int(height * 0.45), int(width * 0.7), int(height * 0.8))
        data = pytesseract.image_to_data(screenshot, output_type=Output.DICT)
        button_pos = find_button_by_text(["yes"] if roll_double else ["no"], region=region, data=data)

    if button_pos:
        x, y = button_pos
    else:
        x_ratio, y_ratio = resolve_button_ratio(key, overrides)
        x = int(width * x_ratio)
        y = int(height * y_ratio)
    click_at(x, y)
    time.sleep(0.25)


def compare_stats(old_stats, new_stats, required_stats_count):
    ignore_keywords = ["capacity", "replace", "keep"]

    def should_ignore(stat):
        stat_lower = stat.lower()
        return any(kw in stat_lower for kw in ignore_keywords)

    old_dict = {stat_name(s): (s, parse_value(s)) for s in old_stats if stat_name(s) and not should_ignore(s)}
    new_dict = {stat_name(s): (s, parse_value(s)) for s in new_stats if stat_name(s) and not should_ignore(s)}

    for name, (new_full, new_val) in new_dict.items():
        if name in old_dict:
            old_full, old_val = old_dict[name]
            if new_val is not None and old_val is not None:
                if new_val > old_val:
                    print(f"  - {Fore.GREEN}{new_full}{Style.RESET_ALL} (better than {old_full})")
                elif new_val < old_val:
                    print(f"  - {Fore.RED}{new_full}{Style.RESET_ALL} (worse than {old_full})")
                else:
                    print(f"  - {new_full} (unchanged)")
            else:
                if new_full.strip() == old_full.strip():
                    print(f"  - {new_full} (unchanged)")
                else:
                    print(f"  - {Fore.BLUE}{new_full}{Style.RESET_ALL} (changed from '{old_full}')")
        else:
            print(f"  - {Fore.BLUE}{new_full}{Style.RESET_ALL} (new stat!)")

    for name, (old_full, _) in old_dict.items():
        if name not in new_dict:
            print(f"  - {Fore.YELLOW}{old_full}{Style.RESET_ALL} (stat missing in new amulet)")

    print(f"Required stats threshold: {required_stats_count}")
def capture_and_process(config):
    img = capture_screen()
    cropped = find_amulet_region(img)
    save_cropped_image(cropped)

    old_stats, new_stats = extract_amulet_stats(cropped)
    required_stats_count = int(config["stats"].get("required_stats_count", 1))

    print("\n=== Supreme Star Amulet OCR Result ===")
    print("OLD AMULET:")
    for stat in old_stats:
        print("  -", stat)

    print("\nNEW AMULET:")
    for stat in new_stats:
        print("  -", stat)

    print("\nCOMPARISON:")
    compare_stats(old_stats, new_stats, required_stats_count)

    old_passives = [s for s in old_stats if "passive" in s.lower()]
    new_passives = [s for s in new_stats if "passive" in s.lower()]
    return old_stats, new_stats, old_passives, new_passives
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_and_prepare_config(show_summary=True):
    config = load_config()
    if show_summary:
        print_config_settings(config)
    validate_config(config)
    return config

def print_config_settings(config):
    content = "\n=== Loaded Config Settings ===\n"
    content += "Stats:\n"
    enabled_stats = 0
    for stat, entry in config["stats"].items():
        if stat in ("Stat", "Required stats count", "required_stats_count"):
            continue
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        enabled, value, mode = entry
        if not enabled:
            continue
        enabled_stats += 1
        mode_str = "REQUIRED" if mode == 1 else "OPTIONAL"
        content += f"  - {stat}: ENABLED, value: {value}, mode: {mode_str}\n"
    content += f"Total enabled stats: {enabled_stats}/5\n\n"

    content += "Passives:\n"
    primary_count = 0
    secondary_count = 0
    for passive, value in config["passives"].items():
        if passive == "Passive":
            continue
        try:
            mode = int(value)
        except Exception:
            continue
        if mode == 1:
            primary_count += 1
            content += f"  - {passive}: PRIMARY REQUIRED\n"
        elif mode == 2:
            secondary_count += 1
            content += f"  - {passive}: SECONDARY REQUIRED\n"
        elif mode == 3:
            content += f"  - {passive}: IGNORED\n"
    content += f"Total primary passives: {primary_count}/2\n"
    content += f"Secondary passive options: {secondary_count}\n\n"

    automation_cfg = config.get("automation", {})
    if automation_cfg:
        auto_roll = automation_cfg.get("Auto roll", False)
        roll_double = automation_cfg.get("Roll double passives", False)
        better_mode = automation_cfg.get("Better mode", False)
        button_positions = automation_cfg.get("Button positions", {})

        auto_roll_status = "ENABLED" if auto_roll else "DISABLED"
        roll_double_status = "YES" if roll_double else "NO"
        better_status = "ON" if better_mode else "OFF"

        content += "Automation:\n"
        content += f"  - Auto roll: {auto_roll_status}\n"
        content += f"  - Roll double passives: {roll_double_status}\n"
        content += f"  - Better mode: {better_status}\n"

        if isinstance(button_positions, dict) and button_positions:
            content += "  - Button overrides:\n"
            for name, value in button_positions.items():
                content += f"      {name}: {value}\n"
        content += "\n"

    print(content)
    if SENDWEBHOOK:        requests.post(
            WEBHOOKURL,
            json={"content": f"```{content}```"},
            headers={"Content-Type": "application/json"}
        )

def validate_config(config):
    max_values = {
        "Pollen": 20,
        "White Pollen": 70,
        "Red Pollen": 70,
        "Blue Pollen": 70,
        "Bee Gather Pollen": 70,
        "Instant Conversion": 12,
        "Convert Rate": 1.25,
        "Bee Ability Rate": 7,
        "Critical Chance": 7
    }
    min_values = {
        "Pollen": 5,
        "White Pollen": 15,
        "Red Pollen": 15,
        "Blue Pollen": 15,
        "Bee Gather Pollen": 15,
        "Instant Conversion": 3,
        "Convert Rate": 1.05,
        "Bee Ability Rate": 1,
        "Critical Chance": 1
    }
    errors = []

    enabled_stats = 0
    required_stats = 0
    optional_stats = 0
    required_stats_count = config["stats"].get("required_stats_count", 1)
    for stat, value in config["stats"].items():
        if stat in ("Stat", "Required stats count", "required_stats_count"):
            continue
        if not isinstance(value, list) or len(value) < 3:
            continue
        enabled, val, mode = value
        if not enabled:
            continue
        if mode not in (1, 2):
            errors.append(f"Stat '{stat}' has invalid mode {mode}. Use 1 or 2.")
            continue
        if isinstance(val, str):
            val = val.replace("%", "").replace("x", "")
        try:
            numeric_val = float(val)
        except Exception:
            errors.append(f"Stat '{stat}' value '{value}' is not a valid number.")
            continue
        if numeric_val > max_values[stat]:
            errors.append(f"Stat '{stat}' value {numeric_val} is above max ({max_values[stat]}).")
        if mode == 1:
            enabled_stats += 1
            required_stats += 1
        elif mode == 2:
            optional_stats += 1
    if enabled_stats > 5:
        errors.append(f"More than 5 required stats are enabled ({enabled_stats}/5).")
    if (required_stats + optional_stats) < required_stats_count:
        errors.append(f"Not enough required/optional stats enabled: {required_stats + optional_stats} (need at least {required_stats_count}).")

    required_passives = []
    secondary_passives = []
    for passive, value in config["passives"].items():
        if passive == "Passive":
            continue
        if not isinstance(value, list) or len(value) < 2:
            continue
        enabled, mode = value
        if not enabled:
            continue
        if mode not in (1, 2, 3):
            errors.append(f"Passive '{passive}' has invalid mode {mode}. Use 1, 2, or 3.")
            continue
        if mode == 1:
            required_passives.append(passive)
        elif mode == 2:
            secondary_passives.append(passive)
    if len(required_passives) > 2:
        errors.append(f"Too many required passives are enabled ({len(required_passives)}/2).")

    if errors:
        print("\n[CONFIG ERROR] The following issues were found in config.json:")
        for err in errors:
            print(" -", err)
        print("\nPlease fix these errors and restart the script.")
        exit(1)

def is_amulet_accepted(old_stats, new_stats, old_passives, new_passives, config):
    stat_cfg = {k: v for k, v in config["stats"].items() if isinstance(v, list) and len(v) >= 3}
    passive_cfg = {k: v for k, v in config["passives"].items() if isinstance(v, list) and len(v) >= 2}

    required_stats_count = int(config["stats"].get("required_stats_count", 1))
    decline_reasons = []

    def add_reason(reason: str) -> None:
        if reason not in decline_reasons:
            decline_reasons.append(reason)

    def build_stat_map(stats):
        mapping = {}
        for entry in stats:
            name = stat_name(entry)
            if not name:
                continue
            mapping[name] = (entry, parse_value(entry))
        return mapping

    def normalize_passives(passive_lines):
        names = []
        for passive in passive_lines:
            name = passive.strip().replace("+Passive:", "").replace("Passive:", "").strip().title()
            if name:
                names.append(name)
        return names

    old_stat_map = build_stat_map(old_stats)
    new_stat_map = build_stat_map(new_stats)
    old_passive_list = normalize_passives(old_passives)
    new_passive_list = normalize_passives(new_passives)
    old_passive_set = set(old_passive_list)

    better_mode_enabled = bool(config.get("automation", {}).get("Better mode", False))

    if better_mode_enabled:
        reasons = []
        missing_stats = []
        regressed_stats = []

        old_keys = set(old_stat_map.keys())
        new_keys = set(new_stat_map.keys())

        for name in old_keys:
            if name not in new_stat_map:
                missing_stats.append(name)
                continue
            old_full, old_val = old_stat_map[name]
            new_full, new_val = new_stat_map[name]
            if old_val is not None and new_val is not None:
                if new_val < old_val:
                    regressed_stats.append(name)
                    continue
            elif new_full.strip() == old_full.strip():
                regressed_stats.append(name)
                continue

        extra_stats = sorted(name for name in new_keys if name not in old_stat_map)
        missing_passives = [p for p in old_passive_set if p not in new_passive_list]

        if missing_stats:
            reasons.append("Better mode: missing stats -> " + ", ".join(sorted(missing_stats)))
        if regressed_stats:
            reasons.append("Better mode: non-improved stats -> " + ", ".join(sorted(regressed_stats)))
        for name in extra_stats:
            reasons.append(f"Extra stat '{name}' present in new amulet.")
        for passive_name in missing_passives:
            reasons.append(f"Required passive '{passive_name}' missing from amulet.")

        if reasons:
            print("[DECLINED] SSA was declined for the following reasons:")
            decline_content = "** **\n\n **Declined:**\n"
            for reason in reasons:
                print(" - ", reason)
                decline_content += " - " + str(reason) + "\n"
            if SENDWEBHOOK:                
                with open("current_amulet.png", "rb") as f:
                    requests.post(
                        WEBHOOKURL,
                        files={"file": f},
                        data={"content": (decline_content + "```")}
                    )

            return False



        print("[ACCEPTED] SSA is better than or equal to old amulet")
        if SENDWEBHOOK:            
            with open("current_amulet.png", "rb") as f:
                        requests.post(
                            WEBHOOKURL,
                            files={"file": f},
                            data={"content": "Accepted amulet!"}
                        )
        return True

    stat_count = 0
    for raw_stat in new_stats:
        name = stat_name(raw_stat)
        if not name:
            continue
        cfg = stat_cfg.get(name)
        if cfg and not cfg[0]:
            add_reason(f"Disabled stat '{name}' found in amulet.")

    for stat_key, cfg in stat_cfg.items():
        if not isinstance(cfg, list) or len(cfg) < 3:
            continue
        enabled, min_val, mode = cfg
        if not enabled:
            continue
        new_entry = new_stat_map.get(stat_key)

        if mode == 1:
            if new_entry is None:
                add_reason(f"Required stat '{stat_key}' missing from amulet.")
                continue
            new_val = new_entry[1]
            if new_val is not None and new_val < min_val:
                add_reason(f"Stat '{stat_key}' value {new_val} is below required minimum ({min_val}).")
                continue
            stat_count += 1
        elif mode == 2:
            if new_entry is None:
                continue
            new_val = new_entry[1]
            if new_val is not None and new_val < min_val:
                add_reason(f"Stat '{stat_key}' value {new_val} is below required minimum ({min_val}).")
                continue
            stat_count += 1

    if stat_count < required_stats_count:
        add_reason(f"Required stats threshold not met: {stat_count}/{required_stats_count} qualifying stats.")

    primary_required = [name for name, cfg in passive_cfg.items() if cfg[0] and cfg[1] == 1]
    secondary_required = [name for name, cfg in passive_cfg.items() if cfg[0] and cfg[1] == 2]

    unknown_passives = [name for name in new_passive_list if name not in passive_cfg]
    for name in unknown_passives:
        add_reason(f"Passive '{name}' is not configured in config.json.")

    remaining_passives = [name for name in new_passive_list if name in passive_cfg and passive_cfg[name][0]]
    for req in primary_required:
        if req in remaining_passives:
            remaining_passives.remove(req)
        else:
            add_reason(f"Required passive '{req}' missing from amulet.")

    if secondary_required:
        if not any(name in secondary_required for name in remaining_passives):
            expected = ", ".join(secondary_required)
            add_reason(f"Secondary passive missing. Expected one of: {expected}.")

    if decline_reasons:
        print("[DECLINED] SSA was declined for the following reasons:")
        decline_content = "** **\n\n **Declined:**\n"
        for reason in decline_reasons:
            print(" - ", reason)
            decline_content += " - " + str(reason) + "\n"
        if SENDWEBHOOK:            
            with open("current_amulet.png", "rb") as f:
                requests.post(
                WEBHOOKURL,
                files={"file": f},
                data={"content": decline_content}
            )

        return False
    
    if SENDWEBHOOK:        
        with open("current_amulet.png", "rb") as f:
            requests.post(
                WEBHOOKURL,
                files={"file": f},
                data={"content": "Accepted amulet!"}
            )
    print("[ACCEPTED] SSA meets all requirements!")
    return True

def perform_amulet_roll(config, roll_double, stop_event, button_overrides, attempt):

    if stop_event.is_set():
        return False

    initial_pause = 0.9 if attempt == 1 else 0.35
    time.sleep(initial_pause)
    click_double_passive_option(roll_double, button_overrides, attempt)
    time.sleep(1.0)

    old_stats, new_stats, old_passives, new_passives = capture_and_process(config)
    accepted = is_amulet_accepted(old_stats, new_stats, old_passives, new_passives, config)

    if not accepted and not stop_event.is_set():
        print("[DEBUG] Declined - sending 'E' to reroll")
        keyboard.send('e')
        time.sleep(0.9)

    return accepted


def automation_loop(config, stop_event):
    automation_cfg = config.get("automation", {})
    auto_roll_enabled = automation_cfg.get("Auto roll", True)
    if not auto_roll_enabled:
        print("[INFO] Auto roll disabled in config. Waiting for next start command.")
        return

    roll_double = bool(automation_cfg.get("Roll double passives", False))
    button_overrides = automation_cfg.get("Button positions", {})
    attempt = 1

    print("[INFO] Opening first SSA prompt...")
    print("[DEBUG] Sending 'E' to open prompt")
    keyboard.send('e')
    time.sleep(1.0)

    while not stop_event.is_set():
        print(f"\n[INFO] Starting roll attempt #{attempt} (double passives {"ON" if roll_double else "OFF"})")
        accepted = perform_amulet_roll(config, roll_double, stop_event, button_overrides, attempt)

        if stop_event.is_set():
            break

        if accepted:
            print("[INFO] Accepted amulet detected. Stopping automation.")
            stop_event.set()
            break

        attempt += 1
        time.sleep(0.25)


if __name__ == "__main__":
    os.system('cls')
    config = load_and_prepare_config(show_summary=True)
    print("[INFO] Script ready. Press '=' to begin automated rolling. Press '-' to stop.")

    stop_event = threading.Event()
    start_event = threading.Event()

    def trigger_start():
        if not stop_event.is_set():
            start_event.set()

    keyboard.add_hotkey('-', lambda: stop_event.set())
    keyboard.add_hotkey('=', trigger_start)

    try:
        while not stop_event.is_set():
            if not start_event.wait(timeout=0.1):
                continue
            if stop_event.is_set():
                break
            start_event.clear()
            config = load_and_prepare_config(show_summary=False)
            print("[INFO] Configuration reloaded from disk. Starting automation...")
            automation_loop(config, stop_event)
            break
    finally:
        keyboard.unhook_all_hotkeys()
        print("[INFO] Script stopped.")
