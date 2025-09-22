# Auto SSA Roller

Automatically rolls Supreme Star Amulets until the desired requirements from config.json are met.
Does not automatically claim the SSA (I didn't want to test that because I don't want to accidentally lose my own SSA lol)

If I feel like it I will update this to add full automation (auto claiming SSA)

## Prerequisites
- **Python 3.8+**
- **Tesseract OCR** installed locally (default path: C:\Program Files\Tesseract-OCR\tesseract.exe). Update the path in main.py if Tesseract is somewhere else.

## Installation
1. Clone or download this repository.
2. Dependencies are installed automatically when main.py runs, but you can pre-install them:
   `
   pip install pyautogui pytesseract Pillow opencv-python numpy keyboard colorama
   `
3. Ensure Tesseract OCR is installed (it is an external installation).

## Configuration
Edit config.json to control behaviour:
- stats: enable/disable stats and set minimum values. Modes: 1 (required), 2 (optional). 
required_stats_count defines how many qualifying stats must appear.
- passives: enable passives and choose mode (1 required, 2 secondary required, 3 ignored).
- utomation:
  - Auto roll: true/false to enable auto rolling.
  - Roll double passives: self explanatory
  - Better mode: when true, config requirements are ignored and the script only accepts a new SSA if every stat from the old SSA is present and better or the same.

## Usage
1. Configure config.json to your liking.
2. Run main.py 
3. Press = to activate the auto-roller. You may have to click "no" for the first prompt or it will get stuck on it. You'll notice if it happens. I'm not sure how to solve this problem.
4. Press - at any time to halt the script.

## Notes
- Screenshots are saved as current_amulet.png for debugging.
- Logs are printed in console.
- Ensure Roblox remains the active window during rolling.

