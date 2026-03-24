# Dragon Oracle AR Assistant

Dragon Oracle is an Augmented Reality (AR) assistant designed to read, analyze, and map game boards utilizing computer vision. By identifying game tiles and classifying unique patterns or cards, it assists players by memorizing boards locally, offering a live AR viewfinder via webcam or a companion mobile web layout.

## Features
- **Local Computer Vision Pipeline**: Detects game grids, contour lines, and individual tiles using OpenCV, with ArUco marker perspective warping for clean image parsing.
- **Card Memorization & Classification**: Captures frames on the fly, remembers card positions over the session, and updates the board accurately.
- **Web Companion App**: Allows you to view a composite board on your PC while using your mobile phone’s camera to capture changes. 
- **Local AR Mode**: Native OpenCV window overlay displaying the memory path calculated via spiral patterns.
- **Fully Local & Private**: All image processing and classification happens locally on your machine—no cloud API keys necessary, ensuring your privacy completely.

## Prerequisites
- Python 3.9+ 
- A connected webcam (for the local AR mode) or a mobile device on the same network (for the Web Companion App).

## Installation

1. Create a virtual environment (Optional, but recommended)
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install the required packages via `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You have two main ways to use the Drag Oracle assistant:

### 1. Web Companion Mode (Recommended)
This approach runs a FastAPI web application. You take pictures on your mobile phone, and view the memory board overlaid on your PC.

Run the web app from the main project folder:
```bash
python -m dragon_oracle.web_app
```
Then follow the terminal output instructing you how to connect. The terminal will provide two URLs:
- **Phone Camera Capture**: `http://<your-local-ip>:8000` (Open this on your phone's browser)
- **PC Board Overlay**: `http://localhost:8000/composite` (Open this on your PC screen)

### 2. Local AR Application
Use this approach if you want to use a webcam attached to your PC directly or process static images in a directory.

Start the AR App (uses webcam 0 by default):
```bash
python -m dragon_oracle.main --camera 0
```
Run with a static directory or test image:
```bash
python -m dragon_oracle.main --image ./test_images/
```

**AR Controls:**
- `Space`: Pause/resume processing
- `d`: Show/Hide debug visualization windows (shows grid extraction step)
- `s`: Switch spiral pattern calculations (Clockwise vs. Counter-Clockwise)
- `q` or `ESC`: Quit

## Security Considerations
- **No API Keys or Secrets**: This tool is designed to utilize local computer vision libraries to identify tiles rather than depending on external cloud providers.
- **Local Network Access**: The web companion app binds to port `8000` via `0.0.0.0`, meaning it's reachable by other devices on your local network/WiFi. Ensure you run this application on a trusted private network.
