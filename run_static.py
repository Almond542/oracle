"""Quick script to run the Dragon Oracle pipeline on static images.

Usage:
    python run_static.py                          # all test images
    python run_static.py test_images/IMG_xxx.jpg  # single image
    python run_static.py --debug                  # with debug windows

Controls:
    Space  - pause/resume
    n      - next image
    d      - toggle debug windows
    s      - toggle spiral direction
    q/Esc  - quit
"""
import sys
from dragon_oracle.main import main

if __name__ == "__main__":
    # Default to test_images directory if no --image arg provided
    if "--image" not in sys.argv and len(sys.argv) > 1:
        # Check if first positional arg is an image path
        first_arg = sys.argv[1]
        if not first_arg.startswith("--"):
            sys.argv = [sys.argv[0], "--image", first_arg] + sys.argv[2:]
    elif "--image" not in sys.argv:
        sys.argv.append("--image")
        sys.argv.append("test_images")

    # Default to no-aruco mode for static images
    if "--no-aruco" not in sys.argv:
        sys.argv.append("--no-aruco")

    main()
