"""
Dragon Oracle — FastAPI web server (screenshot edition).

All on localhost, no SSL needed.
http://localhost:8000
"""
import base64
import os
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from dragon_oracle.board import Board
from dragon_oracle.capture import crop_region, split_into_cells, take_full_screenshot
from dragon_oracle.config import OracleConfig
from dragon_oracle.detect import detect_flipped_cells

# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------

config = OracleConfig()
board = Board(config)

reference_cells: Optional[List[List[np.ndarray]]] = None
calibrated: bool = False

app = FastAPI(title="Dragon Oracle")
_template_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_template_dir)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@app.post("/calibrate/screenshot")
async def calibrate_screenshot():
    """Take a full-screen screenshot and return as base64 JPEG."""
    img = take_full_screenshot()
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return JSONResponse(content={"image": b64, "width": w, "height": h})


@app.post("/calibrate/set-region")
async def calibrate_set_region(body: dict):
    """Save the board region and grid dimensions."""
    global config, board

    config.region_left = int(body["left"])
    config.region_top = int(body["top"])
    config.region_width = int(body["width"])
    config.region_height = int(body["height"])
    config.board_rows = int(body.get("rows", 6))
    config.board_cols = int(body.get("cols", 10))

    board = Board(config)
    return JSONResponse(content={"status": "ok"})


@app.post("/calibrate/set-reference")
async def calibrate_set_reference():
    """Capture the board region (all cards face-down) as the reference."""
    global reference_cells, calibrated

    if config.region_width == 0 or config.region_height == 0:
        return JSONResponse(status_code=400,
                            content={"error": "Set region first."})

    screenshot = take_full_screenshot()
    board_img = crop_region(screenshot, config)
    reference_cells = split_into_cells(board_img, config.board_rows, config.board_cols)
    calibrated = True
    return JSONResponse(content={
        "status": "ok",
        "cells": config.board_rows * config.board_cols,
    })


# ---------------------------------------------------------------------------
# Capture — screenshot + detect + auto-place
# ---------------------------------------------------------------------------

@app.post("/capture")
async def capture():
    """Take a screenshot, detect flipped cards, auto-place them."""
    if not calibrated or reference_cells is None:
        return JSONResponse(status_code=400,
                            content={"error": "Not calibrated yet."})

    screenshot = take_full_screenshot()
    board_img = crop_region(screenshot, config)

    cards = detect_flipped_cells(
        board_img, reference_cells, config, board.occupied_cells_set()
    )

    placed = []
    skipped = []
    for card in cards:
        result = board.auto_place(card["jpeg"], card["row"], card["col"])
        pos = {"row": card["row"], "col": card["col"]}
        if result == "ok":
            placed.append(pos)
        else:
            skipped.append(pos)

    return JSONResponse(content={
        "detected": len(cards),
        "placed": len(placed),
        "skipped": len(skipped),
        "positions": placed,
        "revealed": board.revealed_count,
        "total": board.total_positions,
    })


# ---------------------------------------------------------------------------
# Status / Move / Reset
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    return JSONResponse(content={
        **board.to_json(),
        "calibrated": calibrated,
    })


@app.post("/move")
async def move_card(body: dict):
    fr = int(body.get("from_row", -1))
    fc = int(body.get("from_col", -1))
    tr = int(body.get("to_row", -1))
    tc = int(body.get("to_col", -1))
    result = board.move_card(fr, fc, tr, tc)
    return JSONResponse(content={"status": result})


@app.post("/reset")
async def reset():
    board.clear()
    return JSONResponse(content={"status": "ok"})


@app.post("/recalibrate")
async def recalibrate():
    global calibrated, reference_cells
    calibrated = False
    reference_cells = None
    board.clear()
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n  Dragon Oracle (Screenshot Edition)")
    print(f"  Grid: {config.board_rows} x {config.board_cols}")
    print("  Open: http://localhost:8000\n")

    uvicorn.run(app, host="127.0.0.1", port=8000)
