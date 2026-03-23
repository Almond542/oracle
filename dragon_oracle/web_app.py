import cv2
import numpy as np
import socket
import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import Response, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from dragon_oracle.config import OracleConfig
from dragon_oracle.tiles import TileDetector
from dragon_oracle.memory import CardClassifier, MemoryBoard

# Initialize components
config = OracleConfig()
tile_detector = TileDetector(config)
classifier = CardClassifier(config)
board = MemoryBoard(config)

latest_grid_image = None

# FastAPI app
app = FastAPI(title="Dragon Oracle Assistant")
template_dir = os.path.join(os.path.dirname(__file__), "templates")

# Mount templates (assuming there is a templates folder)
templates = Jinja2Templates(directory=template_dir)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("mobile.html", {"request": request})


@app.post("/capture")
async def capture(image: UploadFile = File(...)):
    global latest_grid_image

    if not image:
        return JSONResponse(status_code=400, content={"message": "No image provided"})

    img_bytes = await image.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image"})

    # Resize to processing resolution
    frame = cv2.resize(
        frame, (config.process_width, config.process_height)
    )

    tiles = tile_detector.detect(frame)
    if not tiles:
        return JSONResponse(content={
            "error": "No tiles detected - try a clearer photo",
            "cards_revealed": board.revealed_count,
        })

    # MemoryBoard handles filtering, grid mapping, classification, storage
    grid_image = board.process_capture(frame, tiles, classifier)
    latest_grid_image = grid_image

    _, buf = cv2.imencode(".jpg", grid_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/composite")
async def get_composite():
    if latest_grid_image is None:
        return JSONResponse(status_code=404, content={"message": "No grid yet - capture a photo first"})

    _, buf = cv2.imencode(
        ".jpg", latest_grid_image, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/reset")
async def reset():
    global latest_grid_image
    board.clear()
    latest_grid_image = None
    print("Memory cleared")
    return {"status": "ok", "message": "Memory cleared"}


@app.get("/status")
async def get_status():
    rows, cols = board.grid_size
    return {
        "cards_revealed": board.revealed_count,
        "total_positions": board.total_positions,
        "grid": f"{rows}x{cols}",
    }


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    rows, cols = board.grid_size
    print(f"\n  Dragon Oracle - Memory Board ({rows}x{cols} grid)")
    print(f"  Open on phone: http://{local_ip}:8000")
    print(f"  Grid on PC:    http://localhost:8000/composite\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
