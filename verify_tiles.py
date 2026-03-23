import cv2
import os
import glob
from dragon_oracle.config import OracleConfig
from dragon_oracle.tiles import TileDetector
from dragon_oracle.grid import GridClusterer

def main():
    config = OracleConfig()
    detector = TileDetector(config)
    clusterer = GridClusterer(config)
    
    test_dir = "test_images"
    images = glob.glob(os.path.join(test_dir, "*.jpg"))
    if not images:
        print("No test images found.")
        return
        
    for path in images[:3]:
        print(f"\nProcessing {os.path.basename(path)}...")
        frame = cv2.imread(path)
        if frame is None:
            continue
            
        frame = cv2.resize(frame, (config.process_width, config.process_height))
            
        # 1. Detect Tiles (now using HoughLines under the hood)
        tiles = detector.detect(frame)
        print(f"  Detected {len(tiles)} virtual tiles from grid intersections.")
        
        # 2. Cluster into Grid
        if tiles:
            centers = [t.center for t in tiles]
            grid_result = clusterer.cluster(centers)
            print(f"  Clustered into Grid: {grid_result.num_rows} rows x {grid_result.num_cols} cols")
        else:
            print("  No grid could be formed.")

if __name__ == "__main__":
    main()
