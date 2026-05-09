from ultralytics import YOLO

# 1. Use the nano version (yolo11n-seg) - it's the lightest
model = YOLO("yolo11n-seg.pt") 

# 2. Use stream=True to prevent RAM buildup
# 3. Reduce imgsz to 320 or 480 to save memory
results = model.predict(
    source="/mnt/NewVolume/Internship/Week-3/traffic_simple.mp4",
    show=False, 
    save=True, 
    stream=True, 
    imgsz=320, 
    device='cpu' # Use 'cpu' if you don't have a dedicated NVIDIA GPU
)

# Because stream=True is a generator, we must loop through it to actually run
for r in results:
    pass
