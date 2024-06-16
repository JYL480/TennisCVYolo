from ultralytics import YOLO
# from pathlib import path


input_video_path = ("data/input_video.mp4")
print(input_video_path)

model = YOLO('yolov8x.pt')

result = model.predict(input_video_path, save=True)

print(result)