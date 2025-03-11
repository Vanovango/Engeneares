"""
Используя пред-обученную модель 'yolo11n.pt' из библиотеки 'ultralytics' распознайте объекты на изображении 'peoples.png'
Для распознавания используйте метод '.predict'
У данного метода существуют следующие аргументы:
    source - file_path
    show - bool
    save - bool
    conf - float (обводить объекты с данной или большей вероятностью уверенности модели)
В качестве ответа предоставить фото, на котором выделены все объекты с вероятностью более 75%
"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(source="peoples.png", show=True, save=True, conf=0.5)
