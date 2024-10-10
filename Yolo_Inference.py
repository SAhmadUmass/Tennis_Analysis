from ultralytics import YOLO
import shutil
model = YOLO('Models/YoloV5_best.pt')
#'Models/YoloV5_last.pt' -Fine Tuned model trained on tennis balls

result = model.predict('input_videos/image.png',conf=0.2, save = True)

'''
print(result)
print('Boxes:')
for box in result[0].boxes:
    print(box)
'''
#shutil.move('/opt/homebrew/runs', '/Users/shaheerahmad/Documents/Tennis_Analysis')

