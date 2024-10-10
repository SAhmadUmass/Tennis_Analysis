from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class BallTracker():
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        try:
            results = self.model.track(frame, persist=True,conf=0.15)[0]
            ball_dict = {}
            for box in results.boxes:
                if box.xyxy is not None:
                    #track_id = int(box.id.tolist()[0])
                    result = box.xyxy.tolist()[0]
                    # Uncomment if multiple classes are detected
                    # object_cls_id = box.cls.tolist()[0] 
                    # object_cls_name = results.names[object_cls_id]
                    # if object_cls_name == "ball":
                    #x1,y1,x2,y2 = box.xyxy[0] #extract min/max xy pairs
                    

                    ball_dict[1] = result
                else:
                    print("Warning: Detection box is None, skipping this box.")
            return ball_dict
        except Exception as e:
            print(f"Error in detect_frame: {e}")
            return {}
        
    def detect_frames(self, frames):
        ball_detections = []

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        return ball_detections
    
    def extract_ball_positions(self,ball_detections):
        ball_positions = {}
        for frame_idx,ball_dict in enumerate(ball_detections):
            for track_id,bbox in ball_dict.items():
                x1,y1,x2,y2 = bbox
                center_x = int((x1+x2)/2)
                center_y = int((y1+y2)/2)
                if track_id not in ball_positions:
                    ball_positions[track_id] = []
                ball_positions[track_id].append((frame_idx,center_x,center_y)) #find the center and append to the list
        return ball_positions
    
    @staticmethod
    def moving_average(data, window_size=5):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def smooth_ball_positions(self, ball_positions,window_size=5):
        smoothed_positions = {}
        for track_id, positions in ball_positions.items():
            frame_idxs,x,y = zip(*positions)
            smoothed_x = self.moving_average(x,window_size)
            smoothed_y = self.moving_average(y,window_size)
            #adjust frame indices
            adjusted_frame_indices = frame_idxs[window_size-1:]
            smoothed_positions[track_id] = list(zip(adjusted_frame_indices,smoothed_x,smoothed_y))
        return smoothed_positions
    
    def predict_future_positions(self,smoothed_positions, future_steps=30):
        predicted_positions ={}
        for track_id,positions in smoothed_positions.items():
            frame_idxs,smoothed_x , smoothed_y = zip(*positions)
            frame_idxs = np.array(frame_idxs).reshape(-1,1)

            # predict x
            model_x = LinearRegression()
            model_x.fit(frame_idxs,smoothed_x)
            future_frame_idxs = np.arange(frame_idxs[-1][0] + 1, frame_idxs[-1][0] + future_steps + 1).reshape(-1, 1)
            predicted_x = model_x.predict(future_frame_idxs)

            # predict y
            model_y = LinearRegression()
            model_y.fit(frame_idxs,smoothed_y)
            predicted_y = model_y.predict(future_frame_idxs)

            predicted_positions[track_id] = list(zip(predicted_x.astype(int),predicted_y.astype(int)))

        return predicted_positions
    
    def visualize_trajectory(self,frame,trajectory, color=(255,0,0),thickness = 2):
        for i in range(1,len(trajectory)):
            cv2.line(frame, trajectory[i-1],trajectory[i],color,thickness)
        return frame





    
