from utils import save_video,read_video
from trackers import PlayerTracker,BallTracker
import cv2

def main():
    input_video_path = "input_videos/short_input_video.mp4"
    video_frames = read_video(input_video_path)

    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='Models/YoloV5_best.pt')
    player_detections = player_tracker.detect_frames(video_frames)
    ball_detections = ball_tracker.detect_frames(video_frames)

    ball_positions = ball_tracker.extract_ball_positions(ball_detections)
    smoothed_ball_positions = ball_tracker.smooth_ball_positions(ball_positions,window_size=5)
    predicted_ball_positions = ball_tracker.predict_future_positions(smoothed_ball_positions,future_steps=30)


    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections,ball_detections)

    for track_id,smoothed in smoothed_ball_positions.items():
        smoothed_coords = [(int(x),int(y)) for _,x,y in smoothed]
        predicted = predicted_ball_positions.get(track_id,[])

        for i in range(1, len(smoothed_coords)):
            cv2.line(output_video_frames[-1],smoothed_coords[i-1],smoothed_coords[i], (0,255,255), 2)
        for i in range(1, len(predicted)):
            cv2.line(output_video_frames[-1],predicted[i-1],predicted[i],(255,0,0),2)
        
    if video_frames:
        save_video(video_frames, 'output_videos/output_video.mp4')

if __name__ == "__main__":
    main()
