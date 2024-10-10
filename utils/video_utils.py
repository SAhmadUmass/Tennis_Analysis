import os,cv2


def save_video(output_video_frames, output_video_path):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    if not out.isOpened():
        print(f"Error: Cannot open video writer for {output_video_path}")
        return
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
    print(f'Video saved to {output_video_path}')



def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return []
    
    print(f"Successfully opened {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()  # Ensure release is called as a function
    return frames

