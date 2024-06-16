# Now we have to analyse the video frame by frame! as save it in frames!!
from utils import (read_video, save_video)
from tracker import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt

def main():
    video_path = "data/input_video.mp4"
    video_frames = read_video(video_path)


    # We create an instance of this first!
    player_tracker = PlayerTracker("yolov8x")
    # Place in the model we trained!!
    ball_tracker = BallTracker("models/last.pt")
    player_detection = player_tracker.detect_frames(video_frames
                                                    , read_from_stub=True,
                                                    stub_path="tracker_stubs/player_detections.pkl")

    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stub=True,
                                                stub_path="tracker_stubs/ball_detections.pkl") 
    
    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)


    # Court Line detector 
    court_line_detector_model = CourtLineDetector("models/restnet50.pth")

    # We give it 1 frame, bcause the court will not shift!
    court_kps = court_line_detector_model.predict(video_frames[0])

    # Filter out the players
    # You should only have 2 players!
    player_detection = player_tracker.choose_and_filter_players(court_kps, player_detection)

    # Initialise the mini court!
    mini_court = MiniCourt(video_frames[0])

    # Detect ball hits
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detection)
    print(ball_shot_frames)


    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detection, 
                                                                                                          ball_detection,
                                                                                                          court_kps)


    # Draw the output (All the outputs are cascaded!!)
    # Note that you video frames in the parameters should be from the previous draw ons
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detection)
    output_video_frames =player_tracker.draw_bboxes(output_video_frames, player_detection)
    

    # Draw the keypoints
    # You will use the previous output frames to add in the kps
    output_video_frames = court_line_detector_model.draw_keypoints_on_video(video_frames=output_video_frames, keypoints=court_kps)


    # Draw the mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    


    # Place the frame number on each frame in top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 255), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")



if __name__ == "__main__": 
    main()