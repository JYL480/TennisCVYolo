from ultralytics import YOLO
import cv2
import numpy as np
import pickle

from utils import (get_center_of_bbox, measure_distance)


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        
        # This will give the dictionary of the tracked player!
        player_detections_first_frame = player_detections[0]


        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            # Will create a dictionary and will filter out those player_id that is in the chose player list
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            
            # Append this dictionary and append this dict to the list!
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)


            # This min_distance is for comparison!
            min_distance = float('inf')
            # You will for loop to all the keypoints on the court! 
            # and you will see which one is the closest!
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        # Will extra out the index 1 distances from the tuple and will sort it
        # Oh and the use of lambda allows you to quickly create a throwaway function?
        # Which will allow you to get all the 1 index in the list
        distances.sort(key = lambda x: x[1])
        
        # Choose the first 2 tracks
        # As these 2 are supposed to be the closest and on the court!
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict
    
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # This is to store the convert the python object into a pickle format
        # Which is into a byte stream!! So the numpy array of frames will be converted
        # Then when you want to use it, you can load it back!
        # YOu will need way less time to process the image as you will load from 
        # stub file!
        if read_from_stub and stub_path is not None:
            # Open the pickle file using rb (read binary)
            with open(stub_path, 'rb') as f:
                # load the pickle file in byte stream format return the numpy array
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        # If pickle not found, then we will create one and write in binary
        # with pickle.dump!
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections
    
    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # We get the axis of each persons box min and max x and y coordinates
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        # Hmm we are always returning a list
        return output_video_frames
