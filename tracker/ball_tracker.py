from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    

    def detect_frame(self,frame):
        # We will predict the ball instead of track as there is only 1 ball
        # note that persits is only for trackign!
        results = self.model.predict(frame)[0]

        ball_dict = {}
      
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict
    
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        # This is to store the convert the python object into a pickle format
        # Which is into a byte stream!! So the numpy array of frames will be converted
        # Then when you want to use it, you can load it back!
        # YOu will need way less time to process the image as you will load from 
        # stub file!
        if read_from_stub and stub_path is not None:
            # Open the pickle file using rb (read binary)
            with open(stub_path, 'rb') as f:
                # load the pickle file in byte stream format return the numpy array
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        # If pickle not found, then we will create one and write in binary
        # with pickle.dump!
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections #return the xyxy coordinate of the ball!
    
    def draw_bboxes(self,video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # We get the axis of each persons box min and max x and y coordinates
                cv2.putText(frame, f"Tennis Ball {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        # Hmm we are always returning a list
        return output_video_frames
    
    # TO track the ball!Where you will return the row index of where the ball was hit!
    # The row can be used in df to get the x and y coordinates
    def get_ball_shot_frames(self,ball_positions):

        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0
        minimum_change_frames_for_hit = 25
        # There is a min because we cannot get all the frames as this 24fps!

        # To get the middle y cooridnate of the ball from the min and mac y coordinates
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        # Get the rolling mean of the middle y coordinate with window 5 to see the changes!
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        # Get the difference between consecutive elements
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                # Plus 1 which is account for the next frame!
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1


        # Get a list of the indexes when the ball was hit!
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        # index_list  = [df_ball_positions[df_ball_positions["ball_hit"]==1].index]
        return frame_nums_with_ball_hits

    # This is to make the ball tracking smoother!!
    def interpolate_ball_positions(self, ball_positions):

            # Rmb that ball_position is from the ball tracker and it returns a dictornary with key = 1 as the ball position
            # we will get the coordinates, if non is found then will replace with empty list
            ball_positions = [x.get(1,[]) for x in ball_positions]
            # convert the list into pandas dataframe
            df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

            # interpolate the missing values
            # Note that interpolate() and bfill() are only found on pandas!
            df_ball_positions = df_ball_positions.interpolate() #This will fill in the NaN numbers
            df_ball_positions = df_ball_positions.bfill() #Another method to fill in any remaining NaN numbers at the end and start

            # Convert the dataframe into a list of dictionaries
            # {1:x}, 1 is the id and x is the boundary box
            # change the numpy array and into a list!
            ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

            return ball_positions