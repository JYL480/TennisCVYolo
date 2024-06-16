# here we will create the a simulation of the court!!

import cv2
import sys

# sys.path.append("../")
import constants
from utils import *
import numpy as np

class MiniCourt():
    def __init__(self, frame):

        # Rmb that these are pixels!! YOu have to convert them!!
        # We will create the boundaries and buffer of the mini court on the screen
        
        # This is the width of the white box
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        
        # This from the box of the white rectangle thing to the edge of the video
        self.buffer = 50
        self.padding_court=20
    
        # When we call the MiniCourt constructor, we would want the functions below to be called!
        self.set_canvas_bg_box_position(frame)
        
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()


    # To get the person position on the mini court!
    def convert_bounding_boxes_to_mini_court_coordinates(self,player_boxes, ball_boxes, original_court_key_points ):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }


        # We containt a list of dicts!
        output_player_boxes= []
        output_ball_boxes= []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]

            # So that we are able to find the xy coordinates of the ball to a player!
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():

                # Note that we are getting the foot position as it is the center of the player and closest to the box
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels from the foot!
                # We will only use the 4 key points that are closest to the foot 2 ends and the 2 middles of the court
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels
                # Because the players are constantly crouching and moving, so the hieght might change
                # We will use the max height of the player within a 20 frames to get the max hefiht!
                frame_index_min = max(0, frame_num-20) # We go back 20 frames
                frame_index_max = min(len(player_boxes), frame_num+50) # We forward 50 frames
                
                # Below will have a list of the heights of the boxes which accounts for the height of the human!
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range (frame_index_min,frame_index_max)]
                # We want to get the max player height from the list!
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position


                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes
    
    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        # Distance from player to the kps!
        distance_from_keypoint_x_meters = convert_pixels_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixels_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = convert_meters_to_pixels(distance_from_keypoint_x_meters,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        mini_court_y_distance_pixels = convert_meters_to_pixels(distance_from_keypoint_y_meters,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        # You add the distance from the key point and food position to the nearest kps x and y cooridnates!
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    # Draw cooridnates of the ball and player on the mini court
    def draw_points_on_mini_court(self,frames, positions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames
    # You will call this to draw the rectangle box!
    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    # To draw the court out
    def draw_court(self,frame):
        # Draw out all the KPS
        for i in range(0,len(self.drawing_key_points),2): #You have 2 steps because each x and y is 1 set
             x = int(self.drawing_key_points[i])
             y = int(self.drawing_key_points[i+1])
             frame = cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # draw Lines, where you will loop through the self.lines which contains the kps 
        for line in self.lines:
            # c
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            frame = cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
        
        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        frame = cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # The black will be 0s 
        # Draw the rectangle which has the frame shape!
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        # Copy the frame so that it will not be overriden
        out = frame.copy()
        
        alpha=0.5 # this will give the translucency
        mask = shapes.astype(bool)
        # Will AND and mask Non zero will become 1s and become white!
        # Add weighted to add the alpha translucency!!
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def set_court_lines(self):
        # create a list of tuples of the court lines 
        # Im guess you will loop through this when drawing the kps
                self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]
    
    # create a function to draw on the video
    def set_canvas_bg_box_position(self,frame):
        
        # X = width Y = Height 

        # So that the frame wil not be overriden!
        frame = frame.copy()

        # here we will get the position from the edge of the frame to the white box
        
        #The side from the right
        self.end_x = frame.shape[1] - self.buffer #.shape[1] is the width!!
        
        # The buffer from the top + length of the width court
        self.end_y = self.buffer + self.drawing_rectangle_height

        # The side from the left
        self.start_x = self.end_x - self.drawing_rectangle_width

        # The top edge of the frame to top width of the white rectangle
        self.start_y = self.end_y - self.drawing_rectangle_height

    # Draw out the court lines 
    def set_mini_court_position(self):

        # here are the positions of the mini tennis courts!
        self.court_start_x = self.start_x + self.padding_court
        self.court_end_x = self.end_x - self.padding_court

        self.court_start_y = self.start_y + self.padding_court
        self.court_end_y = self.end_y - self.padding_court

        # The widht of the mini court!
        self.court_drawing_width = self.court_end_x - self.court_start_x

    # Draw the kps on the mini court!
    def set_court_drawing_key_points(self):
        # Here you will draw out the 28 kps on the mini court
        
        # You will create an list of 28 '0's Which will represent both x and y 
        drawing_key_points = [0]*28

    # All the points are based on the markings on the video!!
        # point 0 
        drawing_key_points[0],drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)

        # point 1
        drawing_key_points[2],drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)

        # point 2
        drawing_key_points[4],drawing_key_points[5] = int(self.court_start_x), int(self.court_start_y 
        + 2*(convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)))

        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 

        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_key_points[9] = drawing_key_points[1] 
        
        #point 5

        drawing_key_points[10] = drawing_key_points[4] + convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_key_points[11] = drawing_key_points[5] 
       
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_key_points[13] = drawing_key_points[3] 
        
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_key_points[15] = drawing_key_points[7] 
        
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH,constants.DOUBLE_LINE_WIDTH,self.court_drawing_width)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    # Getter methods for the court distance
    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points