import torch
from torchvision import transforms, models
import cv2
import numpy as np

# This KPS are very good for the court line detector
# We use these coordinates to determine who is on the court, who is the closest etc...

class CourtLineDetector:
    def __init__(self, model_param_path):
        self.weights = models.ResNet50_Weights.DEFAULT

        self.model = models.resnet50(weights=self.weights)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_param_path, map_location='cpu'))

        # Create a transform to transfomr the input frames!!
        # The order matters!
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, frame):

        # We have to transform the input frame first!!
        # Note that everything when we read the imgae with cv2
        # It will be in cv2 format!
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_transformed = self.transform(frame_rgb).unsqueeze(0) # We are unsqueezing to add the B in B C W H

        self.model.eval()
        with torch.inference_mode():
            results = self.model(frame_transformed)
        
        # We then remove the Batch dimension
        kps = results.squeeze().cpu().numpy()

        # Now we want to get the Height and width as we have to adjust it
        # As we resize the frame to 224
    
        # note that opencv will arrange the image in H W Colour channels format
        OG_h, OG_w, _ = frame_rgb.shape

        # We will resize the kps from the large video to what we resize it to
        # Think of it as OG_w/224 is the ratio and we are multiple the coord kps[] * ratio
        kps[::2] *= OG_w/224
        kps[1::2] *= OG_h/224

        return kps
    
    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            # (x,y-10) is at the bottom left corner of the box - 10 to create a buffer to plave this text above the point
            # str(i//2) this will be string text
            # RMB that in CV2 it is BGR format!
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # cv.circle(	img, center, radius, color[, thickness[, lineType[, shift]]]	) ->	img
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    # Draw the keypoints on the video
    # RMB that to draw on the video you will need to return a array of frames back!!
    # In here you will draw_keypoints on the frame with the above function
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)
        return output_frames
