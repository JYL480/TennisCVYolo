# This will convert pixels to meters
# Meaning what is the distance corresponding to 1 pixel on the image

def convert_pixels_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    
    # Simple math!!
    # Just use the ratio
    return (pixel_distance * reference_height_in_meters)/ reference_height_in_pixels

def convert_meters_to_pixels(meters_distance, reference_height_in_meters, reference_height_in_pixels):
    
    # Simple math!!
    # Just use the ratio
    return (meters_distance * reference_height_in_pixels) / reference_height_in_meters