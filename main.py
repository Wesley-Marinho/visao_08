import cv2
import numpy as np

# Stereo camera parameters
baseline = 0.1  # distance between cameras in meters
focal_length = 700  # focal length in pixels (calibrated value)
min_object_size = 1000  # Minimum area of the object in pixels
max_object_size = 5000  # Maximum area of the object in pixels (adjust as needed)

# Disparity parameters
block_size = 15   # Size of the block window. Must be odd.
min_disp = 0     # Minimum possible disparity value
num_disp = 16    # Maximum disparity minus minimum disparity

# Initialize webcams
cap1 = cv2.VideoCapture(0)  # Left camera
cap2 = cv2.VideoCapture(2)  # Right camera

# Create StereoBM object with custom parameters
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

while True:
    # Capture frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error: Unable to capture images from one or both cameras.")
        break
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute disparity map
    disparity = stereo.compute(gray1, gray2)
    
    # Normalize disparity for visualization
    disparity_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_display = np.uint8(disparity_display)
    
    # Convert disparity to binary image
    _, binary_disparity = cv2.threshold(disparity_display, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_disparity, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours based on area and find the largest one
        valid_contours = [contour for contour in contours if min_object_size < cv2.contourArea(contour) < max_object_size]
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Get bounding box of largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract disparity values within the bounding box
            box_disparity = disparity[y:y+h, x:x+w]
            valid_disparity_values = box_disparity[box_disparity > 0]
            
            if valid_disparity_values.size > 0:
                # Calculate the average disparity within the bounding box
                avg_disparity = np.mean(valid_disparity_values)
                
                # Calculate distance in meters
                distance_meters = (focal_length * baseline) / avg_disparity if avg_disparity > 0 else float('inf')
                
                # Convert distance to centimeters
                distance_cm = distance_meters * 100
                
                # Draw a rectangle around the largest object
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display the distance on the frame
                cv2.putText(frame1, f'Distance: {distance_cm:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame2, f'Distance: {distance_cm:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the results
    cv2.imshow('Disparity', disparity_display)
    #cv2.imshow('Left Frame with Rectangle', frame1)
    cv2.imshow('Right Frame with Rectangle', frame2)
    
    # Wait for 33 ms to achieve ~30 FPS
    if cv2.waitKey(33) & 0xFF == 27:
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
