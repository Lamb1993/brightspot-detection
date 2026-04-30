import cv2
import imutils
import time
import numpy as np
from skimage import measure
from imutils import contours

# image_open highlights bright spots in an image. 
def image_open():
    print("Opening image...")
    radius = 73

    image = cv2.imread(r'mats/bright.png')  # Load image into 'src'
    # image = cv2.imread(r'C:/Users/Gregory\Documents/Projects/Headlight Tracking/.venv/mats/bright.png')  # Load image into 'src'
    # cv2.imshow("Image", src) # display the image
    # cv2.waitKey(0) # wait for key press from the user before closing

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # load the image and convert it to grayscale
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", gray)
    cv2.waitKey(0)
    cv2.destroyWindow("Image")

    # the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)

    # display the results of the naive attempt
    cv2.imshow("Naive", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Naive")

    # apply a Gaussian blur to the image then find the brightest
    # region
    gray = cv2.GaussianBlur(gray,  (11, 11), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    image = orig.copy()
    print(maxLoc)
    print(maxVal)
    cv2.circle(image, maxLoc, radius, (255, 0, 0), 2) # args["radius"] must be odd numbered

    # display the results of our newly improved method
    cv2.imshow("Robust", image)
    cv2.waitKey(0)

    # cv2.imwrite("outputimage.jpg", image)

# video_open detects bright spots in each video frame 
def video_open():
    print("Opening video...")
    frameCounter = 0 # frame counter
    
    cap = cv2.VideoCapture('mats/mybright.mp4')  # Load video into 'cap'
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original video FPS
    delay = int(100 / fps) if fps > 0 else 25
    
    print("Total frames: ", length)
    print("Original FPS: ", fps, "Delay: ", delay, "ms")

    while cap.isOpened():
        res, frame = cap.read()  # Read a frame from the video into 'frame'
        frame_1080 = cv2.resize(frame, (1920, 1080))

        if not res:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frameCounter = frameCounter + 1

        if frameCounter % 3 == 0: # only process every 3rd frame to reduce processing time.
            gray = cv2.cvtColor(frame_1080, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)  # Apply Gaussian blur to the grayscale image

            myThresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]  # Threshold the blurred grayscale image to create a binary image
            myThresh = cv2.erode(myThresh, None, iterations=2)  # Erode the thresholded image to remove small blobs of noise
            myThresh = cv2.dilate(myThresh, None, iterations=4)  # Dilate the eroded image to restore the size of the bright regions

            mask = connected_component_analysis(myThresh) # Perform connected component analysis
            contoured = draw_labeled_blobs(mask, frame_1080)

            cv2.imshow('Frame', contoured)  # Display the processed frame
        else:
            cv2.imshow('Frame', frame_1080)  # Display the original frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break

    print("video_open() finished")
    cap.release()  # Release the video capture object

# perform connected component analysis to filter out small blobs of noise and keep only the larger bright regions
def connected_component_analysis(video_object): # should learn about this more!
    labels = measure.label(video_object, connectivity=2, background=0)
    mask = np.zeros(video_object.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(video_object.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    
    return mask

# draw_labeled_blobs finds contours of the bright regions and draws circles around them in the original video frame.
def draw_labeled_blobs(video_object, original_frame):   
    cnts = cv2.findContours(video_object.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts: # if no contours are found, return the video frame without any modification
        print('No contours found.')
        return video_object
    
    cnts = contours.sort_contours(cnts)[0]

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(original_frame, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
        cv2.putText(original_frame, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return original_frame


video_open()