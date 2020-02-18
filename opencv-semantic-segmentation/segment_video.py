# USAGE
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/massachusetts.mp4 --output output/massachusetts_output.avi
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/toronto.mp4 --output output/toronto_output.avi

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
from tkinter.filedialog import askopenfilename, Tk

# construct the argument parse and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt file containing class labels")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-s", "--show", type=int, default=1,
	help="whether or not to display frame to screen")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default=500,
	help="desired width (in pixels) of input image")
args = vars(ap.parse_args())
'''
def segment_video(vs, file_path):
    # load the class label names
    #CLASSES = open(args["classes"]).read().strip().split("\n")
    CLASSES = open("enet-cityscapes/enet-classes.txt").read().strip().split("\n")
    
    # if a colors file was supplied, load it from disk
    '''
    if args["colors"]:
    	COLORS = open(args["colors"]).read().strip().split("\n")
    	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    	COLORS = np.array(COLORS, dtype="uint8")
    
    # otherwise, we need to randomly generate RGB colors for each class
    # label
    else:
    	# initialize a list of colors to represent each class label in
    	# the mask (starting with 'black' for the background/unlabeled
    	# regions)
    	np.random.seed(42)
    	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
    		dtype="uint8")
    	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    '''
    COLORS = open("enet-cityscapes/enet-colors.txt").read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")
        
    # load our serialized model from disk
    print("[INFO] loading model...")
    #net = cv2.dnn.readNet(args["model"])
    net = cv2.dnn.readNet("enet-cityscapes/enet-model.net")
    
    # initialize the video stream and pointer to output video file
    #vs = cv2.VideoCapture(args["video"])
        
    writer = None
    in_writer = None
    
    # try to determine the total number of frames in the video file
    
    try:
    	prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
    		else cv2.CAP_PROP_FRAME_COUNT
    	total = int(vs.get(prop))
    	print("[INFO] {} total frames in video".format(total))
    
    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
    	print("[INFO] could not determine # of frames in video")
    	total = -1
    
    # Create a video writer for saving the video output`
    out_path = "output/" + file_path.split("/")[-1].split(".")[0] + "_out.mp4"
    in_path = "input/" + file_path.split("/")[-1].split(".")[0] + ".mp4"
    print(out_path)
    
    # initialize the legend visualization
    legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")
    # loop over the class names + colors
    for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
    	# draw the class name + color on the legend
    	color = [int(c) for c in color]
    	cv2.putText(legend, className, (5, (i * 25) + 17),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    	cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
    		tuple(color), -1)
        
    cv2.imshow("Legend", legend)
    # loop over frames from the video file stream
    while True:
    	# read the next frame from the file
    	(grabbed, frame) = vs.read()
    
    	# if the frame was not grabbed, then we have reached the end
    	# of the stream
    	if not grabbed:
    		break
    
    	# construct a blob from the frame and perform a forward pass
    	# using the segmentation model
    	#frame = imutils.resize(frame, width=args["width"])
    	frame = imutils.resize(frame, width=500)
    	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
    		swapRB=True, crop=False)
    	net.setInput(blob)
    	start = time.time()
    	output = net.forward()
    	end = time.time()
    
    	# infer the total number of classes along with the spatial
    	# dimensions of the mask image via the shape of the output array
    	(numClasses, height, width) = output.shape[1:4]
    
    	# our output class ID map will be num_classes x height x width in
    	# size, so we take the argmax to find the class label with the
    	# largest probability for each and every (x, y)-coordinate in the
    	# image
    	classMap = np.argmax(output[0], axis=0)
    
    	# given the class ID map, we can map each of the class IDs to its
    	# corresponding color
    	mask = COLORS[classMap]
    
    	# resize the mask such that its dimensions match the original size
    	# of the input frame
    	mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
    		interpolation=cv2.INTER_NEAREST)
    
    	# perform a weighted combination of the input frame with the mask
    	# to form an output visualization
    	output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")
                  
    	# check if the video writer is None
    	if writer is None:
    		# initialize our video writer
    		fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    		writer = cv2.VideoWriter(out_path, fourcc, 10,
    			(output.shape[1], output.shape[0]), True)
    
    		# some information on processing single frame
    		if total > 0:
    			elap = (end - start)
    			print("[INFO] single frame took {:.4f} seconds".format(elap))
    			print("[INFO] estimated total time: {:.4f}".format(
    				elap * total))
        
    	if in_writer is None:
    		# initialize our video writer
    		fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    		in_writer = cv2.VideoWriter(in_path, fourcc, 10,
    			(frame.shape[1], frame.shape[0]), True)

        # write the output frame to disk
    	in_writer.write(frame)
        
    	# write the output frame to disk
    	writer.write(output)
    
    
    	# check to see if we should display the output frame to our screen
    	'''
        if args["show"] > 0:
    		cv2.imshow("Frame", output)
    		key = cv2.waitKey(1) & 0xFF
    
    		# if the `q` key was pressed, break from the loop
    		if key == ord("q"):
    			break
        '''
    	cv2.imshow("Output", output)
    	cv2.imshow("Frame", frame)
    	key = cv2.waitKey(1) & 0xFF
        
    	if cv2.getWindowProperty('Frame',cv2.WND_PROP_VISIBLE) < 1 \
                or cv2.getWindowProperty('Output',cv2.WND_PROP_VISIBLE) < 1 \
                or (cv2.getWindowProperty('Legend',cv2.WND_PROP_VISIBLE) < 1):        
    		break  
    
    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break
    
    # release the file pointers
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    writer.release()
    vs.release()