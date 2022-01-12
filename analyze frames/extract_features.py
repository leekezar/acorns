import cv2
import mediapipe as mp
import numpy as np

class SignFrame:
	def __init__(self, h, w, mp_results):
		lhl = [mp_results.left_hand_landmarks.landmark[i]
				for i in range(21)]
		rhl = [mp_results.left_hand_landmarks.landmark[i]
				for i in range(21)]
		self.hands = [(coord.x * w, coord.y * h)
				for coord in lhl + rhl]
		
		fl 	= [mp_results.face_landmarks.landmark[i]
				for i in range(478)]
		self.face = [(coord.x * w, coord.y * h) 
				for coord in fl]

		bl 	= [mp_results.pose_landmarks.landmark[i] \
				for i in range(33)]
		self.body = [(coord.x * w, coord.y * h) 
				for coord in bl]

		self.vec = []

		for (x,y) in self.hands + self.face + self.body:
			self.vec.append(x)
			self.vec.append(y)


def get_frames(vid_path):
	video = cv2.VideoCapture(vid_path)
	video.set(5,10)
	time = 0
	fps = 24
	increment = 1000/fps
	video.set(cv2.CAP_PROP_POS_MSEC, time)
	success, image = video.read()
	frames = []
	while success:
		time += increment
		video.set(cv2.CAP_PROP_POS_MSEC, time)
		success, image = video.read()
		frames.append(image)
		if len(frames) > 100:
			break

	return frames

def get_features(vid_path):
	frames = get_frames(vid_path)
	sfs = []
	with mp.solutions.holistic.Holistic(
		model_complexity		= 2,
		enable_segmentation		= True,
		refine_face_landmarks 	= True) as holistic:
		for idx, frame in enumerate(frames):
			h,w,_ = frame.shape
			
			results = holistic.process(
				cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

			if results.left_hand_landmarks and \
				results.right_hand_landmarks:
				sf = SignFrame(h, w, results)
				sfs.append(sf)
	return sfs

		
features = get_features("_0fO5ETSwyg-5-rgb_front.mp4")
hand_features = []
for sf in features:
	hand_features.append([])
	for x,y in sf.hands:
		hand_features[-1].extend([x,y])
hand_features = np.array(hand_features)

# TO-DO: Save frames as images
# TO-DO: Save frame path as signframe object property
# TO-DO: normalize coords by base of the palm
# TO-DO: Cluster and save each cluster into its own folder
# TO-DO: Evaluate quality of different clustering techniques