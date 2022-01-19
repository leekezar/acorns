import cv2, glob, random, json
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# constants for location parameters
EYE_L = 3
EYE_R = 6
EAR_L = 7
EAR_R = 8
NOSE = 0
MOUTH_L = 9
MOUTH_R = 10
SHOULDER_L = 11
SHOULDER_R = 12
ELBOW_L = 13
ELBOW_R = 14
WRIST_L = 15
WRIST_R = 16
WAIST_L = 23
WAIST_R = 24

class SignFrame:
	def __init__(self, h, w, mp_results, frame):
		def process_hand(coords):
			# extract the landmarks
			coords = [coords[i] for i in range(21)]

			# convert from [0,1] to [0,h] or [0,w]
			coords = [(coord.x * w, coord.y * h) if coord else None \
				for coord in coords ]

			# save the palm's location before the next transformation ruins it
			palm = coords[0]

			# crop
			hand_left = min([c[0] for c in coords])
			hand_bottom = min([c[1] for c in coords])
			coords = [(c[0] - hand_left, c[1] - hand_bottom) for c in coords]

			return coords, palm

		self.frame = frame

		if mp_results.left_hand_landmarks:
			self.lhl, self.lpalm = process_hand(mp_results.left_hand_landmarks.landmark)
		else:
			self.lhl = [None for _ in range(21)]
			self.lpalm = None

		if mp_results.right_hand_landmarks:
			self.rhl, self.rpalm = process_hand(mp_results.right_hand_landmarks.landmark)
		else:
			self.rhl = [None for _ in range(21)]
			self.rpalm = None
		if mp_results.face_landmarks:
			fl 	= [mp_results.face_landmarks.landmark[i]
					for i in range(478)]
			self.face = [(coord.x * w, coord.y * h) 
					for coord in fl]
		else:
			self.face = [None for _ in range(478)]

		if mp_results.pose_landmarks:
			bl 	= [mp_results.pose_landmarks.landmark[i] \
					for i in range(33)]
			self.body = [(coord.x * w, coord.y * h) 
					for coord in bl]
		else:
			self.body = [None for _ in range(33)]

		if self.rpalm and self.body[0]:
			self.rlocs = [
				distance.euclidean(self.rpalm, self.body[EYE_L]),
				distance.euclidean(self.rpalm, self.body[EYE_R]),
				distance.euclidean(self.rpalm, self.body[EAR_L]),
				distance.euclidean(self.rpalm, self.body[EAR_R]),
				distance.euclidean(self.rpalm, self.body[NOSE]),
				distance.euclidean(self.rpalm, self.body[MOUTH_L]),
				distance.euclidean(self.rpalm, self.body[MOUTH_R]),
				distance.euclidean(self.rpalm, self.body[SHOULDER_L]),
				distance.euclidean(self.rpalm, self.body[SHOULDER_R]),
				distance.euclidean(self.rpalm, self.body[ELBOW_L]),
				distance.euclidean(self.rpalm, self.body[ELBOW_R]),
				distance.euclidean(self.rpalm, self.body[WRIST_L]),
				distance.euclidean(self.rpalm, self.body[WRIST_R]),
				distance.euclidean(self.rpalm, self.body[WAIST_L]),
				distance.euclidean(self.rpalm, self.body[WAIST_R])
				]
		else:
			self.rlocs = [None for _ in range(15)]

		if self.lpalm and self.body[0]:
			self.llocs = [
				distance.euclidean(self.lpalm, self.body[EYE_L]),
				distance.euclidean(self.lpalm, self.body[EYE_R]),
				distance.euclidean(self.lpalm, self.body[EAR_L]),
				distance.euclidean(self.lpalm, self.body[EAR_R]),
				distance.euclidean(self.lpalm, self.body[NOSE]),
				distance.euclidean(self.lpalm, self.body[MOUTH_L]),
				distance.euclidean(self.lpalm, self.body[MOUTH_R]),
				distance.euclidean(self.lpalm, self.body[SHOULDER_L]),
				distance.euclidean(self.lpalm, self.body[SHOULDER_R]),
				distance.euclidean(self.lpalm, self.body[ELBOW_L]),
				distance.euclidean(self.lpalm, self.body[ELBOW_R]),
				distance.euclidean(self.lpalm, self.body[WRIST_L]),
				distance.euclidean(self.lpalm, self.body[WRIST_R]),
				distance.euclidean(self.lpalm, self.body[WAIST_L]),
				distance.euclidean(self.lpalm, self.body[WAIST_R])
				]
		else:
			self.llocs = [None for _ in range(15)]


class SignVideo:
	def __init__(self, vid_path):
		self.features = self.get_features(vid_path)
		if len(self.features) > 50:
			self.norm_features = self.sample_frames(self.features, 50)
			self.lmov = [frame.lpalm for frame in self.norm_features if frame.lpalm]
			self.rmov = [frame.rpalm for frame in self.norm_features if frame.rpalm]
		else:
			self.norm_features = None
			self.lmov = [frame.lpalm for frame in self.features if frame.lpalm]
			self.rmov = [frame.rpalm for frame in self.features if frame.rpalm]


	def get_features(self, vid_path):
		def get_frames():
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

		frames = get_frames()
		sfs = []
		with mp.solutions.holistic.Holistic(
			model_complexity		= 2,
			enable_segmentation		= True,
			refine_face_landmarks 	= True) as holistic:
			for idx, frame in enumerate(frames):
				if frame is None:
					break

				h,w,_ = frame.shape
				
				results = holistic.process(
					cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

				sf = SignFrame(h, w, results, frame)
				sfs.append(sf)
		return sfs

	def sample_frames(self, sfs, n):
		return [sfs[i] for i in sorted(random.sample(range(len(sfs)), n))]

data = {}
count = 0
for vid_path in glob.glob("./videos/*.mp4"):
	vid = SignVideo(vid_path)
	if not vid.norm_features:
		continue

	data[vid_path] = {
		"left_handshape" : [f.lhl for f in vid.norm_features],
		"right_handshape" : [f.rhl for f in vid.norm_features],
		"left_movement" : vid.lmov,
		"right_movement" : vid.rmov,
		"left_location" : [f.llocs for f in vid.norm_features],
		"right_location" : [f.rlocs for f in vid.norm_features],
		"face" : [f.face for f in vid.norm_features],
		"pose" : [f.body for f in vid.norm_features]
	}
	count += 1
	print(count)
	
json.dump(data, open("sample_features.json", "w+"), indent=4)


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# half = int(len(vid.features) / 2)
# for frame_idx in range(half, half+10):
# 	face_body = vid.features[frame_idx].body #+ vid.features[-10].face
# 	X = [c[0] for c in face_body]
# 	Y = [c[1] * -1 for c in face_body]
# 	labels = [i for i in range(len(face_body))]

# 	fix, ax = plt.subplots()
# 	ax.scatter(X,Y,s=1)
# 	for i,label in enumerate(labels):
# 		ax.annotate(label,(X[i],Y[i]))
# 	plt.show()

# TO-DO: Save frames as images
# TO-DO: Save frame path as signframe object property
# TO-DO: normalize coords by base of the palm
# TO-DO: Cluster and save each cluster into its own folder
# TO-DO: Evaluate quality of different clustering techniques


