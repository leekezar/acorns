import cv2, glob, random, json, math, string
import pandas as pd
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# constants for location parameters
face2id = {
	"EYE_L": 3,
	"EYE_R": 6,
	"EAR_L": 7,
	"EAR_R": 8,
	"NOSE": 0,
	"MOUTH_L": 9,
	"MOUTH_R": 10,
	"SHOULDER_L": 11,
	"SHOULDER_R": 12,
	"ELBOW_L": 13,
	"ELBOW_R": 14,
	"WRIST_L": 15,
	"WRIST_R": 16,
	"WAIST_L": 23,
	"WAIST_R": 24
}

asl_lex = pd.read_csv(open("ASL-LEX View Data.csv"))
used_ids = []

class SignFrame:
	def __init__(self, h, w, mp_results, frame):
		# from https://stackoverflow.com/a/34374437/15930256
		def rotate_hand(origin, point, angle):
			ox, oy = origin
			px, py = point

			qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
			qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
			return (qx, qy)

		def process_hand(coords, palm, rotate, scale, ratio):
			# extract the landmarks
			coords = [coords[i] for i in range(21)]

			# convert from [0,1] to [0,h] or [0,w]
			coords = [(coord.x * w, coord.y * h) if coord else None \
				for coord in coords ]

			if rotate:
				palm = coords[0]
				mid_base = coords[9]
				mid_len = math.dist(palm,mid_base)
				new_mid_tip = (palm[0], palm[1] - mid_len)
				dist = math.dist(mid_base,new_mid_tip)
				theta = math.acos(((dist**2) / (-2 * (mid_len**2))) + 1)
				# always rotates counter-clockwise,
				# so if the rotation is easiest clockwise,
				# then theta should be += pi
				if mid_base[0] > palm[0]:
					theta = theta * -1
				coords = [rotate_hand(palm,c,theta) for c in coords]
				assert abs(coords[0][0] - coords[9][0]) < 2

			if scale:
				width = max([c[0] for c in coords]) - \
					min([c[0] for c in coords])
				scale_factor = 1/width
				coords = [(c[0] * scale_factor, c[1] * scale_factor)
					for c in coords]
			
			hand_left = min([c[0] for c in coords])
			hand_bottom = min([c[1] for c in coords])
			coords = [(c[0] - hand_left, c[1] - hand_bottom)
				for c in coords]

			if ratio:
				distances = [
					distance.euclidean(coords[4], coords[0]), # thumb/wrist
					distance.euclidean(coords[8], coords[0]), # index/wrist
					distance.euclidean(coords[12], coords[0]), # middle/wrist
					distance.euclidean(coords[16], coords[0]), # ring/wrist
					distance.euclidean(coords[20], coords[0]), # pinky/wrist
					distance.euclidean(coords[4], coords[1]), # thumb/base
					distance.euclidean(coords[8], coords[5]), # index/base
					distance.euclidean(coords[12], coords[9]), # middle/base
					distance.euclidean(coords[16], coords[13]), # ring/base
					distance.euclidean(coords[20], coords[17]), # pinky/base
					distance.euclidean(coords[4], coords[8]), # thumb/index
					distance.euclidean(coords[4], coords[12]), # thumb/middle
					distance.euclidean(coords[4], coords[16]), # thumb/ring
					distance.euclidean(coords[4], coords[20]), # thumb/pinky
					distance.euclidean(coords[4], coords[17]), # thumb/pbase
				]
				scale_factor = 1 / max(distances)
				distances = [d * scale_factor for d in distances]
				
				return distances

			else:
				return coords

		def save_frame():
			self.frame = frame
			self.frame_id = None
			while not self.frame_id or self.frame_id in used_ids:
				self.frame_id = ''.join(random.choice(string.ascii_lowercase + \
					string.digits) for _ in range(5))
			if random.random() < 0.01:
				self.path = self.frame_id + ".jpg"
				cv2.imwrite("./frames/" + self.path, frame)
			else:
				self.path = None

		save_frame()

		if mp_results.left_hand_landmarks:
			lh_coords = mp_results.left_hand_landmarks.landmark
			
			self.lpalm = (lh_coords[0].x * w, lh_coords[0].y * h)

			self.lhl = {
				"raw" : 					[(c.x*w, c.y*h) 
												for c in lh_coords],
				# "crop" : 					process_hand(lh_coords,0,0,0,0),
				# "crop_rotate" : 			process_hand(lh_coords,0,1,0,0),
				"crop_rotate_scale" : 		process_hand(lh_coords,0,1,1,0),
				"crop_rotate_scale_ratio" : process_hand(lh_coords,0,1,1,1),
				"crop_scale_ratio" : 		process_hand(lh_coords,0,0,1,1),
				# "crop_rotate_ratio" : 		process_hand(lh_coords,0,1,0,1),
				"crop_scale" : 				process_hand(lh_coords,0,0,1,0)
			}

		else:
			self.lhl = None
			self.lpalm = None

		if mp_results.right_hand_landmarks:
			rh_coords = mp_results.right_hand_landmarks.landmark

			self.rpalm = (rh_coords[0].x * w, rh_coords[0].y * h)

			self.rhl = {
				"raw" : 					[(c.x*w, c.y*h) 
												for c in rh_coords],
				# "crop" : 					process_hand(rh_coords,0,0,0,0),
				# "crop_rotate" : 			process_hand(rh_coords,0,1,0,0),
				"crop_rotate_scale" : 		process_hand(rh_coords,0,1,1,0),
				"crop_rotate_scale_ratio" : process_hand(rh_coords,0,1,1,1),
				"crop_scale_ratio" : 		process_hand(rh_coords,0,0,1,1),
				# "crop_rotate_ratio" : 		process_hand(rh_coords,0,1,0,1),
				"crop_scale" : 				process_hand(rh_coords,0,0,1,0)
			}

		else:
			self.rhl = None
			self.rpalm = None

		if mp_results.face_landmarks:
			fl 	= [mp_results.face_landmarks.landmark[i]
					for i in range(478)]
			self.face = [(coord.x * w, coord.y * h) 
					for coord in fl]
		else:
			self.face = None

		if mp_results.pose_landmarks:
			bl 	= [mp_results.pose_landmarks.landmark[i]
					for i in range(33)]
			self.body = [(coord.x * w, coord.y * h) 
					for coord in bl]
		else:
			self.body = None

		if self.rpalm and self.body:
			self.rlocs = [
				distance.euclidean(self.rpalm, self.body[face_id]) \
					for face_id in face2id.values()
				]
		else:
			self.rlocs = None

		if self.lpalm and self.body:
			self.llocs = [
				distance.euclidean(self.lpalm, self.body[face_id]) \
					for face_id in face2id.values()
				]
		else:
			self.llocs = None

class SignVideo:
	def __init__(self, vid_path):
		self.features = self.get_features(vid_path)

		if len(self.features) > 50:
			self.features = self.sample_frames(self.features, 50)
		
		self.lmov = [frame.lpalm for frame in self.features 
			if frame.lpalm]
		self.rmov = [frame.rpalm for frame in self.features 
			if frame.rpalm]



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

label2id = {}
id2label = {}
def get_lexical_features(word):
	def get_id(param, label):
		if param not in label2id.keys():
			label2id[param] = {}
			id2label[param] = {}

		if label not in label2id[param].keys():
			n = len(label2id[param])
			label2id[param][label] = n
			id2label[param][n] = label

		return label2id[param][label]

	row = asl_lex.loc[asl_lex["Lemma ID"] == word.replace("-","_")]

	# this means the word was not found
	if len(row["Frequency"]) == 0:
		return None

	selected_params = [
		"Handshape","Selected Fingers", "Flexion", "Spread", "Spread Change",
		"Thumb Position", "Thumb Contact", "Sign Type", "Path Movement",
		"Repeated Movement", "Major Location", "Minor Location",
		"Second Minor Location", "Contact", "Nondominant Handshape", 
		"Wrist Twist"]


	feats = { 
		param.lower() : get_id(param.lower(),str(row[param].iloc[0])) 
			for param in selected_params
	}

	return feats

data = {}
count = 0
for vid_path in glob.glob("./high_freq/*.mp4"):
	name = vid_path.split("\\")[1].replace(".mp4", "")
	word = name.split("_")[0]

	# print(name)

	lex_feats = get_lexical_features(word)
	if lex_feats == None:
		continue

	vid = SignVideo(vid_path)

	# data[name] = {
	# 	"left_handshape" : [f.lhl for f in vid.norm_features if f.lpalm],
	# 	"right_handshape" : [f.rhl for f in vid.norm_features if f.rpalm],
	# 	"left_movement" : vid.lmov,
	# 	"right_movement" : vid.rmov,
	# 	"left_location" : [f.llocs for f in vid.norm_features if f.llocs[0]],
	# 	"right_location" : [f.rlocs for f in vid.norm_features if f.rlocs[0]],
	# 	"face" : [f.face for f in vid.norm_features if f.face[0]],
	# 	"pose" : [f.body for f in vid.norm_features if f.body[0]],
	# 	"lexical" : lex_feats
	# }

	data[name] = {
		"path" : vid_path,
		"lexical" : lex_feats,
		"movement" : { 
			"left_movement" : 	vid.lmov,
			"right_movement" : 	vid.rmov
		},
		
		"frames" : [
			{
				"path" : 			f.path,
				"left_handshape" : 	f.lhl,
				"right_handshape" : f.rhl,
				"left_location" : 	f.llocs,
				"right_location" : 	f.rlocs,
				"face" : 			f.face,
				"pose" : 			f.body
			} for f in vid.features
		]
	}

	count += 1
	print(name,count)
	
json.dump(data, open("sample_features_all.json", "w+"), indent=4)