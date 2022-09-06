# check the arguments are correct
import sys, os

if len(sys.argv) != 3:
	sys.exit("Please provide a video directory to read from and a file to write to.")

vid_dir = sys.argv[1]

if not os.path.isdir(vid_dir):
	sys.exit("Video directory is not a valid path.")

out_file = sys.argv[2]

# initialize libraries/packages
import cv2, glob, random, json, math, string, itertools
from collections import Counter
import pandas as pd
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from scipy.stats import zscore
random.seed(1)

# constants for phoneme management
phoneme_gt = pd.read_csv(open("./ASL-LEX View Data.csv"))
label2id = json.load(open("./asl_lex_labels.json"))
id2label = { param : { v : k for k,v in label2id[param].items() } \
	for param in label2id.keys() }
PHONEME_CATS = [
	"Handshape","Selected Fingers", "Flexion", "Spread", "Spread Change",
	"Thumb Position", "Thumb Contact", "Sign Type", "Path Movement",
	"Repeated Movement", "Major Location", "Minor Location",
	"Second Minor Location", "Contact", "Nondominant Handshape", 
	"Wrist Twist", "Sign Onset", "Sign Offset", "Handshape Morpheme 2"]
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

# constants for morpheme management
semantic_labels = pd.read_csv("plf_lexica.csv")

# TO-DO
url_data = pd.read_csv(open("videos.csv"))

class SignFrame:
	def __init__(self, keypoints, frame):
		h, w, _ = frame.shape
		self.h, self.w, _ = frame.shape
		# convert raw X,Y,Z coordinates to a more useful representation
		def process_hand(coords, distances):
			# extract the landmarks
			coords = [coords[i] for i in range(21)]

			# convert from [0,1] to [0,h] or [0,w]
			coords = [(coord.x * self.w, coord.y * self.h, coord.z) \
				if coord else None for coord in coords ]

			# scale by dividing by the width
			hand_width = max([c[0] for c in coords]) - \
				min([c[0] for c in coords])
			scale_factor_w = 1/hand_width

			hand_height = max([c[1] for c in coords]) - \
				min([c[1] for c in coords])
			scale_factor_h = 1/hand_height


			coords = [(c[0] * scale_factor_w, c[1] * scale_factor_h, c[2])
					for c in coords]
			
			# crop by translating bottom left corner of the hand to origin
			hand_left = min([c[0] for c in coords])
			hand_bottom = min([c[1] for c in coords])
			coords = [(c[0] - hand_left, c[1] - hand_bottom, c[2])
				for c in coords]

			if distances:
				return [distance.euclidean(p1, p2)
					for p1, p2 in itertools.combinations(coords, 2)]

			else:
				return coords

		def save_frame():
			self.frame = frame
			self.frame_id = None
			while not self.frame_id or self.frame_id in used_ids:
				self.frame_id = ''.join(random.choice(string.ascii_lowercase + \
					string.digits) for _ in range(5))
			#if random.random() < 0.01:
			# self.path = self.frame_id + ".jpg"
			# cv2.imwrite("./frames/" + self.path, frame)
			#else:
			#	self.path = None

		# save_frame()


		""" FIRST-ORDER FEATURES """


		# save the left hand coordinates
		if keypoints.left_hand_landmarks:
			lh_coords = keypoints.left_hand_landmarks.landmark
			
			self.lpalm = (lh_coords[0].x * w, lh_coords[0].y * h, \
				lh_coords[0].z)

			self.lhl = {
				"raw" : 					[(c.x*w, c.y*h, c.z) 
												for c in lh_coords],
				"distances" : 				process_hand(lh_coords,1),
				"crop_scale" : 				process_hand(lh_coords,0)
			}

		else:
			self.lhl = None
			self.lpalm = None

		# save the right hand coordinates
		if keypoints.right_hand_landmarks:
			rh_coords = keypoints.right_hand_landmarks.landmark

			self.rpalm = (rh_coords[0].x * w, rh_coords[0].y * h, \
				rh_coords[0].z)

			self.rhl = {
				"raw" : 					[(c.x*w, c.y*h, c.z) 
												for c in rh_coords],
				"distances" : 				process_hand(rh_coords,1),
				"crop_scale" : 				process_hand(rh_coords,0)
			}

		else:
			self.rhl = None
			self.rpalm = None

		# save the face coordinates
		if keypoints.face_landmarks:
			fl 	= [keypoints.face_landmarks.landmark[i]
					for i in range(468)]
			self.face = [(coord.x * w, coord.y * h, coord.z) 
					for coord in fl]
		else:
			self.face = None

		# save the body coordiantes
		if keypoints.pose_landmarks:
			bl 	= [keypoints.pose_landmarks.landmark[i]
					for i in range(33)]
			self.body = [(coord.x * w, coord.y * h, coord.z) 
					for coord in bl]
		else:
			self.body = None


		""" SECOND-ORDER FEATURES """


		# if this frame has a right hand and body, save the right location
		if self.rpalm and self.body:
			self.rlocs = [
				distance.euclidean(self.rpalm, self.body[face_id]) \
					for face_id in face2id.values()
				]
		else:
			self.rlocs = None

		# if this frame has a left hand and body, save the left location
		if self.lpalm and self.body:
			self.llocs = [
				distance.euclidean(self.lpalm, self.body[face_id]) \
					for face_id in face2id.values()
				]
		else:
			self.llocs = None

class SignVideo:
	def __init__(self, sign, vid_path, url = None):
		self.sign = sign
		self.path = vid_path
		self.url = url

		# this distinction is useful for onset/offset information
		# which, in turn, yields cleaner data
		if "asllex" in vid_path:
			self.is_asllex_video = True
		else:
			self.is_asllex_video = False

		# load and process the video
		self.frames = self.process_frames()

		# if len(self.frames) > 50:
		# 	self.frames = self.sample_frames(self.features, 50)
		
		# save the movement
		self.lmov = [frame.lpalm for frame in self.frames \
			if frame and frame.lpalm]
		self.rmov = [frame.rpalm for frame in self.frames \
			if frame and frame.rpalm]

	def process_frames(self):

		# divide the video into frames
		def get_frames():
			if self.is_asllex_video:
				onset = self.sign.phonemes["sign_onset"]
				offset = self.sign.phonemes["sign_offset"]
			else:
				onset, offset = None, None

			video = cv2.VideoCapture(self.path)
			video.set(5,10)
			time = 0
			fps = 30
			increment = 1000/fps
			video.set(cv2.CAP_PROP_POS_MSEC, time)
			success, image = video.read()
			frames = []
			while success:
				time += increment

				if onset and time < onset:
					continue

				if offset and time > offset:
					break

				video.set(cv2.CAP_PROP_POS_MSEC, time)
				success, image = video.read()

				frames.append(image)

			return frames

		frames = get_frames()

		# sign frames
		processed_frames = []
		with mp.solutions.holistic.Holistic() as holistic:
			for frame in frames:
				if frame is None:
					continue

				keypoints = holistic.process(
					cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

				processed_frame = SignFrame(keypoints, frame)

				processed_frames.append(processed_frame)

		return processed_frames

	def sample_frames(self, n_frames):
		valid_frames = [f for f in self.frames if f.rhl and f.rlocs]
		return [valid_frames[i] for i in \
			sorted(random.sample(range(len(valid_frames)), n_frames))]
	
	def vectorize(self, n_frames):
		vec = []
		frames = self.sample_frames(n_frames)
		vec.extend([f.rhl["distances"] for f in frames])
		vec.extend([f.rlocs for f in frames])
		vec = [b for a in vec if a for b in a if b]
		return vec

class Sign:
	def __init__(self, i, glosses):
		self.id = i
		self.glosses = glosses
		self.phonemes = self.get_phonemes()
		self.has_phonemes = self.phonemes != None

		# self.morphemes = self.get_morphemes()
		# self.has_morphemes = self.morphemes != {}

		self.examples = []

		# TO-DO: insert stuff about visualizing aggregate data for a sign

	def get_phonemes(self):
		# get the numeric ID for an English label given a parameter
		# adaptive ! --- if label not found, will create one and save it
		def get_id(param, label):
			if param in ["sign onset", "sign offset"]:
				return int(label)

			if param not in label2id.keys():
				label2id[param] = {}
				id2label[param] = {}

			if label not in label2id[param].keys():
				n = len(label2id[param])
				label2id[param][label] = n
				id2label[param][n] = label
				json.dump(label2id, open("./asl_lex/asl_lex_labels.json","w"))

			return label2id[param][label]

		for gloss in self.glosses:
			phonemes = phoneme_gt.loc[gloss == phoneme_gt["Lemma ID"]]
			if len(phonemes["Frequency"]) != 0:
				return { 
					"_".join(feat.lower().split()) : get_id(feat.lower(), \
						str(phonemes[feat].iloc[0])) for feat in PHONEME_CATS
				}

			# TO-DO: this code currently ignores signs that have variants
			# i.e. the asl lex entry id ends with _1, _2, ...
			# will need to compare the videos to figure out which variant it is
			
		return None

	def get_morphemes(self):
		morphemes = { }

		for gloss in self.glosses:
			for token in gloss.split("_"):
				curr_morphemes = dict(semantic_labels.loc[token == 
					semantic_labels["Entry ID"]])

				if len(curr_morphemes["E-0"]) != 1:
					continue

				filtered_morphemes = [k for k,v in dict(curr_morphemes).items() \
					if ("L-" in k or "M-" in k) and int(v) > 0]

				for morpheme in filtered_morphemes:
					morphemes[morpheme] = 1

		return morphemes

	def verify_examples(self):
		if len(self.examples) <= 2:
			return

		vectors = []

		# get the lengths to filter short/long ones
		lengths = [len([f for f in v.frames if f.rhl and f.rlocs]) for v in self.examples]
		len_zscore = zscore(lengths)

		# this integer will be the resulting length for all vectors
		min_len = min([lengths[i] for i in range(len(lengths)) \
			if abs(len_zscore[i]) < 1])

		# get valid vectors
		valid_examples = []
		for i,example in enumerate(self.examples):
			# skip the videos that are too long or too short
			if abs(len_zscore[i]) > 1:
				continue

			vectors.append(example.vectorize(min_len))
			valid_examples.append(example)

		z_scores = np.mean(zscore(vectors), axis=1)

		for i,example in enumerate(valid_examples):
			print(f"\t{example.path}\t{z_scores[i]:.3f}\t|",end="")
			
			for j,example2 in enumerate(valid_examples):
				if i>j:
					print("\t     ",end="")
				elif i == j:
					print("\t1.000",end="")
				else:
					print(f"\t{sim(example, example2):.3f}",end="")
			
			print()
		

def save_as_json(signs, filename):
	dsigns = {}

	for sign in signs.values():
		dsigns[sign.id] = []

		for instance in sign.examples:

			file_data = {
				"path" : instance.path,
				"glosses" : sign.glosses,
				"url" : instance.url
			}

			video_data = {
				"mov_l" : instance.lmov,
				"mov_r" : instance.rmov,

				"frames" : [
					{
						"hs_l" : 	f.lhl,
						"hs_r" : 	f.rhl,
						"loc_l" : 	f.llocs,
						"loc_r" : 	f.rlocs,
						"face" : 	f.face,
						"pose" : 	f.body
					} for f in instance.frames
				]
			}
			
			phoneme_data = { "phonemes" : sign.phonemes }

			# morpheme_data = { "morphemes" : sign.morphemes }

			dsigns[sign.id].append(
				{**file_data, **video_data, **phoneme_data}#, **morpheme_data}
			)

	json.dump(dsigns, open(filename, "w+"), indent=4)
	print("Saved", filename)

def sim(v1, v2):
	vid_dist = 0.0
	n_frames = min(len(v1.frames), len(v2.frames))
	if n_frames == 0:
		return -1

	for i in range(n_frames):
		hs1 = v1.frames[i].rhl
		hs2 = v2.frames[i].rhl
		loc1 = v1.frames[i].rlocs
		loc2 = v2.frames[i].rlocs

		if hs1 and hs2:
			frame_dist = 0.0
			for (coord1, coord2) in zip(hs1["distances"], hs2["distances"]):
				frame_dist += distance.euclidean(coord1, coord2)
			frame_dist /= len(hs1["distances"])

			vid_dist += frame_dist

		if loc1 and loc2:
			frame_dist = 0.0

			for (coord1, coord2) in zip(loc1, loc2):
				frame_dist += distance.euclidean(coord1, coord2)
			frame_dist /= len(loc1)

			vid_dist += frame_dist


	vid_dist /= 2 * min(len(v1.frames), len(v2.frames))

	if vid_dist == 0.0:
		pass

	return vid_dist

def get_gloss_inst(filename):
	if "_" in filename:
		instance_id = int(filename.split("_")[1])	
		gloss = filename.split("_")[0]
	else:
		instance_id = 0
		gloss = filename
	return instance_id, gloss

def get_url(gloss, instance_id):
	url_rows = url_data.loc[gloss == url_data["gloss"]]

	if instance_id == 0:
		for url in list(dict(url_rows["url"]).values()):
			if "player.vimeo.com" in url:
				return url

	elif instance_id <= len(url_rows):
		return list(dict(url_rows["url"]).values())[instance_id - 1]
	
	return None

def get_unique_id():
	sign_id = None
	while sign_id in used_ids or not sign_id:
		sign_id = ''.join(random.choice(string.ascii_lowercase + \
					string.digits) for _ in range(5))
	used_ids.append(sign_id)
	return sign_id

def get_sample(paths, n_samples = 5, allow_more=True):
	sorted_paths = {}
	for i, vid_path in enumerate(paths):
		sample_label = vid_path.split("/")[-1]
		gloss = sample_label.split(".")[0]

		inst,gloss = get_gloss_inst(gloss)

		if gloss not in sorted_paths.keys():
			sorted_paths[gloss] = []

		sorted_paths[gloss].append(vid_path)

	sample_paths = []

	sorted_paths = list(sorted_paths.items())
	# random.shuffle(sorted_paths)

	for gloss, paths in sorted_paths:
		if n_samples == 0:
			sample_paths.append((gloss, paths))
		elif len(paths) >= n_samples:
			if allow_more:
				sample_paths.append((gloss, paths))
			else:
				sample_paths.append((gloss, paths[:n_samples]))

	return sample_paths

used_ids = []
# get all of the .m* videos in the provided directory
paths = [p for p in glob.glob(vid_dir + "/*.m*")]
paths = sorted(paths)

# load the videos that have already been processed
# we'll skip these later
used_glosses = []
saved_data_paths = glob.glob("./"+out_file+"_*.json")
for i,f in enumerate(saved_data_paths):
	print(f"Loading file {i+1}/{len(saved_data_paths)}")
	saved_data = json.load(open(f))
	
	for sign in saved_data.values():
		used_glosses.extend(sign[0]["glosses"])

# get a sample & process them
paths_sample = get_sample(paths,4)
data = {}
for i, (gloss, paths) in enumerate(paths_sample):
	if gloss in used_glosses:
		print("Skipping", gloss)
		continue
	print(f"[{i}/{len(paths_sample)}]\t{gloss}",end = " ")
	sys.stdout.flush()
	# get a unique identifier (glosses are not unique)
	d = get_unique_id()

	s = Sign(d, [gloss])
	for j, path in enumerate(paths):
		s.examples.append(SignVideo(s,path))
		print(str(j), end = " ")
		sys.stdout.flush()

	print()

	# verify that the examples are in fact the same sign
	s.verify_examples()

	data[d] = s

	# save every 10 signs
	# if i % 10 == 0 and i > 0:
		# save_as_json(data, out_file + "_" + str(int(i / 10)) +  ".json")
		# data = {}

# save what remains since last save-point
# save_as_json(data, out_file + ".json")










