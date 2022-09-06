import json
import glob
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import mode

VERBOSE = True

def get_data(glob_pattern, max_files=0):
	data = {}
	paths = glob.glob(glob_pattern)
	for i, path in enumerate(paths):
		if max_files > 0 and i >= max_files:
			break

		if VERBOSE:
			print(f"Loading data file {i}/{len(paths)}")

		data.update(json.load(open(path)))

	return data

def format_data(data, window_size=30, step_size=1):
	formatted_data = {
		"hs" : {"x":[], "y":[]},
		"loc" : {"x":[], "y":[]},
		"mov" : {"x":[], "y":[]}
	}

	for i, (sign_id, examples) in enumerate(data.items()):
		if VERBOSE:
			print(f"Formatting data for sign {i}/{len(data)}")

		for j,example in enumerate(examples):
			if not example["phonemes"]:
				if VERBOSE:
					print(f"Skipping {sign_id}_{j} because it has no phonemes.")
				continue
			# handshape X,Y pairs
			x_hs = [frame["hs_r"]["distances"] for frame in \
						example["frames"] if frame["hs_r"]]
			
			# get the middle half by cutting first and last quarter
			x_min = int(len(x_hs)/4)
			x_max = x_min*3
			x_hs = x_hs[x_min:x_max]

			if len(x_hs) > 0:
				formatted_data["hs"]["x"].extend(x_hs)
				y_hs = [example["phonemes"]["handshape"]]*len(x_hs)
				formatted_data["hs"]["y"].extend(y_hs)

			
			# location X,Y pairs
			x_loc = [frame["loc_r"] for frame in \
						example["frames"] if frame["loc_r"]]
			# x_min = int(len(x_loc)/4)
			# x_max = x_min*3
			# x_loc = x_loc[x_min:x_max]
			if len(x_loc) > 0:
				formatted_data["loc"]["x"].extend(x_loc)
				y_loc = [example["phonemes"]["minor_location"]]*len(x_loc)
				formatted_data["loc"]["y"].extend(y_loc)

			
			# movement X,Y pairs
			x_mov = [frame["hs_r"]["raw"][0][:2] for frame in \
						example["frames"] if frame["hs_r"]]

			if len(x_mov) > window_size:
				windows = [x_mov[k:k+window_size] 
					for k in range(0,len(x_mov)-window_size,step_size)]
				# flatten the (n,window_size,2) array into 2 dims
				windows = np.reshape(windows, (-1,window_size*2))
				formatted_data["mov"]["x"].extend(windows)
				y_mov = [example["phonemes"]["path_movement"]]*len(windows)
				formatted_data["mov"]["y"].extend(y_mov)

	return formatted_data

def learn_handshape_rf(train_data, window_size=30, step_size=1):
	clf = RandomForestClassifier()
	y_pred = cross_val_predict(clf, 
		train_data["x"], 
		train_data["y"],
		cv=5, method="predict_proba")
	y_pred_top = np.argmax(y_pred,axis=1)

	# divide the predictions into windows in order to use mode()
	windows_pred = [mode(y_pred_top[start:start+window_size])[0]
		for start in range(0,len(y_pred_top),step_size)]
	windows_true = [mode(train_data["y"]\
		[start:start+ window_size])[0]
		for start in range(0,len(y_pred_top),step_size)]

	acc = accuracy_score(windows_true, windows_pred)
	prec = precision_score(windows_true, windows_pred, average='macro')
	rec = recall_score(windows_true, windows_pred, average='macro')

	if VERBOSE:
		print(f"Window {window_size}+{step_size}\t--> \
			acc: {acc:.4f}, \tprec:{prec:.4f}, \trec: {rec:.4f}, \
			\tf1: {2*(prec*rec)/(prec+rec):.4f}") 

	clf.fit(train_data["x"], train_data["y"])
	return clf

def learn_movement_rf(train_data, window_size=30, step_size=1):
	clf = RandomForestClassifier()
	y_pred = cross_val_predict(clf, 
		train_data["x"], 
		train_data["y"],
		cv=5, method="predict_proba")
	y_pred_top = np.argmax(y_pred,axis=1)

	acc = accuracy_score(train_data["y"], y_pred_top)
	prec = precision_score(train_data["y"], y_pred_top, average='macro')
	rec = recall_score(train_data["y"], y_pred_top, average='macro')

	if VERBOSE:
		print(f"Window {window_size}+{step_size}\t--> \
			acc: {acc:.4f}, \tprec:{prec:.4f}, \trec: {rec:.4f}, \
			\tf1: {2*(prec*rec)/(prec+rec):.4f}") 

	clf.fit(train_data["x"], train_data["y"])
	return clf

def learn_location_rf(train_data, window_size=30, step_size=1):
	clf = RandomForestClassifier()
	y_pred = cross_val_predict(clf, 
		train_data["x"], 
		train_data["y"],
		cv=5, method="predict_proba")
	y_pred_top = np.argmax(y_pred,axis=1)

	# divide the predictions into windows in order to use mode()
	windows_pred = [mode(y_pred_top[start:start+window_size])[0]
		for start in range(0,len(y_pred_top),step_size)]
	windows_true = [mode(train_data["y"]\
		[start:start+ window_size])[0]
		for start in range(0,len(y_pred_top),step_size)]

	acc = accuracy_score(windows_true, windows_pred)
	prec = precision_score(windows_true, windows_pred, average='macro')
	rec = recall_score(windows_true, windows_pred, average='macro')

	if VERBOSE:
		print(f"Window {window_size}+{step_size}\t--> \
			acc: {acc:.4f}, \tprec:{prec:.4f}, \trec: {rec:.4f}, \
			\tf1: {2*(prec*rec)/(prec+rec):.4f}") 

	clf.fit(train_data["x"], train_data["y"])
	return clf

def learn_f2p_rf(train_data):
	hs_model = learn_handshape_rf(train_data["hs"])
	loc_model = learn_location_rf(train_data["loc"])
	mov_model = learn_movement_rf(train_data["mov"])
	return hs_model, loc_model, mov_model
	
asllex_train_data = get_data("./asllex_*.json")
asllex_train_data = format_data(asllex_train_data)
hs_model, loc_model, mvt_model = learn_f2p_rf(asllex_train_data)

pickle.dump(hs_model, open("hs_model.pickle", "wb+"))
pickle.dump(loc_model, open("loc_model.pickle", "wb+"))
pickle.dump(mvt_model, open("mov_model.pickle", "wb+"))