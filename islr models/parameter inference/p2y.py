import pickle, json, glob
from scipy.stats import mode
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from random import shuffle
from collections import Counter
import math

id2n = {} #json.load(open("id2n.json"))

def get_n(sign_id):
	if sign_id not in id2n.keys():
		id2n[sign_id] = len(id2n)
		json.dump(id2n, open("id2n.json", "w+"), indent=4)
	return id2n[sign_id]

def get_id(n):
	n2id = {v:k for k,v in id2n.items()}
	if n not in n2id.keys():
		return None
	return n2id[n]

def label_data(glob_pattern, hs_model, loc_model, mov_model, 
		window_size = 30, step_size = 1, max_files = 200):

	paths = glob.glob(glob_pattern)[:max_files]

	train_data = []
	test_data = []

	for i, path in enumerate(paths):
		# print(f"Predicting file {i+1}/{len(paths)}")

		data = json.load(open(path))
		for sign_id, examples in data.items():
			
			valid_examples = []

			for j,example in enumerate(examples):
				# predict handshape
				x_hs = [frame["hs_r"]["distances"] for frame in \
						example["frames"] if frame["hs_r"]]
				x_hs = x_hs[int(len(x_hs)/4):int(len(x_hs)/4)*3]
				if len(x_hs) == 0:
					continue
				y_pred_hs = mode(hs_model.predict(x_hs))[0]

				# predict location
				x_loc = [frame["loc_r"] for frame in \
						example["frames"] if frame["loc_r"]]
				if len(x_loc) == 0:
					continue
				y_pred_loc = mode(loc_model.predict(x_loc))[0]

				# predict movement
				x_mov = [frame["hs_r"]["raw"][0][:2] for frame in \
						example["frames"] if frame["hs_r"]]
				
				if len(x_mov) < window_size:
					continue
				elif len(x_mov) == window_size:
					windows = x_mov
				else:
					windows = [x_mov[k:k+window_size] for k in \
						range(0,len(x_mov)-window_size,step_size)]

				# flatten the (n,window_size,2) array into 2 dims
				windows = np.reshape(windows, (-1,window_size*2))
				y_pred_mov = mode(mov_model.predict(windows))[0]


				n = get_n(example["gloss"][0])
				instance = {"hs":y_pred_hs[0],"loc":y_pred_loc[0],
						"mov":y_pred_mov[0],"sign":n}

				valid_examples.append(instance)

			if len(valid_examples) != 5:
				continue

			shuffle(valid_examples)

			train_data.extend(valid_examples[:3])
			test_data.extend(valid_examples[3:])

			# for instance in valid_examples:
			# 	print(f'gloss:{get_id(instance["sign"])}\ths:{instance["hs"]}\tloc:{instance["loc"]}\tmov:{instance["mov"]}')
			# print("")

		if i > 0 and i % 10 == 0:
			hl2y, lm2y, mh2y = learn_p2y_prob(train_data)
			eval_p2y_prob(hl2y, lm2y, mh2y, test_data)

	return train_data, test_data

def smooth(d,a=0.01):
	for k1 in d.keys():
		for k2 in d[k1].keys():
			d[k1][k2] = (1-a)*d[k1][k2] + a*1
	return d

# compute the joint probability of P(id|hs,loc,mov)
def learn_p2y_prob(train_data):
	hl2y = {}
	lm2y = {}
	mh2y = {}

	# TODO: include single-factor probability (prior prob) p(y|x) where x is one factor
	# TODO: use single-factors to weight the probabilities p(y|x)p(x)
	# TODO: inject numpy arrays
	for i,item in enumerate(train_data):
		hl = str(item["hs"]) + str(item["loc"])
		lm = str(item["loc"]) + str(item["mov"])
		mh = str(item["mov"]) + str(item["hs"])

		y = item["sign"]

		if hl not in hl2y.keys():
			hl2y[hl] = Counter()

		hl2y[hl][y] += 1

		if lm not in lm2y.keys():
			lm2y[lm] = Counter()

		lm2y[lm][y] += 1

		if mh not in mh2y.keys():
			mh2y[mh] = Counter()

		mh2y[mh][y] += 1

	for hl in hl2y.keys():
		tot = sum(hl2y[hl].values())

		for y in hl2y[hl].keys():
			hl2y[hl][y] = hl2y[hl][y] / tot

	for lm in lm2y.keys():
		tot = sum(lm2y[lm].values())
		
		for y in lm2y[lm].keys():
			lm2y[lm][y] = lm2y[lm][y] / tot

	for mh in mh2y.keys():
		tot = sum(mh2y[mh].values())
		
		for y in mh2y[mh].keys():
			mh2y[mh][y] = mh2y[mh][y] / tot

	return smooth(hl2y), smooth(lm2y), smooth(mh2y)

def eval_p2y_prob(hl2y, lm2y, mh2y, test_data):
	t1 = 0
	t5 = 0
	t10 = 0

	y_all = set([i["sign"] for i in test_data])

	y_preds = []
	y_trues = []
	for item in test_data:
		hl = str(item["hs"]) + str(item["loc"])
		if hl not in hl2y.keys():
			hl2y[hl] = { y : 1e-10 for y in y_all }

		lm = str(item["loc"]) + str(item["mov"])
		if lm not in lm2y.keys():
			lm2y[lm] = { y : 1e-10 for y in y_all }

		mh = str(item["mov"]) + str(item["hs"])
		if mh not in mh2y.keys():
			mh2y[mh] = { y : 1e-10 for y in y_all }

		y_pred = {
				y_i : 
					-1*math.log(max(hl2y[hl][y_i],1e-10))
					- math.log(max(lm2y[lm][y_i],1e-10))
					- math.log(max(mh2y[mh][y_i],1e-10))
			for y_i in y_all
		}

		y_pred = sorted(y_pred.items(), key=lambda x:x[1])
		y_pred_labels, y_pred_probs = zip(*y_pred)

		y_preds.append(y_pred_labels[0])
		y = item["sign"]
		y_trues.append(y)

		print(y,y_pred_labels[:10])
		
		if y == y_pred_labels[0]:
			t1 += 1

		if y in y_pred_labels[:5]:
			t5 += 1

		if y in y_pred_labels[:10]:
			t10 += 1

	acc = accuracy_score(y_trues, y_preds)
	prec = precision_score(y_trues, y_preds, average='macro')
	rec = recall_score(y_trues, y_preds, average='macro')

	print(f"n_signs: {len(set(y_trues))}, acc: {acc:.4f}, \tprec:{prec:.4f}, rec: {rec:.4f}, \
			f1: {2*(prec*rec)/(prec+rec):.4f}, \
			r@1: {t1/len(test_data):.4f}, \
			r@5: {t5/len(test_data):.4f}, \
			r@10: {t10/len(test_data):.4f}")

hs_model = pickle.load(open("hs_model.pickle", "rb"))
loc_model = pickle.load(open("loc_model.pickle", "rb"))
mov_model = pickle.load(open("mov_model.pickle", "rb"))
train_data, test_data = label_data("./5sample_*.json", hs_model, loc_model, mov_model)
hl2y, lm2y, mh2y = learn_p2y_prob(train_data)
eval_p2y_prob(hl2y, lm2y, mh2y, test_data)






pass