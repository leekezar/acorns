import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
feats = json.load(open("sample_features.json"))

hs = []

ids = []
for vid_id in feats.keys():
	for frame in feats[vid_id]["left_handshape"] + feats[vid_id]["right_handshape"]:
		if not frame[0]:
			continue
		coords = np.array(frame).flatten()
		hs.append(coords)
		ids.append(vid_id)

hs = np.array(hs)

print("before dimredux: ",np.shape(hs))
redux_hs = PCA(n_components=10).fit_transform(hs)
print("after dimredux: ",np.shape(redux_hs))

kmeans = KMeans(n_clusters = 30).fit(redux_hs)

results = zip(ids, kmeans.labels_)

results = sorted(results, key = lambda x: x[1])

for vid,label in results:
	print(vid,label)