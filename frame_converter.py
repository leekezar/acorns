from os import listdir
from os.path import isfile, join
import os
import cv2
print(cv2.__version__)
folder = "./word_signs/"
word_files = [f for f in listdir(folder) if isfile(join(folder, f))]
folders = [f for f in listdir(folder) if not isfile(join(folder, f))]
for f in word_files:
	
	os.mkdir(os.path.join(folder, f[:-4] + "_frames"))

	vidcap = cv2.VideoCapture(os.path.join(folder,f))
	count = 0
	while True:
		success, image = vidcap.read()
		if not success:
			break
		cv2.imwrite(os.path.join(folder,f[:-4] +"_frames",str(count) + ".jpg"), image)
		count += 1

	print("Processed " + f)
