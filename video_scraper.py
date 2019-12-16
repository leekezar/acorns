import pandas as pd
from openpyxl import load_workbook
import requests
import time
import sys

wb = load_workbook(filename = 'urls.xlsx')
sheet_ranges = wb["Sheet1"]
df = pd.DataFrame(sheet_ranges.values)
glosses = df[1]
urls = df[11]
gloss_to_url = {}

gloss = ""
warning_given = 0

def download_url(url, filename):
	req = requests.get(url)
	with open("./word_signs/" + filename, "wb+") as f:
		f.write(req.content)

def parse_url_cell(text):
	return text[12:-9]

for i, url in enumerate(urls):
	if i < 4117:
		continue
	try:
		if glosses[i] != None and glosses[i].strip() != "":
			gloss = glosses[i]

		# remove invalid characters
		invalid_chars = ["\\", ":", "*", "?", '"', "<", ">", "|"]
		for char in invalid_chars:
			gloss = gloss.replace(char, "_")

		# skip cells that don't have a URL in them
		if "http" not in url:
			continue

		# trim the '=HYPERLINK("' (12 chars) and '", "MOV")' (9 chars)
		urls[i] = parse_url_cell(urls[i])

		# for each word in the gloss, download and save appropriately
		for word in gloss.split("/"):
			if word not in gloss_to_url.keys():
				gloss_to_url[word] = []

			gloss_to_url[word].append(urls[i])

			name = word + "_" + str(len(gloss_to_url[word])) + ".mov"

			download_url(urls[i], name)
			print("Saved %s (%.4f)" % (name, i/len(urls)))
			sys.stdout.flush()

	except UserWarning as e:
		print("Ignoring UserWarning...")

	except Exception as e:
		print(e)
		continue
		for i in range(100):
			print("\a")
			time.sleep(1)