import scrapy, time, json
from string import ascii_lowercase
BASE = "https://signasl.org"

def get_between(text, start, end, preserve_end = False):
	return text.split(start)[1].split(end)[0] + (end if preserve_end else "")

def parse_letter_page(letter, html):
	next_pages = []

	for line in html.split("\n"):
		# gets the other pages for this letter
		# e.g. <a href='/dictionary/a/1'>1</a></li><li><a href='/dictionary/a/2'>2</a> ...
		if "dictionary/" + letter + "/" in line:
			links = line.split("/dictionary/" + letter + "/")[2:]
			for link in links:
				num = link.split("'")[0]
				if num != "1" and "li" not in num:
					next_pages.append(BASE + "/dictionary/" + letter + "/" + num)
		
		# gets the current signs on this page
		elif "<td>" in line:
			url = get_between(line, "href=\"", "\">")
			next_pages.append(BASE + url)
		else:
			continue
	
	# remove duplicates
	return list(set(next_pages))

def parse_sign_page(sign, html):
	videos = []			# the videos found on this page thusfar
	curr_def = None		# the current definition (sometimes there are several)
	in_body = False		# whether ths current line is within the body

	for line in html.split("\n"):
		if "<body>" in line:
			in_body = True

		if not in_body:
			continue

		# update the current definition if one is present
		if "How to sign:</b> " in line:
			curr_def = get_between(line, "</b>", "<br />")

		# the different variations of video elements
		if ".mp4" in line or "youtube.com/watch" in line:
			if "content" in line:
				vid_url = get_between(line, "content=\"", "\"")
			elif "src" in line:
				vid_url = get_between(line, "src=\"", "\"")
		else:
			continue

		videos.append((sign, curr_def, vid_url))

	# remove duplicates
	return list(set(videos))

def save_videos(videos):
	print(f"Saving {len(videos)} videos for {videos[0][0]}")
	with open("./videos.csv", "a+") as file:
		for video in videos:
			if video[1]:
				file.write(video[0] + "," + video[1].replace(",","") + "," + video[2] + "\n")
			else:
				file.write(video[0] + ",," + video[2] + "\n")

class SignASLSpider(scrapy.Spider):
	name = 'sign_asl_spider'
	start_urls = ['https://www.signasl.org/dictionary/' + c for c in ascii_lowercase]
	# start_urls = ['https://www.signasl.org/dictionary/a']
	
	download_delay = 5.0

	def parse(self, response):
		# not interested in processing http as well as https -- only use the latter
		if "http://" in response.url:
			yield None

		# the kw for pages showing the signs of a particular letter
		if "dictionary" in response.url:
			letter = response.url.split("/")[-1]
			next_pages = parse_letter_page(letter, response.text)
			for url in next_pages:
				yield response.follow(url, self.parse)

		# the kw for pages showing the variations of a single sign
		elif "sign" in response.url:
			sign = response.url.split("/")[-1]
			videos = parse_sign_page(sign, response.text)
			if len(videos) > 0:
				save_videos(videos)

		return None