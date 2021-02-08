from pytube import YouTube
import cv2
import os

class FrameExtractor:
    VID_PREFIX = './data/vids'
    CC_PREFIX = './data/captions'
    FRAME_PREFIX = './data/frames'
    FILE_GLOSSARY = './data/related_files.csv'

    def __init__(self):
        # create list of processed_urls
        self.processed_urls = []

        # fill in list
        with open(self.FILE_GLOSSARY, 'r') as ifile:
            ifile.readline() # skip the header
            for entry in ifile:
                url = entry.split(',')[1]
                self.processed_urls.append(url)

    def vid_downloaded(self, url):
        return url in self.processed_urls

    def frame_capture(self, vid_path, fname):
        vidcap = cv2.VideoCapture(vid_path)
        vidcap.set(5, 10)

        path = f'{self.FRAME_PREFIX}/{fname}_frames'
        if not os.path.isdir(path):
            os.mkdir(path)

        count = 0
        increment = 1000/10 # number of milliseconds/fps
        time = 0
        vidcap.set(cv2.CAP_PROP_POS_MSEC, time)
        success, image = vidcap.read()
        while success:
            count += 1
            cv2.imwrite(f'{path}/frame{count}.png', image)
            if count%500 == 0:
                print(f'{count} frames captured......')
            time += increment
            vidcap.set(cv2.CAP_PROP_POS_MSEC, time)
            success, image = vidcap.read()

        print(f'all {count} frames captured from {fname}.mp4...')

    def save_video(self, yt_obj):
        stream = yt_obj.streams

        # choose the highest quality mp4 with no audio
        vid = stream.filter(only_video=True, file_extension='mp4').order_by('resolution').desc().first()

        # create filename and remove all unneeded chars
        new_filename = vid.title.replace(' ', '_').lower()
        accepted = 'abcdefghijklmnopqrstuvwxyz1234567890_'
        for c in new_filename:
            if c not in accepted:
                new_filename = new_filename.replace(c, '')

        vid.download(self.VID_PREFIX, new_filename)

        print(f'{new_filename}.mp4 downloaded...')

        return new_filename

    def save_captions(self, yt_obj, fname):
        if len(yt_obj.captions) == 0:
            print('WARNING: video has no captions')
            print(f'creating empty file {fname}.srt...')
            open(f'{self.CC_PREFIX}/{fname}.srt', 'w').close()
            return

        caption = yt_obj.captions['en']

        with open(f'{self.CC_PREFIX}/{fname}.srt', 'w') as ofile:
            print(caption.generate_srt_captions(), file=ofile)

        print(f'{fname}.srt downloaded...')

    def process_url(self, url):
        if self.vid_downloaded(url):
            return

        print(f'----- Processing vid @ {url} -----')
        yt = YouTube(url)
        fname = self.save_video(yt)
        self.frame_capture(f'{self.VID_PREFIX}/{fname}.mp4', fname)
        self.save_captions(yt, fname)

        with open(self.FILE_GLOSSARY, 'a') as ofile:
            print(fname, file=ofile, end=',')
            print(url, file=ofile, end=',')
            print(f'{self.VID_PREFIX}/{fname}.mp4', file=ofile, end=',')
            print(f'{self.CC_PREFIX}/{fname}.srt', file=ofile, end=',')
            print(f'{self.FRAME_PREFIX}/{fname}_frames', file=ofile)

        print(f'file glossary @ {self.FILE_GLOSSARY} updated...')

        print(f'----- Successfully processed as {fname} -----')

def main():
    path = input('please enter a video filepath: ')
    new_filename = path[path.rfind('/')+1:path.rfind('.')].replace(' ', '_').lower()
    accepted = 'abcdefghijklmnopqrstuvwxyz1234567890_'
    for c in new_filename:
        if c not in accepted:
            new_filename = new_filename.replace(c, '')

    try:
        FrameExtractor().frame_capture(path, new_filename)
    except Exception as e:
        print("an error occured:")
        print(e)
        print("terminating processes")
    finally:
        print("goodbye!")

if __name__ == '__main__':
    main()
