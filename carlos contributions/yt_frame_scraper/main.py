import frameExtractor as fe
from pytube import Playlist
import re

def process_channel(videos_link):
    print(f'/***** Processing Channel @ {videos_link} *****/')

    # create a list of video urls
    list_videos = Playlist(videos_link)

    # vars to keep track of videos downloaded
    video_count = 0
    total_video = len(list_videos)
    print(f'{total_video} Videos Found')

    # init processor
    processor = fe.FrameExtractor()

    # loop through and process videos
    for video in list_videos:
        video_count += 1
        if processor.vid_downloaded(video):
            print(f'>>> Video {video_count}/{total_video} already downloaded')
        else:
            processor.process_url(video)
            print(f'>>> Video {video_count}/{total_video} done')

    print(f'/***** Finished Processing Channel @ {videos_link} *****/')

def main():
    ifile = open('./data/input_channels.txt', 'r')
    ofile = open('./data/processed_channels.txt', 'a+')

    processed_channels = []
    for channel in ofile:
        processed_channels.append(channel)

    for channel in ifile:
        if channel not in processed_channels:
            process_channel(channel.strip())
            print(channel, file=ofile)
            processed_channels.append(channel)

if __name__ == '__main__':
    main()
