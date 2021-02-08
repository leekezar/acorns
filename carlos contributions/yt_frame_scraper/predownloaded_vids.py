import frameExtractor as fe
from pytube import Playlist
import os

def main():
    extractor = fe.FrameExtractor()

    dir = input('please enter a path to the directory: ')
    ofile = open('./data/processed_videos.txt', 'a+')

    processed_videos = []
    for vid in ofile:
        processed_videos.append(vid)

    print(f'/***** Processing Videos @ {dir} *****/')

    for file in os.listdir(dir):
        if file in processed_videos:
            print(f'>>> {file} already processed')
        if file not in processed_videos and (file.endswith('.mov') or file.endswith('.mp4')):
            new_filename = file[:file.rfind('.')].replace(' ', '_').lower()
            accepted = 'abcdefghijklmnopqrstuvwxyz1234567890_'
            for c in new_filename:
                if c not in accepted:
                    new_filename = new_filename.replace(c, '')
            extractor.frame_capture(f'{dir}/{file}', new_filename)
            print(file, file=ofile)
            processed_videos.append(file)
            print(f'>>> {file} has now been processed')
        print(f'>>> Preparing to process the next video')

    print('/***** All Videos Processed *****/')
    print(f'/***** {len(processed_videos)} Total Videos Processed *****/')
    print(f'Now terminating... goodbye!')


if __name__ == '__main__':
    main()
