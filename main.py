from pytube import YouTube
import requests
import json
import os
import pandas as pd


"""
Download videos
"""



def load_credentials(cred_file):
    """
    Load credentials
    """
    with open(cred_file, 'r') as f:
        data = f.read()

    print(data)
    json_data = json.loads(data)
    return json_data

def search_channel_videos(channel_id, yt_key, page_token=None):
    """
    Retrieve video from channel
    """

    if page_token != None:
        url = 'https://www.googleapis.com/youtube/v3/search?order=date&pageToken={token}&part=snippet&channelId={ch_id}&maxResults=50&key={yt_key}'.format(ch_id=channel_id, yt_key=yt_key, token=page_token)

    else:
        url = 'https://www.googleapis.com/youtube/v3/search?order=date&part=snippet&channelId={ch_id}&maxResults=50&key={yt_key}'.format(ch_id=channel_id, yt_key=yt_key)

    results = requests.get(url)

    # print("results.text: ", results.json())
    return results.json()


def download_captions(video_id, yt_key):
    """
    Use the youtube api to download captions

    :param video_id: Video id for youtube video
    :type video_id: string
    """

    url = 'https://www.googleapis.com/youtube/v3/captions/{ID}?key={API_KEY}'.format(ID=video_id, API_KEY=YT_KEY)

    res = requests.get(url)
    print("res.text: ", res.text)
    return res


def main():
    """
    Main
    """


    def receiv_channel_id(x):

        try:
            channel_id = x.split('channel/')[1]
        except IndexError:
            channel_id = None

        return channel_id


    # Loading csv to loop through channel_ids
    csv_name = 'source_channels.csv'
    target_channels = pd.read_csv(csv_name)

    print("target_channels: ", target_channels['url'].iloc[0].split('channel/'))

    chan_ids = target_channels['url'].transform(receiv_channel_id)

    clean_chan = chan_ids.dropna().to_list()
    print("chan_ids: ", clean_chan)


    video_dir = 'videos'
    caption_dir = 'captions'

    credentials = load_credentials('credentials.json')

    yt_key = credentials['yt_key']


    for channel_id in clean_chan:
        page_token = None
        while True:
            # Grabbing channel videos
            try:
                results = search_channel_videos(channel_id, yt_key, page_token)
                page_token = results['nextPageToken']
                print("page_token: ", page_token)

                tot_video_ids = [x['id']['videoId'] for x in results['items']]
                print("tot_video_ids: ", tot_video_ids)
            except KeyError:
                break

            for video_id in tot_video_ids:
                print("video_id: ", video_id)
                # print("sample_video: ", sample_video)
                # Getting caption
                video_url = 'https://www.youtube.com/watch?v={0}'.format(video_id)
                yt = YouTube(video_url)
                en_caption = yt.captions.get_by_language_code('en')
                xml_capt = en_caption.xml_captions

                # Writing caption to save
                with open(caption_dir+'/caption_{0}.xml'.format(video_id), 'w') as f:
                    f.write(xml_capt)

                # Getting video
                # downloading to directory
                dl = yt.streams.first().download(output_path=video_dir, filename='{0}.mp4'.format(video_id))




if __name__ == '__main__':
    main()
