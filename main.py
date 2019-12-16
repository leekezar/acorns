from pytube import YouTube
import pytube
import requests
import json
import os
import pandas as pd

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

    # load previously saved videos
    with open("./videos_meta.json", "r") as meta_file:
        try:
            meta_info = json.load(meta_file)
            channels = list(meta_info.keys())
        except json.decoder.JSONDecodeError:
            meta_info = {}
            channels = []

    for channel_id in clean_chan:
        print("Downloading from %s" % channel_id)

        if channel_id in channels:
            continue

        page_token = None
        while True:
            # Grabbing channel videos
            try:
                results = search_channel_videos(channel_id, yt_key, page_token)
                page_token = results['nextPageToken']
                #print("page_token: ", page_token)

                tot_video_ids = [x['id']['videoId'] for x in results['items']]
                #print("tot_video_ids: ", tot_video_ids)
            except KeyError:
                print("Couldn't get new page.")
                break

            for video_id in tot_video_ids:
                print("video_id: %s\t" % video_id, end="")

                # print("sample_video: ", sample_video)
                # Getting caption
                video_url = 'https://www.youtube.com/watch?v={0}'.format(video_id)

                try:
                    yt = YouTube(video_url)

                    title = yt.title
                    duration = yt.player_config_args['player_response']['videoDetails']['lengthSeconds']
                    desc = yt.description
                    en_caption = yt.captions.get_by_language_code('en')
                    
                    if en_caption:
                        xml_capt = en_caption.xml_captions
                    else:
                        print("No captions.")
                        continue

                    # Writing caption to save
                    with open(caption_dir+'/caption_{0}.xml'.format(video_id), 'w') as f:
                        f.write(xml_capt)

                    # Getting video
                    # downloading to directory
                    dl = yt.streams.first().download(output_path=video_dir, filename=video_id)
                    
                    meta_info[video_id] = {
                        "channel_id": channel_id,
                        "title": title,
                        "description": desc,
                        "duration": duration
                    }

                    with open("./videos_meta.json", "w") as meta_file:
                        json.dump(meta_info, meta_file)
                        print("Added " + video_id)
                    
                except pytube.exceptions.VideoUnavailable:
                    print("Video Unavailable")
                    continue

                except pytube.exceptions.RegexMatchError:
                    print("Couldn't find video")
                    continue

                except:
                    print("Couldn't download, unknown error")
                    continue

        print("Done with %s", channel_id)
        channels.append(channel_id)


if __name__ == '__main__':
    main()
