# YouTube Frame Scraper
***Carlos Lao***

## How to use:
1. Add channels to download from to input_channels.txt
    1. Go to the channel homepage
    2. Click "PLAY ALL" by "Uploads" (this will generate a playlist with all videos)
    3. Click on the newly generated playlist title (should be something along the lines of "Uploads from _channel name_")
    4. Copy and paste the playlist link (format: _https://www.youtube.com/playlist?list=someRandomCharacters_)
2. Run `main.py`

## Known issues:
- Sometimes encounter the error `mmco: unref short failure` on some frames. I think it has something to do with the mp4 file being corrupted. Nonetheless, the error does not terminate the program so it's still functional.
