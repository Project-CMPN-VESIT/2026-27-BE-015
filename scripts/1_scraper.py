import yt_dlp
import os

def download_youtube_video(url, output_folder="Obama_Dataset/01_raw_youtube"):
    os.makedirs(output_folder, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best', 
        'merge_output_format': 'mp4', 
        'outtmpl': f'{output_folder}/%(id)s_%(upload_date)s.%(ext)s',
        
        # --- THE FIX IS HERE ---
        # This forces yt-dlp to use the WEB or ANDROID client instead of the broken iOS one
        'extractor_args': {'youtube':['client=web,android']}, 
        
        'ignoreerrors': True,
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"\n--- Downloading from: {url} ---")
        ydl.download([url])
        print("--- Download Complete! ---\n")

url = "https://www.youtube.com/watch?v=lbwlVwNWLzU" 
download_youtube_video(url)