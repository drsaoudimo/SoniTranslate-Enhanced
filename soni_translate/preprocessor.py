from .utils import remove_files
import os
import shutil
import subprocess
import time
import shlex
import sys
import json
from .logging_setup import logger

# Constants for codecs
ERROR_INCORRECT_CODEC_PARAMETERS = ["prores", "ffv1", "msmpeg4v3", "wmv2", "theora"]
TESTED_CODECS = ["h264", "h265", "hevc", "vp9", "mpeg4", "mpeg2video", "mjpeg"]

class OperationFailedError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        super().__init__(message)
        self.message = message

def run_subprocess(command, check_file=None):
    """Run a subprocess command and check for errors."""
    command = shlex.split(command)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    output, errors = process.communicate()
    time.sleep(1)
    if process.returncode != 0 or (check_file and not os.path.exists(check_file)):
        raise OperationFailedError(f"Error executing command:\n{errors.decode('utf-8')}")
    return output.decode('utf-8')

def get_video_codec(video_file):
    """Get the codec of the given video file."""
    command = rf'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of json "{video_file}"'
    try:
        codec_info = json.loads(run_subprocess(command))
        return codec_info['streams'][0]['codec_name']
    except Exception as error:
        logger.debug(str(error))
        return None

def preprocess_audio(base_audio, preview=False):
    """Preprocess the audio from the given base audio file."""
    base_audio = base_audio.strip()
    audio_wav = "audio.wav"
    remove_files([audio_wav])

    if preview:
        logger.warning("Creating a preview video of 10 seconds. To disable this option, go to advanced settings and turn off preview.")
        command = f'ffmpeg -y -i "{base_audio}" -ss 00:00:20 -t 00:00:10 -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_wav}'
    else:
        command = f'ffmpeg -y -i "{base_audio}" -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_wav}'

    run_subprocess(command, check_file=audio_wav)

def preprocess_video(video, preview=False):
    """Preprocess the given video file."""
    video = video.strip()
    output_file = "Video.mp4"
    remove_files([output_file])

    if os.path.exists(video):
        if preview:
            logger.warning("Creating a preview video of 10 seconds. To disable this option, go to advanced settings and turn off preview.")
            command = f'ffmpeg -y -i "{video}" -ss 00:00:20 -t 00:00:10 -c:v libx264 -c:a aac -strict experimental {output_file}'
        else:
            video_codec = get_video_codec(video)
            if video.endswith(".mp4") or video_codec in TESTED_CODECS:
                shutil.copy(video, output_file)
                return output_file
            else:
                logger.warning("File does not have the '.mp4' extension or a supported codec. Converting video to mp4 (codec: h264).")
                command = f'ffmpeg -y -i "{video}" -c:v libx264 -c:a aac -strict experimental {output_file}'
    else:
        raise OperationFailedError("Video file does not exist")

    run_subprocess(command, check_file=output_file)
    return output_file

def download_media(video, output_file, preview=False):
    """Download media from a given video link."""
    if preview:
        logger.warning("Creating a preview from the link, 10 seconds. To disable this option, go to advanced settings and turn off preview.")
        command = f'yt-dlp -f "mp4" --downloader ffmpeg --downloader-args "ffmpeg_i: -ss 00:00:20 -t 00:00:10" --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {output_file} {video}'
    else:
        command = f'yt-dlp -f "mp4" --force-overwrites --max-downloads 1 --no-warnings --no-playlist --no-abort-on-error --ignore-no-formats-error --restrict-filenames -o {output_file} {video}'

    run_subprocess(command, check_file=output_file)
    return output_file

def audio_video_preprocessor(preview, video, output_file, audio_wav, use_cuda=False):
    """Preprocess audio and video based on input parameters."""
    video = video.strip()
    previous_files_to_remove = [output_file, "audio.webm", audio_wav]
    remove_files(previous_files_to_remove)

    if os.path.exists(video):
        output_file = preprocess_video(video, preview)
        command = f'ffmpeg -y -i {output_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_wav}'
        run_subprocess(command, check_file=audio_wav)
    else:
        output_file = download_media(video, output_file, preview)
        command = f'ffmpeg -y -i {output_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_wav}'
        run_subprocess(command, check_file=audio_wav)
