import os
import zipfile
import rarfile
import shutil
import subprocess
import shlex
import sys
from .logging_setup import logger
from urllib.parse import urlparse
from IPython.utils import capture
import re

# File extensions
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg", ".3gp"]
AUDIO_EXTENSIONS = [".mp3", ".wav", ".aiff", ".aif", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".alac", ".pcm", ".opus", ".ape", ".amr", ".ac3", ".vox", ".caf"]
SUBTITLE_EXTENSIONS = [".srt", ".vtt", ".ass"]

# Function to run shell commands
def run_command(command):
    logger.debug(command)
    if isinstance(command, str):
        command = shlex.split(command)

    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    }
    process = subprocess.Popen(command, **sub_params)
    output, errors = process.communicate()
    if process.returncode != 0:
        logger.error("Error executing command")
        raise Exception(errors.decode())

# Function to print directory tree
def print_tree_directory(root_dir, indent=""):
    if not os.path.exists(root_dir):
        logger.error(f"{indent} Invalid directory or file: {root_dir}")
        return

    items = os.listdir(root_dir)
    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        is_last_item = index == len(items) - 1
        if os.path.isfile(item_path) and item_path.endswith(".zip"):
            with zipfile.ZipFile(item_path, "r") as zip_file:
                print(f"{indent}{'└──' if is_last_item else '├──'} {item} (zip file)")
                zip_contents = zip_file.namelist()
                for zip_item in sorted(zip_contents):
                    print(f"{indent}{'    ' if is_last_item else '│   '}{zip_item}")
        else:
            print(f"{indent}{'└──' if is_last_item else '├──'} {item}")
            if os.path.isdir(item_path):
                new_indent = indent + ("    " if is_last_item else "│   ")
                print_tree_directory(item_path, new_indent)

# Function to upload model list
def upload_model_list():
    weight_root = "weights"
    models = [f"weights/{name}" for name in os.listdir(weight_root) if name.endswith(".pth")]
    if models:
        logger.debug(models)

    index_root = "logs"
    index_paths = [None] + [f"logs/{name}" for name in os.listdir(index_root) if name.endswith(".index")]
    if index_paths:
        logger.debug(index_paths)

    return models, index_paths

# Function for manual download
def manual_download(url, dst):
    if "drive.google" in url:
        logger.info("Drive URL detected")
        if "folders" in url:
            logger.info("Folder download")
            os.system(f'gdown --folder "{url}" -O {dst} --fuzzy -c')
        else:
            logger.info("Single file download")
            os.system(f'gdown "{url}" -O {dst} --fuzzy -c')
    elif "huggingface" in url:
        logger.info("HuggingFace URL detected")
        if "/blob/" in url or "/resolve/" in url:
            url = url.replace("/blob/", "/resolve/")
            download_manager(url=url, path=dst, overwrite=True, progress=True)
        else:
            os.system(f"git clone {url} {dst+'repo/'}")
    elif "http" in url:
        logger.info("HTTP URL detected")
        download_manager(url=url, path=dst, overwrite=True, progress=True)
    elif os.path.exists(url):
        logger.info("Local path detected")
        copy_files(url, dst)
    else:
        logger.error(f"No valid URL: {url}")

# Function to download files from a list of URLs
def download_list(text_downloads):
    try:
        urls = [elem.strip() for elem in text_downloads.split(",")]
    except Exception as error:
        raise ValueError(f"No valid URL. {str(error)}")

    create_directories(["downloads", "logs", "weights"])
    path_download = "downloads/"
    for url in urls:
        manual_download(url, path_download)

    # Print directory tree
    print("####################################")
    print_tree_directory("downloads", indent="")
    print("####################################")

    # Process downloaded files
    select_zip_and_rar_files("downloads/")
    models, _ = upload_model_list()

    # Cleanup
    remove_directory_contents("downloads/repo")
    return f"Downloaded = {models}"

# Function to extract and move files from zip and rar archives
def select_zip_and_rar_files(directory_path="downloads/"):
    zip_files = [f for f in os.listdir(directory_path) if f.endswith(".zip")]
    rar_files = [f for f in os.listdir(directory_path) if f.endswith(".rar")]

    # Extract files
    for file_name in zip_files:
        file_path = os.path.join(directory_path, file_name)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(directory_path)

    for file_name in rar_files:
        file_path = os.path.join(directory_path, file_name)
        with rarfile.RarFile(file_path, "r") as rar_ref:
            rar_ref.extractall(directory_path)

    # Move extracted files
    move_files_with_extension(directory_path, ".index", "logs/")
    move_files_with_extension(directory_path, ".pth", "weights/")

    return "Download complete"

# Helper function to move files with a specific extension
def move_files_with_extension(src_dir, extension, destination_dir):
    for root, _, files in os.walk(src_dir):
        for file_name in files:
            if file_name.endswith(extension):
                source_file = os.path.join(root, file_name)
                destination = os.path.join(destination_dir, file_name)
                shutil.move(source_file, destination)

# Helper function to check file extensions
def is_file_with_extensions(string_path, extensions):
    return any(string_path.lower().endswith(ext) for ext in extensions)

# Specific file type checks
def is_video_file(string_path):
    return is_file_with_extensions(string_path, VIDEO_EXTENSIONS)

def is_audio_file(string_path):
    return is_file_with_extensions(string_path, AUDIO_EXTENSIONS)

def is_subtitle_file(string_path):
    return is_file_with_extensions(string_path, SUBTITLE_EXTENSIONS)

# Function to get directory files
def get_directory_files(directory):
    audio_files = []
    video_files = []
    sub_files = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            if is_audio_file(item_path):
                audio_files.append(item_path)
            elif is_video_file(item_path):
                video_files.append(item_path)
            elif is_subtitle_file(item_path):
                sub_files.append(item_path)

    logger.info(f"Files in path ({directory}): {str(audio_files + video_files + sub_files)}")
    return audio_files, video_files, sub_files

# Function to get valid files from paths
def get_valid_files(paths):
    valid_paths = []
    for path in paths:
        if os.path.isdir(path):
            audio_files, video_files, sub_files = get_directory_files(path)
            valid_paths.extend(audio_files)
            valid_paths.extend(video_files)
            valid_paths.extend(sub_files)
        else:
            valid_paths.append(path)

    return valid_paths

# Function to extract video links
def extract_video_links(link):
    params_dlp = {"quiet": False, "no_warnings": True, "noplaylist": False}
    try:
        from yt_dlp import YoutubeDL
        with capture.capture_output() as cap:
            with YoutubeDL(params_dlp) as ydl:
                ydl.extract_info(link, download=False, process=True)

        urls = re.findall(r'\[youtube\] Extracting URL: (.*?)\n', cap.stdout)
        logger.info(f"List of videos in ({link}): {str(urls)}")
    except Exception as error:
        logger.error(f"{link} >> {str(error)}")
        urls = [link]

    return urls

# Function to get link list
def get_link_list(link):
    try:
        urls = [elem.strip() for elem in link.split(",")]
    except Exception as error:
        raise ValueError(f"No valid URL. {str(error)}")

    valid_urls = []
    for url in urls:
        extracted_urls = extract_video_links(url)
        valid_urls.extend(extracted_urls)

    return valid_urls
