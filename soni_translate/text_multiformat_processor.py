from typing import List
import os
import re
import string
import copy
import sf
from PIL import Image, ImageOps, ImageDraw, ImageFont
import srt
from .logging_setup import logger
from .utils import get_writer, remove_files, run_command, remove_directory_contents
import soundfile as sf

# Constants
punctuation_list = list(string.punctuation + "¡¿«»„”“”‚‘’「」『』《》（）【】〈〉〔〕〖〗〘〙〚〛⸤⸥⸨⸩")
symbol_list = punctuation_list + ["", "..", "..."]

COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "light_gray": (200, 200, 200),
    "light_blue": (173, 216, 230),
    "light_green": (144, 238, 144),
    "light_yellow": (255, 255, 224),
    "light_pink": (255, 182, 193),
    "lavender": (230, 230, 250),
    "peach": (255, 218, 185),
    "light_cyan": (224, 255, 255),
    "light_salmon": (255, 160, 122),
    "light_green_yellow": (173, 255, 47),
}
BORDER_COLORS = ["dynamic"] + list(COLORS.keys())


def extract_from_srt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            srt_content = file.read()
    except Exception as error:
        logger.error(str(error))
        fixed_file = "fixed_sub.srt"
        remove_files(fixed_file)
        fix_sub = f'ffmpeg -i "{file_path}" "{fixed_file}" -y'
        run_command(fix_sub)
        with open(fixed_file, "r", encoding="utf-8") as file:
            srt_content = file.read()

    subtitle_generator = srt.parse(srt_content)
    srt_content_list = list(subtitle_generator)
    if not srt_content_list:
        raise Exception("No data found in srt subtitle file")
    return srt_content_list


def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove content within square brackets
    text = re.sub(r'<comment>.*?</comment>', '', text)  # Remove content within <comment> tags
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'♫.*?♫|♪.*?♪', '', text)  # Remove "♫" and "♪" content
    text = text.replace("\n", ". ")  # Replace newline characters with an empty string
    text = text.replace('"', '')  # Remove double quotation marks
    text = re.sub(r"\s+", " ", text.strip())  # Collapse multiple spaces and replace with a single space
    return "" if any(symbol in text for symbol in symbol_list) else text


def srt_file_to_segments(file_path, speaker=False):
    try:
        srt_content_list = extract_from_srt(file_path)
    except Exception as error:
        logger.error(str(error))
        raise

    segments = [
        {
            "text": clean_text(str(segment.content)),
            "start": float(segment.start.total_seconds()),
            "end": float(segment.end.total_seconds())
        }
        for segment in srt_content_list
        if clean_text(str(segment.content))
    ]

    if not segments:
        raise Exception("No data found in srt subtitle file")

    if speaker:
        segments = [{**seg, "speaker": "SPEAKER_00"} for seg in segments]

    return {"segments": segments}


def dehyphenate(lines: List[str], line_no: int) -> List[str]:
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix):]
    return lines


def remove_hyphens(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]

    line_numbers = [line_no for line_no, line in enumerate(lines[:-1]) if line.endswith("-")]

    for line_no in line_numbers:
        lines = dehyphenate(lines, line_no)

    return "\n".join(lines)


def pdf_to_txt(pdf_file, start_page, end_page):
    from pypdf import PdfReader

    with open(pdf_file, "rb") as file:
        reader = PdfReader(file)
        logger.debug(f"Total pages: {reader.get_num_pages()}")
        text = ""

        start_page_idx = max((start_page-1), 0)
        end_page_inx = min((end_page), (reader.get_num_pages()))
        document_pages = reader.pages[start_page_idx:end_page_inx]
        logger.info(
            f"Selected pages from {start_page_idx} to {end_page_inx}: "
            f"{len(document_pages)}"
        )

        for page in document_pages:
            text += remove_hyphens(page.extract_text())
    return text


def docx_to_txt(docx_file):
    from docx import Document

    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def replace_multiple_elements(text, replacements):
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    replaced_text = pattern.sub(
        lambda match: replacements[match.group(0)], text
    )

    replaced_text = re.sub(r"\s+", " ", replaced_text)
    return replaced_text


def document_preprocessor(file_path, is_string, start_page, end_page):
    if not is_string:
        file_ext = os.path.splitext(file_path)[1].lower()

    if is_string:
        text = file_path
    elif file_ext == ".pdf":
        text = pdf_to_txt(file_path, start_page, end_page)
    elif file_ext == ".docx":
        text = docx_to_txt(file_path)
    elif file_ext == ".txt":
        with open(file_path, "r", encoding='utf-8', errors='replace') as file:
            text = file.read()
    else:
        raise Exception("Unsupported file format")

    replacements = {
        "、": "、 ",
        "。": "。 ",
    }
    text = replace_multiple_elements(text, replacements)

    return text.strip(), text


def segments_to_plain_text(result_diarize):
    complete_text = ' '.join(seg["text"] for seg in result_diarize["segments"])

    txt_file_path = "./text_translation.txt"
    with open(txt_file_path, "w", encoding='utf-8', errors='replace') as txt_file:
        txt_file.write(complete_text)

    return txt_file_path, complete_text


def create_image_with_text_and_subimages(
    text,
    subimages,
    width,
    height,
    text_color,
    background_color,
    output_file
):
    image = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_x = (width - (text_bbox[2] - text_bbox[0])) / 2
    text_y = (height - (text_bbox[3] - text_bbox[1])) / 2

    draw.text((text_x, text_y), text, fill=text_color, font=font)

    for subimage in subimages:
        try:
            image.paste(subimage, (0, 0), subimage)
        except Exception as e:
            logger.error(str(e))
            continue

    image.save(output_file)


def main( audio_path, output_directory, video_path, sub_file, audio_fps=16000, speaker=False, start=1, end=1):
    from pathlib import Path

    remove_directory_contents(output_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_path)
    video_path = Path(video_path)

    audio_path2 = output_directory / "audio.wav"

    # srtfile to txt
    subtitle_srt = srt_file_to_segments(sub_file, speaker)
    txt_file_path, txt_data = segments_to_plain_text(subtitle_srt)

    # docx & pdf & txt to txt
    document_txt, _ = document_preprocessor(audio_path, True, start, end)

    # Text merge
    complete_txt_data = f"{txt_data}\n\n{document_txt}"
    txt_file_path = "./translate.txt"
    with open(txt_file_path, "w", encoding='utf-8', errors='replace') as txt_file:
        txt_file.write(complete_txt_data)

    # Draw Picture
    create_image_with_text_and_subimages(
        complete_txt_data,
        [],
        1300,
        1300,
        COLORS["blue"],
        COLORS["white"],
        "./1.jpg"
    )
    # TXT to WAV
    run_command(
        f"ffmpeg -f concat -i {audio_path2} -ar {audio_fps} -vn -acodec pcm_s16le -y {output_directory}/audio.wav"
    )
