import os
import sys
import time
import copy
import json
import torch
import logging
import hashlib
import argparse
from pydub import AudioSegment
import gradio as gr
from voice_main import ClassVoices
from soni_translate.logging_setup import (
    logger,
    set_logging_level,
    configure_logging_libs,
)
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import (
    audio_segmentation_to_voice,
    edge_tts_voices_list,
    coqui_xtts_voices_list,
    piper_tts_voices_list,
    create_wav_file_vc,
    accelerate_segments,
)
from soni_translate.translate_segments import (
    translate_text,
    TRANSLATION_PROCESS_OPTIONS,
    DOCS_TRANSLATION_PROCESS_OPTIONS
)
from soni_translate.preprocessor import (
    audio_video_preprocessor,
    audio_preprocessor,
)
from soni_translate.postprocessor import (
    OUTPUT_TYPE_OPTIONS,
    DOCS_OUTPUT_TYPE_OPTIONS,
    sound_separate,
    get_no_ext_filename,
    media_out,
    get_subtitle_speaker,
)
from soni_translate.language_configuration import (
    LANGUAGES,
    UNIDIRECTIONAL_L_LIST,
    LANGUAGES_LIST,
    BARK_VOICES_LIST,
    VITS_VOICES_LIST,
    OPENAI_TTS_MODELS,
)
from soni_translate.utils import (
    remove_files,
    download_list,
    upload_model_list,
    download_manager,
    run_command,
    is_audio_file,
    is_subtitle_file,
    copy_files,
    get_valid_files,
    get_link_list,
    remove_directory_contents,
)
from soni_translate.mdx_net import (
    UVR_MODELS,
    MDX_DOWNLOAD_LINK,
    mdxnet_models_dir,
)
from soni_translate.speech_segmentation import (
    ASR_MODEL_OPTIONS,
    COMPUTE_TYPE_GPU,
    COMPUTE_TYPE_CPU,
    find_whisper_models,
    transcribe_speech,
    align_speech,
    diarize_speech,
    diarization_models,
)
from soni_translate.text_multiformat_processor import (
    BORDER_COLORS,
    srt_file_to_segments,
    document_preprocessor,
    determine_chunk_size,
    plain_text_to_segments,
    segments_to_plain_text,
    process_subtitles,
    linguistic_level_segments,
    break_aling_segments,
    doc_to_txtximg_pages,
    page_data_to_segments,
    update_page_data,
    fix_timestamps_docs,
    create_video_from_images,
    merge_video_and_audio,
)
from soni_translate.languages_gui import language_data, news

configure_logging_libs()  # noqa

directories = [
    "downloads",
    "logs",
    "weights",
    "clean_song_output",
    "_XTTS_",
    f"audio2{os.sep}audio",
    "audio",
    "outputs",
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(BARK_VOICES_LIST.keys())
        self.list_vits = list(VITS_VOICES_LIST.keys())
        self.list_openai_tts = OPENAI_TTS_MODELS
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = piper_tts_voices_list() if self.piper_enabled else []
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = coqui_xtts_voices_list() if self.xtts_enabled else []
        list_tts = self.list_coqui_xtts + sorted(
            self.list_edge
            + self.list_bark
            + self.list_vits
            + self.list_openai_tts
            + self.list_vits_onnx
        )
        return list_tts

def prog_disp(msg, percent, is_gui, progress=None):
    logger.info(msg)
    if is_gui:
        progress(percent, desc=msg)

def warn_disp(wrn_lang, is_gui):
    logger.warning(wrn_lang)
    if is_gui:
        gr.Warning(wrn_lang)

class SoniTrCache:
    def __init__(self):
        self.cache = {
            'media': [[]],
            'refine_vocals': [],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_data = {
            'media': [],
            'refine_vocals': [],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_keys = list(self.cache.keys())
        self.first_task = self.cache_keys[0]
        self.last_task = self.cache_keys[-1]

        self.pre_step = None
        self.pre_params = []

    def set_variable(self, variable_name, value):
        setattr(self, variable_name, value)

    def task_in_cache(self, step: str, params: list, previous_step_data: dict):
        self.pre_step_cache = None

        if step == self.first_task:
            self.pre_step = None

        if self.pre_step:
            self.cache[self.pre_step] = self.pre_params

            # Fill data in cache
            self.cache_data[self.pre_step] = copy.deepcopy(previous_step_data)

        self.pre_params = params

        if params == self.cache[step]:
            logger.debug(f"In cache: {str(step)}")

            # Set the var needed for next step
            # Recovery from cache_data the current step
            for key, value in self.cache_data[step].items():
                self.set_variable(key, copy.deepcopy(value))
                logger.debug(
                    f"Chache load: {str(key)}"
                )

            self.pre_step = step
            return True

        else:
            logger.debug(f"Flush next and caching {str(step)}")
            selected_index = self.cache_keys.index(step)

            for idx, key in enumerate(self.cache.keys()):
                if idx >= selected_index:
                    self.cache[key] = []
                    self.cache_data[key] = {}

            # The last is now previous
            self.pre_step = step
            return False

    def clear_cache(self, media, force=False):
        self.cache["media"] = self.cache["media"] if len(self.cache["media"]) else [[]]

        if media != self.cache["media"][0] or force:
            self.cache = {key: [] for key in self.cache}
            self.cache["media"] = [[]]
            logger.info("Cache flushed")

def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:18]

def check_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "To use GPT for translation, please set up your OpenAI API key "
            "as an environment variable in Linux as follows: "
            "export OPENAI_API_KEY='your-api-key-here'. Or change the "
            "translation process in Advanced settings."
        )

class SoniTranslate(SoniTrCache):
    def __init__(self, cpu_mode=False):
        super().__init__()
        os.environ["SONITR_DEVICE"] = "cpu" if cpu_mode else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = os.environ.get("SONITR_DEVICE")
        self.result_diarize = None
        self.align_language = None
        self.result_source_lang = None
        self.edit_subs_complete = False
        self.voiceless_id = None
        self.burn_subs_id = None

        self.vci = ClassVoices(only_cpu=cpu_mode)

        self.tts_voices = self.get_tts_voice_list()

        logger.info(f"Working in: {self.device}")

    def get_tts_voice_list(self):
        try:
            from piper import PiperVoice  # Lazy import
            piper_enabled = bool(PiperVoice)
        except ImportError:
            piper_enabled = False

        try:
            import xtts  # Lazy import
            xtts_enabled = bool(xtts)
        except ImportError:
            xtts_enabled = False

        return TTS_Info(piper_enabled, xtts_enabled).tts_list()

    def save_json_result(self, filename, data, create=False):
        filename = filename if filename.lower().endswith('.json') else f"{filename}.json"

        file_exists = os.path.isfile(filename)
        if create or file_exists:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                logger.info(f"Saved json file: {filename}")

        if not file_exists:
            logger.info(f"{filename} not found, skipping...")

    def separate_audio(self, files, progress=gr.Progress()):
        progress(0, desc="Processing with demucs or mdx model...")

        valid_files = get_valid_files(files, [".mp3", ".wav", ".flac", ".mp4", ".m4a"])
        links = get_link_list(valid_files, ("demucs_model", "mdx_model"))

        separate_audio_list = []
        for file_path, link in zip(valid_files, links):
            valid_hash = get_hash(file_path)
            destination = f"{mdxnet_models_dir}/{valid_hash}.wav"

            if os.path.isfile(destination):
                logger.info(f"Skipping {file_path} as {destination} already exists")
                continue

            command = f'python3 -m demucs.separate -n {link} "{file_path}" -o {mdxnet_models_dir}'
            logger.info(f"Running command: {command}")
            run_command(command)
            separate_audio_list.append(destination)

            progress((len(separate_audio_list) / len(valid_files)) * 100, desc="Separating audio...")

        progress(100, desc="Audio separation complete")
        return separate_audio_list
