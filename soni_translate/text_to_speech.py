from gtts import gTTS
import edge_tts, asyncio, json, glob  # noqa
from tqdm import tqdm
import librosa, os, re, torch, gc, subprocess  # noqa
from .language_configuration import (
    fix_code_language,
    BARK_VOICES_LIST,
    VITS_VOICES_LIST,
)
from .utils import (
    download_manager,
    create_directories,
    copy_files,
    rename_file,
    remove_directory_contents,
    remove_files,
    run_command,
)
import numpy as np
from typing import Any, Dict
from pathlib import Path
import soundfile as sf
import platform
import logging
import traceback
from .logging_setup import logger


class TTS_OperationError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        self.message = message
        super().__init__(self.message)


def verify_saved_file_and_size(filename):
    if not os.path.exists(filename):
        raise TTS_OperationError(f"File '{filename}' was not saved.")
    if os.path.getsize(filename) == 0:
        raise TTS_OperationError(
            f"File '{filename}' has a zero size. "
            "Related to incorrect TTS for the target language"
        )


def error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename):
    traceback.print_exc()
    logger.error(f"Error: {str(error)}")
    try:
        from tempfile import TemporaryFile

        tts = gTTS(segment["text"], lang=fix_code_language(TRANSLATE_AUDIO_TO))
        f = TemporaryFile()
        tts.write_to_fp(f)

        f.seek(0)

        audio_data, samplerate = sf.read(f)
        f.close()
        sf.write(
            filename, audio_data, samplerate, format="ogg", subtype="vorbis"
        )

        logger.warning(
            'TTS auxiliary will be utilized '
            f'rather than TTS: {segment["tts_name"]}'
        )
        verify_saved_file_and_size(filename)
    except Exception as error:
        logger.critical(f"Error: {str(error)}")
        sample_rate_aux = 22050
        duration = float(segment["end"]) - float(segment["start"])
        data = np.zeros(int(sample_rate_aux * duration)).astype(np.float32)
        sf.write(
            filename, data, sample_rate_aux, format="ogg", subtype="vorbis"
        )
        logger.error("Audio will be replaced -> [silent audio].")
        verify_saved_file_and_size(filename)


def pad_array(array, sr):
    if isinstance(array, list):
        array = np.array(array)

    if not array.shape[0]:
        raise ValueError("The generated audio does not contain any data")

    valid_indices = np.where(np.abs(array) > 0.001)[0]

    if len(valid_indices) == 0:
        logger.debug(f"No valid indices: {array}")
        return array

    try:
        pad_indice = int(0.1 * sr)
        start_pad = max(0, valid_indices[0] - pad_indice)
        end_pad = min(len(array), valid_indices[-1] + 1 + pad_indice)
        padded_array = array[start_pad:end_pad]
        return padded_array
    except Exception as error:
        logger.error(str(error))
        return array


def edge_tts_voices_list():
    try:
        completed_process = subprocess.run(
            ["edge-tts", "--list-voices"], capture_output=True, text=True
        )
        lines = completed_process.stdout.strip().split("\n")
    except Exception as error:
        logger.debug(str(error))
        lines = []

    voices = []
    for line in lines:
        if line.startswith("Name: "):
            voice_entry = {}
            voice_entry["Name"] = line.split(": ")[1]
        elif line.startswith("Gender: "):
            voice_entry["Gender"] = line.split(": ")[1]
            voices.append(voice_entry)

    formatted_voices = [
        f"{entry['Name']}-{entry['Gender']}" for entry in voices
    ]

    if not formatted_voices:
        logger.warning(
            "The list of Edge TTS voices could not be obtained, "
            "switching to an alternative method"
        )
        tts_voice_list = asyncio.new_event_loop().run_until_complete(
            edge_tts.list_voices()
        )
        formatted_voices = sorted(
            [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
        )

    if not formatted_voices:
        logger.error("Can't get EDGE TTS - list voices")

    return formatted_voices


def segments_edge_tts(filtered_edge_segments, TRANSLATE_AUDIO_TO, is_gui):
    for segment in tqdm(filtered_edge_segments["segments"]):
        speaker = segment["speaker"]  # noqa
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"]

        filename = f"audio/{start}.ogg"
        temp_file = filename[:-3] + "mp3"

        logger.info(f"{text} >> {filename}")
        try:
            if is_gui:
                asyncio.run(
                    edge_tts.Communicate(
                        text, "-".join(tts_name.split("-")[:-1])
                    ).save(temp_file)
                )
            else:
                command = f'edge-tts -t "{text}" -v "{tts_name.replace("-Male", "").replace("-Female", "")}" --write-media "{temp_file}"'
                run_command(command)
            verify_saved_file_and_size(temp_file)

            data, sample_rate = sf.read(temp_file)
            data = pad_array(data, sample_rate)

            sf.write(
                file=filename,
                samplerate=sample_rate,
                data=data,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)

        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)


def segments_bark_tts(
    filtered_bark_segments, TRANSLATE_AUDIO_TO, model_id_bark="suno/bark-small"
):
    from transformers import AutoProcessor, BarkModel
    from optimum.bettertransformer import BetterTransformer

    device = os.environ.get("SONITR_DEVICE")
    torch_dtype_env = torch.float16 if device == "cuda" else torch.float32

    model = BarkModel.from_pretrained(
        model_id_bark, torch_dtype=torch_dtype_env
    ).to(device)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(
        model_id_bark, return_tensors="pt"
    )

    if device == "cuda":
        model = BetterTransformer.transform(model, keep_original_model=False)

    sampling_rate = model.generation_config.sample_rate

    for segment in tqdm(filtered_bark_segments["segments"]):
        speaker = segment["speaker"]  # noqa
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"]

        inputs = processor(text, voice_preset=BARK_VOICES_LIST[tts_name]).to(
            device
        )

        filename = f"audio/{start}.ogg"
        logger.info(f"{text} >> {filename}")
        try:
            with torch.inference_mode():
                speech_output = model.generate(
                    **inputs,
                    do_sample=True,
                    fine_temperature=0.4,
                    coarse_temperature=0.8,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            data_tts = pad_array(
                speech_output.cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            sf.write(
                file=filename,
                samplerate=sampling_rate,
                data=data_tts,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        finally:
            del inputs, speech_output, data_tts
            gc.collect()
            torch.cuda.empty_cache()

    try:
        del processor
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()


def uromanize(input_string):
    if not os.path.exists("./uroman"):
        logger.info(
            "Clonning repository uroman https://github.com/isi-nlp/uroman.git"
            " for romanize the text"
        )
        process = subprocess.Popen(
            ["git", "clone", "https://github.com/isi-nlp/uroman.git"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
    script_path = os.path.join("./uroman", "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    uromanized_string = stdout.decode().strip()
    return uromanized_string
