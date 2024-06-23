# Imports organized logically
import os
import gc
import soundfile as sf
from IPython.utils import capture # noqa
from .language_configuration import EXTRA_ALIGN, INVERTED_LANGUAGES
from .logging_setup import logger
from .postprocessor import sanitize_file_name
from .utils import remove_directory_contents, run_command
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH as DAMT, DEFAULT_ALIGN_MODELS_HF as DAMHF
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch


# Configuration moved to a separate file if applicable

ASR_MODEL_OPTIONS = [
    "tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "Systran/faster-distil-whisper-large-v3", "tiny.en", "base.en",
    "small.en", "medium.en", "distil-small.en", "distil-medium.en", "OpenAI_API_Whisper"
]

COMPUTE_TYPE_GPU = [
    "default", "auto", "int8", "int8_float32", "int8_float16", "int8_bfloat16", "float16", "bfloat16", "float32"
]

COMPUTE_TYPE_CPU = [
    "default", "auto", "int8", "int8_float32", "int16", "float32"
]

WHISPER_MODELS_PATH = './WHISPER_MODELS'


def openai_api_whisper(input_audio_file, source_lang=None, chunk_duration=1800):
    info = sf.info(input_audio_file)
    duration = info.duration

    output_directory = "./whisper_api_audio_parts"
    os.makedirs(output_directory, exist_ok=True)
    remove_directory_contents(output_directory)

    if duration > chunk_duration:
        # Split the audio file into smaller chunks with chunk_duration
        cm = f'ffmpeg -i "{input_audio_file}" -f segment -segment_time {chunk_duration} -c:a libvorbis "{output_directory}/output%03d.ogg"'
        run_command(cm)
        # Get list of generated chunk files
        chunk_files = sorted([f"{output_directory}/{f}" for f in os.listdir(output_directory) if f.endswith('.ogg')])
    else:
        one_file = f"{output_directory}/output000.ogg"
        cm = f'ffmpeg -i "{input_audio_file}" -c:a libvorbis {one_file}'
        run_command(cm)
        chunk_files = [one_file]

    # Transcript
    segments = []
    language = source_lang if source_lang else None

    for i, chunk in enumerate(chunk_files):
        # Handling client creation here to ensure resource management
        from openai import OpenAI
        client = OpenAI()

        try:
            with open(chunk, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

                try:
                    transcript_dict = transcription.model_dump()
                except Exception as e:
                    transcript_dict = transcription.to_dict()

                if language is None:
                    logger.info(f'Language detected: {transcript_dict["language"]}')
                    language = TO_LANGUAGE_CODE[transcript_dict["language"]]

                chunk_time = chunk_duration * i

                for seg in transcript_dict["segments"]:
                    if "start" in seg.keys():
                        segments.append({
                            "text": seg["text"],
                            "start": seg["start"] + chunk_time,
                            "end": seg["end"] + chunk_time,
                        })

        except Exception as e:
            logger.error(f"Error processing chunk {chunk}: {str(e)}")

    audio = whisperx.load_audio(input_audio_file)
    result = {"segments": segments, "language": language}

    return audio, result


def find_whisper_models():
    path = WHISPER_MODELS_PATH
    folders = [folder for folder in os.listdir(path)
               if os.path.isdir(os.path.join(path, folder)) and 'model.bin' in os.listdir(os.path.join(path, folder))]

    return folders


def transcribe_speech(audio_wav, asr_model, compute_type, batch_size, SOURCE_LANGUAGE, literalize_numbers=True, segment_duration_limit=15):
    """
    Transcribe speech using a whisper model.
    """
    if asr_model == "OpenAI_API_Whisper":
        if literalize_numbers:
            logger.info("OpenAI's API Whisper does not support the literalization of numbers.")
        return openai_api_whisper(audio_wav, SOURCE_LANGUAGE)

    # Handle model download and loading
    if asr_model not in ASR_MODEL_OPTIONS:
        base_dir = WHISPER_MODELS_PATH
        os.makedirs(base_dir, exist_ok=True)
        model_dir = os.path.join(base_dir, sanitize_file_name(asr_model))

        if not os.path.exists(model_dir):
            try:
                from ctranslate2.converters import TransformersConverter

                quantization = "float32"
                converter = TransformersConverter(asr_model, low_cpu_mem_usage=True,
                                                  copy_files=["tokenizer_config.json", "preprocessor_config.json"])
                converter.convert(model_dir, quantization=quantization, force=False)
            except Exception as error:
                if "File tokenizer_config.json does not exist" in str(error):
                    converter._copy_files = ["tokenizer.json", "preprocessor_config.json"]
                    converter.convert(model_dir, quantization=quantization, force=True)
                else:
                    raise error

        asr_model = model_dir
        logger.info(f"ASR Model: {str(model_dir)}")

    # Load and transcribe audio
    model = whisperx.load_model(asr_model, os.environ.get("SONITR_DEVICE"),
                                compute_type=compute_type, language=SOURCE_LANGUAGE,
                                asr_options={"initial_prompt": None, "suppress_numerals": literalize_numbers})

    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(audio, batch_size=batch_size, chunk_size=segment_duration_limit, print_progress=True)

    if result["language"] == "zh" and not prompt:
        result["language"] = "zh-TW"
        logger.info("Chinese - Traditional (zh-TW)")

    del model
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return audio, result


def align_speech(audio, result):
    """
    Aligns speech segments based on the provided audio and result metadata.
    """
    DAMHF.update(DAMT)  # lang align
    if result["language"] not in DAMHF.keys() and result["language"] not in EXTRA_ALIGN.keys():
        logger.warning("Automatic detection: Source language not compatible with align")
        raise ValueError(f"Detected language {result['language']} incompatible, you can select the source language to avoid this error.")

    if result["language"] in EXTRA_ALIGN.keys() and EXTRA_ALIGN[result["language"]] == "":
        lang_name = INVERTED_LANGUAGES.get(result["language"], result["language"])
        logger.warning(f"No compatible wav2vec2 model found for the language '{lang_name}', skipping alignment.")
        return result

    model_a, metadata = whisperx.load_align_model(language_code=result["language"],
                                                 device=os.environ.get("SONITR_DEVICE"),
                                                 model_name=None if result["language"] in DAMHF.keys() else EXTRA_ALIGN[result["language"]])

    result = whisperx.align(result["segments"], model_a, metadata, audio,
                            os.environ.get("SONITR_DEVICE"), return_char_alignments=True, print_progress=False)

    del model_a
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return result


def reencode_speakers(result):
    if result["segments"][0]["speaker"] == "SPEAKER_00":
        return result

    speaker_mapping = {}
    counter = 0

    logger.debug("Reencode speakers")

    for segment in result["segments"]:
        old_speaker = segment["speaker"]
        if old_speaker not in speaker_mapping:
            speaker_mapping[old_speaker] = f"SPEAKER_{counter:02d}"
            counter += 1
        segment["speaker"] = speaker_mapping[old_speaker]

    return result


def diarize_speech(audio_wav, result, min_speakers, max_speakers, YOUR_HF_TOKEN,
                   model_name="pyannote/speaker-diarization@2.1"):
    """
    Performs speaker diarization on speech segments.
    """
    if max(min_speakers, max_speakers) > 1 and model_name:
        try:
            diarize_model = whisperx.DiarizationPipeline(model_name=model_name, use_auth_token=YOUR_HF_TOKEN,
                                                         device=os.environ.get("SONITR_DEVICE"))
            diarize_segments = diarize_model(audio_wav, min_speakers=min_speakers, max_speakers=max_speakers)
            result_diarize = whisperx.assign_word_speakers(diarize_segments, result)

            for segment in result_diarize["segments"]:
                if "speaker" not in segment:
                    segment["speaker"] = "SPEAKER_00"
                    logger.warning(f"No speaker detected in {segment['start']}. First TTS will be used for the segment text: {segment['text']}")

            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()  # noqa

        except Exception as error:
            error_str = str(error)
            gc.collect()
            torch.cuda.empty_cache()  # noqa

            if "'NoneType' object has no attribute 'to'" in error_str:
                if model_name == diarization_models["pyannote_2.1"]:
                    raise ValueError("Accept the license agreement for using Pyannote 2.1. You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation. Get your KEY TOKEN here: https://hf.co/settings/tokens.")
                elif model_name == diarization_models["pyannote_3.1"]:
                    raise ValueError("New Licence Pyannote 3.1: You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization-3.1 and https://huggingface.co/pyannote/segmentation-3.0.")

            else:
                raise error

    else:
        result_diarize = result
        result_diarize["segments"] = [{**item, "speaker": "SPEAKER_00"} for item in result_diarize["segments"]]

    return reencode_speakers(result_diarize)
