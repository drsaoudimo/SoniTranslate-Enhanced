from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
import re
import json
import time
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import tiktoken
from openai import OpenAI

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]


def translate_iterative(segments, target, source=None):
    segments_ = copy.deepcopy(segments)
    source = source or "auto"
    logger.debug("Source language: %s", source)
    translator = GoogleTranslator(source=source, target=target)

    for line in tqdm(segments_):
        text = line["text"]
        translated_line = translator.translate(text.strip())
        line["text"] = translated_line

    return segments_


def verify_translate(segments, segments_copy, translated_lines, target, source):
    if len(segments) == len(translated_lines):
        for line, translated_line in zip(segments_copy, translated_lines):
            logger.debug("%s >> %s", line["text"], translated_line.strip())
            line["text"] = translated_line.replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        logger.error("Translation failed, switching to google_translate iterative. %s", len(segments), len(translated_lines))
        return translate_iterative(segments, target, source)


def translate_batch(segments, target, chunk_size=2000, source=None):
    segments_copy = copy.deepcopy(segments)
    source = source or "auto"
    logger.debug("Source language: %s", source)

    text_lines = [line["text"].strip() for line in segments_copy]

    text_merge, global_text_list = [], []
    actual_chunk, actual_text_list = "", []
    for one_line in text_lines:
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk:
                actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            global_text_list.append(actual_text_list)
            actual_chunk = one_line
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    progress_bar = tqdm(total=len(segments), desc="Translating")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []

    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translator.translate(text.strip())
            split_text = translated_line.split("|||||")
            if len(split_text) != len(text_iterable):
                logger.debug("Chunk fixing iteratively. Len chunk: %d, expected: %d", len(split_text), len(text_iterable))
                split_text = [translator.translate(txt.strip()) for txt in text_iterable]
                progress_bar.update(len(split_text))
            split_list.append(split_text)
            progress_bar.update(len(split_text))
        progress_bar.close()
    except Exception as error:
        progress_bar.close()
        logger.error(str(error))
        logger.warning("Translation in chunks failed, switching to iterative.")
        return translate_iterative(segments, target, source)

    translated_lines = list(chain.from_iterable(split_list))
    return verify_translate(segments, segments_copy, translated_lines, target, source)


def call_gpt_translate(client, model, system_prompt, user_prompt, original_text=None, batch_lines=None):
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    logger.debug("Result: %s", result)

    try:
        translation = json.loads(result)
    except Exception as error:
        match_result = re.search(r'\{.*?\}', result)
        if match_result:
            logger.error(str(error))
            json_str = match_result.group(0)
            translation = json.loads(json_str)
        else:
            raise error

    if batch_lines:
        for conversation in translation.values():
            if isinstance(conversation, dict):
                conversation = list(conversation.values())[0]
            if len(conversation) == batch_lines:
                break

        fix_conversation_length = [{k: v} for line in conversation for k, v in line.items()]

        logger.debug("Data batch: %s", fix_conversation_length)
        logger.debug("Lines Received: %d, expected: %d", len(fix_conversation_length), batch_lines)

        return fix_conversation_length

    else:
        if isinstance(translation, dict):
            translation = list(translation.values())[0]
        if isinstance(translation, list):
            translation = translation[0]
        if isinstance(translation, set):
            translation = list(translation)[0]
        if not isinstance(translation, str):
            raise ValueError(f"No valid response received: {str(translation)}")

        return translation


def gpt_sequential(segments, model, target, source=None):
    translated_segments = copy.deepcopy(segments)
    client = OpenAI()
    progress_bar = tqdm(total=len(segments), desc="Translating")

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip() if source else ""

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    system_prompt = "Machine translation designed to output the translated_text JSON."

    for line in translated_segments:
        text = line["text"].strip()
        user_prompt = f"Translate the following {lang_sc} text into {lang_tg}, write the fully translated text and nothing more:\n{text}"

        time.sleep(0.5)

        try:
            translated_text = call_gpt_translate(client, model, system_prompt, user_prompt)
        except Exception as error:
            logger.error("%s >> The text of segment %s is being corrected with Google Translate", str(error), line["start"])
            translator = GoogleTranslator(source=fixed_source, target=fixed_target)
            translated_text = translator.translate(text.strip())

        line["text"] = translated_text.strip()
        progress_bar.update(1)

    progress_bar.close()
    return translated_segments


def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    token_batch_limit = max(100, (token_batch_limit - 40) // 2)
    progress_bar = tqdm(total=len(segments), desc="Translating")
    segments_copy = copy.deepcopy(segments)
    encoding = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip() if source else ""

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    name_speaker = "ABCDEFGHIJKL"
    translated_lines, text_data_dict, count_sk = [], [], {char: 0 for char in name_speaker}
    num_tokens = 0

    for i, line in enumerate(segments_copy):
        text = line["text"]
        speaker = line["speaker"]
        index_sk = int(speaker[-2:])
        character_sk = name_speaker[index_sk]
        count_sk[character_sk] += 1
        code_sk = character_sk + str(count_sk[character_sk])
        text_data_dict.append({code_sk: text})
        num_tokens += len(encoding.encode(text)) + 7

        if num_tokens >= token_batch_limit or i == len(segments_copy) - 1:
            try:
                batch_lines = len(text_data_dict)
                batch_conversation = {"conversation": copy.deepcopy(text_data_dict)}
                num_tokens = 0
                text_data_dict, count_sk = [], {char: 0 for char in name_speaker}

                system_prompt = f"Machine translation designed to output the translated_conversation key JSON containing a list of {batch_lines} items."
                user_prompt = f"Translate each of the following text values in conversation{' from' if lang_sc else ''} {lang_sc} to {lang_tg}:\n{batch_conversation}"
                logger.debug("Prompt: %s", user_prompt)

                conversation = call_gpt_translate(client, model, system_prompt, user_prompt, original_text=batch_conversation, batch_lines=batch_lines)

                if len(conversation) < batch_lines:
                    raise ValueError("Incomplete result received. Batch lines: %d, expected: %d", len(conversation), batch_lines)

                translated_lines.extend([list(translated_text.values())[0] for translated_text in conversation])
                progress_bar.update(batch_lines)

            except Exception as error:
                logger.error(str(error))
                logger.warning("Batch process failed. Switching to iterative mode for batch lines.")
                progress_bar.update(batch_lines)
                for item in text_data_dict:
                    line_text = list(item.values())[0]
                    translated_lines.append(call_gpt_translate(client, model, system_prompt, user_prompt, original_text=line_text))

    progress_bar.close()
    return verify_translate(segments, segments_copy, translated_lines, target, source)


def translate(segments, target, process="google_translator_batch", source=None):
    process_options = {
        "google_translator_batch": translate_batch,
        "google_translator": translate_iterative,
        "gpt-3.5-turbo-0125_batch": gpt_batch,
        "gpt-3.5-turbo-0125": gpt_sequential,
        "gpt-4-turbo-preview_batch": gpt_batch,
        "gpt-4-turbo-preview": gpt_sequential,
        "disable_translation": lambda x, y, z, w=None: x,
    }
    return process_options.get(process, "disable_translation")(segments, target, source)
