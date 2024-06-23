import gc
import hashlib
import os
import queue
import threading
import json
import shlex
import sys
import subprocess
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

try:
    from .utils import (
        remove_directory_contents,
        create_directories,
    )
except ImportError:  # noqa
    from utils import (
        remove_directory_contents,
        create_directories,
    )
from .logging_setup import logger

try:
    import onnxruntime as ort
except ImportError as error:
    logger.error(str(error))

stem_naming = {
    "Vocals": "Instrumental",
    "Other": "Instruments",
    "Instrumental": "Vocals",
    "Drums": "Drumless",
    "Bass": "Bassless",
}

class MDXModel:
    def __init__(
        self,
        device,
        dim_f,
        dim_t,
        n_fft,
        hop=1024,
        stem_name=None,
        compensation=1.000,
    ):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(
            window_length=self.n_fft, periodic=True
        ).to(device)

        self.freq_pad = torch.zeros(
            [1, self.dim_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 4, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
        )
        return x.reshape([-1, 2, self.chunk_size])

class MDX:
    DEFAULT_SR = 44100
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(self, model_path: str, params: MDXModel, processor=0):
        self.device = torch.device(f"cuda:{processor}") if processor >= 0 else torch.device("cpu")
        self.provider = ["CUDAExecutionProvider"] if processor >= 0 else ["CPUExecutionProvider"]

        self.model = params
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        self.ort.run(None, {"input": torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        self.process = lambda spec: self.ort.run(None, {"input": spec.cpu().numpy()})[0]

        self.prog = None

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:  # noqa
            model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()

        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        if combine:
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:
                    processed_wave = segment[:, start:end]
                else:
                    processed_wave = np.concatenate(
                        (processed_wave, segment[:, start:end]), axis=-1
                    )
        else:
            processed_wave = []
            sample_count = wave.shape[-1]
            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count
            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        wave_p = np.concatenate(
            (
                np.zeros((2, trim)),
                wave,
                np.zeros((2, pad)),
                np.zeros((2, trim)),
            ),
            1,
        )

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = (
                    processed_wav[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()
                )
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)

        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(
                target=self._process_wave, args=(mix_waves, trim, pad, q, c)
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.prog.close()

        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [
            list(wave.values())[0]
            for wave in sorted(processed_batches, key=lambda d: list(d.keys())[0])
        ]
        assert len(processed_batches) == len(waves), "Incomplete processed batches, please reduce batch size!"
        return self.segment(processed_batches, True, chunk)

def run_mdx(
    model_params,
    output_dir,
    model_path,
    filename,
    exclude_main=False,
    exclude_inversion=False,
    suffix=None,
    invert_suffix=None,
    denoise=False,
    keep_orig=True,
    m_threads=2,
    device_base="cuda",
):
    if device_base == "cuda":
        device = torch.device("cuda:0")
        processor_num = 0
        device_properties = torch.cuda.get_device_properties(0)
        logger.info(
            f"Using {device_properties.name} with Total Memory of {round(device_properties.total_memory / (1024 ** 3))} GB"
        )
    elif device_base == "cpu":
        device = torch.device("cpu")
        processor_num = -1
    else:
        logger.error("Invalid Device Argument")
        raise ValueError("Invalid Device Argument")

    sample_rate = MDX.DEFAULT_SR
    mix, sample_rate = librosa.load(filename, sr=sample_rate, mono=False)
    logger.info(f"Loading Wave File with Sample Rate of {sample_rate}")

    model = MDX(model_path=model_path, params=model_params, processor=processor_num)

    logger.info(f"Starting Wave File Segmentation with {m_threads} Threads")

    final_wave = model.process_wave(mix, m_threads)
    main_stem = model_params.stem_name
    alt_stem = stem_naming[main_stem]
    main_stem_final = f"{main_stem}" if not suffix else f"{main_stem}_{suffix}"

    main_stem = f"{main_stem_final}.wav"
    sf.write(os.path.join(output_dir, main_stem), final_wave.T, sample_rate)

    if exclude_main:
        logger.info(f"Skipping Instrumental Stem Generation, Exclude Main is Enabled")
    else:
        logger.info(f"Starting Instrumental Stem Generation")
        model_params.compensation = 1 / model_params.compensation
        alt_wave = model.process_wave(mix - final_wave, m_threads)

        invert_stem = f"{alt_stem}" if not invert_suffix else f"{alt_stem}_{invert_suffix}"
        alt_stem = f"{invert_stem}.wav"

        sf.write(os.path.join(output_dir, alt_stem), alt_wave.T, sample_rate)

    if exclude_inversion:
        logger.info(f"Skipping Additional Inversion Phase, Exclude Inversion is Enabled")
    else:
        logger.info(f"Starting Additional Inversion Phase")
        alt_wave = model.process_wave(final_wave, m_threads)
        invert_stem = f"{main_stem_final}" if not invert_suffix else f"{main_stem_final}_{invert_suffix}"
        main_stem_final = f"{invert_stem}_inverted.wav"
        sf.write(os.path.join(output_dir, main_stem_final), alt_wave.T, sample_rate)

    logger.info(f"Process Completed, Saved to {output_dir}")

    if not keep_orig:
        logger.info(f"Deleting Original Files from {output_dir}")
        remove_directory_contents(output_dir)

    logger.info("Done!")
