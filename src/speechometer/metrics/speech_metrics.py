"""Reusable speech metric wrappers.

Some metrics were adopted from the DASB benchmark
https://github.com/speechbrain/benchmarks

DNSMOS, STOI, PESQ, Mel distance, STFT distance, and ASR perplexity are
adapted from the Apache-2.0-licensed ``lucadellalib/audiocodecs`` project:
https://github.com/lucadellalib/audiocodecs

Authors
 * Artem Ploujnikov 2026
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
import csv
import json
import os
from os import PathLike
import re
import string
from pathlib import Path
from typing import TextIO, Any

import numpy as np
import torch
import torchaudio

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.integrations.huggingface import Whisper
from speechbrain.utils.fetching import fetch
from speechbrain.utils.importutils import LazyModule
from speechbrain.utils.logger import get_logger
from speechbrain.utils.metric_stats import ErrorRateStats, MetricStats
from speechometer.models.utmos import UTMOSModel
from transformers import (
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from speechometer.stats import descriptive_statistics
from speechometer.utils import undo_padding

logger = get_logger(__name__)

nisqa = LazyModule("nisqa", "torchmetrics.functional.audio.nisqa", None)
bleu = LazyModule("bleu", "speechbrain.integrations.nlp.bleu", None)
librosa = LazyModule("librosa", "librosa", None)
onnxruntime = LazyModule("onnxruntime", "onnxruntime", None)
pesq = LazyModule("pesq", "torchmetrics.functional.audio.pesq", None)
stoi = LazyModule("stoi", "torchmetrics.functional.audio.stoi", None)

RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)
SEPARATOR_WIDTH = 80

ASR_METRICS = ["wer", "cer", "dwer", "dcer"]
ASR_METRIC_KIND = {
    "wer": "word",
    "cer": "character",
    "dwer": "word",
    "dcer": "character",
}

ASR_METRIC_TARGETS = {
    "wer": "text",
    "cer": "text",
    "dwer": "ground_pred",
    "dcer": "ground_pred",
}

ASR_WHISPER_DEFAULT_SOURCE = "openai/whisper-small"

SPKSIM_WAVLM_DEFAULT_MODEL_HUB = "microsoft/wavlm-base-sv"

UTMOS_SAMPLE_RATE = 16000
UTMOS_DEFAULT_SOURCE = "chaanks/UTMOS"
UTMOS_DEFAULT_SOURCE_BASE = "chaanks/wav2vec2-small"
UTMOS_DEFAULT_MODEL_NAME = "utmos.ckpt"
UTMOS_DEFAULT_SAVE_DIR = "./pretrained_models"
UTMOS_DEFAULT_JUDGE_ID = 288
UTMOS_DEFAULT_DOMAIN_ID = 0

AUDIOCODECS_METRIC_SAMPLE_RATE = 16000
PESQ_MINIMUM_SCORE = -0.5
DNSMOS_INPUT_LENGTH = 9.01
DNSMOS_DEFAULT_MODEL_PATH = Path(__file__).with_name("model_v8.onnx")
ASR_PERPLEXITY_DEFAULT_MODEL_HUB = "meta-llama/Llama-3.2-1B"


def _pesq_has_no_utterances(error: Exception) -> bool:
    """Whether a PESQ backend error means the waveform contains no speech.

    ``cypesq`` normally exposes this as ``NoUtterancesError``. Some versions
    instead leak a ``ValueError`` while converting the resulting NaN.
    """
    return error.__class__.__name__ == "NoUtterancesError" or (
        isinstance(error, ValueError)
        and str(error) == "cannot convert float NaN to integer"
    )


class SpeechMetricStats(MetricStats, ABC):
    @abstractmethod
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Evaluates a batch of samples

        Arguments
        ---------
        ids : list
            The utterance IDs
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list | None
            Text labels corresponding to the waveforms
        wavs_ref : torch.Tensor | None
            A batch of waveforms (ground truth)
        length_ref : torch.Tensor | None
            Relative lengths (ground truth)
        sample_rate : int | None
            The sample rate of the waveforms
        sample_rate_ref : int | None
            The sample rate of the reference waveforms
        language : str | None
            The language identifier, if applicable
        """
        pass

    def write_report(self, filestream: TextIO, **kwargs: dict) -> None:
        """Outputs a detailed CSV report for the metric

        Arguments
        ---------
        filestream : TextIO
            The target stream
        **kwargs: dict
            Metric-specific arguments
        """
        pass

    def write_reports(self, path: str | PathLike) -> None:
        """Outputs all relevant reports for the metric to the specified path

        Arguments
        ---------
        path : str | PathLike
            The path where reports will be output
        """
        pass

    def summarize(self, field: str | None = None) -> Any:
        """Computes the summary if not already computed and returns
        either the entire summary or one of the available fields

        Arguments
        ---------
        field : str | None
            The name of the field to be retreived

        Returns
        -------
        result : Any
            a dictionary with computed statistics if no field is provided
            or a specific value from that dictionary if one is provided
        """
        summary = self._summarize()
        return summary.get(field) if field is not None else summary

    def _summarize(self) -> dict:
        """Computes the summary

        Returns
        -------
        summary : dict
            A dictionary of computed statistics
        """
        if not self.summary:
            self.summary = descriptive_statistics(
                self.scores
            )
        return self.summary

    def write_stats(self, filestream: TextIO):
        """Outputs high-level summary statistics

        Arguments
        ---------
        filestream : TextIO
            The target stream
        """
        summary = self._summarize()
        json.dumps(summary, filestream, indent=4)


class SingleMetricStats(SpeechMetricStats):
    def __init__(self):
        self.clear()
        self.report_key = type(self).__name__.replace(
            "Stats", ""
        ).lower()

    def append_scores(
        self,
        ids: list,
        scores: torch.Tensor | list,
        key: str = "score"
    ) -> None:
        """Adds scores from a metrc

        Argument
        --------
        ids : list
            A list of data identifiers
        scores : torch.Tensor | list
            A single tensor of scores or a list
            of dicts
        key : str
            If a tensor is provided, this will be
            used as a dictionary key
        """
        if torch.is_tensor(scores):
            scores = [
                {key: score}
                for score in scores.cpu().tolist()
            ]
        self.scores.extend(scores)
        self.ids.extend(ids)

    def write_report(self, filestream: TextIO, **kwargs: dict) -> None:
        """Outputs a detailed CSV report for the metric

        Arguments
        ---------
        filestream : TextIO
            The target stream
        **kwargs: dict
            Metric-specific arguments
        """
        if not self.scores:
            return
        columns = ["id"] + list(self.scores[0].keys())
        writer = csv.DictWriter(filestream, fieldnames=columns)
        writer.writeheader()
        for uttid, scores in zip(self.ids, self.scores):
            row = {"id": uttid, **scores}
            writer.writerow(row)

    def write_reports(self, path: str | PathLike) -> None:
        """Outputs all relevant reports for the metric to the specified path

        Arguments
        ---------
        path : str | PathLike
            The path where reports will be output
        """
        path = Path(path)
        file_name = path / f"{self.report_key}.csv"
        with open(file_name, "w") as report_file:
            self.write_report(report_file)


class Transcriber(ABC):
    """An ASR transcriber wrapper"""
    @abstractmethod
    def transcribe(
        self,
        wavs: torch.Tensor,
        length: torch.Tensor,
        sample_rate: int,
        language: str = None
    ) -> list:
        """Makes an ASR prediction

        Arguments
        ---------
        wavs : torch.Tensor
            Raw waveforms
        length : torch.Tensor
            Relative lengths
        sample_rate : int
            The sample rate of the waveform
        language : str
            The language identifier

        Returns
        -------
        predictions : list
            The text predictions
        """
        pass


class WhisperTranscriber(Transcriber):
    """A Transcriber implementation for Whisper
    
    Attributes
    ----------
    source : str | None
        The source (path or HuggingFace hub) of the
        Whisper model to use
    model : torch.nn.Module | Non
        A pre-loaded Whisper model. Useful for situations
        where a single training recipe uses Whisper for
        more than only the metric or for multiple metrics
    sample_rate : int
        The sample rate of the Whisper model
    min_decode_ratio : float
        The minimum decode ratio
    max_decode_ratio : float
        The maximum decode ratio
    unbatch : bool
        Whether to "undo" batches and process
        one example at a time. This has been known to
        improve performance
    run_opts : dict | None
        Runtime options

        "device": the device identifier
    """
    def __init__(
        self,
        source: str | None = None,
        model: torch.nn.Module | None = None,
        save_path: str | PathLike | None = None,
        sample_rate: int = 22050,
        min_decode_ratio: float = 0.0,
        max_decode_ratio: float = 1.0,
        unbatch: bool = True,
        run_opts: dict | None = None,
    ):
        if source is None:
            source = ASR_WHISPER_DEFAULT_SOURCE
        if run_opts is None:
            run_opts = {}
        if save_path is None:
            save_path = "."
        if model is not None:
            self.model = model
        else:
            self.model = Whisper(
                source,
                save_path,
                sample_rate,
                freeze=True,
                freeze_encoder=True,
            )
        self.sample_rate = sample_rate
        self.model.tokenizer.set_prefix_tokens("english", "transcribe", False)
        self.searcher = S2SWhisperGreedySearcher(
            self.model,
            min_decode_ratio=min_decode_ratio,
            max_decode_ratio=max_decode_ratio,
        )
        device = run_opts.get("device", next(self.model.parameters()).device)
        self.unbatch = unbatch
        self.to(device)

    def transcribe(
        self,
        wavs: torch.Tensor,
        length: torch.Tensor,
        sample_rate: int,
        language: str = None
    ) -> list:
        """Makes an ASR prediction

        Arguments
        ---------
        wavs : torch.Tensor
            Raw waveforms
        length : torch.Tensor
            Relative lengths
        sample_rate : int
            The sample rate of the waveform
        language : str
            The language identifier

        Returns
        -------
        predictions : list
            The text predictions
        """
        if self.unbatch:
            full_length = torch.ones(1, device=wavs.device)
            wavs = undo_padding(
                wavs, length
            )
            result = [
                self._transcribe(
                    wavs=wavs_item.unsqueeze(0),
                    length=full_length,
                    sample_rate=sample_rate,
                    language=language
                )[0]
                for wavs_item in wavs
            ]
        else:
            result = self._transcribe(
                wavs=wavs,
                length=length,
                sample_rate=sample_rate,
                language=language
            )
        return result

    def _transcribe(
        self,
        wavs: torch.Tensor,
        length: torch.Tensor,
        sample_rate: int,
        language: str = None
    ) -> list:
        """Makes an ASR prediction

        Arguments
        ---------
        wavs : torch.Tensor
            Raw waveforms
        length : torch.Tensor
            Relative lengths
        sample_rate : int
            The sample rate of the waveform
        language : str
            The language identifier

        Returns
        -------
        predictions : list
            The text predictions
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        if language is not None:
            self.model.tokenizer.set_prefix_tokens(
                language=language, task="transcribe", predict_timestamps=False
            )
            self.searcher.set_task("transcribe")
        wavs = torchaudio.functional.resample(
            wavs, sample_rate, self.sample_rate
        )
        wavs = self.model.pad_or_trim(wavs)
        mels = self.model.log_mel_spectrogram(wavs)
        enc_out = self.model.forward_encoder(mels)
        predictions, _, _, _ = self.searcher(enc_out.detach(), length)
        predictions = self.model.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        predictions = [normalize(text) for text in predictions]
        return predictions

    def to(self, device: str | torch.device) -> "Transcriber":
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.device
            the target device

        Returns
        -------
        result : Transcriber
            The evaluator, on the correct device
        """
        self.model = self.model.to(device)
        return self


class ASRStats(SpeechMetricStats):
    """A base class for ASR-based evaluators

    Arguments
    ---------
    transcriber : Transcriber | Callable | None
        The ASR wrapper to use
    save_path: str | PathLike | None
        The path to save the transcriber model
        (if the default model is used)
    run_opts : dict | None
        Run options for the transcriber
    """
    def __init__(
        self,
        transcriber: Transcriber | Callable | None = None,
        save_path: str | PathLike | None = None,
        run_opts: dict | None = None
    ):
        self.ids = []
        self.metrics = self._init_metrics()
        if transcriber is None:
            self.transcriber = WhisperTranscriber(
                save_path=save_path,
                run_opts=run_opts
            )
        elif isinstance(transcriber, Transcriber):
            self.transcriber = transcriber
        else:
            self.transcriber = transcriber(run_opts=run_opts)
        self.clear()

    def _init_metrics(self):
        return {
            key: ErrorRateStats(
                split_tokens=ASR_METRIC_KIND[key] == "character"
            )
            for key in ASR_METRICS
        }

    def clear(self):
        """Clears the metrics"""
        self.ids = []
        for metric in self.metrics.values():
            metric.clear()
        self.summary = {}

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None
    ):
        """Evaluates a batch of samples

        Arguments
        ---------
        ids : list
            The utterance IDs
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list | None
            Text labels corresponding to the waveforms
        wavs_ref : torch.Tensor | None
            A batch of waveforms (ground truth)
        length_ref : torch.Tensor | None
            Relative lengths (ground truth)
        sample_rate : int | None
            The sample rate of the waveforms
        sample_rate_ref : int | None
            The sample rate of the reference waveforms
        language : str | None
            The language identifier, if applicable
        """
        self.ids.extend(ids)
        if sample_rate_ref is None:
            sample_rate_ref = sample_rate
        predictions = self.transcriber.transcribe(
            wavs=wavs,
            length=length,
            sample_rate=sample_rate,
            language=language
        )
        predictions_words = self._split_words(predictions)
        text_words = self._split_words(text)
        predictions_ref = self.transcriber.transcribe(
            wavs_ref,
            length_ref,
            sample_rate_ref,
            language,
        )
        predictions_ref_words = self._split_words(predictions_ref)
        self._update_metrics(
            self.metrics,
            ids,
            predictions_words,
            predictions_ref_words,
            text_words
        )

    def _update_metrics(
        self,
        metrics: dict[str, ErrorRateStats],
        ids: list,
        predictions_words: list,
        predictions_ref_words: list,
        text_words: list
    ):
        for key, metric in metrics.items():
            target_kind = ASR_METRIC_TARGETS[key]
            if target_kind == "text":
                target_words = text_words
            else:
                target_words = predictions_ref_words
            metric.append(ids, predictions_words, target_words)

    def _split_words(self, items: str) -> list:
        return [normalize(utt_seq).split(" ") for utt_seq in items]

    def summarize(self, field: str | None = None) -> Any:
        """Computes the summary if not already computed and returns
        either the entire summary or one of the available fields

        Arguments
        ---------
        field : str | None
            The name of the field to be retreived

        Returns
        -------
        result : Any
            a dictionary with computed statistics if no field is provided
            or a specific value from that dictionary if one is provided
        """
        summary = self._summarize()
        return summary.get(field) if field is not None else summary

    def _summarize(self) -> dict:
        scores = {key: metric.scores for key, metric in self.metrics.items()}
        summary = {
            stat_key: value
            for key, item_scores in scores.items()
            for stat_key, value in descriptive_statistics(
                item_scores, "WER", key
            ).items()
        }
        micro_stats = {
            f"{key}_micro": metric.summarize("WER")
            for key, metric in self.metrics.items()
        }
        summary.update(micro_stats)
        return summary

    def to(self, device: str | torch.device) -> "ASRStats":
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.device
            the target device

        Returns
        -------
        result : ASRStats
            The evaluator, on the correct device
        """
        self.model = self.model.to(device)
        return self

    def write_report(
        self,
        filestream: TextIO,
        **kwargs: dict
    ):
        """Write metric statistics to a file-like object.

        Arguments
        ---------
        filestream : TextIO
            An open file or file-like object to which stats will be written.
        **kwargs : dict
            Method-specific arguments

            Supported:
            key: the metric key
        """
        key = kwargs.pop("key")
        if key is not None:
            metric = self.metrics[key]
            metric.write_stats(filestream)
        else:
            for key, metric in self.metrics.items():
                print(key, file=filestream)
                print(file=filestream)
                metric.write_stats(filestream)

    def write_reports(self, path: str | PathLike) -> None:
        """Outputs all relevant reports for the metric to the specified path

        Arguments
        ---------
        path : str | PathLike
            The path where reports will be output
        """
        path = Path(path)
        for key in self.stats_keys:
            file_name = path / f"{key}_report.txt"
            with open(file_name, "w") as report_file:
                self.write_report(report_file, key=key)

    stats_keys = ASR_METRICS


class UTMOSStats(SingleMetricStats):
    """A metric implementing UTMOS

    Arguments
    ---------
    sample_rate : int
        The audio sample rate
    source : str, optional
        The HuggingFace hube name for the encoder
    source_base : str, optional
        The source for the base model (wav2vec2)
    save_path : str | PathLike | None
        The path where the model will be saved
    model_name : str, optional
        The name of the model
    features_dim : int, optional
        The features dimension
    num_domains : int, optional
        The number of domains
    domain_dim : int, optional
        The dimension of each domain
    num_judges : int, optional
        The number of "judges"
    judge_dim : int, optional
        The dimension of each judge
    decoder_hidden_size : int, optional
        The size of the decoder hidden state
    domain_id : int, optional
        The domain identifier
    judge_id : int, optional
        The judge identifier
    run_opts : dict | None
        Run options when instantiating the metric
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        source: str = None,
        source_base: str = None,
        save_path: str | PathLike | None = None,
        model_name: str = "utmos.ckpt",
        features_dim: int = 768,
        num_domains: int = 3,
        domain_dim: int = 128,
        num_judges: int = 3000,
        judge_dim: int = 128,
        decoder_hidden_size: int = 512,
        domain_id: int = None,
        judge_id: int = None,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.clear()

        if source is None:
            source = UTMOS_DEFAULT_SOURCE
        if source_base is None:
            source_base = UTMOS_DEFAULT_SOURCE_BASE
        if model_name is None:
            model_name = UTMOS_DEFAULT_MODEL_NAME
        if save_path is None:
            save_path = UTMOS_DEFAULT_SAVE_DIR
        if domain_id is None:
            domain_id = UTMOS_DEFAULT_DOMAIN_ID
        if judge_id is None:
            judge_id = UTMOS_DEFAULT_JUDGE_ID
        if sample_rate is None:
            sample_rate = UTMOS_SAMPLE_RATE

        encoder_path = Path(save_path)
        encoder_path.mkdir(parents=True, exist_ok=True)
        self.model = UTMOSModel(
            source=source_base,
            save_path=encoder_path.as_posix(),
            features_dim=features_dim,
            num_domains=num_domains,
            domain_dim=domain_dim,
            num_judges=num_judges,
            judge_dim=judge_dim,
            decoder_hidden_size=decoder_hidden_size,
        )

        # Download utmos model checkpoint
        fetch(model_name, source, save_path)
        model_path = Path(save_path) / model_name
        assert model_path.exists()

        # Load weights
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.domain_id = domain_id
        self.judge_id = judge_id

        if run_opts:
            device = run_opts.get("device")
            if device:
                self.model.to(device)

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None
    ):
        """Evaluates a batch of samples

        Arguments
        ---------
        ids : list
            The utterance IDs
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list | None
            Text labels corresponding to the waveforms
        wavs_ref : torch.Tensor | None
            A batch of waveforms (ground truth)
        length_ref : torch.Tensor | None
            Relative lengths (ground truth)
        sample_rate : int | None
            The sample rate of the waveforms
        sample_rate_ref : int | None
            The sample rate of the reference waveforms
        language : str | None
            The language identifier, if applicable
        """
        if wavs.dim() > 2:
            wavs = wavs.squeeze()

        # Resample
        hyp_audio = wavs
        if sample_rate is not None:
            hyp_audio = torchaudio.functional.resample(
                wavs, sample_rate, self.sample_rate
            )

        self.model.device = hyp_audio.device
        self.model.to(hyp_audio.device)
        output = self.model(hyp_audio)
        self.append_scores(ids, output)


def _resample_audio(
    wavs: torch.Tensor,
    sample_rate: int | None,
    default_sample_rate: int,
    target_sample_rate: int = AUDIOCODECS_METRIC_SAMPLE_RATE,
) -> torch.Tensor:
    """Resample audio, using a metric's configured rate as the fallback."""
    source_rate = default_sample_rate if sample_rate is None else sample_rate
    if source_rate == target_sample_rate:
        return wavs
    return torchaudio.functional.resample(wavs, source_rate, target_sample_rate)


def _unpadded_audio(
    wavs: torch.Tensor, length: torch.Tensor | None
) -> list[torch.Tensor]:
    """Return individual unpadded waveforms."""
    if length is None:
        return list(wavs)
    return undo_padding(wavs, length)


class DNSMOSStats(SingleMetricStats):
    """Deep Noise Suppression Mean Opinion Score (DNSMOS P.808).

    This is the DNSMOS implementation used by the audiocodecs SLM recipe.
    The bundled ONNX model is evaluated over 9.01-second windows at 16 kHz.

    Arguments
    ---------
    sample_rate : int
        Default input sample rate.
    model : object | None
        An existing ONNX Runtime session.
    model_path : str | PathLike | None
        Path to the DNSMOS P.808 ONNX model.
    """

    def __init__(
        self,
        sample_rate: int = AUDIOCODECS_METRIC_SAMPLE_RATE,
        model: object | None = None,
        model_path: str | PathLike | None = None,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        if model is None:
            model_path = (
                DNSMOS_DEFAULT_MODEL_PATH
                if model_path is None
                else Path(model_path)
            )
            if not Path(model_path).is_file():
                raise FileNotFoundError(
                    f"DNSMOS model not found at {model_path}. Reinstall speechometer "
                    "with package data or provide model_path."
                )
            session_options = onnxruntime.SessionOptions()
            session_options.inter_op_num_threads = os.cpu_count()
            session_options.intra_op_num_threads = os.cpu_count()
            model = onnxruntime.InferenceSession(
                str(model_path), sess_options=session_options
            )
        self.model = model

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor | None,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Evaluate a batch and append one P.808 MOS score per waveform."""
        if wavs.ndim != 2:
            raise ValueError("DNSMOS expects waveforms shaped [batch, time]")
        wavs = _resample_audio(wavs, sample_rate, self.sample_rate)
        scores = [
            {"p808_mos": self._score(wav.cpu().numpy())}
            for wav in _unpadded_audio(wavs, length)
        ]
        self.append_scores(ids, scores)

    def _score(self, audio: np.ndarray) -> float:
        sample_rate = AUDIOCODECS_METRIC_SAMPLE_RATE
        required_samples = int(DNSMOS_INPUT_LENGTH * sample_rate)
        if audio.size == 0:
            return float("nan")
        if audio.size < required_samples:
            repeats = int(np.ceil(required_samples / audio.size))
            audio = np.tile(audio, repeats)

        num_hops = (
            int(np.floor(audio.size / sample_rate) - DNSMOS_INPUT_LENGTH) + 1
        )
        scores = []
        for index in range(max(num_hops, 0)):
            start = index * sample_rate
            segment = audio[start : start + required_samples]
            if segment.size < required_samples:
                continue
            features = np.asarray(
                self._audio_melspec(segment[:-160]), dtype=np.float32
            )[None]
            scores.append(
                float(self.model.run(None, {"input_1": features})[0][0][0])
            )
        return float(np.mean(scores)) if scores else float("nan")

    @staticmethod
    def _audio_melspec(audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=AUDIOCODECS_METRIC_SAMPLE_RATE,
            n_fft=321,
            hop_length=160,
            n_mels=120,
        )
        return ((librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40).T


class STOIStats(SingleMetricStats):
    """Short-time objective intelligibility (STOI)."""

    def __init__(
        self,
        sample_rate: int = AUDIOCODECS_METRIC_SAMPLE_RATE,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor | None,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Evaluate STOI against reference waveforms."""
        if wavs_ref is None:
            raise ValueError("STOI requires wavs_ref")
        if length_ref is None:
            length_ref = length
        wavs = _resample_audio(wavs, sample_rate, self.sample_rate)
        wavs_ref = _resample_audio(wavs_ref, sample_rate_ref, self.sample_rate)
        scores = []
        for hyp, ref in zip(
            _unpadded_audio(wavs, length),
            _unpadded_audio(wavs_ref, length_ref),
        ):
            size = min(hyp.numel(), ref.numel())
            score = stoi.short_time_objective_intelligibility(
                hyp[:size].cpu(),
                ref[:size].cpu(),
                AUDIOCODECS_METRIC_SAMPLE_RATE,
            ).float()
            scores.append(score)
        self.append_scores(ids, torch.stack(scores))


class PESQStats(SingleMetricStats):
    """Wide-band perceptual evaluation of speech quality (PESQ)."""

    def __init__(
        self,
        sample_rate: int = AUDIOCODECS_METRIC_SAMPLE_RATE,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor | None,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Evaluate PESQ against reference waveforms."""
        if wavs_ref is None:
            raise ValueError("PESQ requires wavs_ref")
        if length_ref is None:
            length_ref = length
        wavs = _resample_audio(wavs, sample_rate, self.sample_rate)
        wavs_ref = _resample_audio(wavs_ref, sample_rate_ref, self.sample_rate)
        scores = []
        for utterance_id, hyp, ref in zip(
            ids,
            _unpadded_audio(wavs, length),
            _unpadded_audio(wavs_ref, length_ref),
        ):
            size = min(hyp.numel(), ref.numel())
            try:
                score = pesq.perceptual_evaluation_speech_quality(
                    hyp[:size], ref[:size], AUDIOCODECS_METRIC_SAMPLE_RATE, "wb"
                ).cpu()
            except Exception as error:
                # cypesq raises this for silent / too-short synthesized audio.
                # Do not hide other PESQ errors, which may indicate bad inputs.
                if not _pesq_has_no_utterances(error):
                    raise
                logger.warning(
                    "PESQ found no utterances for %s; using its minimum score (%s).",
                    utterance_id,
                    PESQ_MINIMUM_SCORE,
                )
                score = torch.tensor(PESQ_MINIMUM_SCORE)
            scores.append(score)
        self.append_scores(ids, torch.stack(scores))


class MelDistanceStats(SingleMetricStats):
    """L2 distance between log-Mel spectrograms."""

    def __init__(
        self,
        sample_rate: int = AUDIOCODECS_METRIC_SAMPLE_RATE,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 320,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIOCODECS_METRIC_SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor | None,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Evaluate Mel distance against reference waveforms."""
        if wavs_ref is None:
            raise ValueError("Mel distance requires wavs_ref")
        if length_ref is None:
            length_ref = length
        wavs = _resample_audio(wavs, sample_rate, self.sample_rate)
        wavs_ref = _resample_audio(wavs_ref, sample_rate_ref, self.sample_rate)
        self.mel_spec.to(wavs.device)
        self.amplitude_to_db.to(wavs.device)
        scores = []
        for hyp, ref in zip(
            _unpadded_audio(wavs, length),
            _unpadded_audio(wavs_ref, length_ref),
        ):
            size = min(hyp.numel(), ref.numel())
            hyp_mel = self.amplitude_to_db(self.mel_spec(hyp[:size]))
            ref_mel = self.amplitude_to_db(self.mel_spec(ref[:size]))
            scores.append((hyp_mel - ref_mel).norm(dim=0).mean().cpu())
        self.append_scores(ids, torch.stack(scores))


class STFTDistanceStats(SingleMetricStats):
    """L2 distance between log-magnitude STFT spectrograms."""

    def __init__(
        self,
        sample_rate: int = AUDIOCODECS_METRIC_SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 320,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor | None,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Evaluate STFT distance against reference waveforms."""
        if wavs_ref is None:
            raise ValueError("STFT distance requires wavs_ref")
        if length_ref is None:
            length_ref = length
        wavs = _resample_audio(wavs, sample_rate, self.sample_rate)
        wavs_ref = _resample_audio(wavs_ref, sample_rate_ref, self.sample_rate)
        self.amplitude_to_db.to(wavs.device)
        window = torch.hann_window(self.n_fft, device=wavs.device)
        scores = []
        for hyp, ref in zip(
            _unpadded_audio(wavs, length),
            _unpadded_audio(wavs_ref, length_ref),
        ):
            size = min(hyp.numel(), ref.numel())
            hyp_stft = torch.stft(
                hyp[:size],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            ).abs()
            ref_stft = torch.stft(
                ref[:size],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            ).abs()
            hyp_db = self.amplitude_to_db(hyp_stft)
            ref_db = self.amplitude_to_db(ref_stft)
            scores.append((hyp_db - ref_db).norm(dim=0).mean().cpu())
        self.append_scores(ids, torch.stack(scores))


class ASRPerplexityStats(SingleMetricStats):
    """Perplexity of ASR transcripts under a causal language model.

    The corpus-level ``perplexity`` summary is token-weighted. Descriptive
    statistics over per-utterance perplexities are included alongside it.

    Arguments
    ---------
    model_hub : str
        Hugging Face causal language model identifier.
    sample_rate : int
        Default input sample rate.
    transcriber : Transcriber | Callable | None
        ASR transcriber; defaults to the project's Whisper wrapper.
    asr_model_hub : str
        Whisper model size or full Hugging Face identifier.
    model : torch.nn.Module | None
        Existing causal language model.
    tokenizer : object | None
        Existing tokenizer for the causal language model.
    """

    def __init__(
        self,
        model_hub: str = ASR_PERPLEXITY_DEFAULT_MODEL_HUB,
        sample_rate: int = AUDIOCODECS_METRIC_SAMPLE_RATE,
        save_path: str | PathLike | None = None,
        model: torch.nn.Module | None = None,
        tokenizer: object | None = None,
        asr_model_hub: str = "small",
        asr_model: Transcriber | Callable | None = None,
        transcriber: Transcriber | Callable | None = None,
        run_opts: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        cache_dir = None if save_path is None else str(save_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_hub, cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_hub, cache_dir=cache_dir
        )

        transcriber = transcriber or asr_model
        if transcriber is None:
            source = (
                asr_model_hub
                if "/" in asr_model_hub
                else f"openai/whisper-{asr_model_hub}"
            )
            self.transcriber = WhisperTranscriber(
                source=source,
                save_path=save_path,
                sample_rate=AUDIOCODECS_METRIC_SAMPLE_RATE,
                run_opts=run_opts,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {"min_decode_ratio", "max_decode_ratio", "unbatch"}
                },
            )
        elif isinstance(transcriber, Transcriber):
            self.transcriber = transcriber
        else:
            self.transcriber = transcriber(run_opts=run_opts)
        self.clear()

    def clear(self):
        """Clear accumulated scores, transcripts, and token counts."""
        super().clear()
        self.texts = []
        self.perplexities = []
        self.counts = []

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor | None,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Transcribe audio and append per-utterance perplexities."""
        if length is None:
            length = torch.ones(len(wavs), device=wavs.device)
        if sample_rate is None:
            sample_rate = self.sample_rate
        texts = self.transcriber.transcribe(
            wavs=wavs,
            length=length,
            sample_rate=sample_rate,
            language=language,
        )
        device = wavs.device
        self.model.to(device)
        self.model.eval()
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        labels = input_ids[..., 1:].contiguous()
        mask = attention_mask[..., 1:].contiguous()
        counts = mask.sum(dim=1)
        log_perplexities = (
            torch.nn.functional.cross_entropy(
                logits[..., :-1, :].movedim(-1, -2),
                labels,
                reduction="none",
            )
            * mask
        ).sum(dim=1) / counts.clamp_min(1)
        valid = (counts > 0) & log_perplexities.isfinite()
        if not valid.any():
            return

        valid_items = valid.cpu().tolist()
        valid_ids = [item for item, keep in zip(ids, valid_items) if keep]
        valid_texts = [item for item, keep in zip(texts, valid_items) if keep]
        log_perplexities = log_perplexities[valid]
        counts = counts[valid]
        perplexities = log_perplexities.exp().cpu().tolist()
        self.append_scores(
            valid_ids,
            [{"perplexity": value} for value in perplexities],
        )
        self.texts.extend(valid_texts)
        self.perplexities.extend(perplexities)
        self.counts.extend(counts.cpu().tolist())

    def _summarize(self) -> dict:
        if not self.scores:
            return {}
        summary = descriptive_statistics(self.scores)
        log_perplexities = torch.tensor(self.perplexities).log()
        counts = torch.tensor(self.counts)
        summary["perplexity"] = (
            ((log_perplexities * counts).sum() / counts.sum()).exp().item()
        )
        self.summary = summary
        return summary


class NISQAStats(SingleMetricStats):
    """A wrapper for the NISQA metric

    Arguments
    ---------
    hop_length : float
        The hop length
    seg_length : int
        The length of one segment, in frames
    max_segments : int
        The maximum number of segments
    seg_hop : int
        The segment hop
    sample_rate : int
        The sample rate
    run_opts : dict | None
        Run options (e.g., device) used when instantiating the metric.
    """
    def __init__(
        self,
        hop_length: float = 0.01,
        seg_length: int = 15,
        max_segments: int = 1300,
        seg_hop: int = 4,
        sample_rate: int = 16000,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.seg_length = seg_length
        self.max_segments = max_segments
        self.seg_hop = seg_hop
        self.sample_rate = sample_rate
        self.clear()

    def clear(self):
        """Reset accumulated scores and language-specific mappings.

        This clears `ids`, `scores`, language-specific containers, and any
        cached summary.
        """
        super().clear()

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None
    ):
        """Evaluates a batch of samples

        Arguments
        ---------
        ids : list
            The utterance IDs
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list | None
            Text labels corresponding to the waveforms
        wavs_ref : torch.Tensor | None
            A batch of waveforms (ground truth)
        length_ref : torch.Tensor | None
            Relative lengths (ground truth)
        sample_rate : int | None
            The sample rate of the waveforms
        sample_rate_ref : int | None
            The sample rate of the reference waveforms
        language : str | None
            The language identifier, if applicable
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        max_wav_length = int(
            (self.max_segments + (self.seg_length - 1) // self.seg_hop)
            * self.hop_length
            * sample_rate
            * self.seg_hop
        )
        wavs = wavs[..., :max_wav_length]
        results = nisqa.non_intrusive_speech_quality_assessment(
            wavs, sample_rate
        ).detach().to("cpu").numpy().tolist()
        scores = []
        for mos, noisiness, discontinuity, coloration, loudness in results:
            item_scores = {
                "mos": mos,
                "noisiness": noisiness,
                "discontinuity": discontinuity,
                "coloration": coloration,
                "loudness": loudness,
            }
            scores.append(item_scores)
        self.append_scores(ids, scores)


class SpkSimECAPATDNNStats(SingleMetricStats):
    """Speaker Similarity using ECAPA-TDNN

    Arguments
    ---------
    source : str
        The HuggingFace hub or path from which to fetch the model
    save_path : str | PathLike
        The path where the model will be saved
    sample_rate : int
        The default sample rate of the audio files
    model_sample_rate : int
        The sample rate of the model
    run_opts : dict | None
        The run options (the device, etc)
    """
    def __init__(
        self,
        source: str,
        save_path: str | PathLike = None,
        sample_rate: int = 16000,
        model_sample_rate: int = 16000,
        run_opts: dict | None = None
    ):
        super().__init__()
        self.sample_rate = sample_rate
        if run_opts is None:
            run_opts = {}
        self.model = SpeakerRecognition.from_hparams(
            source, savedir=save_path, run_opts=run_opts
        )
        self.clear()
        self.model_sample_rate = model_sample_rate

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None
    ):
        """Evaluates a batch of samples

        Arguments
        ---------
        ids : list
            The utterance IDs
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list | None
            Text labels corresponding to the waveforms
        wavs_ref : torch.Tensor | None
            A batch of waveforms (ground truth)
        length_ref : torch.Tensor | None
            Relative lengths (ground truth)
        sample_rate : int | None
            The sample rate of the waveforms
        sample_rate_ref : int | None
            The sample rate of the reference waveforms
        language : str | None
            The language identifier, if applicable
        """
        assert wavs.ndim == 2
        assert wavs_ref is not None
        assert wavs_ref.ndim == 2
        assert len(wavs) == len(wavs_ref)

        if sample_rate is None:
            sample_rate = self.sample_rate
        if sample_rate != self.model_sample_rate:
            wavs = torchaudio.functional.resample(
                wavs,
                sample_rate,
                self.model_sample_rate
            )

        if sample_rate_ref is None:
            sample_rate_ref = self.sample_rate
        if sample_rate_ref != self.model_sample_rate:
            wavs_ref = torchaudio.functional.resample(
                wavs_ref,
                sample_rate_ref,
                self.model_sample_rate
            )

        assert self.model is not None
        self.model.device = wavs.device
        self.model.to(wavs.device)
        self.model.eval()

        # Encode separately because the hypothesis and reference batches may
        # have different padded time dimensions and relative lengths.
        hyp_embs = self.model.encode_batch(wavs, length, normalize=False)
        ref_embs = self.model.encode_batch(
            wavs_ref, length_ref, normalize=False
        )
        scores = self.model.similarity(hyp_embs, ref_embs)[:, 0]
        self.append_scores(ids, scores)


class SpkSimWavLMStats(SingleMetricStats):
    """Speaker Similarity using WavLM

    Arguments
    ---------
    source : str
        The HuggingFace hub or path from which to fetch the model
    save_path : str | PathLike
        The path where the model will be saved
    sample_rate : int
        The default sample rate of the audio files
    model_sample_rate : int
        The sample rate of the model
    run_opts : dict | None
        The run options (the device, etc)
    """
    def __init__(
        self,
        source: str = SPKSIM_WAVLM_DEFAULT_MODEL_HUB,
        save_path: str | PathLike = None,
        sample_rate: int = 16000,
        model_sample_rate: int = 16000,
        run_opts: dict | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.model = AutoModelForAudioXVector.from_pretrained(
            source, cache_dir=save_path
        )
        if run_opts is None:
            run_opts = {}
        device = run_opts.get("device")
        if device is not None:
            self.model = self.model.to(device)
        self.model_sample_rate = model_sample_rate
        self.clear()

    @torch.no_grad()
    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None
    ):
        """Evaluates a batch of samples

        Arguments
        ---------
        ids : list
            The utterance IDs
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list | None
            Text labels corresponding to the waveforms
        wavs_ref : torch.Tensor | None
            A batch of waveforms (ground truth)
        length_ref : torch.Tensor | None
            Relative lengths (ground truth)
        sample_rate : int | None
            The sample rate of the waveforms
        sample_rate_ref : int | None
            The sample rate of the reference waveforms
        language : str | None
            The language identifier, if applicable
        """
        assert wavs_ref is not None
        assert wavs.ndim == 2
        assert wavs_ref.ndim == 2
        assert len(wavs) == len(wavs_ref)

        if sample_rate is None:
            sample_rate = self.sample_rate
        if sample_rate != self.model_sample_rate:
            wavs = torchaudio.functional.resample(
                wavs,
                sample_rate,
                self.model_sample_rate
            )

        if sample_rate_ref is None:
            sample_rate_ref = self.sample_rate
        if sample_rate_ref != self.model_sample_rate:
            wavs_ref = torchaudio.functional.resample(
                wavs_ref,
                sample_rate_ref,
                self.model_sample_rate
            )

        self.model.to(wavs.device)
        self.model.eval()

        # Build masks and encode separately because the hypothesis and
        # reference batches may have different padded time dimensions.
        attention_mask = None
        if length is not None:
            attention_mask = length_to_mask(
                (length * wavs.shape[-1]).int(),
                max_len=wavs.shape[-1],
            ).long()  # 0 for masked tokens
        attention_mask_ref = None
        if length_ref is not None:
            attention_mask_ref = length_to_mask(
                (length_ref * wavs_ref.shape[-1]).int(),
                max_len=wavs_ref.shape[-1],
            ).long()  # 0 for masked tokens

        hyp_embs = self.model(
            input_values=wavs,
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings
        ref_embs = self.model(
            input_values=wavs_ref,
            attention_mask=attention_mask_ref,
            output_attentions=False,
        ).embeddings
        scores = torch.nn.functional.cosine_similarity(
            hyp_embs, ref_embs, dim=-1
        )

        self.append_scores(ids, scores)


class SpeechBLEUStats(SpeechMetricStats):
    """Statistics for the BLEU metric

    Arguments
    ---------
    transcriber: Transcriber | Callable | None
        An ASR wrapper
    save_path: str | PathLike | None
        The path to save the transcriber model
        (if the default model is used)
    run_opts : dict | None
        Run options for the transcriber
    """
    def __init__(
        self,
        transcriber: Transcriber | Callable | None = None,
        save_path: str | PathLike | None = None,
        run_opts: dict | None = None
    ):
        self.bleu = bleu.BLEUStats()
        self.ids = []
        if transcriber is None:
            self.transcriber = WhisperTranscriber(
                save_path=save_path,
                run_opts=run_opts
            )
        elif isinstance(transcriber, Transcriber):
            self.transcriber = transcriber
        else:
            self.transcriber = transcriber(run_opts=run_opts)
        self.clear()

    def append(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None = None,
        wavs_ref: torch.Tensor | None = None,
        length_ref: torch.Tensor | None = None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None,
    ):
        """Evaluates a batch of samples

        Arguments
        ---------
        ids : list
            The utterance IDs
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list | None
            Text labels corresponding to the waveforms
        wavs_ref : torch.Tensor | None
            A batch of waveforms (ground truth)
        length_ref : torch.Tensor | None
            Relative lengths (ground truth)
        sample_rate : int | None
            The sample rate of the waveforms
        sample_rate_ref : int | None
            The sample rate of the reference waveforms
        language : str | None
            The language identifier, if applicable
        """
        self.ids.extend(ids)
        text = self.transcriber.transcribe(
            wavs=wavs,
            length=length,
            sample_rate=sample_rate,
            language=language
        )
        text_ref = self.transcriber.transcribe(
            wavs=wavs_ref,
            length=length_ref,
            sample_rate=sample_rate,
            language=language
        )
        self.bleu.append(
            ids=ids,
            predict=text,
            targets=[text_ref]
        )

    def summarize(self, field: str | None = None) -> Any:
        """Computes the summary if not already computed and returns
        either the entire summary or one of the available fields

        Arguments
        ---------
        field : str | None
            The name of the field to be retreived

        Returns
        -------
        result : Any
            a dictionary with computed statistics if no field is provided
            or a specific value from that dictionary if one is provided
        """
        if not self.summary:
            self.summary = self.bleu.summarize()
        return self.summary


def normalize(text: str) -> str:
    """Performs text normalization (uppercase, remove whitespace,
    remove punctuation)

    Arguments
    ---------
    text : str
        Unnormalized text

    Returns
    -------
    text : str
        Normalized text
    """
    text = text.upper()
    text = text.strip()
    text = RE_PUNCTUATION.sub("", text)
    return text
