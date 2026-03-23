"""Reusable speech metric wrappers

Some metrics were adopted from the DASB benchmark
https://github.com/speechbrain/benchmarks

Authors
 * Artem Ploujnikov 2026
"""

from abc import ABC, abstractmethod
import csv
import json
from os import PathLike
import re
import string
from pathlib import Path
from typing import TextIO

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
from transformers import AutoModelForAudioXVector

from speechometer.stats import descriptive_statistics

logger = get_logger(__name__)

nisqa = LazyModule("nisqa", "torchmetrics.functional.audio.nisqa", None)

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

    def summarize(self, field: str | None = None) -> any:
        """Computes the summary if not already computed and returns
        either the entire summary or one of the available fields

        Arguments
        ---------
        field : str | None
            The name of the field to be retreived

        Returns
        -------
        result : any
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
            "MetricStats", ""
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


class ASRMetricStats(SpeechMetricStats):
    """A base class for ASR-based evaluators

    Arguments
    ---------
    unbatch : bool
        Whether to undo batches
    """

    def __init__(
        self,
        unbatch: bool = False,
    ):
        self.ids = []
        self.metrics = self._init_metrics()
        self.unbatch = unbatch
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
        if self.unbatch:
            batch_size = len(wavs)
            length_abs = (length * wavs.size(1)).int()
            length_ref_abs = (length_ref * wavs.size(1)).int()
            for idx in range(batch_size):
                self._evaluate_samples(
                    ids[idx: idx + 1],
                    wavs=wavs[idx:idx + 1, : length_abs[idx].item()],
                    length=torch.ones(1, device=wavs.device),
                    text=text[idx:idx + 1],
                    wavs_ref=wavs_ref[
                        idx:idx + 1, : length_ref_abs[idx].item()
                    ],
                    length_ref=torch.ones(1, device=wavs_ref.device),
                    sample_rate=sample_rate,
                    sample_rate_ref=sample_rate_ref,
                    language=language,
                )
        else:
            self._evaluate_samples(
                wavs,
                length,
                text,
                wavs_ref,
                length_ref,
                sample_rate,
                sample_rate_ref,
                language=language,
            )

    def _evaluate_samples(
        self,
        ids: list,
        wavs: torch.Tensor,
        length: torch.Tensor,
        text: list | None,
        wavs_ref: torch.Tensor | None,
        length_ref: torch.Tensor | None,
        sample_rate: int | None = None,
        sample_rate_ref: int | None = None,
        language: str | None = None
    ):
        """Evaluates a batch of samples. This function is meant
        to be used internally. evaluate_samples will call
        it multiple times if unbatch is enabled.

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
        language: str | None
            The language identifier, if applicable
        """
        predictions = self.predict(wavs, length, sample_rate, language)
        predictions_words = self._split_words(predictions)
        text_words = self._split_words(text)
        predictions_ref = self.predict(
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
        return [utt_seq.split(" ") for utt_seq in items]

    def summarize(self, field: str = None) -> dict:
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

    def normalize(self, text: str) -> str:
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

    def to(self, device: str | torch.device) -> "ASRMetricStats":
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.device
            the target device

        Returns
        -------
        result : ASRMetricStats
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


class WhisperASR(ASRMetricStats):
    """A speech evaluator implementation based on Whisper ASR

    Arguments
    ---------
    source : str | None
        The source directory
    model: torch.nn.Module | None
        a pretrained Whisper model
    save_path : str | PathLike | None
        The path where Whisper will be saved
    sample_rate: int
        The audio sample rate
    min_decode_ratio : float, optional
        The minimum decode ratio
    max_decode_ratio : float, optional
        The maximum decode ratio
    run_opts : dict | None
        Run options for the Whisper model
    unbatch : bool, optional
        If enabled, which is the default, the implementation
        will evaluate samples one by one with a batch size of
        1 and then "reassemble" the original batch. This is
        sometimes needed because batched inference has been
        found to result in decreased performance, primarily
        due to masks not being applied to convolutional layers
    """

    def __init__(
        self,
        source: str | None = None,
        model: torch.nn.Module | None = None,
        save_path: str | PathLike | None = None,
        sample_rate: int = 22050,
        min_decode_ratio: float = 0.0,
        max_decode_ratio: float = 1.0,
        run_opts: dict | None = None,
        unbatch: bool = True,
    ):
        super().__init__(unbatch=unbatch)
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
        self.to(device)

    def predict(
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
            Waveforms
        length : torch.Tensor
            Negative lengths
        sample_rate : int
            The sample rate of the waveform
        language : str
            The language identifier (Whisper-compatible)

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
        predictions = [self.normalize(text) for text in predictions]
        return predictions


class UTMOS(SingleMetricStats):
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


class NISQA(SingleMetricStats):
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


class SpkSimECAPATDNN(SingleMetricStats):
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
            source, savedir=save_path, **run_opts
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
        assert wavs.shape == wavs_ref.shape
        assert wavs.ndim == 2

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

        # Concatenate
        audio = torch.cat([wavs, wavs_ref])
        if length is not None:
            length = torch.cat([length, length])

        self.model.device = wavs.device
        self.model.to(wavs.device)
        self.model.eval()

        # Forward
        embs = self.model.encode_batch(audio, length, normalize=False)
        hyp_embs, ref_embs = embs.split([len(wavs), len(wavs_ref)])
        scores = self.model.similarity(hyp_embs, ref_embs)[:, 0]
        self.append_scores(ids, scores, language=language)


class SpkSimWavLM(SingleMetricStats):
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
        assert wavs.ndim == 2
        assert wavs_ref.ndim == 2
        if wavs.size(1) != wavs_ref.size(1):
            min_length = min(wavs.size(1), wavs_ref.size(1))
            wavs = wavs[:, :min_length]
            wavs_ref = wavs_ref[:, :min_length]

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

        # Concatenate
        audio = torch.cat([wavs, wavs_ref])
        if length is not None:
            length = torch.cat([length, length])

        # Attention mask
        attention_mask = None
        if length is not None:
            abs_length = length * audio.shape[-1]
            attention_mask = length_to_mask(
                abs_length.int()
            ).long()  # 0 for masked tokens

        # Forward
        embs = self.model(
            input_values=audio,
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings

        hyp_embs, ref_embs = embs.split([len(wavs), len(wavs_ref)])
        scores = torch.nn.functional.cosine_similarity(
            hyp_embs, ref_embs, dim=-1
        )

        self.ids += ids
        self.append_scores(ids, scores)
