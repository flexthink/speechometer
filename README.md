# speechometer

Reusable SpeechBrain-compatible metrics for evaluating generated and
reconstructed speech.

## Metrics

The SLM metric suite from
[lucadellalib/audiocodecs](https://github.com/lucadellalib/audiocodecs) is
available from `speechometer.metrics.speech_metrics`:

- `UTMOSStats` and `DNSMOSStats` for non-intrusive speech quality
- `STOIStats` and `PESQStats` for reference-based intelligibility and quality
- `MelDistanceStats` and `STFTDistanceStats` for spectral distortion
- `ASRPerplexityStats` for transcript language-model perplexity
- `SpkSimWavLMStats` and `SpkSimECAPATDNNStats` for speaker similarity

All metric wrappers implement the common `SpeechMetricStats.append(...)`,
`summarize(...)`, and `write_reports(...)` interface.

## Attribution

The DNSMOS, STOI, PESQ, Mel distance, STFT distance, and ASR perplexity
implementations were adapted from Luca Della Libera's Apache-2.0-licensed
[audiocodecs](https://github.com/lucadellalib/audiocodecs) SLM recipe. The
original copyright and license notice is retained in [NOTICE](NOTICE), and a
copy of Apache License 2.0 is included in `LICENSES/Apache-2.0.txt`.
