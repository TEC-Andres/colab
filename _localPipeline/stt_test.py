#!/usr/bin/env python3
"""
Local CPU STT test for RoBorregos/home2 issue #1011.

Usage:
    python stt_test.py --audio path/to/file.wav
    python stt_test.py --record 5
    python stt_test.py --batch ./test_audio/
    python stt_test.py --benchmark
    python stt_test.py --benchmark --benchmark-audio path/to/file.wav
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

# ── Resolve imports from local copies ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from device_utils import detect_device_and_compute_type
from transcriber_faster_whisper import WhisperModel


# ── Model singleton ────────────────────────────────────────────────────────
_model = None
_model_name = None


def get_model(model_name: str) -> WhisperModel:
    global _model, _model_name
    if _model is None or _model_name != model_name:
        device, compute_type = detect_device_and_compute_type()
        print(f"Loading model '{model_name}' on {device} ({compute_type}) ...")
        _model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=os.path.join(SCRIPT_DIR, "models"),
        )
        _model_name = model_name
        print("Model loaded.\n")
    return _model


# ── Core transcription ─────────────────────────────────────────────────────
def transcribe_file(
    audio_path: str,
    model_name: str = "base.en",
    language: str = "en",
    task: str = "transcribe",
    vad: bool = True,
    word_timestamps: bool = True,
    hotwords: str = "",
    initial_prompt: str = "",
) -> dict:
    model = get_model(model_name)

    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language,
        task=task,
        vad_filter=vad,
        word_timestamps=word_timestamps,
        hotwords=hotwords or None,
        initial_prompt=initial_prompt or None,
    )

    all_segments = []
    full_text_parts = []
    for seg in (segments or []):
        words = []
        if seg.words:
            words = [
                {
                    "word": w.word,
                    "confidence": round(w.probability, 4),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                }
                for w in seg.words
            ]
        all_segments.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "words": words,
            }
        )
        full_text_parts.append(seg.text)

    elapsed = time.time() - t0
    duration = info.duration if info else 0.0
    full_text = "".join(full_text_parts).strip()

    result = {
        "text": full_text,
        "segments": all_segments,
        "language": info.language if info else "unknown",
        "language_probability": round(info.language_probability, 4) if info else 0,
        "audio_duration": round(duration, 3),
        "processing_time": round(elapsed, 3),
        "real_time_factor": round(elapsed / duration, 4) if duration > 0 else None,
    }
    return result


def print_result(result: dict, label: str = "") -> None:
    header = f"  {label} " if label else "  "
    print(f"\n{'='*60}")
    print(f"{header}Transcription Result")
    print(f"{'='*60}")
    print(f"  Text:       {result['text']}")
    print(f"  Language:   {result['language']} ({result['language_probability']:.2%})")
    print(f"  Duration:   {result['audio_duration']}s")
    print(f"  RTF:        {result['real_time_factor']}x" if result['real_time_factor'] is not None else "  RTF:        N/A")
    print(f"  Time:       {result['processing_time']}s")
    print(f"  Segments:   {len(result['segments'])}")
    for seg in result["segments"]:
        print(f"    [{seg['start']:.2f} -> {seg['end']:.2f}] {seg['text']}")
        for w in seg["words"][:5]:
            print(f"      {w['word']:<20} conf={w['confidence']:.2%}")
        if len(seg["words"]) > 5:
            print(f"      ... and {len(seg['words']) - 5} more words")
    print(f"{'='*60}\n")


# ── Record from mic ────────────────────────────────────────────────────────
def record_audio(duration: int, output_path: str, sample_rate: int = 16000) -> str:
    import sounddevice as sd
    import soundfile as sf

    print(f"Recording {duration}s from microphone ... speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    print("Recording done.")
    sf.write(output_path, audio.flatten(), sample_rate)
    print(f"Saved to {output_path}")
    return output_path


# ── Batch transcription ────────────────────────────────────────────────────
def batch_transcribe(audio_dir: str, model_name: str, language: str) -> list:
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {audio_dir}")
        return []
    print(f"Found {len(wav_files)} .wav files.\n")
    results = []
    for wav in wav_files:
        result = transcribe_file(wav, model_name=model_name, language=language)
        print_result(result, label=os.path.basename(wav))
        results.append({"file": wav, **result})
    return results


# ── Benchmark ──────────────────────────────────────────────────────────────
def benchmark(audio_path: str, model_name: str, n_runs: int = 3) -> None:
    print(f"Benchmarking '{model_name}' on {n_runs} runs ...")
    latencies = []
    for i in range(n_runs):
        result = transcribe_file(audio_path, model_name=model_name)
        latencies.append(result["processing_time"])
        print(f"  Run {i+1}: {result['processing_time']}s (RTF {result['real_time_factor']})")

    avg = np.mean(latencies)
    dur = result["audio_duration"]
    print(f"\n--- Benchmark Summary ---")
    print(f"  Audio duration:   {dur}s")
    print(f"  Avg latency:      {avg:.3f}s")
    print(f"  Avg RTF:          {avg/dur:.4f}x" if dur > 0 else "  Avg RTF: N/A")
    print(f"  Throughput:       {dur/avg:.2f}x realtime" if avg > 0 else "  Throughput: N/A")


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Local CPU STT test for home2 issue #1011",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="Path to a .wav file to transcribe")
    group.add_argument("--record", type=int, metavar="SECONDS", help="Record N seconds from mic, then transcribe")
    group.add_argument("--batch", type=str, metavar="DIR", help="Transcribe all .wav files in a directory")
    group.add_argument("--benchmark", action="store_true", help="Benchmark transcription speed")

    parser.add_argument("--model", type=str, default="base.en",
                        help="Whisper model size (default: base.en)")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code (default: en)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task: transcribe or translate (default: transcribe)")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument("--no-word-timestamps", action="store_true", help="Disable word-level timestamps")
    parser.add_argument("--hotwords", type=str, default="", help="Hotwords hint for the model")
    parser.add_argument("--initial-prompt", type=str, default="", help="Initial prompt for the model")
    parser.add_argument("--benchmark-audio", type=str, default=None,
                        help="Audio file for benchmark (default: generates synthetic audio)")
    parser.add_argument("--benchmark-runs", type=int, default=3,
                        help="Number of benchmark runs (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save transcription result to a JSON file")

    args = parser.parse_args()

    vad = not args.no_vad
    word_timestamps = not args.no_word_timestamps

    if args.audio:
        if not os.path.isfile(args.audio):
            print(f"Error: file not found: {args.audio}")
            sys.exit(1)
        result = transcribe_file(
            args.audio,
            model_name=args.model,
            language=args.language,
            task=args.task,
            vad=vad,
            word_timestamps=word_timestamps,
            hotwords=args.hotwords,
            initial_prompt=args.initial_prompt,
        )
        print_result(result)

    elif args.record is not None:
        rec_path = os.path.join(SCRIPT_DIR, "recorded.wav")
        record_audio(args.record, rec_path)
        result = transcribe_file(
            rec_path,
            model_name=args.model,
            language=args.language,
            task=args.task,
            vad=vad,
            word_timestamps=word_timestamps,
            hotwords=args.hotwords,
            initial_prompt=args.initial_prompt,
        )
        print_result(result)

    elif args.batch:
        batch_transcribe(args.batch, args.model, args.language)

    elif args.benchmark:
        if args.benchmark_audio:
            if not os.path.isfile(args.benchmark_audio):
                print(f"Error: file not found: {args.benchmark_audio}")
                sys.exit(1)
            audio_path = args.benchmark_audio
        else:
            # Generate a 10s synthetic audio for quick benchmarking
            sr = 16000
            t = np.linspace(0, 10, sr * 10, endpoint=False)
            synthetic = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
            import soundfile as sf
            audio_path = os.path.join(SCRIPT_DIR, "_bench_synth.wav")
            sf.write(audio_path, synthetic, sr)
            print(f"Generated synthetic test audio: {audio_path}")
        benchmark(audio_path, args.model, n_runs=args.benchmark_runs)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
