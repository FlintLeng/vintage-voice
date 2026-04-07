#!/usr/bin/env python3
"""
VintageVoice — Speech Generation with Historical Voice Presets

Generate speech that sounds like it's from the 1930s-1950s.
Uses F5-TTS with vintage fine-tuned weights.
"""
import argparse
import os
import torch
import torchaudio


# Reference audio clips for each preset (included in model release)
PRESET_REFS = {
    "transatlantic": "refs/transatlantic_ref.wav",
    "newsreel": "refs/newsreel_narrator_ref.wav",
    "fireside": "refs/fdr_fireside_ref.wav",
    "radio_drama": "refs/radio_drama_ref.wav",
    "edison": "refs/edison_cylinder_ref.wav",
    "wartime": "refs/wartime_broadcast_ref.wav",
    "announcer": "refs/radio_announcer_ref.wav",
}


def generate_speech(
    text,
    preset="transatlantic",
    model_path=None,
    ref_audio=None,
    output_path="output.wav",
    device="cuda:0",
):
    """Generate vintage-styled speech from text"""

    # Use preset reference audio if no custom ref provided
    if ref_audio is None:
        ref_audio = PRESET_REFS.get(preset)
        if ref_audio and not os.path.exists(ref_audio):
            # Check in model directory
            if model_path:
                model_dir = os.path.dirname(model_path)
                ref_audio = os.path.join(model_dir, ref_audio)

    print(f"VintageVoice Generation")
    print(f"  Preset: {preset}")
    print(f"  Text: {text[:80]}...")
    print(f"  Reference: {ref_audio}")
    print(f"  Output: {output_path}")

    try:
        from f5_tts.api import F5TTS

        # Load model with vintage weights
        tts = F5TTS(device=device)
        if model_path and os.path.exists(model_path):
            print(f"  Loading vintage weights: {model_path}")
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            tts.model.load_state_dict(state_dict, strict=False)

        # Generate with reference audio
        wav, sr, _ = tts.infer(
            ref_file=ref_audio,
            ref_text="",  # Auto-transcribe reference
            gen_text=text,
            file_wave=output_path,
        )

        print(f"  Generated {len(wav)/sr:.1f}s of audio at {sr}Hz")
        return output_path

    except ImportError:
        print("\n  F5-TTS not installed. Install: pip install f5-tts")
        print("  For now, generating with reference audio cloning via torchaudio...")

        # Fallback: basic generation pipeline demo
        print("  (Full generation requires F5-TTS package)")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate vintage-styled speech")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--preset", default="transatlantic", choices=list(PRESET_REFS.keys()))
    parser.add_argument("--model", default=None, help="Path to fine-tuned model weights")
    parser.add_argument("--ref-audio", default=None, help="Custom reference audio clip")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    generate_speech(
        text=args.text,
        preset=args.preset,
        model_path=args.model,
        ref_audio=args.ref_audio,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
