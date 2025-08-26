import os
import re
import wave
import contextlib
from datetime import datetime, timedelta
from pydub import AudioSegment, silence
from pydub.scipy_effects import band_pass_filter

# Path to your directory containing WAV and TXT files
AUDIO_DIR = "./"  # Change this to your actual directory

# Regex to extract timestamps from the TXT file
TIMESTAMP_PATTERN = r"Start Time:\s*(.*?)\n.*?End Time:\s*(.*?)\n"

def parse_timestamps(txt_file):
    """Extracts the start and end timestamps from a given txt file."""
    with open(txt_file, "r") as f:
        content = f.read()

    match = re.search(TIMESTAMP_PATTERN, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse timestamps from {txt_file}")

    start_time = datetime.strptime(match.group(1).strip(), "%Y-%m-%d %H:%M:%S.%f")
    end_time = datetime.strptime(match.group(2).strip(), "%Y-%m-%d %H:%M:%S.%f")

    return start_time, end_time

def split_audio(audio_file, start_time, end_time):
    """Splits the given audio file based on silence and saves each call separately with timestamp-based filenames."""
    audio = AudioSegment.from_wav(audio_file)
    audio = band_pass_filter(audio, low_cutoff_freq=300, high_cutoff_freq=3000)

    # Detect non-silent chunks (returns [(start, end), ...] in milliseconds)
    min_silence_len = 500  # 500ms silence = new radio call
    silence_thresh = -40  # dBFS threshold
    non_silent_chunks = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    for i, (start_ms, end_ms) in enumerate(non_silent_chunks):
        # Convert to seconds
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0

        # Compute absolute timestamps
        abs_start_time = start_time + timedelta(seconds=start_sec)
        abs_end_time = start_time + timedelta(seconds=end_sec)

        # Format timestamp as filename-safe string
        timestamp_str = abs_start_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Trim microseconds to milliseconds
        out_wav = f"../processed_audio/{timestamp_str}.wav"
        out_txt = f"../processed_audio/{timestamp_str}.txt"

        # Extract and save the chunk
        chunk = audio[start_ms:end_ms]
        chunk.export(out_wav, format="wav")

        # Save timestamps
        with open(out_txt, "w") as f:
            f.write(f"Start Time:\n{abs_start_time}\nEnd Time:\n{abs_end_time}\n")

        print(f"Saved: {out_wav}, {out_txt} (from {abs_start_time} to {abs_end_time})")


def process_directory(audio_dir):
    """Processes all WAV and TXT files in a given directory."""
    for dirpath, _, filenames in os.walk(audio_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                base_name = os.path.splitext(filename)[0]
                txt_file = os.path.join(dirpath, f"{base_name}.txt")
                wav_file = os.path.join(dirpath, filename)

                if os.path.exists(txt_file):
                    print(f"Processing: {filename}")
                    try:
                        start_time, end_time = parse_timestamps(txt_file)
                        split_audio(wav_file, start_time, end_time)
                    except ValueError as e:
                        print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    os.makedirs('../processed_audio', exist_ok=True)
    process_directory(AUDIO_DIR)
