import os
import wave
import contextlib

# Path to your directory containing the processed .wav and .txt files
PROCESSED_DIR = "processed_audio"
DURATION_THRESHOLD = 1  # in seconds

deleted_count = 0

for dirpath, _, filenames in os.walk(PROCESSED_DIR):
    for filename in filenames:
        if filename.endswith(".wav"):
            wav_path = os.path.join(dirpath, filename)
            txt_path = os.path.splitext(wav_path)[0] + ".txt"

            # Measure duration
            try:
                with contextlib.closing(wave.open(wav_path, 'r')) as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
            except Exception as e:
                print(f"Could not read {wav_path}: {e}")
                continue

            # If shorter than threshold, delete .wav and .txt
            if duration < DURATION_THRESHOLD:
                try:
                    os.remove(wav_path)
                    print(f"Deleted: {wav_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {wav_path}: {e}")

                if os.path.exists(txt_path):
                    try:
                        os.remove(txt_path)
                        print(f"Deleted: {txt_path}")
                    except Exception as e:
                        print(f"Failed to delete {txt_path}: {e}")

print(f"\nTotal .wav files deleted: {deleted_count}")
print(f"Total .txt files deleted: {deleted_count}")
