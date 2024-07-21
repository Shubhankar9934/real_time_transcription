import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from queue import Queue
import speech_recognition as sr
from time import sleep
import threading
from Real_Time_Trans_Project.helper.helper import setup_microphone, load_models, setup_recorder, transcribe_audio ,record_callback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--default_microphone", default='Stereo Mix',
                        help="Default microphone name for SpeechRecognition. "
                             "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    source, mic_name = setup_microphone(args.default_microphone)
    if source is None:
        print(f"Microphone with name '{args.default_microphone}' not found.")
        return

    print(f"Using microphone: {mic_name}")

    audio_model, translation_model, tokenizer = load_models(args.model, args.non_english)

    data_queue = Queue()
    recorder = sr.Recognizer()
    setup_recorder(recorder, args.energy_threshold)

    with source:
        recorder.adjust_for_ambient_noise(source)

    recorder.listen_in_background(source, lambda _, audio: record_callback(_, audio, data_queue), phrase_time_limit=args.record_timeout)
    print("Model loaded.\n")

    transcribe_thread = threading.Thread(target=transcribe_audio, args=(data_queue, audio_model, translation_model, tokenizer, args.phrase_timeout))
    transcribe_thread.start()

    while True:
        sleep(1)

if __name__ == "__main__":
    main()
