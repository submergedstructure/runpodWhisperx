import runpod
import os
import time
import whisperx
import gc 
import base64
import tempfile
import requests
import traceback

# CHECK THE ENV VARIABLES FOR DEVICE AND COMPUTE TYPE
device = os.environ.get('DEVICE', 'cuda') # cpu if on Mac
compute_type = os.environ.get('COMPUTE_TYPE', 'float16') #int8 if on Mac
batch_size = 16 # reduce if low on GPU mem

def base64_to_tempfile(base64_data):
    """
    Decode base64 data and write it to a temporary file.
    Returns the path to the temporary file.
    """
    # Decode the base64 data to bytes
    audio_data = base64.b64decode(base64_data)

    # Create a temporary file and write the decoded data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    with open(temp_file.name, 'wb') as file:
        file.write(audio_data)

    return temp_file.name

def download_file(url):
    """
    Download a file from a URL to a temporary file and return its path.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file from URL")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def handler(event):
    """
    Run inference on the model.

    Args:
        event (dict): The input event containing the audio data.
            The event should have the following structure:
            {
                "input": {
                    "audio_base_64": str,  # Base64-encoded audio data (optional)
                    "audio_url": str       # URL of the audio file (optional)
                    "language_code": str   # Language code (optional), default is "pl"
                    "diarize": bool        # Whether to perform diarization (optional), default is False
                    "min_speakers": int    # Minimum number of speakers for diarization (optional)
                    "max_speakers": int    # Maximum number of speakers for diarization (optional)
                    "transcribe_options": dict  # Options to pass to model.transcribe (optional)
                }
            }
            Either "audio_base_64" or "audio_url" must be provided.
    """
    job_input = event['input']
    job_input_audio_base_64 = job_input.get('audio_base_64')
    job_input_audio_url = job_input.get('audio_url')
    if job_input_audio_base_64:
        # If there is base64 data
        audio_input = base64_to_tempfile(job_input_audio_base_64)
    elif job_input_audio_url and job_input_audio_url.startswith('http'):
        # If there is an URL
        audio_input = download_file(job_input_audio_url)
    else:
        return "No audio input provided"
    
    language_code = job_input.get('language_code', 'pl')
    diarize = job_input.get('diarize', False)
    min_speakers = job_input.get('min_speakers')
    max_speakers = job_input.get('max_speakers')
    transcribe_options = job_input.get('transcribe_options', {})

    try:
        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("large-v3", device, compute_type=compute_type, language=language_code)
        # Load the audio
        audio = whisperx.load_audio(audio_input)
        # Transcribe the audio
        result = model.transcribe(audio, batch_size=batch_size, language=language_code, print_progress=True, **transcribe_options)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
        print(result["segments"])

        # 3. Optionally perform diarization
        if diarize:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.environ.get('YOUR_HF_TOKEN'), device=device)
            if min_speakers is not None and max_speakers is not None:
                diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            else:
                diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(diarize_segments)
            print(result["segments"]) # segments are now assigned speaker IDs

        # after alignment
        return result
    except Exception as e:
        return f"Error transcribing audio: {str(e)}, Args: {e.args}, Traceback: {''.join(traceback.format_tb(e.__traceback__))}"

runpod.serverless.start({
    "handler": handler
})