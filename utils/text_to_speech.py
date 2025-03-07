import requests
import json
import subprocess
import base64
import hashlib
import os

def synthesize_text_to_speech(text, output_dir="audio_cache"):
    """
    Synthesizes text to speech using Google Cloud Text-to-Speech API.
    Generates a filename based on a hash of the input text.
    
    Args:
        text (str): The text to convert to speech
        output_dir (str): Directory to save audio files
    
    Returns:
        str: Filename (hash) if successful, None otherwise
    """
    # Create hash of the text for filename
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    filename = f"{text_hash}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # Check if file already exists (caching)
    if os.path.exists(filepath):
        return text_hash
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the Google Cloud project ID
    project = subprocess.check_output(
        ["gcloud", "config", "list", "--format=value(core.project)"]
    ).decode().strip()

    # Get an access token
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"]
    ).decode().strip()

    # Prepare the JSON payload
    payload = {
        "input": {
            "text": text
        },
        "voice": {
            "languageCode": "en-US",
            "name": "en-US-Chirp3-HD-Puck"
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16"
        }
    }

    # Set the headers
    headers = {
        "Content-Type": "application/json",
        "X-Goog-User-Project": project,
        "Authorization": f"Bearer {token}",
    }

    # Make the POST request
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raises an error if the request fails

    # The response contains a base64-encoded audio content
    audio_content = response.json().get("audioContent")

    with open(filepath, "wb") as out_file:
        out_file.write(base64.b64decode(audio_content))
    return text_hash

if __name__ == "__main__":
    text_to_speak = (
        "Pocket Flow is A 100-line minimalist LLM framework. humm. This is a test. humm."
    )
    result = synthesize_text_to_speech(text_to_speak)
    print(f"Result hash: {result}") 