import os
import uuid
import time
import json
import urllib.request
import logging
import wave
import pyaudio
import google.generativeai as genai
import requests
import boto3
import threading
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from azure.cognitiveservices.speech import AudioConfig, SpeechConfig, SpeechSynthesizer, ResultReason

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multilingual_translator")

# AWS configuration
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
bucket_name = os.getenv("S3_BUCKET_NAME")

# Azure Translator configuration
azure_translator_api_key = os.getenv("AZURE_TRANSLATOR_API_KEY")
azure_translator_region = os.getenv("AZURE_TRANSLATOR_REGION")
azure_translator_endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")

# Azure Speech Services configuration
azure_speech_api_key = os.getenv("AZURE_SPEECH_API_KEY")
azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
azure_speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize S3 and Transcribe clients
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)
transcribe_client = boto3.client(
    "transcribe",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_OUTPUT_FOLDER'] = 'static/audio'  # New folder for audio output
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_OUTPUT_FOLDER'], exist_ok=True)

# Global variables to store state
current_session_id = None
current_english_text = None
is_recording = False
audio_frames = []

# List of supported languages for AWS Transcribe
SUPPORTED_INPUT_LANGUAGES = {
    "te-IN": "Telugu",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "en-US": "English",
    "ml-IN": "Malayalam",
    "kn-IN": "Kannada"
}

# List of supported output languages for Azure Translator and TTS
SUPPORTED_OUTPUT_LANGUAGES = {
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "pa": "Punjabi",
    "en": "English"
}

# Azure TTS voice mapping
AZURE_TTS_VOICES = {
    "ta": "ta-IN-Pallavi",
    "te": "te-IN-Chitra",
    "hi": "hi-IN-Kalpana",
    "ml": "ml-IN-Sobhana",
    "kn": "kn-IN-Sapna",
    "mr": "mr-IN-Aarohi",
    "gu": "gu-IN-Niranjan",
    "bn": "bn-IN-Bashkar",
    "pa": "pa-IN-Baldev",
    "en": "en-US-Aria"
}

def validate_wav_file(file_path):
    """Validates if the file is a valid WAV audio file."""
    try:
        with wave.open(file_path, "rb") as wav_file:
            logger.info(
                f"Valid WAV file - Channels: {wav_file.getnchannels()}, Sample Rate: {wav_file.getframerate()}, Frames: {wav_file.getnframes()}"
            )
        return True
    except wave.Error as e:
        logger.error(f"Invalid WAV file: {str(e)}")
        return False

def upload_to_s3(file_path, bucket, object_name):
    """Uploads a file to an S3 bucket."""
    try:
        logger.info(f"Uploading {file_path} to S3...")
        with open(file_path, "rb") as file_data:
            s3_client.upload_fileobj(file_data, bucket, object_name)
        logger.info("File uploaded to S3 successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        return False

def transcribe_audio(job_name, file_uri, language_code="te-IN"):
    """Transcribes an audio file using Amazon Transcribe."""
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": file_uri},
            MediaFormat="wav",
            LanguageCode=language_code,
        )
        logger.info(f"Started transcription job: {job_name}")

        while True:
            job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            status = job["TranscriptionJob"]["TranscriptionJobStatus"]
            logger.info(f"Job status: {status}")

            if status == "COMPLETED":
                transcript_uri = job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                response = urllib.request.urlopen(transcript_uri, timeout=30)
                data = json.loads(response.read())
                
                logger.info(f"Full transcript data: {json.dumps(data, indent=2)}")
                
                if "results" in data and "transcripts" in data["results"] and data["results"]["transcripts"]:
                    text = data["results"]["transcripts"][0]["transcript"]
                    if text:
                        logger.info(f"Transcription completed: {text}")
                        return text
                    else:
                        logger.error("Transcription returned empty text.")
                        return None
                else:
                    logger.error(f"Unexpected transcript format: {data}")
                    return None

            elif status == "FAILED":
                error_reason = job["TranscriptionJob"].get("FailureReason", "Unknown reason")
                logger.error(f"Transcription failed: {error_reason}")
                return None

            time.sleep(5)

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None

def correct_and_translate(source_text, source_lang):
    """Translates source text to English using Gemini API with context awareness."""
    try:
        with open("./Polyhouse Ontology.ttl", "r") as f:
            onto = f.read()
        
        prompt = f"""
        I'll give you an ontology file and a {SUPPORTED_INPUT_LANGUAGES.get(source_lang, 'unknown language')} text. The {SUPPORTED_INPUT_LANGUAGES.get(source_lang, 'unknown language')} text might have errors related to specific terms in the ontology.

        First, analyze the ontology file to understand its domain and key terms.
        Then, examine the {SUPPORTED_INPUT_LANGUAGES.get(source_lang, 'unknown language')} text for words that might be misused or misspelled based on context.
        Finally, return ONLY the corrected English translation. Don't include any explanations or additional text.

        Ontology file:
        {onto}

        Tamil text: {source_text}
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            translated_text = response.text.strip()
            logging.info(f"Translated to English: {translated_text}")
            return translated_text
        else:
            logging.error("Error: No valid response from the model.")
            return "Error: No valid translation received."
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return f"Error: {str(e)}"

def translate_to_target_language(english_text, target_lang):
    """Translates English text to target language using Azure Translator."""
    try:
        endpoint = "https://api.cognitive.microsofttranslator.com"
        path = "/translate"
        constructed_url = endpoint + path
        
        params = {
            'api-version': '3.0',
            'from': 'en',
            'to': target_lang
        }
        
        headers = {
            'Ocp-Apim-Subscription-Key': azure_translator_api_key,
            'Ocp-Apim-Subscription-Region': azure_translator_region,
            'Content-type': 'application/json'
        }
        
        body = [{'text': english_text}]
        
        response = requests.post(constructed_url, params=params, headers=headers, json=body)
        
        if response.status_code == 200:
            translated_text = response.json()[0]["translations"][0]["text"]
            logging.info(f"Translated to {target_lang}: {translated_text}")
            return translated_text
        else:
            logging.error(f"Azure Translation failed: {response.text}")
            return "Error: Translation failed."
            
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return f"Error: {str(e)}"

def text_to_speech(text, target_lang):
    """Converts text to speech using Azure Speech Services."""
    try:
        # Configure Azure Speech
        speech_config = SpeechConfig(subscription=azure_speech_api_key, region=azure_speech_region)
        speech_config.speech_synthesis_voice_name = AZURE_TTS_VOICES.get(target_lang, "en-US-Aria")
        
        # Generate unique filename
        audio_filename = f"output_{uuid.uuid4()}.wav"
        audio_path = os.path.join(app.config['AUDIO_OUTPUT_FOLDER'], audio_filename)
        
        # Set audio output
        audio_config = AudioConfig(filename=audio_path)
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # Synthesize speech
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            logger.info(f"Speech synthesized and saved to {audio_path}")
            return f"/static/audio/{audio_filename}"
        else:
            logger.error(f"Speech synthesis failed: {result.reason}")
            if result.reason == ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(f"Cancellation reason: {cancellation_details.reason}")
                logger.error(f"Error details: {cancellation_details.error_details}")
            return None
            
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return None

def process_audio(audio_path, input_language):
    """Processes audio file: validates, uploads to S3, transcribes, and translates."""
    global current_session_id, current_english_text

    try:
        if not validate_wav_file(audio_path):
            logger.error("Invalid WAV file. Exiting.")
            return {"status": "error", "message": "Invalid WAV file"}

        session_id = str(uuid.uuid4())
        current_session_id = session_id
        s3_file_name = f"{session_id}.wav"
        s3_uri = f"s3://{bucket_name}/{s3_file_name}"

        if not upload_to_s3(audio_path, bucket_name, s3_file_name):
            logger.error("Failed to upload to S3. Exiting.")
            return {"status": "error", "message": "Failed to upload to S3"}

        source_text = transcribe_audio(session_id, s3_uri, input_language)
        if not source_text or not source_text.strip():
            logger.error("Transcription failed.")
            return {"status": "error", "message": "Transcription failed"}

        english_text = correct_and_translate(source_text, input_language)
        if not english_text:
            logger.error("Translation to English failed.")
            return {"status": "error", "message": "Translation to English failed"}

        current_english_text = english_text

        return {
            "status": "success",
            "source_text": source_text,
            "english_text": english_text
        }

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', 
                          input_languages=SUPPORTED_INPUT_LANGUAGES, 
                          output_languages=SUPPORTED_OUTPUT_LANGUAGES)

@app.route('/start-recording', methods=['POST'])
def start_recording():
    global is_recording, audio_frames

    try:
        is_recording = True
        audio_frames = []

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
        )
        logger.info("Recording started...")

        while is_recording:
            data = stream.read(CHUNK)
            audio_frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        logger.info("Recording stopped.")

        return jsonify({"status": "success", "message": "Recording stopped successfully."})

    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@app.route('/stop-recording', methods=['POST'])
def stop_recording():
    global is_recording, audio_frames

    try:
        is_recording = False

        output_file = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.wav")
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(audio_frames))

        logger.info(f"Audio saved to {output_file}")

        input_language = request.form.get('input_language', 'te-IN')
        result = process_audio(output_file, input_language)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error stopping recording: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@app.route('/translate-to-language', methods=['POST'])
def translate_to_language():
    try:
        data = request.get_json()
        target_language = data.get('target_language')
        
        if not current_english_text:
            return jsonify({"status": "error", "message": "No English text available for translation"})
            
        if not target_language:
            return jsonify({"status": "error", "message": "No target language specified"})
            
        translated_text = translate_to_target_language(current_english_text, target_language)
        
        if translated_text:
            return jsonify({
                "status": "success", 
                "translated_text": translated_text,
                "target_language": SUPPORTED_OUTPUT_LANGUAGES.get(target_language, target_language)
            })
        else:
            return jsonify({"status": "error", "message": "Translation failed"})
            
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech_route():
    try:
        data = request.get_json()
        text = data.get('text')
        target_language = data.get('target_language')
        
        if not text or not target_language:
            return jsonify({"status": "error", "message": "Text and target language are required"})
            
        audio_url = text_to_speech(text, target_language)
        
        if audio_url:
            return jsonify({
                "status": "success",
                "audio_url": audio_url
            })
        else:
            return jsonify({"status": "error", "message": "Text-to-speech conversion failed"})
            
    except Exception as e:
        logger.error(f"TTS route error: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5050)  # Changed port to 5050