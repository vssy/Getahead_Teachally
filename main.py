import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {
    'wav', 'mp3', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'aac', 'ogg', 'wma'}
app.config['GCS_BUCKET'] = 'sample_joe_recordings'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def upload_to_gcs(file_path, filename):
    client = storage.Client()
    bucket = client.bucket(app.config['GCS_BUCKET'])
    blob = bucket.blob(filename)
    blob.upload_from_filename(file_path)
    return f'gs://{app.config["GCS_BUCKET"]}/{filename}'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error('No file part in the request')
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            gcs_uri = upload_to_gcs(file_path, filename)
            transcript = generate(gcs_uri)
            return jsonify({'transcript': transcript})
        except Exception as e:
            logger.error(f'Error generating transcript: {e}')
            return jsonify({'error': 'Error generating transcript'}), 500
    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/transcriptions')
def transcriptions():
    return render_template('transcriptions.html')


# Additional API endpoint to serve transcription data

@app.route('/api/get_transcription', methods=['GET'])
def get_transcription():
    # Mock response for now
    transcript = "Sample transcription text."
    return jsonify({'transcript': transcript})


def generate(gcs_uri):
    vertexai.init(project="wordscape-399515", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-pro-001",
        system_instruction=["You are an interaction analysis AI. Based on the following transcript of a classroom session, identify and categorize the interactions between the teacher and students. Label each segment of the dialogue as 'Teacher' or 'Student'. Highlight key moments such as questions, answers, and any significant pauses or interruptions. Summarize the types of interactions and provide a count of each type. Transcribe and identify the speakers between the teacher and the student, keep grammar mistakes if any."]
    )

    audio_part = generative_models.Part.from_uri(
        gcs_uri, mime_type="audio/wav")

    text1 = """You are an interaction analysis AI. Based on the following transcript of a classroom session, identify and categorize the interactions between the teacher and students. Label each segment of the dialogue as 'Teacher' or 'Student'. Highlight key moments such as questions, answers, and any significant pauses or interruptions. Summarize the types of interactions and provide a count of each type. Transcribe and identify the speakers between the teacher and the student, keep grammar mistakes if any."""

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    responses = model.generate_content(
        [audio_part, text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    transcript = "".join(response.text for response in responses)
    return transcript


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
