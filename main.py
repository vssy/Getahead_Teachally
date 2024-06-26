import re
import uuid
from google.cloud import storage
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai
import os
import logging
from dotenv import load_dotenv
import json
import marko
import mimetypes

load_dotenv()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'aac', 'mp3', 'wav', 'wma', 'ogg'}
app.config['GCS_BUCKET'] = 'sample_joe_recordings'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_google_credentials():
    environment = os.environ.get('ENVIRONMENT')
    if environment == 'production':
        credentials_content = os.environ.get(
            'GOOGLE_APPLICATION_CREDENTIALS_CONTENT')
        if credentials_content:
            credentials_dict = json.loads(credentials_content)
            with open('/tmp/gcp-credentials.json', 'w') as f:
                json.dump(credentials_dict, f)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/gcp-credentials.json'
        else:
            logger.error(
                'Google application credentials content not found in environment variables.')
    else:
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        else:
            logger.error(
                'Google application credentials not found in environment variables.')


setup_google_credentials()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def upload_to_gcs(file_path, filename):
    client = storage.Client()
    bucket = client.bucket(app.config['GCS_BUCKET'])
    blob = bucket.blob(filename)
    blob.upload_from_filename(file_path)
    mime_type, _ = mimetypes.guess_type(filename)
    return f'gs://{app.config["GCS_BUCKET"]}/{filename}', mime_type


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
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
                gcs_uri, mime_type = upload_to_gcs(file_path, filename)
                transcript_id = generate(gcs_uri, mime_type)
                return jsonify({'transcript_id': transcript_id})
            except Exception as e:
                logger.error(f'Error generating transcript: {e}')
                return jsonify({'error': 'Error generating transcript', 'details': str(e)}), 500
        return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        return jsonify({'error': 'Unexpected error', 'details': str(e)}), 500


@app.route('/transcriptions')
def transcriptions():
    return render_template('transcriptions.html')


# Additional API endpoint to serve transcription data

@app.route('/api/get_transcription', methods=['GET'])
def get_transcription():
    transcript_id = request.args.get('transcript_id')
    if not transcript_id:
        logger.error('No transcript ID provided')
        return jsonify({'error': 'No transcript ID provided'}), 400

    try:
        transcript_filename = f'transcripts/{transcript_id}.txt'
        if not os.path.exists(transcript_filename):
            logger.error(
                f'Transcription file not found: {transcript_filename}')
            return jsonify({'error': 'Transcription file not found'}), 404
        with open(transcript_filename, 'r') as f:
            transcript = f.read()

        # Parse markdown to HTML using marko
        transcript_html = marko.convert(transcript)

        return jsonify({'transcript': transcript_html})
    except Exception as e:
        logger.error(f'Error reading transcription file: {e}')
        return jsonify({'error': 'Error reading transcription file'}), 500


def generate(gcs_uri, mime_type):
    vertexai.init(project="wordscape-399515", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-flash-001",
        system_instruction=[
            "You are an accurate analysis AI, designed to enhance teaching quality."]
    )

    audio_part = generative_models.Part.from_uri(gcs_uri, mime_type=mime_type)

    text1 = """Transcribe and identify the speakers between the teacher and the student. Keep every grammatical mistake as it is. You should accurately transcribe so that we can analyse their performance later on. Label each segment of the dialogue as 'Teacher' or 'Student'.
    
    format example:
    <p>Teacher: Yeah, but first, how are you today? </p>
    <p>Student: Great! Super busy! </p>

    """
    # text2 = """You are an interaction analysis AI. Based on the following transcript of a classroom session, identify and categorize the interactions between the teacher and students. Label each segment of the dialogue as 'Teacher' or 'Student'. Highlight key moments such as questions, answers, and any significant pauses or interruptions. Summarize the types of interactions and provide a count of each type. Transcribe and identify the speakers between the teacher and the student, keep grammar mistakes if any."""

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
    transcript_id = str(uuid.uuid4())  # Generate a unique ID
    transcript_filename = f'transcripts/{transcript_id}.txt'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(transcript_filename), exist_ok=True)

    with open(transcript_filename, 'w') as f:
        f.write(transcript)
    logger.info(f'Transcription file saved: {transcript_filename}')
    return transcript_id


@app.route('/analysis')
def analysis():
    transcript_id = request.args.get('transcript_id')
    if not transcript_id:
        return "No transcript ID provided.", 400
    return render_template('analysis.html', transcript_id=transcript_id)


@app.route('/api/get_analysis', methods=['POST'])
def get_analysis():
    data = request.get_json()
    transcript_id = data.get('transcript_id')
    if not transcript_id:
        logger.error('No transcript ID provided')
        return jsonify({'error': 'No transcript ID provided'}), 400

    try:
        transcript_filename = f'transcripts/{transcript_id}.txt'
        logger.info(f'Checking for transcription file: {transcript_filename}')
        if not os.path.exists(transcript_filename):
            logger.error(
                f'Transcription file not found: {transcript_filename}')
            return jsonify({'error': 'Transcription file not found'}), 404

        logger.info(f'Opening transcription file: {transcript_filename}')
        with open(transcript_filename, 'r') as f:
            transcript = f.read()
            logger.debug(f'Transcript content: {transcript}')

        # Send the transcript to the AI for analysis
        analysis = perform_analysis(transcript)

        return jsonify({'analysis': json.dumps(analysis)})
    except Exception as e:
        logger.error(f'Error performing analysis: {str(e)}')
        return jsonify({'error': 'Error performing analysis'}), 500


def perform_analysis(transcript):
    vertexai.init(project="wordscape-399515", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-flash-001",
        system_instruction=[
            "You are an accurate analysis AI, designed to enhance teaching quality."]
    )

    analysis_prompt = f"""
    You are an interaction analysis AI. Based on the following transcript of a classroom session, provide constructive feedback to the teacher in the following JSON format. Each criterion should have a title, a score out of 10, and an array of feedback items.

    [
        {{
            "title": "Efficient Greetings",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Language Immersion",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Thinking Time",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Opportunities for Self-Correction",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Balance of Opinions",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Direct Error Correction",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Positive Reinforcement",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Constructive Corrections",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Enhancing Vocabulary",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }},
        {{
            "title": "Active Listening",
            "score": 0,
            "details": [
                "Provide detailed feedback here."
            ]
        }}
    ]

    Here is the transcript:
    {transcript}
    """

    logger.debug(f'Analysis prompt: {analysis_prompt}')

    generation_config = {
        "max_output_tokens": 8000,
        "temperature": 1,
        "top_p": 0.95,
    }

    # Create a Part object with the analysis prompt
    text_part = generative_models.Part.from_text(analysis_prompt)

    logger.info('Generating analysis...')
    response = model.generate_content(
        [text_part],
        generation_config=generation_config,
    )

    # Extract the generated text from the response
    analysis_text = response.text
    logger.debug(f'Generated analysis: {analysis_text}')

    # Remove the ```json ... ``` wrapping if present
    if analysis_text.startswith("```json") and analysis_text.endswith("```"):
        analysis_text = analysis_text[7:-3].strip()

    # Add a check to ensure the response is properly formatted JSON
    if analysis_text.strip().startswith('[') and analysis_text.strip().endswith(']'):
        try:
            # Attempt to parse the JSON
            analysis = json.loads(analysis_text)
        except json.JSONDecodeError as e:
            logger.error(f'JSONDecodeError: {str(e)}')
            logger.error(f'Raw analysis text: {analysis_text}')
            return {'error': 'Error decoding JSON', 'details': str(e), 'raw_text': analysis_text}
    else:
        logger.error('Generated text is not valid JSON')
        logger.error(f'Raw analysis text: {analysis_text}')
        return {'error': 'Generated text is not valid JSON', 'raw_text': analysis_text}

    return analysis


def parse_analysis(analysis_text):
    analysis = []
    sections = re.split(r'\n\n', analysis_text)

    for section in sections:
        if section.strip():
            lines = section.strip().split('\n')
            title = lines[0].strip()
            details = '\n'.join(lines[1:]).strip()
            analysis.append({
                'title': title,
                'details': details
            })

    return analysis


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists('transcripts'):
        os.makedirs('transcripts')
    app.run(debug=True)
