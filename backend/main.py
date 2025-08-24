from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
import requests
import os
import utils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API keys and config from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Send audio to Deepgram for transcription
    audio_data = await file.read()
    transcript = utils.deepgram_transcribe(audio_data, file.content_type, DEEPGRAM_API_KEY)
    return {"transcript": transcript}

@app.post("/extract/")
def extract_clinical_info(transcript: str = Form(...)):
    # Use OpenAI or similar model to process transcript
    structured_data = utils.extract_medical_notes(transcript, OPENAI_API_KEY)
    return structured_data

# Add endpoints for EHR export, summaries, analytics, etc.

