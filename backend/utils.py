import requests
import openai

def deepgram_transcribe(audio_data, mimetype, api_key):
    headers = {"Authorization": f"Token {api_key}"}
    response = requests.post(
        "https://api.deepgram.com/v1/listen",
        headers=headers,
        data=audio_data,
        params={"punctuate": True, "model": "general"},
    )
    return response.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")

def extract_medical_notes(transcript, api_key):
    # Use OpenAI API to generate structured medical notes
    openai.api_key = api_key
    prompt = f"""You are a clinical documentation AI. Extract and structure this consultation into relevant fields: Symptoms, Diagnosis, Medications, Treatment Plan, Follow-up Steps.

    Conversation Transcript:
    {transcript}

    Output as JSON."""
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text
