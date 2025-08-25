let mediaRecorder;
let audioChunks = [];

document.getElementById('record-btn').onclick = async () => {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

    document.getElementById('record-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
};

document.getElementById('stop-btn').onclick = () => {
    mediaRecorder.stop();
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        document.getElementById('audio').src = URL.createObjectURL(audioBlob);

        // Send audio to backend for transcription
        const formData = new FormData();
        formData.append('file', audioBlob, 'audio.webm');
        const res = await fetch('http://localhost:8000/transcribe/', { method: 'POST', body: formData });
        const data = await res.json(); // Ensure this line is present to parse the JSON response
        document.getElementById('transcript').value = data.transcript;
    };
    document.getElementById('record-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
};

document.getElementById('extract-btn').onclick = async () => {
    const transcript = document.getElementById('transcript').value;
    const formData = new FormData();
    formData.append('transcript', transcript);
    const res = await fetch('http://localhost:8000/extract/', { method: 'POST', body: formData });
    const data = await res.json(); // Ensure this line is present to parse the JSON response
    document.getElementById('notes').innerText = JSON.stringify(data, null, 2);
};
