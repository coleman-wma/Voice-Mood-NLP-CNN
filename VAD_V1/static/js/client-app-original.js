let rec = null;
let audioStream = null;

const recordButton = document.getElementById("recordButton");
const transcribeButton = document.getElementById("transcribeButton");

recordButton.addEventListener("click", startRecording);
transcribeButton.addEventListener("click", transcribeText);

function startRecording() {

    let constraints = { audio: true, video:false }

    recordButton.disabled = true;
    transcribeButton.disabled = false;

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        const audioContext = new window.AudioContext();
        audioStream = stream;
        const input = audioContext.createMediaStreamSource(stream);
        rec = new Recorder(input, { numChannels:1 })
        rec.record()
    }).catch(function(err) {
        recordButton.disabled = false;
        transcribeButton.disabled = true;
    });
}

function transcribeText() {
    transcribeButton.disabled = true;
    recordButton.disabled = false;
    rec.stop();
    audioStream.getAudioTracks()[0].stop();
    rec.exportWAV(uploadSoundData);
}

function uploadSoundData(blob) {
    let filename = new Date().toISOString();
    let xhr = new XMLHttpRequest();
    xhr.onload = function(e) {
        if(this.readyState === 4) {
            document.getElementById("output").innerHTML = `<br><br><strong>Result: </strong>${e.target.responseText}`
        }
    };
    let formData = new FormData();
    formData.append("audio_data", blob, filename);
    xhr.open("POST", "/upload_sound", true);
    xhr.send(formData);
}