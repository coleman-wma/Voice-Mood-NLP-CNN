var rec = null;
var audioStream = null;

const recordButton = document.getElementById("recordButton");
const transcribeButton = document.getElementById("transcribeButton");
const predictButton = document.getElementById("predictButton");

recordButton.addEventListener("click", startRecording);
transcribeButton.addEventListener("click", transcribeText);
predictButton.addEventListener("click", predictNow);

function startRecording() {

    var constraints = { audio: true, video:false }

    recordButton.disabled = true;
    transcribeButton.disabled = false;
    predictButton.disabled = true;

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
    recordButton.disabled = true;
    predictButton.disabled = false;
    rec.stop();
    audioStream.getAudioTracks()[0].stop();
    rec.exportWAV(sendData);
}

// This from: https://github.com/mattdiamond/Recorderjs/issues/188
// How to post to flask/python
function sendData(blob) {

	// sends data to flask url /upload_sound as a post with data blob - in format for wav file, hopefully. it is a promise
	fetch("/upload_sound", {
	method: "post",
	body: blob
	});
}

function predictNow() {
    transcribeButton.disabled = true;
    recordButton.disabled = false;
    predictButton.disabled = true;
    //window.open=("result.html");
}