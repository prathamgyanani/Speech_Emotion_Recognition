const audioRecorder = {
  async recordAudio() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    let localAudioChunks = [];

    mediaRecorder.start();
    mediaRecorder.ondataavailable = (event) => {
      if (typeof event.data === "undefined") return;
      if (event.data.size === 0) return;
      localAudioChunks.push(event.data);
    };

    mediaRecorder.stop();
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(localAudioChunks, { type: "audio/webm" });
      const audioURL = URL.createObjectURL(audioBlob);
      const userInput = document.getElementById("user-input").value;
      // Send the recorded audio and user input to the Flask backend for prediction
      fetch("/predict", {
        method: "POST",
        body: new FormData().append("audio", audioBlob).append("user_input", userInput),
      })
        .then((response) => response.json())
        .then((prediction) => {
          // Display the prediction on the webpage
          document.getElementById("prediction").textContent = prediction;
        });
    };
  },
};

document.getElementById("record-button").addEventListener("click", () => {
  audioRecorder.recordAudio();
});