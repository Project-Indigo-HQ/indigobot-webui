document.addEventListener("DOMContentLoaded", function () {
    const micButton = document.createElement("button");
    micButton.innerText = "🎤";
    micButton.style.fontSize = "24px";
    micButton.style.cursor = "pointer";
    micButton.style.border = "none";
    micButton.style.background = "none";

    document.querySelector(".cl-input").prepend(micButton);

    let mediaRecorder;
    let audioChunks = [];

    micButton.addEventListener("click", async function () {
        if (!mediaRecorder) {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                const formData = new FormData();
                formData.append("audio", audioBlob, "speech.wav");

                const response = await fetch("/stt", {
                    method: "POST",
                    body: formData,
                });

                const result = await response.json();
                document.querySelector("textarea").value = result.text || "Could not recognize speech.";
                document.querySelector("textarea").dispatchEvent(new Event("input"));
                audioChunks = [];
            };
        }

        if (mediaRecorder.state === "inactive") {
            mediaRecorder.start();
            micButton.innerText = "🛑 Recording...";
        } else {
            mediaRecorder.stop();
            micButton.innerText = "🎤";
        }
    });
});