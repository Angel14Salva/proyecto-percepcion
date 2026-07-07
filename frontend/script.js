const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const result = document.getElementById("result");

const API = "https://clasificador-residuos-backend.onrender.com/predict";

const MIN_DELAY_MS = 0;

navigator.mediaDevices
    .getUserMedia({ video: { width: 640, height: 480 } })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        console.error(err);
    });

video.addEventListener("loadedmetadata", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
});

const tempCanvas = document.createElement("canvas");
const tempCtx = tempCanvas.getContext("2d");

function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detections.length) {
        result.innerHTML = "Buscando residuo...";
        return;
    }

    result.innerHTML = "";

    detections.forEach((det) => {
        const x1 = det.bbox[0];
        const y1 = det.bbox[1];
        const x2 = det.bbox[2];
        const y2 = det.bbox[3];

        ctx.strokeStyle = "lime";
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = "lime";
        ctx.font = "22px Arial";
        ctx.fillText(`${det.class} ${(det.confidence * 100).toFixed(1)}%`, x1, y1 - 10);

        result.innerHTML += `♻️ <b>${det.class}</b> (${(det.confidence * 100).toFixed(1)}%)<br>`;
    });
}

async function detectLoop() {
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    tempCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    tempCanvas.toBlob(
        async (blob) => {
            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");

            try {
                const response = await fetch(API, { method: "POST", body: formData });
                const data = await response.json();
                drawDetections(data.detections || []);
            } catch (error) {
                console.error(error);
            }

            if (MIN_DELAY_MS > 0) {
                setTimeout(detectLoop, MIN_DELAY_MS);
            } else {
                detectLoop();
            }
        },
        "image/jpeg",
        0.8
    );
}

video.addEventListener("loadedmetadata", () => detectLoop(), { once: true });
