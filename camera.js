const video = document.querySelector('.input_video');
const canvas = document.querySelector('.output_canvas');
const ctx = canvas.getContext('2d');
const outputText = document.getElementById('output');
const subtitlesBox = document.getElementById('subtitles');
const startButton = document.getElementById('startCameraBtn');

let helloFrames = [];
let lastGesture = "";
let lastTimestamp = 0;

// تحميل بيانات الإشارة "أهلاً"
fetch('hello_reference.json')
  .then(res => res.json())
  .then(data => { helloFrames = data; });

// دالة تطبيع النقاط
function normalizeLandmarks(landmarks) {
  const points = landmarks.map(lm => [lm.x, lm.y]);
  const center = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0, 0])
                        .map(v => v / points.length);
  const norm = Math.sqrt(points.reduce((sum, [x, y]) => sum + (x - center[0])**2 + (y - center[1])**2, 0));
  return points.map(([x, y]) => [(x - center[0]) / (norm + 1e-6), (y - center[1]) / (norm + 1e-6)]).flat();
}

// مقارنة بين الإشارة الحالية والمراجع
function isHello(current, references, threshold = 0.65) {
  for (const ref of references) {
    if (current.length !== ref.length) continue;
    let diff = 0;
    for (let i = 0; i < current.length; i++) {
      diff += (current[i] - ref[i]) ** 2;
    }
    diff = Math.sqrt(diff);
    if (1 - diff > threshold) return true;
  }
  return false;
}

// إعداد Mediapipe Hands
const hands = new Hands({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.6
});

// النتائج في كل فريم
hands.onResults(results => {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  let currentText = "لا توجد يد 🙌";

  if (results.multiHandLandmarks?.length) {
    const landmarks = results.multiHandLandmarks[0];
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00' });
    drawLandmarks(ctx, landmarks, { color: '#FF0000' });

    const normalized = normalizeLandmarks(landmarks);
    if (isHello(normalized, helloFrames)) {
      currentText = "أهلاً وسهلاً 👋";

      const now = Date.now();
      if (lastGesture !== currentText || now - lastTimestamp > 1500) {
        const span = document.createElement('span');
        span.textContent = currentText;
        span.style.fontSize = '20px';
        span.style.color = '#ffd700';
        span.style.padding = '6px 12px';
        span.style.background = '#1a1a2e';
        span.style.borderRadius = '10px';
        span.style.whiteSpace = 'nowrap';
        subtitlesBox.appendChild(span);

        lastGesture = currentText;
        lastTimestamp = now;
      }
    } else {
      currentText = "لا أفهم الإشارة 🤷‍♂️";
    }
  }

  outputText.textContent = currentText;
  ctx.restore();
});

// بدء الكاميرا عند الضغط على الزر
startButton.addEventListener('click', () => {
  startButton.style.display = 'none';
  const camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480
  });
  camera.start();
});
