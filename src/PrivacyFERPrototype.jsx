import { useEffect, useMemo, useRef, useState } from "react";

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const MP_VERSION = "0.10.3";
const WASM_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;

const CONNECTIONS = [
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
  [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
  [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
  [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
  [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
  [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154],
  [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159],
  [159, 158], [158, 157], [157, 173], [173, 133],
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381],
  [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386],
  [386, 385], [385, 384], [384, 398], [398, 362],
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314],
  [314, 405], [405, 321], [321, 375], [375, 291],
  [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317],
  [317, 402], [402, 318], [318, 324], [324, 308],
  [70, 63], [63, 105], [105, 66], [66, 107],
  [336, 296], [296, 334], [334, 293], [293, 300],
  [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19],
  [19, 94], [94, 2], [2, 164], [164, 0], [0, 11], [11, 12], [12, 13], [13, 14],
];

const FEATURE_POINTS = {
  eyes: [33, 133, 362, 263, 159, 145, 386, 374],
  nose: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
  mouth: [61, 291, 0, 17, 13, 14, 78, 308],
  jaw: [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454],
};

function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max);
}

function getColor(index) {
  if (FEATURE_POINTS.eyes.includes(index)) return "#38bdf8";
  if (FEATURE_POINTS.nose.includes(index)) return "#22c55e";
  if (FEATURE_POINTS.mouth.includes(index)) return "#fb7185";
  if (FEATURE_POINTS.jaw.includes(index)) return "#a78bfa";
  return "#facc15";
}

function shiftPoint(p, index, noise) {
  if (!p) return p;
  const dx = ((index % 5) - 2) * noise * 0.0011;
  const dy = (((index + 2) % 5) - 2) * noise * 0.0011;
  return {
    x: clamp(p.x + dx, 0, 1),
    y: clamp(p.y + dy, 0, 1),
    z: p.z || 0,
  };
}

function Pill({ children, active, danger }) {
  return (
    <span
      style={{
        padding: "9px 13px",
        borderRadius: 999,
        fontSize: 13,
        fontWeight: 800,
        border: `1px solid ${danger ? "#fecaca" : active ? "#bfdbfe" : "#e2e8f0"}`,
        background: danger ? "#fff1f2" : active ? "#eff6ff" : "#f8fafc",
        color: danger ? "#be123c" : active ? "#1d4ed8" : "#475569",
      }}
    >
      {children}
    </span>
  );
}

function Metric({ label, value, note }) {
  return (
    <div
      style={{
        padding: 16,
        borderRadius: 20,
        border: "1px solid #e2e8f0",
        background: "rgba(255,255,255,0.9)",
      }}
    >
      <div style={{ color: "#64748b", fontSize: 13, fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 900, marginTop: 6 }}>{value}</div>
      <div style={{ color: "#64748b", fontSize: 13, lineHeight: 1.5, marginTop: 4 }}>{note}</div>
    </div>
  );
}

export default function PrivacyFERPrototype() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const graphRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(-1);

  const [loading, setLoading] = useState(true);
  const [modelReady, setModelReady] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(true);
  const [noise, setNoise] = useState(0);
  const [landmarkCount, setLandmarkCount] = useState(0);
  const [graphQuality, setGraphQuality] = useState(100);
  const [error, setError] = useState("");
  const [frame, setFrame] = useState({ width: 960, height: 720 });

  const exposure = privacyMode ? "Reduced" : "Visible";

  const summary = useMemo(() => {
    if (!cameraOn) return "Camera inactive";
    if (!faceDetected) return "Scanning for face";
    return "Live graph generated";
  }, [cameraOn, faceDetected]);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      try {
        setLoading(true);
        const vision = await import(
          `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/+esm`
        );

        const { FaceLandmarker, FilesetResolver } = vision;
        const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);

        const landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_PATH,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setModelReady(true);
          setError("");
        }
      } catch (e) {
        console.error(e);
        setError("Model could not load. Refresh once or use the poster QR as backup.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadModel();

    return () => {
      cancelled = true;
      stopCamera();
    };
  }, []);

  function stopCamera() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
    setFaceDetected(false);
    setLandmarkCount(0);
    clearCanvas(overlayRef);
    clearCanvas(graphRef, "Enable camera to generate graph");
  }

  function clearCanvas(ref, text = "") {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (text) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#94a3b8";
      ctx.font = "700 18px Arial";
      ctx.textAlign = "center";
      ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    }
  }

  async function startCamera() {
    try {
      setError("");

      if (!modelReady) {
        setError("Model is still loading. Try again in a few seconds.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const width = video.videoWidth || 960;
      const height = video.videoHeight || 720;
      setFrame({ width, height });

      [overlayRef.current, graphRef.current].forEach((canvas) => {
        if (canvas) {
          canvas.width = width;
          canvas.height = height;
        }
      });

      setCameraOn(true);
      lastTimeRef.current = -1;
      loop();
    } catch (e) {
      console.error(e);
      setError("Camera access failed. Allow browser camera permission.");
    }
  }

  function drawOverlay(face) {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width, height } = frame;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    if (!face) return;

    ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
    ctx.lineWidth = 1.4;

    CONNECTIONS.forEach(([a, b]) => {
      if (!face[a] || !face[b]) return;
      ctx.beginPath();
      ctx.moveTo((1 - face[a].x) * width, face[a].y * height);
      ctx.lineTo((1 - face[b].x) * width, face[b].y * height);
      ctx.stroke();
    });

    ctx.fillStyle = "rgba(255,255,255,0.95)";
    for (let i = 0; i < face.length; i += 8) {
      const p = face[i];
      ctx.beginPath();
      ctx.arc((1 - p.x) * width, p.y * height, 2.1, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawGraph(face) {
    const canvas = graphRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width, height } = frame;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#020617");
    gradient.addColorStop(0.55, "#0f172a");
    gradient.addColorStop(1, "#111827");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(148,163,184,0.12)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 55) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 55) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!face) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "800 22px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for live facial structure", width / 2, height / 2);
      return;
    }

    const cx = width / 2;
    const cy = height / 2;
    const scale = 1.15;

    const xs = face.map((p) => p.x);
    const ys = face.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    function mapPoint(p, index) {
      const q = shiftPoint(p, index, noise);
      const nx = (q.x - minX) / (maxX - minX || 1);
      const ny = (q.y - minY) / (maxY - minY || 1);
      return {
        x: cx + (0.5 - nx) * width * 0.42 * scale,
        y: cy + (ny - 0.5) * height * 0.72 * scale,
        z: q.z,
      };
    }

    ctx.save();
    ctx.shadowColor = "rgba(59,130,246,0.45)";
    ctx.shadowBlur = 8;
    ctx.strokeStyle = "rgba(96,165,250,0.6)";
    ctx.lineWidth = 1.5;

    CONNECTIONS.forEach(([a, b]) => {
      if (!face[a] || !face[b]) return;
      const p1 = mapPoint(face[a], a);
      const p2 = mapPoint(face[b], b);

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    });
    ctx.restore();

    for (let i = 0; i < face.length; i += 2) {
      const p = mapPoint(face[i], i);
      ctx.fillStyle = getColor(i);
      ctx.beginPath();
      ctx.arc(p.x, p.y, FEATURE_POINTS.jaw.includes(i) ? 2.2 : 2.8, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "rgba(15,23,42,0.75)";
    ctx.fillRect(22, 22, 310, 44);
    ctx.fillStyle = "#e0f2fe";
    ctx.font = "800 16px Arial";
    ctx.textAlign = "left";
    ctx.fillText("Graph representation: identity reduced", 38, 50);

    const legend = [
      ["Eyes", "#38bdf8"],
      ["Nose", "#22c55e"],
      ["Mouth", "#fb7185"],
      ["Jawline", "#a78bfa"],
      ["Other", "#facc15"],
    ];

    legend.forEach(([label, color], idx) => {
      const x = 38 + idx * 105;
      const y = height - 34;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#cbd5e1";
      ctx.font = "700 13px Arial";
      ctx.fillText(label, x + 12, y + 4);
    });
  }

  function loop() {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !landmarker) return;

    if (video.readyState >= 2 && video.currentTime !== lastTimeRef.current) {
      lastTimeRef.current = video.currentTime;

      const result = landmarker.detectForVideo(video, performance.now());
      const face = result.faceLandmarks?.[0];

      setFaceDetected(Boolean(face));
      setLandmarkCount(face ? face.length : 0);
      setGraphQuality(face ? Math.max(55, 100 - noise * 3) : 0);

      drawOverlay(face);
      drawGraph(face);
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  return (
    <div style={{ minHeight: "100vh", background: "#eef4ff", color: "#0f172a", fontFamily: "Inter, system-ui, Arial" }}>
      <div style={{ maxWidth: 1440, margin: "0 auto", padding: 24 }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 18 }}>
          <Pill active={modelReady}>{loading ? "Loading model" : "Model ready"}</Pill>
          <Pill active={cameraOn}>{cameraOn ? "Camera active" : "Camera inactive"}</Pill>
          <Pill active={faceDetected}>{faceDetected ? "Face detected" : "No face"}</Pill>
          <Pill active={privacyMode} danger={!privacyMode}>{privacyMode ? "Privacy mode on" : "Raw visible"}</Pill>
        </div>

        <section
          style={{
            background: "linear-gradient(135deg,#ffffff,#f8fbff)",
            border: "1px solid #dbeafe",
            borderRadius: 30,
            padding: 28,
            marginBottom: 22,
            boxShadow: "0 18px 45px rgba(15,23,42,0.08)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", gap: 24, flexWrap: "wrap" }}>
            <div style={{ maxWidth: 850 }}>
              <div style={{ display: "inline-block", padding: "9px 14px", borderRadius: 999, background: "#dbeafe", color: "#1d4ed8", fontWeight: 900, marginBottom: 14 }}>
                Advanced live research prototype
              </div>
              <h1 style={{ fontSize: "clamp(2.4rem,5vw,4.6rem)", lineHeight: 1, margin: 0 }}>
                Privacy-Preserving Facial Graph Analysis
              </h1>
              <p style={{ color: "#475569", fontSize: 19, lineHeight: 1.7, marginTop: 18 }}>
                Live facial input is converted into a graph-based structural representation,
                reducing dependence on identity-rich raw imagery while preserving useful facial structure.
              </p>
            </div>

            <div style={{ minWidth: 280, background: "#f8fafc", border: "1px solid #dbeafe", borderRadius: 24, padding: 20 }}>
              <div style={{ color: "#64748b", fontWeight: 800 }}>System summary</div>
              <div style={{ fontSize: 24, fontWeight: 950, marginTop: 8 }}>{summary}</div>
              <div style={{ color: "#475569", lineHeight: 1.8, marginTop: 10 }}>
                Landmarks: {landmarkCount}<br />
                Graph quality: {graphQuality}%<br />
                Identity exposure: {exposure}
              </div>
            </div>
          </div>
        </section>

        <main style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 22 }}>
          <aside style={{ background: "#ffffff", border: "1px solid #dbeafe", borderRadius: 26, padding: 22, boxShadow: "0 14px 35px rgba(15,23,42,0.06)" }}>
            <h2 style={{ margin: "0 0 16px", fontSize: 20 }}>Controls</h2>

            <button onClick={cameraOn ? stopCamera : startCamera} style={buttonStyle("#2563eb", "#ffffff")}>
              {cameraOn ? "Stop camera" : "Enable camera"}
            </button>

            <button onClick={() => setPrivacyMode((p) => !p)} style={{ ...buttonStyle("#ffffff", "#0f172a"), border: "1px solid #cbd5e1", marginTop: 12 }}>
              {privacyMode ? "Show raw input" : "Hide raw input"}
            </button>

            <div style={{ marginTop: 24 }}>
              <label style={{ fontWeight: 900 }}>Landmark noise simulation: {noise}</label>
              <input type="range" min="0" max="12" value={noise} onChange={(e) => setNoise(Number(e.target.value))} style={{ width: "100%", marginTop: 12 }} />
            </div>

            <div style={{ marginTop: 24, padding: 16, borderRadius: 18, background: "#f8fafc", border: "1px solid #e2e8f0", lineHeight: 1.7, color: "#475569" }}>
              <b>Pipeline</b><br />
              Raw face → landmark detection → graph abstraction → reduced identity exposure
            </div>

            {error && <div style={{ marginTop: 16, color: "#9a3412", background: "#fff7ed", border: "1px solid #fdba74", borderRadius: 16, padding: 14 }}>{error}</div>}
          </aside>

          <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22 }}>
            <Panel title="Raw input" subtitle="Live webcam feed with landmark overlay">
              <div style={mediaBox}>
                <video ref={videoRef} playsInline muted autoPlay style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)", filter: privacyMode ? "blur(12px) brightness(0.7)" : "none", opacity: cameraOn ? 1 : 0.12 }} />
                <canvas ref={overlayRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
                {!cameraOn && <CenterText text="Camera inactive" />}
              </div>
            </Panel>

            <Panel title="Graph abstraction" subtitle="Identity-reduced structural representation">
              <div style={mediaBox}>
                <canvas ref={graphRef} style={{ width: "100%", height: "100%", display: "block" }} />
              </div>
            </Panel>
          </section>
        </main>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginTop: 22 }}>
          <Metric label="Representation" value="Graph" note="Structure over raw identity appearance." />
          <Metric label="Landmarks" value={landmarkCount} note="Detected facial structure points." />
          <Metric label="Graph quality" value={`${graphQuality}%`} note="Stability under simulated noise." />
          <Metric label="Exposure" value={exposure} note="Raw input visibility level." />
        </div>
      </div>
    </div>
  );
}

function Panel({ title, subtitle, children }) {
  return (
    <div style={{ background: "#ffffff", border: "1px solid #dbeafe", borderRadius: 26, padding: 18, boxShadow: "0 14px 35px rgba(15,23,42,0.06)" }}>
      <h2 style={{ margin: 0, fontSize: 20 }}>{title}</h2>
      <p style={{ margin: "6px 0 14px", color: "#64748b", fontWeight: 600 }}>{subtitle}</p>
      {children}
    </div>
  );
}

function CenterText({ text }) {
  return (
    <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#cbd5e1", fontWeight: 900, fontSize: 20 }}>
      {text}
    </div>
  );
}

function buttonStyle(bg, color) {
  return {
    width: "100%",
    padding: "15px 16px",
    borderRadius: 16,
    border: "none",
    background: bg,
    color,
    cursor: "pointer",
    fontWeight: 900,
    fontSize: 16,
  };
}

const mediaBox = {
  position: "relative",
  borderRadius: 22,
  overflow: "hidden",
  border: "1px solid #dbeafe",
  background: "#0f172a",
  aspectRatio: "4 / 3",
};
