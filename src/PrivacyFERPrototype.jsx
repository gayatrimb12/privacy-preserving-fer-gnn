import { useEffect, useMemo, useRef, useState } from "react";

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const MP_VERSION = "0.10.3";
const WASM_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;

const OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];
const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33];
const RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263];
const MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61];
const NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2];

const GROUPS = [
  { name: "face boundary", points: OUTLINE, color: "#60a5fa" },
  { name: "eyes", points: LEFT_EYE, color: "#22c55e" },
  { name: "eyes", points: RIGHT_EYE, color: "#22c55e" },
  { name: "nose", points: NOSE, color: "#facc15" },
  { name: "mouth", points: MOUTH, color: "#fb7185" },
];

const SELECTED_POINTS = Array.from(
  new Set([...OUTLINE, ...LEFT_EYE, ...RIGHT_EYE, ...MOUTH, ...NOSE])
);

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function pointColor(index) {
  if (LEFT_EYE.includes(index) || RIGHT_EYE.includes(index)) return "#22c55e";
  if (MOUTH.includes(index)) return "#fb7185";
  if (NOSE.includes(index)) return "#facc15";
  if (OUTLINE.includes(index)) return "#60a5fa";
  return "#cbd5e1";
}

function shiftPoint(point, index, noise) {
  if (!point) return point;
  const dx = ((index % 5) - 2) * noise * 0.001;
  const dy = (((index + 3) % 5) - 2) * noise * 0.001;
  return {
    x: clamp(point.x + dx, 0, 1),
    y: clamp(point.y + dy, 0, 1),
    z: point.z || 0,
  };
}

function Chip({ children, active, warning }) {
  return (
    <span
      style={{
        padding: "8px 11px",
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 700,
        border: `1px solid ${warning ? "#fed7aa" : active ? "#bfdbfe" : "#e2e8f0"}`,
        background: warning ? "#fff7ed" : active ? "#eff6ff" : "#ffffff",
        color: warning ? "#c2410c" : active ? "#1d4ed8" : "#475569",
      }}
    >
      {children}
    </span>
  );
}

function Stat({ label, value, note }) {
  return (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #e5e7eb",
        borderRadius: 18,
        padding: 14,
      }}
    >
      <div style={{ fontSize: 12, color: "#64748b", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, marginTop: 5 }}>{value}</div>
      <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.4, marginTop: 4 }}>{note}</div>
    </div>
  );
}

function Panel({ title, subtitle, children }) {
  return (
    <section
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: 22,
        padding: 16,
        boxShadow: "0 12px 28px rgba(15,23,42,0.055)",
      }}
    >
      <div style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0, fontSize: 18, lineHeight: 1.2 }}>{title}</h2>
        <p style={{ margin: "4px 0 0", color: "#64748b", fontSize: 13, lineHeight: 1.4 }}>
          {subtitle}
        </p>
      </div>
      {children}
    </section>
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

  const status = useMemo(() => {
    if (!cameraOn) return "Waiting for camera";
    if (!faceDetected) return "Searching for face";
    return "Live graph generated";
  }, [cameraOn, faceDetected]);

  const exposure = privacyMode ? "Reduced" : "Visible";

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
        setError("The landmark model did not load. Refresh once before presenting.");
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
      streamRef.current.getTracks().forEach((track) => track.stop());
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
    drawEmptyGraph();
  }

  function clearCanvas(ref) {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  async function startCamera() {
    try {
      setError("");

      if (!modelReady) {
        setError("Model is still loading. Wait a few seconds and try again.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const width = videoRef.current.videoWidth || 960;
      const height = videoRef.current.videoHeight || 720;
      setFrame({ width, height });

      [overlayRef.current, graphRef.current].forEach((canvas) => {
        if (canvas) {
          canvas.width = width;
          canvas.height = height;
        }
      });

      setCameraOn(true);
      lastTimeRef.current = -1;
      runLoop();
    } catch (e) {
      console.error(e);
      setError("Camera permission was blocked. Allow camera access and reload.");
    }
  }

  function drawEmptyGraph() {
    const canvas = graphRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width || frame.width;
    const height = canvas.height || frame.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "700 22px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Graph view will appear here", width / 2, height / 2);
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

    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(255,255,255,0.75)";

    GROUPS.forEach((group) => {
      ctx.strokeStyle = group.color;
      ctx.globalAlpha = 0.8;

      for (let i = 0; i < group.points.length - 1; i++) {
        const a = group.points[i];
        const b = group.points[i + 1];
        if (!face[a] || !face[b]) continue;

        ctx.beginPath();
        ctx.moveTo((1 - face[a].x) * width, face[a].y * height);
        ctx.lineTo((1 - face[b].x) * width, face[b].y * height);
        ctx.stroke();
      }
    });

    ctx.globalAlpha = 1;
    ctx.fillStyle = "#ffffff";

    SELECTED_POINTS.forEach((index) => {
      const p = face[index];
      if (!p) return;
      ctx.beginPath();
      ctx.arc((1 - p.x) * width, p.y * height, 2.2, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function drawGraph(face) {
    const canvas = graphRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const { width, height } = frame;

    canvas.width = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);

    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, "#020617");
    bg.addColorStop(1, "#172554");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(148,163,184,0.09)";
    ctx.lineWidth = 1;

    for (let x = 0; x < width; x += 70) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    for (let y = 0; y < height; y += 70) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!face) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "800 22px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for facial graph", width / 2, height / 2);
      return;
    }

    const usable = SELECTED_POINTS.filter((i) => face[i]);
    const xs = usable.map((i) => face[i].x);
    const ys = usable.map((i) => face[i].y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    function map(index) {
      const p = shiftPoint(face[index], index, noise);
      const nx = (p.x - minX) / (maxX - minX || 1);
      const ny = (p.y - minY) / (maxY - minY || 1);

      return {
        x: width * 0.5 + (0.5 - nx) * width * 0.62,
        y: height * 0.53 + (ny - 0.5) * height * 0.72,
      };
    }

    ctx.save();
    ctx.shadowColor = "rgba(96,165,250,0.7)";
    ctx.shadowBlur = 10;

    GROUPS.forEach((group) => {
      ctx.strokeStyle = group.color;
      ctx.lineWidth = group.name === "face boundary" ? 2.2 : 1.9;
      ctx.globalAlpha = 0.85;

      for (let i = 0; i < group.points.length - 1; i++) {
        const a = group.points[i];
        const b = group.points[i + 1];

        if (!face[a] || !face[b]) continue;

        const p1 = map(a);
        const p2 = map(b);

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    });

    ctx.restore();

    SELECTED_POINTS.forEach((index) => {
      if (!face[index]) return;
      const p = map(index);

      ctx.fillStyle = pointColor(index);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    ctx.fillStyle = "rgba(15,23,42,0.78)";
    ctx.fillRect(24, 24, 355, 48);

    ctx.fillStyle = "#e0f2fe";
    ctx.font = "800 16px Arial";
    ctx.textAlign = "left";
    ctx.fillText("Structure-only graph representation", 42, 54);

    const legend = [
      ["Face boundary", "#60a5fa"],
      ["Eyes", "#22c55e"],
      ["Nose", "#facc15"],
      ["Mouth", "#fb7185"],
    ];

    legend.forEach(([label, color], index) => {
      const x = 42 + index * 142;
      const y = height - 36;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#dbeafe";
      ctx.font = "700 13px Arial";
      ctx.fillText(label, x + 13, y + 5);
    });
  }

  function runLoop() {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !landmarker) return;

    if (video.readyState >= 2 && video.currentTime !== lastTimeRef.current) {
      lastTimeRef.current = video.currentTime;

      const result = landmarker.detectForVideo(video, performance.now());
      const face = result.faceLandmarks?.[0];

      setFaceDetected(Boolean(face));
      setLandmarkCount(face ? SELECTED_POINTS.length : 0);
      setGraphQuality(face ? Math.max(60, 100 - noise * 3) : 0);

      drawOverlay(face);
      drawGraph(face);
    }

    rafRef.current = requestAnimationFrame(runLoop);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f4f7fb",
        color: "#111827",
        fontFamily: "Inter, system-ui, Arial, sans-serif",
      }}
    >
      <div style={{ maxWidth: 1360, margin: "0 auto", padding: "22px 24px 34px" }}>
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: 18,
            alignItems: "center",
            marginBottom: 18,
          }}
        >
          <div>
            <div style={{ color: "#2563eb", fontWeight: 800, fontSize: 13 }}>
              Privacy-aware facial representation
            </div>
            <h1 style={{ margin: "4px 0 0", fontSize: 34, lineHeight: 1.05 }}>
              Live Graph Abstraction Demo
            </h1>
            <p style={{ margin: "8px 0 0", color: "#64748b", maxWidth: 760, lineHeight: 1.55 }}>
              The system keeps the live camera input separate from the graph view, showing how
              facial structure can be used without depending on full identity-rich imagery.
            </p>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
            <Chip active={modelReady}>{loading ? "Loading model" : "Model ready"}</Chip>
            <Chip active={cameraOn}>{cameraOn ? "Camera active" : "Camera inactive"}</Chip>
            <Chip active={faceDetected}>{faceDetected ? "Face detected" : "No face"}</Chip>
            <Chip active={privacyMode} warning={!privacyMode}>
              {privacyMode ? "Privacy mode on" : "Raw visible"}
            </Chip>
          </div>
        </header>

        <main
          style={{
            display: "grid",
            gridTemplateColumns: "300px 1fr 1fr",
            gap: 18,
            alignItems: "start",
          }}
        >
          <aside
            style={{
              background: "#ffffff",
              border: "1px solid #e2e8f0",
              borderRadius: 22,
              padding: 18,
              boxShadow: "0 12px 28px rgba(15,23,42,0.055)",
            }}
          >
            <h2 style={{ margin: 0, fontSize: 18 }}>Demo controls</h2>
            <p style={{ margin: "6px 0 16px", color: "#64748b", fontSize: 13, lineHeight: 1.45 }}>
              Start with privacy mode on, then reveal the raw input only for comparison.
            </p>

            <button
              onClick={cameraOn ? stopCamera : startCamera}
              style={{
                width: "100%",
                border: "none",
                borderRadius: 15,
                background: "#2563eb",
                color: "#ffffff",
                padding: "14px 15px",
                fontWeight: 800,
                cursor: "pointer",
                fontSize: 15,
              }}
            >
              {cameraOn ? "Stop camera" : "Enable camera"}
            </button>

            <button
              onClick={() => setPrivacyMode((prev) => !prev)}
              style={{
                width: "100%",
                border: "1px solid #cbd5e1",
                borderRadius: 15,
                background: "#ffffff",
                color: "#111827",
                padding: "14px 15px",
                fontWeight: 800,
                cursor: "pointer",
                fontSize: 15,
                marginTop: 10,
              }}
            >
              {privacyMode ? "Show raw input" : "Hide raw input"}
            </button>

            <div style={{ marginTop: 20 }}>
              <label style={{ display: "block", fontWeight: 800, fontSize: 14 }}>
                Landmark noise: {noise}
              </label>
              <input
                type="range"
                min="0"
                max="12"
                value={noise}
                onChange={(e) => setNoise(Number(e.target.value))}
                style={{ width: "100%", marginTop: 10 }}
              />
            </div>

            <div
              style={{
                marginTop: 18,
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                borderRadius: 16,
                padding: 14,
                color: "#475569",
                fontSize: 13,
                lineHeight: 1.65,
              }}
            >
              <b style={{ color: "#111827" }}>Pipeline</b>
              <br />
              Raw face → facial landmarks → graph structure → reduced identity exposure
            </div>

            {error && (
              <div
                style={{
                  marginTop: 14,
                  background: "#fff7ed",
                  border: "1px solid #fdba74",
                  borderRadius: 15,
                  padding: 12,
                  color: "#9a3412",
                  fontSize: 13,
                  lineHeight: 1.5,
                }}
              >
                {error}
              </div>
            )}
          </aside>

          <Panel title="Raw input" subtitle="Live feed with selected facial structure overlay">
            <div style={mediaBox}>
              <video
                ref={videoRef}
                playsInline
                muted
                autoPlay
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  transform: "scaleX(-1)",
                  filter: privacyMode ? "blur(14px) brightness(0.68)" : "none",
                  opacity: cameraOn ? 1 : 0.15,
                  transition: "filter 180ms ease, opacity 180ms ease",
                }}
              />
              <canvas
                ref={overlayRef}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                }}
              />
              {!cameraOn && <CenterText text="Camera inactive" />}
            </div>
          </Panel>

          <Panel title="Graph representation" subtitle="Selected structure-only nodes and edges">
            <div style={mediaBox}>
              <canvas
                ref={graphRef}
                width={frame.width}
                height={frame.height}
                style={{ width: "100%", height: "100%", display: "block" }}
              />
            </div>
          </Panel>
        </main>

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 14,
            marginTop: 18,
          }}
        >
          <Stat label="Current state" value={status} note="Live pipeline status." />
          <Stat label="Graph nodes" value={landmarkCount} note="Selected landmarks, not the full raw face." />
          <Stat label="Graph quality" value={`${graphQuality}%`} note="Estimated stability under simulated noise." />
          <Stat label="Identity exposure" value={exposure} note="Raw visual identity visibility." />
        </section>
      </div>
    </div>
  );
}

function CenterText({ text }) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "grid",
        placeItems: "center",
        color: "#cbd5e1",
        fontWeight: 800,
        fontSize: 18,
      }}
    >
      {text}
    </div>
  );
}

const mediaBox = {
  position: "relative",
  borderRadius: 20,
  overflow: "hidden",
  border: "1px solid #dbeafe",
  background: "#0f172a",
  aspectRatio: "4 / 3",
};
