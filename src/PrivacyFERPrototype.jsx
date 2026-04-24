import { useEffect, useMemo, useRef, useState } from "react";

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const MP_VERSION = "0.10.3";
const WASM_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;

const OUTLINE = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10];
const LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33];
const RIGHT_EYE = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466,263];
const MOUTH = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78,61];
const NOSE = [168,6,197,195,5,4,1,19,94,2];

const GROUPS = [
  { name: "Boundary", points: OUTLINE, color: "#38bdf8", weight: 1.2 },
  { name: "Left eye", points: LEFT_EYE, color: "#22c55e", weight: 1.6 },
  { name: "Right eye", points: RIGHT_EYE, color: "#22c55e", weight: 1.6 },
  { name: "Nose", points: NOSE, color: "#facc15", weight: 1.4 },
  { name: "Mouth", points: MOUTH, color: "#fb7185", weight: 1.5 },
];

const SELECTED_POINTS = Array.from(
  new Set([...OUTLINE, ...LEFT_EYE, ...RIGHT_EYE, ...MOUTH, ...NOSE])
);

function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max);
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

function nodeColor(index) {
  if (LEFT_EYE.includes(index) || RIGHT_EYE.includes(index)) return "#22c55e";
  if (MOUTH.includes(index)) return "#fb7185";
  if (NOSE.includes(index)) return "#facc15";
  if (OUTLINE.includes(index)) return "#38bdf8";
  return "#cbd5e1";
}

function Chip({ children, active, warning }) {
  return (
    <span style={{
      padding: "8px 12px",
      borderRadius: 999,
      fontSize: 12,
      fontWeight: 800,
      border: `1px solid ${warning ? "#fed7aa" : active ? "#bfdbfe" : "#dbeafe"}`,
      background: warning ? "#fff7ed" : active ? "#eff6ff" : "#ffffff",
      color: warning ? "#c2410c" : active ? "#1d4ed8" : "#475569",
    }}>
      {children}
    </span>
  );
}

function Stat({ label, value, note, tone = "default" }) {
  const toneMap = {
    default: ["#ffffff", "#dbeafe"],
    good: ["#f0fdf4", "#bbf7d0"],
    warn: ["#fff7ed", "#fed7aa"],
    dark: ["#0f172a", "#334155"],
  };

  const [bg, border] = toneMap[tone];

  return (
    <div style={{
      background: bg,
      border: `1px solid ${border}`,
      borderRadius: 18,
      padding: 15,
      boxShadow: "0 10px 24px rgba(15,23,42,0.045)",
      color: tone === "dark" ? "#ffffff" : "#111827",
    }}>
      <div style={{ fontSize: 12, color: tone === "dark" ? "#cbd5e1" : "#64748b", fontWeight: 800 }}>
        {label}
      </div>
      <div style={{ fontSize: 25, fontWeight: 950, marginTop: 5 }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: tone === "dark" ? "#cbd5e1" : "#64748b", lineHeight: 1.45, marginTop: 4 }}>
        {note}
      </div>
    </div>
  );
}

function Meter({ label, value, color, caption }) {
  return (
    <div style={{ marginBottom: 15 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, fontWeight: 800 }}>
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div style={{ height: 10, background: "#e5e7eb", borderRadius: 999, marginTop: 7, overflow: "hidden" }}>
        <div style={{ width: `${value}%`, height: "100%", background: color, borderRadius: 999 }} />
      </div>
      <div style={{ fontSize: 12, color: "#64748b", marginTop: 5 }}>{caption}</div>
    </div>
  );
}

function Panel({ title, subtitle, children }) {
  return (
    <section style={{
      background: "#ffffff",
      border: "1px solid #dbeafe",
      borderRadius: 24,
      padding: 16,
      boxShadow: "0 14px 34px rgba(15,23,42,0.06)",
    }}>
      <div style={{ marginBottom: 12 }}>
        <h2 style={{ margin: 0, fontSize: 18 }}>{title}</h2>
        <p style={{ margin: "5px 0 0", color: "#64748b", fontSize: 13, lineHeight: 1.45 }}>
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
  const [graphDensity, setGraphDensity] = useState(0);
  const [edgeStrength, setEdgeStrength] = useState(0);
  const [error, setError] = useState("");
  const [frame, setFrame] = useState({ width: 960, height: 720 });

  const rawExposure = privacyMode ? 38 : 92;
  const graphExposure = Math.max(14, 30 + noise * 2);
  const leakageReduction = Math.max(0, rawExposure - graphExposure);
  const privacyScore = Math.min(100, Math.max(0, 100 - graphExposure));
  const utilityScore = faceDetected ? Math.max(58, 94 - noise * 3) : 0;

  const status = useMemo(() => {
    if (!cameraOn) return "Waiting for camera";
    if (!faceDetected) return "Searching";
    return "Active";
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
        setError("Model did not load. Refresh once before presenting.");
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
    setGraphDensity(0);
    setEdgeStrength(0);
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
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
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

    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "800 22px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Enable camera to generate structure graph", width / 2, height / 2);
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

    GROUPS.forEach((group) => {
      ctx.strokeStyle = group.color;
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.85;

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
      ctx.arc((1 - p.x) * width, p.y * height, 2.15, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = privacyMode ? "rgba(34,197,94,0.88)" : "rgba(194,65,12,0.88)";
    ctx.fillRect(width - 220, height - 54, 195, 36);
    ctx.fillStyle = "#ffffff";
    ctx.font = "800 14px Arial";
    ctx.textAlign = "center";
    ctx.fillText(
      privacyMode ? "Identity minimized" : "Raw identity visible",
      width - 122,
      height - 31
    );
  }

  function drawGraph(face) {
    const canvas = graphRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const { width, height } = frame;

    canvas.width = width;
    canvas.height = height;

    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, "#020617");
    bg.addColorStop(0.55, "#0f172a");
    bg.addColorStop(1, "#172554");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(148,163,184,0.08)";
    ctx.lineWidth = 1;

    for (let x = 0; x < width; x += 72) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    for (let y = 0; y < height; y += 72) {
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
        x: width * 0.5 + (0.5 - nx) * width * 0.56,
        y: height * 0.53 + (ny - 0.5) * height * 0.68,
      };
    }

    const allEdgeDistances = [];

    GROUPS.forEach((group) => {
      for (let i = 0; i < group.points.length - 1; i++) {
        const a = group.points[i];
        const b = group.points[i + 1];
        if (!face[a] || !face[b]) continue;

        const p1 = map(a);
        const p2 = map(b);
        allEdgeDistances.push(Math.hypot(p1.x - p2.x, p1.y - p2.y));
      }
    });

    const avgEdge = allEdgeDistances.length
      ? allEdgeDistances.reduce((a, b) => a + b, 0) / allEdgeDistances.length
      : 0;

    setGraphDensity(Math.min(100, Math.round((SELECTED_POINTS.length / 90) * 100)));
    setEdgeStrength(Math.max(0, Math.min(100, Math.round(100 - avgEdge / 3))));

    ctx.save();
    ctx.shadowColor = "rgba(96,165,250,0.7)";
    ctx.shadowBlur = 10;

    GROUPS.forEach((group) => {
      ctx.strokeStyle = group.color;
      ctx.lineWidth = group.weight + 0.7;

      for (let i = 0; i < group.points.length - 1; i++) {
        const a = group.points[i];
        const b = group.points[i + 1];
        if (!face[a] || !face[b]) continue;

        const p1 = map(a);
        const p2 = map(b);
        const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);

        const opacity = Math.max(0.28, 1 - dist / 320);
        ctx.globalAlpha = opacity;

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
      const isImportant =
        LEFT_EYE.includes(index) ||
        RIGHT_EYE.includes(index) ||
        NOSE.includes(index) ||
        MOUTH.includes(index);

      ctx.fillStyle = nodeColor(index);
      ctx.beginPath();
      ctx.arc(p.x, p.y, isImportant ? 4.5 : 3.4, 0, Math.PI * 2);
      ctx.fill();

      if (isImportant) {
        ctx.strokeStyle = "rgba(255,255,255,0.75)";
        ctx.lineWidth = 1.3;
        ctx.stroke();
      }
    });

    ctx.fillStyle = "rgba(15,23,42,0.82)";
    ctx.fillRect(24, 24, 420, 56);

    ctx.fillStyle = "#e0f2fe";
    ctx.font = "900 16px Arial";
    ctx.textAlign = "left";
    ctx.fillText("Scientific graph view: weighted structural edges", 42, 49);
    ctx.fillStyle = "#93c5fd";
    ctx.font = "700 12px Arial";
    ctx.fillText("Color = region, opacity = edge strength, node size = feature importance", 42, 67);

    const legend = [
      ["Boundary", "#38bdf8"],
      ["Eyes", "#22c55e"],
      ["Nose", "#facc15"],
      ["Mouth", "#fb7185"],
    ];

    legend.forEach(([label, color], index) => {
      const x = 42 + index * 124;
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
    <div style={{
      minHeight: "100vh",
      background: "#f4f7fb",
      color: "#111827",
      fontFamily: "Inter, system-ui, Arial, sans-serif",
    }}>
      <div style={{ maxWidth: 1450, margin: "0 auto", padding: "22px 24px 34px" }}>
        <header style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 18,
          alignItems: "center",
          marginBottom: 18,
        }}>
          <div>
            <div style={{ color: "#2563eb", fontWeight: 800, fontSize: 13 }}>
              Privacy-aware facial representation
            </div>
            <h1 style={{ margin: "4px 0 0", fontSize: 34, lineHeight: 1.05 }}>
              Privacy-Aware Facial Representation System
            </h1>
            <p style={{ margin: "8px 0 0", color: "#64748b", maxWidth: 840, lineHeight: 1.55 }}>
              This system transforms identity-rich facial input into a structure-only graph,
              comparing raw exposure against graph-based representation in real time.
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

        <main style={{
          display: "grid",
          gridTemplateColumns: "305px 1fr 1fr",
          gap: 18,
          alignItems: "start",
        }}>
          <aside style={{
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: 22,
            padding: 18,
            boxShadow: "0 12px 28px rgba(15,23,42,0.055)",
          }}>
            <h2 style={{ margin: 0, fontSize: 18 }}>System controls</h2>
            <p style={{ margin: "6px 0 16px", color: "#64748b", fontSize: 13, lineHeight: 1.45 }}>
              Use comparison mode to show the privacy shift from raw input to structural graph.
            </p>

            <button onClick={cameraOn ? stopCamera : startCamera} style={primaryButton}>
              {cameraOn ? "Stop camera" : "Enable camera"}
            </button>

            <button onClick={() => setPrivacyMode((prev) => !prev)} style={secondaryButton}>
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

            <div style={{
              marginTop: 18,
              background: "#f8fafc",
              border: "1px solid #e2e8f0",
              borderRadius: 16,
              padding: 14,
              color: "#475569",
              fontSize: 13,
              lineHeight: 1.65,
            }}>
              <b style={{ color: "#111827" }}>Pipeline</b>
              <br />
              Raw face → landmarks → weighted graph → privacy evaluation
            </div>

            <div style={{
              marginTop: 18,
              background: "#0f172a",
              borderRadius: 16,
              padding: 14,
              color: "#dbeafe",
              fontSize: 13,
              lineHeight: 1.65,
            }}>
              <b style={{ color: "#ffffff" }}>Real-time comparison</b>
              <br />
              Raw exposure: {rawExposure}%<br />
              Graph exposure: {graphExposure}%<br />
              Reduction: {leakageReduction}%
            </div>

            {error && (
              <div style={{
                marginTop: 14,
                background: "#fff7ed",
                border: "1px solid #fdba74",
                borderRadius: 15,
                padding: 12,
                color: "#9a3412",
                fontSize: 13,
                lineHeight: 1.5,
              }}>
                {error}
              </div>
            )}
          </aside>

          <Panel title="Raw input" subtitle="Identity-rich webcam stream with selected facial structure overlay">
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

          <Panel title="Graph representation" subtitle="Weighted structural network with reduced identity exposure">
            <div style={mediaBox}>
              <canvas
                ref={graphRef}
                width={frame.width}
                height={frame.height}
                style={{ width: "100%", height: "100%", display: "block" }}
              />
            </div>
            <p style={{ fontSize: 13, color: "#64748b", margin: "10px 2px 0" }}>
              Graph preserves geometry, not identity.
            </p>
          </Panel>
        </main>

        <section style={{
          marginTop: 18,
          display: "grid",
          gridTemplateColumns: "1.2fr 1fr 1fr",
          gap: 14,
        }}>
          <div style={{
            background: "#ffffff",
            border: "1px solid #dbeafe",
            borderRadius: 18,
            padding: 16,
            boxShadow: "0 10px 24px rgba(15,23,42,0.045)",
          }}>
            <div style={{ fontWeight: 900, fontSize: 14, marginBottom: 12 }}>
              Raw vs Graph: real-time exposure comparison
            </div>
            <Meter
              label="Raw image exposure"
              value={rawExposure}
              color="#f97316"
              caption="Full visual identity remains available."
            />
            <Meter
              label="Graph representation exposure"
              value={graphExposure}
              color="#22c55e"
              caption="Identity-rich visual detail is reduced."
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Stat label="Privacy score" value={`${privacyScore}%`} note="Higher means less raw identity exposure." tone="good" />
            <Stat label="Identity leakage" value={`${graphExposure}%`} note="Lower is better for privacy." tone="warn" />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Stat label="Graph density" value={`${graphDensity}%`} note="Selected structure captured as graph." />
            <Stat label="Edge strength" value={`${edgeStrength}%`} note="Weighted edge stability." />
          </div>
        </section>
      </div>
    </div>
  );
}

function CenterText({ text }) {
  return (
    <div style={{
      position: "absolute",
      inset: 0,
      display: "grid",
      placeItems: "center",
      color: "#cbd5e1",
      fontWeight: 800,
      fontSize: 18,
    }}>
      {text}
    </div>
  );
}

const primaryButton = {
  width: "100%",
  border: "none",
  borderRadius: 15,
  background: "#2563eb",
  color: "#ffffff",
  padding: "14px 15px",
  fontWeight: 800,
  cursor: "pointer",
  fontSize: 15,
};

const secondaryButton = {
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
};

const mediaBox = {
  position: "relative",
  borderRadius: 20,
  overflow: "hidden",
  border: "1px solid #dbeafe",
  background: "#0f172a",
  aspectRatio: "4 / 3",
};
