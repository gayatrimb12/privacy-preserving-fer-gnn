import { useEffect, useMemo, useRef, useState } from "react";

const BASE_POINTS = [
  [0.50, 0.20],
  [0.38, 0.26],
  [0.62, 0.26],
  [0.33, 0.34],
  [0.67, 0.34],
  [0.40, 0.38],
  [0.60, 0.38],
  [0.43, 0.50],
  [0.57, 0.50],
  [0.50, 0.58],
  [0.36, 0.66],
  [0.44, 0.71],
  [0.50, 0.73],
  [0.56, 0.71],
  [0.64, 0.66],
  [0.30, 0.45],
  [0.70, 0.45],
  [0.50, 0.82]
];

const EDGES = [
  [0, 1], [0, 2],
  [1, 3], [2, 4],
  [3, 5], [4, 6],
  [5, 7], [6, 8],
  [7, 9], [8, 9],
  [10, 11], [11, 12], [12, 13], [13, 14],
  [15, 5], [16, 6],
  [9, 12], [12, 17],
  [7, 11], [8, 13]
];

function jitter(value, amount) {
  return value + (Math.random() - 0.5) * amount;
}

function StatusPill({ label, active = false, warning = false }) {
  const bg = warning ? "#fff7ed" : active ? "#eff6ff" : "#f8fafc";
  const color = warning ? "#c2410c" : active ? "#1d4ed8" : "#475569";
  const border = warning ? "#fdba74" : active ? "#bfdbfe" : "#e2e8f0";

  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: "999px",
        border: `1px solid ${border}`,
        background: bg,
        color,
        fontSize: "0.85rem",
        fontWeight: 700,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </div>
  );
}

function MetricCard({ label, value, subtext }) {
  return (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: "18px",
        padding: "16px",
      }}
    >
      <div style={{ color: "#64748b", fontSize: "0.86rem", marginBottom: "8px" }}>
        {label}
      </div>
      <div style={{ fontWeight: 800, fontSize: "1.4rem", marginBottom: "6px" }}>
        {value}
      </div>
      <div style={{ color: "#475569", fontSize: "0.92rem", lineHeight: 1.6 }}>
        {subtext}
      </div>
    </div>
  );
}

export default function PrivacyFERPrototype() {
  const videoRef = useRef(null);
  const graphCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(true);
  const [noiseLevel, setNoiseLevel] = useState(2);
  const [graphQuality, setGraphQuality] = useState(88);
  const [errorMessage, setErrorMessage] = useState("");
  const [frameSize, setFrameSize] = useState({ width: 640, height: 480 });

  const systemSummary = useMemo(() => {
    if (!cameraOn) return "Camera inactive";
    return "Prototype running";
  }, [cameraOn]);

  const exposureLevel = privacyMode ? "Reduced" : "Visible";

  useEffect(() => {
    return () => {
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setGraphQuality(Math.max(58, 96 - noiseLevel * 4));
  }, [noiseLevel]);

  const clearCanvas = () => {
    const canvas = graphCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 18px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Graph view inactive", width / 2, height / 2);
  };

  const stopCamera = () => {
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
    const canvas = graphCanvasRef.current;
    if (canvas) {
      clearCanvas();
    }
  };

  const startCamera = async () => {
    try {
      setErrorMessage("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 960 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      setFrameSize({ width, height });

      const canvas = graphCanvasRef.current;
      if (canvas) {
        canvas.width = width;
        canvas.height = height;
      }

      setCameraOn(true);
      runGraphLoop();
    } catch (error) {
      console.error(error);
      setErrorMessage("Camera access failed. Please allow webcam permission and try again.");
    }
  };

  const drawGraph = () => {
    const canvas = graphCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = frameSize.width;
    const height = frameSize.height;

    canvas.width = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const faceW = width * 0.34;
    const faceH = height * 0.52;
    const noise = noiseLevel * 0.01;

    const points = BASE_POINTS.map(([px, py]) => {
      const x = centerX + (px - 0.5) * faceW * 2;
      const y = centerY + (py - 0.5) * faceH * 2;
      return {
        x: jitter(x, faceW * noise),
        y: jitter(y, faceH * noise),
      };
    });

    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 1.8;
    ctx.globalAlpha = 0.85;

    EDGES.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
      ctx.stroke();
    });

    ctx.globalAlpha = 1;
    ctx.fillStyle = "#0f172a";

    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "rgba(37,99,235,0.08)";
    ctx.fillRect(16, 16, 228, 34);
    ctx.fillStyle = "#1d4ed8";
    ctx.font = "700 14px Arial";
    ctx.textAlign = "left";
    ctx.fillText("graph-based privacy abstraction", 28, 38);

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.strokeRect(centerX - faceW * 0.8, centerY - faceH * 0.9, faceW * 1.6, faceH * 1.8);
  };

  const runGraphLoop = () => {
    const loop = () => {
      drawGraph();
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f8fafc",
        color: "#0f172a",
        fontFamily:
          'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <div
        style={{
          maxWidth: "1240px",
          margin: "0 auto",
          padding: "32px 20px 48px",
        }}
      >
        <div
          style={{
            marginBottom: "24px",
            display: "flex",
            flexWrap: "wrap",
            gap: "10px",
          }}
        >
          <StatusPill label="Prototype ready" active />
          <StatusPill label={cameraOn ? "Camera active" : "Camera inactive"} active={cameraOn} />
          <StatusPill label="Graph abstraction" active />
          <StatusPill
            label={privacyMode ? "Privacy mode on" : "Privacy mode off"}
            active={privacyMode}
            warning={!privacyMode}
          />
        </div>

        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: "28px",
            padding: "28px",
            boxShadow: "0 10px 30px rgba(15,23,42,0.05)",
            marginBottom: "24px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-start",
              gap: "20px",
              flexWrap: "wrap",
            }}
          >
            <div style={{ maxWidth: "760px" }}>
              <div
                style={{
                  display: "inline-block",
                  padding: "8px 12px",
                  background: "#eff6ff",
                  color: "#1d4ed8",
                  borderRadius: "999px",
                  fontSize: "0.85rem",
                  fontWeight: 700,
                  marginBottom: "14px",
                }}
              >
                Safe fallback demo
              </div>

              <h1
                style={{
                  margin: 0,
                  fontSize: "clamp(2rem, 4vw, 3.4rem)",
                  lineHeight: 1.05,
                }}
              >
                Privacy-Preserving Facial Analysis
              </h1>

              <p
                style={{
                  marginTop: "14px",
                  marginBottom: 0,
                  color: "#475569",
                  lineHeight: 1.8,
                  fontSize: "1rem",
                  maxWidth: "720px",
                }}
              >
                This working prototype demonstrates how a facial recognition pipeline can
                shift from raw identity-heavy input to a graph-style structural
                representation for privacy-aware analysis.
              </p>
            </div>

            <div
              style={{
                minWidth: "250px",
                border: "1px solid #e2e8f0",
                borderRadius: "20px",
                padding: "18px",
                background: "#f8fafc",
              }}
            >
              <div style={{ fontSize: "0.9rem", color: "#64748b", marginBottom: "8px" }}>
                System summary
              </div>
              <div style={{ fontWeight: 700, fontSize: "1.05rem", marginBottom: "10px" }}>
                {systemSummary}
              </div>
              <div style={{ fontSize: "0.95rem", color: "#475569", lineHeight: 1.7 }}>
                Graph quality: {graphQuality}% <br />
                Raw exposure: {exposureLevel} <br />
                Noise level: {noiseLevel}
              </div>
            </div>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(280px, 320px) 1fr",
            gap: "20px",
            alignItems: "start",
          }}
        >
          <div
            style={{
              background: "#ffffff",
              border: "1px solid #e2e8f0",
              borderRadius: "24px",
              padding: "20px",
              boxShadow: "0 10px 30px rgba(15,23,42,0.04)",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "14px" }}>
              Controls
            </div>

            <div style={{ display: "grid", gap: "12px" }}>
              <button
                onClick={cameraOn ? stopCamera : startCamera}
                style={{
                  padding: "14px 16px",
                  borderRadius: "14px",
                  border: "none",
                  background: "#2563eb",
                  color: "#ffffff",
                  cursor: "pointer",
                  fontWeight: 700,
                  fontSize: "0.95rem",
                }}
              >
                {cameraOn ? "Stop camera" : "Enable camera"}
              </button>

              <button
                onClick={() => setPrivacyMode((prev) => !prev)}
                style={{
                  padding: "14px 16px",
                  borderRadius: "14px",
                  border: "1px solid #cbd5e1",
                  background: "#ffffff",
                  color: "#0f172a",
                  cursor: "pointer",
                  fontWeight: 700,
                  fontSize: "0.95rem",
                }}
              >
                {privacyMode ? "Show raw input" : "Hide raw input"}
              </button>
            </div>

            <div style={{ marginTop: "18px" }}>
              <label
                htmlFor="noise"
                style={{
                  display: "block",
                  fontWeight: 700,
                  fontSize: "0.92rem",
                  marginBottom: "10px",
                }}
              >
                Graph noise simulation: {noiseLevel}
              </label>
              <input
                id="noise"
                type="range"
                min="0"
                max="12"
                step="1"
                value={noiseLevel}
                onChange={(e) => setNoiseLevel(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <div
              style={{
                marginTop: "18px",
                padding: "14px",
                borderRadius: "16px",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                color: "#475569",
                lineHeight: 1.7,
                fontSize: "0.94rem",
              }}
            >
              Pipeline:
              <br />
              raw face → abstracted nodes → graph structure → reduced identity exposure
            </div>

            {errorMessage && (
              <div
                style={{
                  marginTop: "16px",
                  padding: "14px",
                  borderRadius: "16px",
                  background: "#fff7ed",
                  border: "1px solid #fdba74",
                  color: "#9a3412",
                  lineHeight: 1.7,
                  fontSize: "0.92rem",
                }}
              >
                {errorMessage}
              </div>
            )}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: "20px",
            }}
          >
            <div
              style={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: "24px",
                padding: "18px",
                boxShadow: "0 10px 30px rgba(15,23,42,0.04)",
              }}
            >
              <div style={{ marginBottom: "12px" }}>
                <div style={{ fontWeight: 700, fontSize: "1rem" }}>Raw input panel</div>
                <div style={{ color: "#64748b", fontSize: "0.9rem", marginTop: "4px" }}>
                  Live webcam feed
                </div>
              </div>

              <div
                style={{
                  position: "relative",
                  borderRadius: "18px",
                  overflow: "hidden",
                  border: "1px solid #dbe4f0",
                  background: "#0f172a",
                  aspectRatio: "4 / 3",
                }}
              >
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
                    opacity: cameraOn ? 1 : 0.12,
                    transition: "filter 180ms ease, opacity 180ms ease",
                  }}
                />
                {!cameraOn && (
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "#cbd5e1",
                      fontWeight: 700,
                      letterSpacing: "0.02em",
                    }}
                  >
                    Camera inactive
                  </div>
                )}
              </div>
            </div>

            <div
              style={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: "24px",
                padding: "18px",
                boxShadow: "0 10px 30px rgba(15,23,42,0.04)",
              }}
            >
              <div style={{ marginBottom: "12px" }}>
                <div style={{ fontWeight: 700, fontSize: "1rem" }}>Graph abstraction panel</div>
                <div style={{ color: "#64748b", fontSize: "0.9rem", marginTop: "4px" }}>
                  Structural representation for downstream analysis
                </div>
              </div>

              <div
                style={{
                  borderRadius: "18px",
                  overflow: "hidden",
                  border: "1px solid #dbe4f0",
                  background: "#ffffff",
                  aspectRatio: "4 / 3",
                }}
              >
                <canvas
                  ref={graphCanvasRef}
                  width={frameSize.width}
                  height={frameSize.height}
                  style={{
                    width: "100%",
                    height: "100%",
                    display: "block",
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        <div
          style={{
            marginTop: "24px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "16px",
          }}
        >
          <MetricCard
            label="System summary"
            value={systemSummary}
            subtext="Current pipeline state."
          />
          <MetricCard
            label="Representation"
            value="Graph-based"
            subtext="Structure emphasized over identity-rich detail."
          />
          <MetricCard
            label="Graph quality"
            value={`${graphQuality}%`}
            subtext="Approximate structural stability under simulated noise."
          />
          <MetricCard
            label="Identity exposure"
            value={exposureLevel}
            subtext="Raw input visibility level."
          />
        </div>
      </div>
    </div>
  );
}
