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

  [70, 63], [63, 105], [105, 66], [66, 107], [336, 296], [296, 334],
  [334, 293], [293, 300],

  [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19],
  [19, 94], [94, 2], [2, 164], [164, 0], [0, 11], [11, 12], [12, 13], [13, 14],

  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405],
  [405, 321], [321, 375], [375, 291],
  [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402],
  [402, 318], [318, 324], [324, 308]
];

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function shiftedPoint(point, index, noiseLevel) {
  if (!point) return point;
  if (!noiseLevel) return point;

  const dx = ((index % 5) - 2) * noiseLevel * 0.0012;
  const dy = (((index + 2) % 5) - 2) * noiseLevel * 0.0012;

  return {
    x: clamp(point.x + dx, 0, 1),
    y: clamp(point.y + dy, 0, 1),
    z: point.z ?? 0,
  };
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
  const rawOverlayCanvasRef = useRef(null);
  const graphCanvasRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [modelReady, setModelReady] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [landmarkCount, setLandmarkCount] = useState(0);
  const [privacyMode, setPrivacyMode] = useState(true);
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const [graphQuality, setGraphQuality] = useState(100);
  const [frameSize, setFrameSize] = useState({ width: 640, height: 480 });

  const systemSummary = useMemo(() => {
    if (!cameraOn) return "Camera inactive";
    if (!modelReady) return "Loading face landmarker";
    if (!faceDetected) return "No face detected";
    return "Face detected · graph generated";
  }, [cameraOn, modelReady, faceDetected]);

  const exposureLevel = privacyMode ? "Reduced" : "Visible";

  useEffect(() => {
    let cancelled = false;

    async function loadLandmarker() {
      try {
        setIsLoadingModel(true);
        setErrorMessage("");

        const vision = await import(
          `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/+esm`
        );

        const { FaceLandmarker, FilesetResolver } = vision;

        if (!FaceLandmarker || !FilesetResolver) {
          throw new Error("MediaPipe module failed to load properly.");
        }

        const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);

        const faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_PATH,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (!cancelled) {
          landmarkerRef.current = faceLandmarker;
          setModelReady(true);
          setErrorMessage("");
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setErrorMessage(
            "Could not load the face landmark model. The UI will still run, but the live graph needs MediaPipe."
          );
        }
      } finally {
        if (!cancelled) {
          setIsLoadingModel(false);
        }
      }
    }

    loadLandmarker();

    return () => {
      cancelled = true;
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const clearCanvas = (canvasRef, text = "View inactive") => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 18px Arial";
    ctx.textAlign = "center";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  };

  const stopCamera = () => {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;

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
    setGraphQuality(100);

    clearCanvas(rawOverlayCanvasRef, "");
    clearCanvas(graphCanvasRef, "Graph view inactive");
  };

  const startCamera = async () => {
    try {
      setErrorMessage("");

      if (!modelReady) {
        setErrorMessage("The face landmark model is still loading.");
        return;
      }

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

      const rawCanvas = rawOverlayCanvasRef.current;
      if (rawCanvas) {
        rawCanvas.width = width;
        rawCanvas.height = height;
      }

      const graphCanvas = graphCanvasRef.current;
      if (graphCanvas) {
        graphCanvas.width = width;
        graphCanvas.height = height;
      }

      setCameraOn(true);
      lastVideoTimeRef.current = -1;
      runInference();
    } catch (error) {
      console.error(error);
      setErrorMessage(
        "Camera access failed. Allow browser camera permission and try again."
      );
    }
  };

  const drawRawOverlay = (landmarks) => {
    const canvas = rawOverlayCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = frameSize.width;
    const height = frameSize.height;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length === 0) return;

    const face = landmarks[0];

    ctx.strokeStyle = "rgba(37,99,235,0.75)";
    ctx.lineWidth = 1.1;
    ctx.globalAlpha = 0.8;

    for (const [a, b] of CONNECTIONS) {
      if (!face[a] || !face[b]) continue;
      ctx.beginPath();
      ctx.moveTo((1 - face[a].x) * width, face[a].y * height);
      ctx.lineTo((1 - face[b].x) * width, face[b].y * height);
      ctx.stroke();
    }

    ctx.globalAlpha = 1;
    ctx.fillStyle = "#ffffff";
    for (let i = 0; i < face.length; i += 6) {
      const p = face[i];
      if (!p) continue;
      ctx.beginPath();
      ctx.arc((1 - p.x) * width, p.y * height, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }

    const xs = face.map((p) => (1 - p.x) * width);
    const ys = face.map((p) => p.y * height);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1.4;
    ctx.strokeRect(minX - 12, minY - 12, maxX - minX + 24, maxY - minY + 24);
  };

  const drawGraphOnly = (landmarks) => {
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

    if (!landmarks || landmarks.length === 0) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "600 18px Arial";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for face landmarks", width / 2, height / 2);
      return;
    }

    const face = landmarks[0];

    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 1.8;
    ctx.globalAlpha = 0.82;

    for (const [a, b] of CONNECTIONS) {
      if (!face[a] || !face[b]) continue;
      const p1 = shiftedPoint(face[a], a, noiseLevel);
      const p2 = shiftedPoint(face[b], b, noiseLevel);

      ctx.beginPath();
      ctx.moveTo((1 - p1.x) * width, p1.y * height);
      ctx.lineTo((1 - p2.x) * width, p2.y * height);
      ctx.stroke();
    }

    ctx.globalAlpha = 1;
    ctx.fillStyle = "#0f172a";

    for (let i = 0; i < face.length; i += 2) {
      const p = shiftedPoint(face[i], i, noiseLevel);
      if (!p) continue;
      const depthSize = 2 + Math.max(0, 1.5 - Math.abs(p.z || 0) * 20);

      ctx.beginPath();
      ctx.arc((1 - p.x) * width, p.y * height, depthSize, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "rgba(37,99,235,0.08)";
    ctx.fillRect(16, 16, 228, 34);
    ctx.fillStyle = "#1d4ed8";
    ctx.font = "700 14px Arial";
    ctx.textAlign = "left";
    ctx.fillText("privacy-aware graph abstraction", 28, 38);
  };

  const runInference = () => {
    const video = videoRef.current;
    const faceLandmarker = landmarkerRef.current;

    if (!video || !faceLandmarker) return;

    const loop = () => {
      if (!videoRef.current || !landmarkerRef.current) return;

      if (video.readyState >= 2) {
        const currentTime = video.currentTime;

        if (currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = currentTime;

          const results = faceLandmarker.detectForVideo(video, performance.now());
          const landmarks = results.faceLandmarks || [];
          const foundFace = landmarks.length > 0;

          setFaceDetected(foundFace);
          setLandmarkCount(foundFace ? landmarks[0].length : 0);
          setGraphQuality(foundFace ? Math.max(55, 100 - noiseLevel * 3) : 0);

          drawRawOverlay(landmarks);
          drawGraphOnly(landmarks);
        }
      }

      animationRef.current = requestAnimationFrame(loop);
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
          <StatusPill
            label={isLoadingModel ? "Loading model" : "Model ready"}
            active={modelReady}
          />
          <StatusPill
            label={cameraOn ? "Camera active" : "Camera inactive"}
            active={cameraOn}
          />
          <StatusPill
            label={faceDetected ? "Face detected" : "No face"}
            active={faceDetected}
          />
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
                Advanced live demo
              </div>

              <h1
                style={{
                  margin: 0,
                  fontSize: "clamp(2rem, 4vw, 3.5rem)",
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
                This prototype compares raw webcam input with a graph-based facial
                representation built from detected landmarks to demonstrate reduced
                identity-heavy visual dependence.
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
              <div
                style={{
                  fontSize: "0.9rem",
                  color: "#64748b",
                  marginBottom: "8px",
                }}
              >
                System summary
              </div>
              <div style={{ fontWeight: 700, fontSize: "1.05rem", marginBottom: "10px" }}>
                {systemSummary}
              </div>
              <div style={{ fontSize: "0.95rem", color: "#475569", lineHeight: 1.7 }}>
                Landmarks: {landmarkCount} <br />
                Graph quality: {graphQuality}% <br />
                Raw exposure: {exposureLevel}
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
                Landmark noise simulation: {noiseLevel}
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
              raw face → detected landmarks → graph abstraction → reduced identity exposure
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
                  Live webcam feed with overlay
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
                <canvas
                  ref={rawOverlayCanvasRef}
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
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
            label="Landmarks"
            value={landmarkCount}
            subtext="Detected structural points."
          />
          <MetricCard
            label="Graph quality"
            value={`${graphQuality}%`}
            subtext="Approximate structural stability under noise."
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
