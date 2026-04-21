import { useEffect, useMemo, useRef, useState } from "react";
const { FaceLandmarker, FilesetResolver } = window;

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";

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

  [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402],
  [402, 318], [318, 324], [324, 308], [78, 191], [191, 80], [80, 81],
  [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 308],

  [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19],
  [19, 94], [94, 2], [2, 164], [164, 0], [0, 11], [11, 12], [12, 13], [13, 14]
];

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function shiftPoint(x, y, index, noiseAmount) {
  if (!noiseAmount) return { x, y };
  const dx = ((index % 5) - 2) * noiseAmount * 0.0012;
  const dy = (((index + 2) % 5) - 2) * noiseAmount * 0.0012;
  return {
    x: clamp(x + dx, 0, 1),
    y: clamp(y + dy, 0, 1)
  };
}

function StatusChip({ label, active = false, warning = false }) {
  const background = warning ? "#fff7ed" : active ? "#eff6ff" : "#f8fafc";
  const border = warning ? "#fdba74" : active ? "#bfdbfe" : "#e2e8f0";
  const color = warning ? "#c2410c" : active ? "#1d4ed8" : "#475569";

  return (
    <div
      style={{
        padding: "10px 12px",
        borderRadius: "999px",
        border: `1px solid ${border}`,
        background,
        color,
        fontWeight: 700,
        fontSize: "0.83rem",
        whiteSpace: "nowrap"
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
        padding: "16px"
      }}
    >
      <div style={{ color: "#64748b", fontSize: "0.86rem", marginBottom: "8px" }}>
        {label}
      </div>
      <div style={{ fontWeight: 800, fontSize: "1.5rem", marginBottom: "6px" }}>
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
  const animationRef = useRef(null);
  const streamRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [modelReady, setModelReady] = useState(false);
  const [loadingModel, setLoadingModel] = useState(true);
  const [cameraOn, setCameraOn] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(true);
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [landmarkCount, setLandmarkCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const [frameSize, setFrameSize] = useState({ width: 640, height: 480 });
  const [graphQuality, setGraphQuality] = useState(100);

  const systemSummary = useMemo(() => {
    if (!cameraOn) return "Camera inactive";
    if (!modelReady) return "Loading landmark model";
    if (!faceDetected) return "No face detected";
    return "Face detected and graph generated";
  }, [cameraOn, modelReady, faceDetected]);

  const exposureLevel = privacyMode ? "Reduced" : "Visible";

  useEffect(() => {
    let cancelled = false;

    async function initializeModel() {
      try {
        setLoadingModel(true);
        setErrorMessage("");

        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_PATH,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        if (!cancelled) {
          faceLandmarkerRef.current = landmarker;
          setModelReady(true);
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setErrorMessage(
            "Failed to load MediaPipe Face Landmarker. Check the dependency install and redeploy."
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingModel(false);
        }
      }
    }

    initializeModel();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  const clearCanvas = () => {
    const canvas = graphCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 18px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Graph view inactive", canvas.width / 2, canvas.height / 2);
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

    if (graphCanvasRef.current) {
      const canvas = graphCanvasRef.current;
      canvas.width = frameSize.width;
      canvas.height = frameSize.height;
      clearCanvas();
    }
  };

  const startCamera = async () => {
    try {
      setErrorMessage("");

      if (!modelReady) {
        setErrorMessage("Model is still loading. Try again in a moment.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 960 },
          height: { ideal: 720 }
        },
        audio: false
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
      lastVideoTimeRef.current = -1;
      runInferenceLoop();
    } catch (error) {
      console.error(error);
      setErrorMessage("Camera access failed. Please allow webcam permission and try again.");
    }
  };

  const drawGraph = (landmarks) => {
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
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.82;

    for (const [a, b] of CONNECTIONS) {
      const p1 = shiftPoint(face[a].x, face[a].y, a, noiseLevel);
      const p2 = shiftPoint(face[b].x, face[b].y, b, noiseLevel);

      ctx.beginPath();
      ctx.moveTo((1 - p1.x) * width, p1.y * height);
      ctx.lineTo((1 - p2.x) * width, p2.y * height);
      ctx.stroke();
    }

    ctx.globalAlpha = 1;
    ctx.fillStyle = "#0f172a";

    for (let i = 0; i < face.length; i += 3) {
      const p = shiftPoint(face[i].x, face[i].y, i, noiseLevel);
      ctx.beginPath();
      ctx.arc((1 - p.x) * width, p.y * height, 2.15, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "rgba(37,99,235,0.08)";
    ctx.fillRect(16, 16, 212, 34);
    ctx.fillStyle = "#1d4ed8";
    ctx.font = "700 14px Arial";
    ctx.textAlign = "left";
    ctx.fillText("graph-based representation", 28, 38);
  };

  const runInferenceLoop = () => {
    const video = videoRef.current;
    const landmarker = faceLandmarkerRef.current;
    if (!video || !landmarker) return;

    const loop = () => {
      if (!videoRef.current || !faceLandmarkerRef.current) return;

      if (video.readyState >= 2) {
        const currentTime = video.currentTime;

        if (currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = currentTime;

          const result = faceLandmarkerRef.current.detectForVideo(
            video,
            performance.now()
          );

          const landmarks = result.faceLandmarks || [];
          const hasFace = landmarks.length > 0;

          setFaceDetected(hasFace);
          setLandmarkCount(hasFace ? landmarks[0].length : 0);
          setGraphQuality(hasFace ? Math.max(58, 100 - noiseLevel * 3) : 0);

          drawGraph(landmarks);
        }
      }

      animationRef.current = requestAnimationFrame(loop);
    };

    loop();
  };

  useEffect(() => {
    if (!cameraOn) return;
    // redraw naturally as next frame comes in
  }, [noiseLevel, cameraOn]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f8fafc",
        fontFamily:
          'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        color: "#0f172a"
      }}
    >
      <div
        style={{
          maxWidth: "1240px",
          margin: "0 auto",
          padding: "28px 20px 48px"
        }}
      >
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "10px",
            marginBottom: "20px"
          }}
        >
          <StatusChip label={loadingModel ? "Loading model" : "Model ready"} active={modelReady} />
          <StatusChip label={cameraOn ? "Camera active" : "Camera inactive"} active={cameraOn} />
          <StatusChip label={faceDetected ? "Face detected" : "No face"} active={faceDetected} />
          <StatusChip
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
            padding: "26px",
            marginBottom: "20px",
            boxShadow: "0 10px 30px rgba(15,23,42,0.05)"
          }}
        >
          <div
            style={{
              display: "inline-block",
              padding: "8px 12px",
              borderRadius: "999px",
              background: "#eff6ff",
              color: "#1d4ed8",
              fontWeight: 700,
              fontSize: "0.82rem",
              marginBottom: "14px"
            }}
          >
            Professional V1 prototype
          </div>

          <h1
            style={{
              margin: 0,
              fontSize: "clamp(2rem, 4.2vw, 3.6rem)",
              lineHeight: 1.05
            }}
          >
            Facial Recognition under Privacy Constraints
          </h1>

          <p
            style={{
              marginTop: "12px",
              marginBottom: 0,
              maxWidth: "860px",
              color: "#475569",
              lineHeight: 1.8,
              fontSize: "1rem"
            }}
          >
            This prototype evaluates identity leakage by comparing live raw facial input
            with a graph-based representation built from facial landmarks. The goal is
            to explore how a facial pipeline can reduce reliance on identity-heavy data.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(280px, 320px) 1fr",
            gap: "20px",
            alignItems: "start"
          }}
        >
          <div
            style={{
              background: "#ffffff",
              border: "1px solid #e2e8f0",
              borderRadius: "24px",
              padding: "20px",
              boxShadow: "0 10px 30px rgba(15,23,42,0.04)"
            }}
          >
            <div style={{ fontWeight: 800, fontSize: "1rem", marginBottom: "16px" }}>
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
                  fontSize: "0.95rem"
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
                  fontSize: "0.95rem"
                }}
              >
                {privacyMode ? "Show raw input" : "Hide raw input"}
              </button>
            </div>

            <div style={{ marginTop: "18px" }}>
              <label
                htmlFor="noise-slider"
                style={{
                  display: "block",
                  fontWeight: 700,
                  fontSize: "0.92rem",
                  marginBottom: "10px"
                }}
              >
                Noise simulation: {noiseLevel}
              </label>
              <input
                id="noise-slider"
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
                fontSize: "0.93rem",
                lineHeight: 1.7
              }}
            >
              Pipeline:
              <br />
              raw face → landmarks → graph abstraction → reduced exposure
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
                  fontSize: "0.92rem",
                  lineHeight: 1.7
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
              gap: "20px"
            }}
          >
            <div
              style={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: "24px",
                padding: "18px",
                boxShadow: "0 10px 30px rgba(15,23,42,0.04)"
              }}
            >
              <div style={{ marginBottom: "12px" }}>
                <div style={{ fontWeight: 800, fontSize: "1rem" }}>Raw input panel</div>
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
                  aspectRatio: "4 / 3"
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
                    filter: privacyMode ? "blur(14px) brightness(0.66)" : "none",
                    opacity: cameraOn ? 1 : 0.12,
                    transition: "filter 180ms ease, opacity 180ms ease"
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
                      fontWeight: 700
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
                boxShadow: "0 10px 30px rgba(15,23,42,0.04)"
              }}
            >
              <div style={{ marginBottom: "12px" }}>
                <div style={{ fontWeight: 800, fontSize: "1rem" }}>Graph representation panel</div>
                <div style={{ color: "#64748b", fontSize: "0.9rem", marginTop: "4px" }}>
                  Structural facial representation
                </div>
              </div>

              <div
                style={{
                  borderRadius: "18px",
                  overflow: "hidden",
                  border: "1px solid #dbe4f0",
                  background: "#ffffff",
                  aspectRatio: "4 / 3"
                }}
              >
                <canvas
                  ref={graphCanvasRef}
                  width={frameSize.width}
                  height={frameSize.height}
                  style={{
                    width: "100%",
                    height: "100%",
                    display: "block"
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        <div
          style={{
            marginTop: "20px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "14px"
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
