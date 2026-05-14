import { useState, useRef } from "react";

type Mode = "text" | "enhance" | "gan";

function App() {
  const [mode, setMode] = useState<Mode>("text");
  const [prompt, setPrompt] = useState("");
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedPreview, setUploadedPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1];
      setUploadedImage(base64);
      setUploadedPreview(result);
      setImage(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setUploadedImage(result.split(",")[1]);
      setUploadedPreview(result);
      setImage(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  }

  async function generate() {
    setLoading(true);
    setError(null);
    setImage(null);

    try {
      let res: Response;

      if (mode === "text") {
        if (!prompt.trim()) { setLoading(false); return; }
        res = await fetch("/api/generate-image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        });
      } else if (mode === "enhance") {
        if (!uploadedImage || !prompt.trim()) { setLoading(false); return; }
        res = await fetch("/api/enhance-image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ imageBase64: uploadedImage, prompt }),
        });
      } else {
        res = await fetch("/api/generate");
      }

      const data = await res.json() as { image?: string; error?: string };
      if (!res.ok || !data.image) {
        setError(data.error ?? "Generation failed. Please try again.");
      } else {
        setImage(data.image);
      }
    } catch {
      setError("Could not reach the server. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  const canGenerate =
    !loading &&
    (mode === "gan" ||
      (mode === "text" && prompt.trim().length > 0) ||
      (mode === "enhance" && !!uploadedImage && prompt.trim().length > 0));

  return (
    <div className="container">
      <h1>Image Generator</h1>
      <p className="subtitle">Generate images from text, enhance a photo, or use your trained GAN.</p>

      <div className="mode-tabs">
        <button className={`tab ${mode === "text" ? "active" : ""}`} onClick={() => setMode("text")}>
          Text to Image
        </button>
        <button className={`tab ${mode === "enhance" ? "active" : ""}`} onClick={() => setMode("enhance")}>
          Enhance Image
        </button>
        <button className={`tab ${mode === "gan" ? "active" : ""}`} onClick={() => setMode("gan")}>
          My GAN Model
        </button>
      </div>

      {mode === "text" && (
        <div className="prompt-area">
          <textarea
            className="prompt-input"
            placeholder="Describe what you want to generate... e.g. 'a cat sitting on a mountain at sunset'"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); generate(); } }}
          />
        </div>
      )}

      {mode === "enhance" && (
        <div className="enhance-area">
          <div
            className={`upload-zone ${uploadedPreview ? "has-image" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {uploadedPreview ? (
              <img src={uploadedPreview} alt="Uploaded" className="upload-preview" />
            ) : (
              <div className="upload-placeholder">
                <span className="upload-icon">↑</span>
                <p>Click or drag & drop an image to upload</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
          </div>
          <textarea
            className="prompt-input"
            placeholder="Describe what changes you want... e.g. 'make it look like winter' or 'add sunglasses'"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); generate(); } }}
          />
        </div>
      )}

      {mode === "gan" && (
        <p className="gan-note">Generates handwritten digits using your trained MNIST GAN.</p>
      )}

      <button onClick={generate} disabled={!canGenerate}>
        {loading ? "Generating..." : mode === "enhance" ? "Enhance Image" : "Generate"}
      </button>

      {loading && mode === "enhance" && (
        <p className="loading-note">Enhancement can take 20–40 seconds on first run…</p>
      )}

      <div className="image-box">
        {image ? (
          <img src={`data:image/png;base64,${image}`} alt="Result" />
        ) : error ? (
          <p className="placeholder" style={{ color: "#e05555" }}>{error}</p>
        ) : (
          <p className="placeholder">
            {mode === "text" && "Describe an image above and click Generate"}
            {mode === "enhance" && "Upload an image and describe the changes you want"}
            {mode === "gan" && "Click Generate to produce handwritten digits"}
          </p>
        )}
      </div>
    </div>
  );
}

export default App;
