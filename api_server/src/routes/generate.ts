import { Router, type Request, type Response } from "express";
import http from "http";
import https from "https";
import { PYTHON_BACKEND_URL } from "../lib/python-backend";

const router = Router();

router.get("/generate", (req: Request, res: Response) => {
  const url = `${PYTHON_BACKEND_URL}/generate`;
  http
    .get(url, (pyRes) => {
      res.status(pyRes.statusCode ?? 200);
      pyRes.pipe(res);
    })
    .on("error", (err) => {
      req.log.error({ err }, "Python backend request failed");
      res.status(503).json({ error: "Model is loading, please try again shortly." });
    });
});

router.post("/generate-image", (req: Request, res: Response) => {
  const { prompt } = req.body as { prompt?: string };
  if (!prompt || typeof prompt !== "string" || prompt.trim().length === 0) {
    res.status(400).json({ error: "A prompt is required." });
    return;
  }
  const encoded = encodeURIComponent(prompt.trim());
  const url = `https://image.pollinations.ai/prompt/${encoded}?width=768&height=768&nologo=true&model=flux`;

  https
    .get(url, (imgRes) => {
      if (imgRes.statusCode !== 200) {
        res.status(502).json({ error: "Image generation failed. Please try again." });
        return;
      }
      const chunks: Buffer[] = [];
      imgRes.on("data", (chunk: Buffer) => chunks.push(chunk));
      imgRes.on("end", () => {
        const b64 = Buffer.concat(chunks).toString("base64");
        res.json({ image: b64 });
      });
    })
    .on("error", (err) => {
      req.log.error({ err }, "Pollinations request failed");
      res.status(502).json({ error: "Image generation failed. Please try again." });
    });
});

router.post("/enhance-image", (req: Request, res: Response) => {
  const { imageBase64, prompt } = req.body as {
    imageBase64?: string;
    prompt?: string;
  };

  if (!imageBase64 || !prompt || prompt.trim().length === 0) {
    res.status(400).json({ error: "Both an image and a prompt are required." });
    return;
  }

  const imageBuffer = Buffer.from(imageBase64, "base64");
  const bodyJson = JSON.stringify({
    inputs: imageBase64,
    parameters: { prompt: prompt.trim() },
  });

  const options = {
    hostname: "api-inference.huggingface.co",
    path: "/models/timbrooks/instruct-pix2pix",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(bodyJson),
      "X-Wait-For-Model": "true",
    },
  };

  void imageBuffer;

  const hfReq = https.request(options, (hfRes) => {
    const chunks: Buffer[] = [];
    hfRes.on("data", (chunk: Buffer) => chunks.push(chunk));
    hfRes.on("end", () => {
      const raw = Buffer.concat(chunks);
      const contentType = hfRes.headers["content-type"] ?? "";

      if (contentType.startsWith("image/")) {
        res.json({ image: raw.toString("base64") });
        return;
      }

      try {
        const parsed = JSON.parse(raw.toString()) as unknown;
        if (
          typeof parsed === "object" &&
          parsed !== null &&
          "error" in parsed
        ) {
          const errMsg = (parsed as { error: string }).error;
          if (errMsg.toLowerCase().includes("loading")) {
            res
              .status(503)
              .json({ error: "Model is warming up, please try again in 30 seconds." });
          } else {
            res.status(502).json({ error: errMsg });
          }
        } else {
          res.status(502).json({ error: "Unexpected response from model." });
        }
      } catch {
        res.status(502).json({ error: "Image enhancement failed. Please try again." });
      }
    });
  });

  hfReq.on("error", (err) => {
    req.log.error({ err }, "HuggingFace request failed");
    res.status(502).json({ error: "Image enhancement failed. Please try again." });
  });

  hfReq.write(bodyJson);
  hfReq.end();
});

export default router;
