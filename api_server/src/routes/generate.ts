import { Router, type Request, type Response } from "express";
import http from "http";
import https from "https";
import crypto from "crypto";
import { PYTHON_BACKEND_URL } from "../lib/python-backend";

const router = Router();

const tempImages = new Map<string, { data: Buffer; mime: string; expires: number }>();

setInterval(() => {
  const now = Date.now();
  for (const [id, entry] of tempImages) {
    if (entry.expires < now) tempImages.delete(id);
  }
}, 30_000);

router.get("/temp-image/:id", (req: Request, res: Response) => {
  const entry = tempImages.get(req.params.id);
  if (!entry || entry.expires < Date.now()) {
    res.status(404).end();
    return;
  }
  res.setHeader("Content-Type", entry.mime);
  res.end(entry.data);
});

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
  const id = crypto.randomUUID();
  tempImages.set(id, {
    data: imageBuffer,
    mime: "image/jpeg",
    expires: Date.now() + 5 * 60 * 1000,
  });

  const domain = process.env.REPLIT_DEV_DOMAIN ?? process.env.REPLIT_DOMAINS?.split(",")[0];
  if (!domain) {
    res.status(500).json({ error: "Server misconfiguration: no public domain available." });
    return;
  }

  const token = process.env.HF_TOKEN;
  if (!token) {
    res.status(500).json({ error: "Missing HuggingFace token." });
    return;
  }

  const bodyJson = JSON.stringify({
    inputs: `https://${domain}/api/temp-image/${id}`,
    parameters: {
      prompt: prompt.trim(),
      guidance_scale: 7.5,
      num_inference_steps: 25,
      image_guidance_scale: 1.5,
    },
  });

  const options = {
    hostname: "api-inference.huggingface.co",
    path: "/models/timbrooks/instruct-pix2pix",
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(bodyJson),
    },
  };

  req.log.info({ id, domain }, "Calling HuggingFace instruct-pix2pix");

  const hfReq = https.request(options, (imgRes) => {
    tempImages.delete(id);
    const chunks: Buffer[] = [];
    imgRes.on("data", (chunk: Buffer) => chunks.push(chunk));
    imgRes.on("end", () => {
      const raw = Buffer.concat(chunks);
      const contentType = imgRes.headers["content-type"] ?? "";

      if (imgRes.statusCode === 200 && contentType.startsWith("image/")) {
        res.json({ image: raw.toString("base64") });
        return;
      }

      try {
        const parsed = JSON.parse(raw.toString()) as unknown;
        if (
          typeof parsed === "object" &&
          parsed !== null &&
          "error" in parsed &&
          typeof (parsed as { error?: unknown }).error === "string"
        ) {
          const message = (parsed as { error: string }).error;
          req.log.warn({ statusCode: imgRes.statusCode, message }, "HuggingFace img2img failed");
          res.status(message.toLowerCase().includes("loading") ? 503 : 502).json({
            error: message.toLowerCase().includes("loading")
              ? "Model is warming up, please try again in a minute."
              : "Image enhancement failed. Please try again.",
          });
          return;
        }
      } catch {
        // fall through
      }

      req.log.warn(
        { statusCode: imgRes.statusCode, contentType },
        "HuggingFace img2img returned unexpected response",
      );
      res.status(502).json({ error: "Image enhancement failed. Please try again." });
    });
  });

  hfReq.on("error", (err) => {
      tempImages.delete(id);
      req.log.error({ err }, "HuggingFace img2img request failed");
      res.status(502).json({ error: "Image enhancement failed. Please try again." });
    });

  hfReq.write(bodyJson);
  hfReq.end();
});

export default router;
