import { spawn } from "child_process";
import { logger } from "./logger";

const PYTHON_PORT = 5100;

export function startPythonBackend() {
  const scriptPath = "/home/runner/workspace/artifacts/gan-backend/app.py";
  const child = spawn("python3", [scriptPath], {
    env: { ...process.env, PORT: String(PYTHON_PORT) },
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout?.on("data", (data: Buffer) => {
    logger.info({ source: "python" }, data.toString().trim());
  });

  child.stderr?.on("data", (data: Buffer) => {
    logger.info({ source: "python" }, data.toString().trim());
  });

  child.on("exit", (code) => {
    logger.warn({ code }, "Python backend exited");
  });

  process.on("exit", () => child.kill());
}

export const PYTHON_BACKEND_URL = `http://127.0.0.1:${PYTHON_PORT}`;
