// Uses a separately checked-out sparkDash; does not vendor its benchmark code.
import path from "node:path";
import fs from "node:fs";
import { pathToFileURL } from "node:url";

if (!process.env.SPARKDASH_DIR) {
  throw new Error("Set SPARKDASH_DIR to a sparkDash checkout with dependencies installed");
}
const modulePath = path.resolve(process.env.SPARKDASH_DIR, "server/collectors/DecodeBench.js");
const { DecodeBenchManager } = await import(pathToFileURL(modulePath).href);
const output = path.resolve(process.env.RESULTS_DIR || "results-local");
fs.mkdirSync(output, { recursive: true });
const manager = new DecodeBenchManager(
  path.join(output, "sparkdash-history.json"),
  path.join(output, "sparkdash-active.json"),
);
const job = manager.start({
  sparkId: "nvidia-fp4-local-k3", lanIp: "127.0.0.1", port: 8001,
  modelId: "ornith", concurrencies: [1, 2, 4], maxTokens: 400,
  promptType: "structured", debug: true,
});
while (true) {
  const current = manager.getJob(job.benchId);
  if (current?.status !== "running") {
    console.log(JSON.stringify(current, null, 2));
    const success = current?.status === "completed"
      && current.results?.length === 3
      && current.results.every(result => result.streamsFailed === 0);
    process.exit(success ? 0 : 1);
  }
  await new Promise(resolve => setTimeout(resolve, 1000));
}
