#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import puppeteer from "puppeteer";
import ffmpegStaticPath from "ffmpeg-static";

const PROJECT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FPS = 30;
const DEFAULT_RESULT_ROOT = path.join(PROJECT_DIR, "experiment");
const DEFAULT_DATASET_ROOT = path.join(PROJECT_DIR, "data");
const DEFAULT_OUTPUT_NAME = "rendered_video.mp4";

function parseIds(spec) {
  const ids = new Set();
  for (const raw of spec.replaceAll("，", ",").split(",")) {
    const part = raw.trim();
    const match = /^(\d+)(?:\s*-\s*(\d+))?$/.exec(part);
    if (!match) throw new Error(`Invalid --task-id value: ${spec}`);
    const start = Number(match[1]);
    const end = match[2] === undefined ? start : Number(match[2]);
    if (start < 1 || end < start) throw new Error(`Invalid --task-id range: ${part}`);
    for (let id = start; id <= end; id += 1) ids.add(id);
  }
  return [...ids].sort((a, b) => a - b);
}

function parseArgs(argv) {
  const options = {
    ids: null,
    model: "",
    resultRoot: DEFAULT_RESULT_ROOT,
    datasetRoot: DEFAULT_DATASET_ROOT,
    outputName: DEFAULT_OUTPUT_NAME,
    settleMs: 800,
    probeTimeoutMs: 8000,
    stableMs: 1200,
    extraCaptureMs: 600,
    overwrite: false,
  };
  const fields = {
    "--model": "model",
    "--result": "resultRoot",
    "--data": "datasetRoot",
    "--output-name": "outputName",
    "--settle-ms": "settleMs",
    "--probe-timeout-ms": "probeTimeoutMs",
    "--stable-ms": "stableMs",
    "--extra-capture-ms": "extraCaptureMs",
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--help") {
      console.log(`Usage: node utils/record_video.mjs --task-id 1,3-10 [options]
  --task-id IDS            Accepts 1, 1-10, 1,3-10, or 1, 3-10 without quotes
  --model NAME             Generated model directory (required if multiple exist)
  --result PATH            Default: ${DEFAULT_RESULT_ROOT}
  --data PATH              Default: ${DEFAULT_DATASET_ROOT}
  --output-name NAME       Default: ${DEFAULT_OUTPUT_NAME}
  --settle-ms N            Pause before recording (default: 800)
  --probe-timeout-ms N     Maximum animation probe time (default: 8000)
  --stable-ms N            Animation discovery quiet time (default: 1200)
  --extra-capture-ms N     Capture margin (default: 600)
  --overwrite             Replace an existing video`);
      process.exit(0);
    }
    if (key === "--overwrite") {
      options.overwrite = true;
      continue;
    }
    if (key === "--task-id") {
      const parts = [];
      while (index + 1 < argv.length && !argv[index + 1].startsWith("--")) {
        parts.push(argv[++index]);
      }
      if (!parts.length) throw new Error("--task-id needs at least one id or range");
      const spec = parts.join(" ").replace(/\s*-\s*/g, "-").replace(/\s*[,，]\s*/g, ",").trim().replace(/\s+/g, ",");
      options.ids = options.ids ? `${options.ids},${spec}` : spec;
      continue;
    }
    const field = fields[key];
    if (!field || index + 1 >= argv.length) throw new Error(`Unknown or incomplete argument: ${key}`);
    options[field] = argv[++index];
  }
  if (!options.ids) throw new Error("--task-id is required (e.g. 1, 1-10, or 1,3-10)");
  options.ids = parseIds(options.ids);
  for (const key of ["settleMs", "probeTimeoutMs", "stableMs", "extraCaptureMs"]) {
    options[key] = Number(options[key]);
    if (!Number.isInteger(options[key]) || options[key] < 0) throw new Error(`${key} must be a non-negative integer`);
  }
  options.resultRoot = path.resolve(options.resultRoot);
  options.datasetRoot = path.resolve(options.datasetRoot);
  if (!options.outputName || path.basename(options.outputName) !== options.outputName) {
    throw new Error("--output-name must be a filename");
  }
  if (!options.outputName.toLowerCase().endsWith(".mp4")) options.outputName += ".mp4";
  return options;
}

async function canExecute(filePath) {
  try {
    await access(filePath, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function resolveBrowserExecutable() {
  const executable = process.env.PUPPETEER_EXECUTABLE_PATH || await puppeteer.executablePath();
  if (await canExecute(executable)) return executable;
  throw new Error("Chrome not found; run npx puppeteer browsers install chrome or set PUPPETEER_EXECUTABLE_PATH");
}

async function resolveFfmpegExecutable() {
  for (const candidate of [process.env.FFMPEG_PATH, ffmpegStaticPath, "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]) {
    if (candidate && await canExecute(candidate)) return candidate;
  }
  return "ffmpeg";
}

function runFfmpeg(executable, args, allowedExitCodes = [0], input = null) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, { windowsHide: true, stdio: [input === null ? "ignore" : "pipe", "pipe", "pipe"] });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    child.on("error", reject);
    child.on("close", (code) => {
      if (allowedExitCodes.includes(code)) resolve({ stdout: Buffer.concat(stdout), stderr: Buffer.concat(stderr).toString() });
      else reject(new Error(`ffmpeg exited ${code}: ${Buffer.concat(stderr).toString().slice(-2000)}`));
    });
    if (input !== null) child.stdin.end(input);
  });
}

async function referenceDurationMs(ffmpeg, videoPath) {
  if (!fs.existsSync(videoPath)) throw new Error(`Reference video not found: ${videoPath}`);
  const result = await runFfmpeg(ffmpeg, ["-i", videoPath], [0, 1]);
  const match = /Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/.exec(result.stderr);
  return match ? Math.round((Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3])) * 1000) : 0;
}

function resolveSample(options, id) {
  const idDir = path.join(options.resultRoot, String(id));
  if (!fs.existsSync(idDir)) throw new Error(`Generated id directory not found: ${idDir}`);
  let modelDir = options.model.toLowerCase().replaceAll("/", "__");
  if (!modelDir) {
    const children = fs.readdirSync(idDir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => entry.name);
    if (children.length !== 1) throw new Error(`Specify --model for ${idDir}; found: ${children.join(", ")}`);
    modelDir = children[0];
  }
  const bundleDir = path.join(idDir, modelDir);
  const htmlPath = path.join(bundleDir, "index.html");
  if (!fs.existsSync(htmlPath)) throw new Error(`Generated HTML not found: ${htmlPath}`);
  const outputPath = path.join(bundleDir, options.outputName);
  const metaPath = outputPath.replace(/\.mp4$/i, ".meta.json");
  const referencePath = path.join(options.datasetRoot, String(id), "ground_truth.mp4");
  const resolutionPath = path.join(options.datasetRoot, String(id), "ground_truth.resolution.json");
  let viewport = { width: 1920, height: 1080 };
  if (fs.existsSync(resolutionPath)) {
    try {
      const resolution = String(JSON.parse(fs.readFileSync(resolutionPath, "utf8")).resolution);
      const match = /^(\d+)x(\d+)$/i.exec(resolution.trim());
      if (match) viewport = { width: Number(match[1]), height: Number(match[2]) };
    } catch { /* Use fallback viewport. */ }
  }
  return { id, modelDir, htmlPath, outputPath, metaPath, referencePath, viewport };
}

const TRACKER_SCRIPT = `(() => {
  const tracker = { start: performance.now(), lastEvent: performance.now(), estimatedMs: 0 };
  const seen = new WeakSet();
  const number = (value) => Number.isFinite(value) ? value : 0;
  function scan() {
    for (const animation of document.getAnimations({ subtree: true })) {
      if (seen.has(animation)) continue;
      seen.add(animation);
      tracker.lastEvent = performance.now();
      const timing = animation.effect?.getComputedTiming?.();
      const end = timing?.endTime;
      const start = number(animation.startTime);
      const cycle = number(timing?.delay) + number(timing?.duration) + number(timing?.endDelay);
      tracker.estimatedMs = Math.max(tracker.estimatedMs, start + (Number.isFinite(end) ? end : cycle));
    }
  }
  for (const name of ['animationstart', 'animationiteration', 'transitionstart']) {
    document.addEventListener(name, (event) => {
      tracker.lastEvent = performance.now();
      const style = getComputedStyle(event.target);
      const values = name.startsWith('transition')
        ? [style.transitionDuration, style.transitionDelay]
        : [style.animationDuration, style.animationDelay];
      const parse = (value) => Math.max(...value.split(',').map((item) => {
        const token = item.trim();
        return (parseFloat(token) || 0) * (token.endsWith('ms') ? 1 : 1000);
      }));
      tracker.estimatedMs = Math.max(tracker.estimatedMs, performance.now() - tracker.start + parse(values[0]) + parse(values[1]));
      scan();
    }, true);
  }
  const observer = new MutationObserver(() => { tracker.lastEvent = performance.now(); scan(); });
  const startObserver = () => {
    if (document.documentElement) {
      observer.observe(document.documentElement, { subtree: true, childList: true, attributes: true });
    }
  };
  startObserver();
  if (!document.documentElement) document.addEventListener('DOMContentLoaded', startObserver, { once: true });
  window.__animwebTracker = () => {
    scan();
    return { estimatedMs: tracker.estimatedMs, silenceMs: performance.now() - tracker.lastEvent };
  };
  window.__animwebPause = () => {
    for (const animation of document.getAnimations({ subtree: true })) {
      try { animation.pause(); animation.currentTime = 0; } catch {}
    }
  };
  window.__animwebPlay = () => {
    for (const animation of document.getAnimations({ subtree: true })) {
      try { animation.currentTime = 0; animation.play(); } catch {}
    }
  };
  Object.defineProperty(Document.prototype, 'hidden', { configurable: true, get: () => false });
  Object.defineProperty(Document.prototype, 'visibilityState', { configurable: true, get: () => 'visible' });
  setInterval(scan, 250);
})();`;

async function preparePage(page, viewport) {
  await page.evaluateOnNewDocument(TRACKER_SCRIPT);
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "no-preference" }]);
  await page.setViewport({ ...viewport, deviceScaleFactor: 1 });
  page.setDefaultNavigationTimeout(60000);
}

async function waitForAssets(page) {
  await page.waitForFunction(() =>
    [...document.images].every((image) => image.complete) &&
    [...(document.body?.querySelectorAll("*") ?? [])].some((element) => {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    }), { timeout: 5000 }).catch(() => {});
  await page.evaluate(() => document.fonts?.ready).catch(() => {});
}

async function wakeUp(page) {
  await page.bringToFront().catch(() => {});
  await page.evaluate(() => {
    window.dispatchEvent(new Event("focus"));
    window.dispatchEvent(new Event("pageshow"));
    window.dispatchEvent(new Event("resize"));
    document.dispatchEvent(new Event("visibilitychange"));
  });
}

async function probeDuration(browser, sample, options) {
  const page = await browser.newPage();
  try {
    await preparePage(page, sample.viewport);
    await page.goto(pathToFileURL(sample.htmlPath).href, { waitUntil: "load" });
    await waitForAssets(page);
    await wakeUp(page);
    await page.evaluate(() => { window.__animwebPause?.(); window.__animwebPlay?.(); });
    const started = Date.now();
    let summary = { estimatedMs: 0, silenceMs: 0 };
    while (Date.now() - started < options.probeTimeoutMs) {
      await new Promise((resolve) => setTimeout(resolve, 250));
      summary = await page.evaluate(() => window.__animwebTracker?.() ?? { estimatedMs: 0, silenceMs: 0 });
      if (summary.estimatedMs > 0 && summary.silenceMs >= options.stableMs) break;
    }
    return summary;
  } finally {
    await page.close();
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const WHITE_SAMPLE_WIDTH = 160;
const WHITE_SAMPLE_HEIGHT = 90;
const WHITE_FRAME_THRESHOLD = 0.985;

function whiteRatio(frame) {
  let whitePixels = 0;
  for (let index = 0; index < frame.length; index += 3) {
    if (frame[index] >= 245 && frame[index + 1] >= 245 && frame[index + 2] >= 245) {
      whitePixels += 1;
    }
  }
  return whitePixels / (frame.length / 3);
}

async function pageOpeningWhiteRatio(ffmpeg, png) {
  const { stdout } = await runFfmpeg(ffmpeg, [
    "-i", "pipe:0", "-vf", `scale=${WHITE_SAMPLE_WIDTH}:${WHITE_SAMPLE_HEIGHT}:flags=area`,
    "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
  ], [0], png);
  const frameSize = WHITE_SAMPLE_WIDTH * WHITE_SAMPLE_HEIGHT * 3;
  if (stdout.length < frameSize) throw new Error("Could not inspect the page opening frame");
  return whiteRatio(stdout.subarray(0, frameSize));
}

async function detectLeadingWhiteFrames(ffmpeg, webmPath) {
  const { stdout } = await runFfmpeg(ffmpeg, [
    "-i", webmPath, "-vf", `fps=${FPS},scale=${WHITE_SAMPLE_WIDTH}:${WHITE_SAMPLE_HEIGHT}:flags=area`,
    "-frames:v", "300", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
  ]);
  const frameSize = WHITE_SAMPLE_WIDTH * WHITE_SAMPLE_HEIGHT * 3;
  let count = 0;
  for (let offset = 0; offset + frameSize <= stdout.length; offset += frameSize) {
    if (whiteRatio(stdout.subarray(offset, offset + frameSize)) < WHITE_FRAME_THRESHOLD) break;
    count += 1;
  }
  return count;
}

async function captureWebm(browser, ffmpeg, sample, options, durationMs, webmPath, fallbackPngPath) {
  const page = await browser.newPage();
  let recorder;
  try {
    await preparePage(page, sample.viewport);
    await wakeUp(page);
    if (options.settleMs) await sleep(options.settleMs);
    recorder = await page.screencast({ path: webmPath, format: "webm", fps: FPS, ffmpegPath: ffmpeg });
    const recorderError = new Promise((_, reject) => recorder.once("error", reject));
    await Promise.race([
      recorderError,
      page.goto(pathToFileURL(sample.htmlPath).href, { waitUntil: "domcontentloaded" }),
    ]);
    // Inspect the actual page, since a white opening can be intentional.
    const openingRatio = await pageOpeningWhiteRatio(ffmpeg, await page.screenshot({ type: "png" }));
    await Promise.race([recorderError, sleep(durationMs)]);
    if (fallbackPngPath) await page.screenshot({ path: fallbackPngPath, type: "png" });
    await recorder.stop();
    recorder = null;
    return openingRatio;
  } finally {
    if (recorder) await recorder.stop().catch(() => {});
    await page.close();
  }
}

async function countFramesAndTrimTail(ffmpeg, webmPath, leadingFrames, trimStaticTail = true) {
  const filter = `fps=${FPS},trim=start_frame=${leadingFrames},setpts=PTS-STARTPTS`;
  const { stdout } = await runFfmpeg(ffmpeg, ["-i", webmPath, "-vf", filter, "-f", "framemd5", "pipe:1"]);
  const hashes = stdout.toString().split("\n").filter((line) => line && !line.startsWith("#"))
    .map((line) => line.split(",").at(-1).trim());
  if (!hashes.length) return null;
  let kept = hashes.length;
  if (trimStaticTail) {
    while (kept > FPS && hashes[kept - 1] === hashes[kept - 2]) kept -= 1;
  }
  return { filter, capturedFrameCount: hashes.length, keptFrameCount: kept, trimmedTrailingStaticFrames: hashes.length - kept };
}

async function renderOne(browser, ffmpeg, sample, options) {
  if (fs.existsSync(sample.outputPath) && fs.statSync(sample.outputPath).size > 0 && !options.overwrite) {
    console.log(`Skipping id=${sample.id}: ${sample.outputPath} already exists`);
    return "skipped";
  }
  const probe = await probeDuration(browser, sample, options);
  const referenceMs = await referenceDurationMs(ffmpeg, sample.referencePath);
  const detectedMs = Math.ceil(probe.estimatedMs || 0);
  const plannedMs = Math.max(detectedMs, referenceMs, 1000) + options.extraCaptureMs;
  const durationMs = Math.ceil(plannedMs / 1000) * 1000;
  const webmPath = `${sample.outputPath}.${process.pid}.recording.webm`;
  const partialPath = `${sample.outputPath}.${process.pid}.partial.mp4`;
  const fallbackPngPath = detectedMs === 0 ? `${sample.outputPath}.${process.pid}.fallback.png` : null;
  console.log(`\nRecording id=${sample.id} (${sample.modelDir}), animation=${detectedMs}ms, reference=${referenceMs}ms, planned=${durationMs}ms`);
  try {
    const openingWhiteRatio = await captureWebm(browser, ffmpeg, sample, options, durationMs, webmPath, fallbackPngPath);
    const leadingWhiteFrames = openingWhiteRatio >= WHITE_FRAME_THRESHOLD
      ? 0 : await detectLeadingWhiteFrames(ffmpeg, webmPath).catch((error) => {
        if (detectedMs > 0) throw error;
        return 0;
      });
    const trim = await countFramesAndTrimTail(ffmpeg, webmPath, leadingWhiteFrames, detectedMs > 0)
      .catch((error) => {
        if (detectedMs > 0) throw error;
        return null;
      });
    const outputFrameCount = detectedMs === 0 ? Math.ceil(durationMs / 1000 * FPS) : trim?.keptFrameCount;
    if (trim) {
      const filter = detectedMs === 0
        ? `${trim.filter},tpad=stop_mode=clone:stop_duration=${durationMs / 1000}`
        : trim.filter;
      await runFfmpeg(ffmpeg, ["-y", "-i", webmPath, "-vf", filter, "-r", String(FPS), "-frames:v", String(outputFrameCount), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", partialPath]);
    } else if (fallbackPngPath) {
      await runFfmpeg(ffmpeg, ["-y", "-loop", "1", "-framerate", String(FPS), "-i", fallbackPngPath,
        "-r", String(FPS), "-frames:v", String(outputFrameCount), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", partialPath]);
    } else {
      throw new Error("Screencast contains no decodable frames");
    }
    fs.renameSync(partialPath, sample.outputPath);
    fs.writeFileSync(sample.metaPath, `${JSON.stringify({
      id: sample.id,
      modelDir: sample.modelDir,
      htmlPath: sample.htmlPath,
      outputPath: sample.outputPath,
      viewport: sample.viewport,
      detectedAnimationDurationMs: detectedMs,
      referenceVideoDurationMs: referenceMs,
      plannedRecordingDurationMs: plannedMs,
      roundedRecordingDurationMs: durationMs,
      pageOpeningWhiteRatio: openingWhiteRatio,
      leadingWhiteFramesRemoved: trim ? leadingWhiteFrames : 0,
      capturedFrameCount: trim?.capturedFrameCount ?? 0,
      keptFrameCount: outputFrameCount,
      trimmedTrailingStaticFrames: trim?.trimmedTrailingStaticFrames ?? 0,
      usedScreenshotFallback: !trim,
      finalRecordingDurationMs: Math.round(outputFrameCount / FPS * 1000),
      createdAt: new Date().toISOString(),
    }, null, 2)}\n`);
    console.log(`Saved ${sample.outputPath}; ${trim ? `removed ${leadingWhiteFrames} leading white and ${trim.trimmedTrailingStaticFrames} trailing repeated frames` : "used a screenshot because the screencast had no usable frames"}`);
    return "done";
  } finally {
    fs.rmSync(webmPath, { force: true });
    fs.rmSync(partialPath, { force: true });
    if (fallbackPngPath) fs.rmSync(fallbackPngPath, { force: true });
  }
}

async function main() {
  const options = parseArgs(process.argv);
  const ffmpeg = await resolveFfmpegExecutable();
  const browser = await puppeteer.launch({
    executablePath: await resolveBrowserExecutable(),
    protocolTimeout: 300000,
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--allow-file-access-from-files",
      "--disable-web-security",
      "--autoplay-policy=no-user-gesture-required",
      "--disable-background-timer-throttling",
      "--disable-backgrounding-occluded-windows",
      "--disable-renderer-backgrounding",
      "--disable-dev-shm-usage",
    ],
  });
  const failures = [];
  let completed = 0;
  let skipped = 0;
  try {
    for (const id of options.ids) {
      try {
        const status = await renderOne(browser, ffmpeg, resolveSample(options, id), options);
        if (status === "skipped") skipped += 1;
        else completed += 1;
      } catch (error) {
        failures.push(id);
        console.error(`Failed id=${id}:`, error);
      }
    }
  } finally {
    await browser.close();
  }
  console.log(`Finished: ${completed} rendered, ${skipped} skipped, ${failures.length} failed`);
  if (failures.length) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
