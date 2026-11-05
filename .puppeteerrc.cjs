const path = require("node:path");

module.exports = {
  cacheDirectory: path.join(__dirname, ".cache", "puppeteer"),
  "chrome-headless-shell": { skipDownload: true },
};
