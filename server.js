import express from "express";

const app = express();
const PORT = process.env.PORT || 3000;

// No API key is used by this version. The AI model runs in the user's browser.
app.use(express.static("public"));

app.get("/api/status", (_req, res) => {
  res.json({
    mode: "browser-local",
    apiKeyRequired: false,
    model: "onnx-community/Qwen3-0.6B-ONNX"
  });
});

app.listen(PORT, () => {
  console.log(`AI Helper running on port ${PORT}`);
  console.log("No API key required — AI inference runs in the browser.");
});
