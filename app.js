import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "onnx-community/SmolLM2-360M-Instruct-ONNX";
const ASSISTANT_NAME = "JARVIS";

const messagesElement = document.getElementById("messages");
const welcomeElement = document.getElementById("welcome");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const micButton = document.getElementById("micButton");
const newChatButton = document.getElementById("newChat");
const historyElement = document.getElementById("history");
const sidebar = document.getElementById("sidebar");
const mobileMenu = document.getElementById("mobileMenu");

let conversation = [];
let currentChatId = null;
let generator = null;
let modelLoading = false;
let isSpeaking = false;
let modelReady = false;

function getChats() {
  try { return JSON.parse(localStorage.getItem("aiHelperChats") || "[]"); }
  catch { return []; }
}
function saveChats(chats) { localStorage.setItem("aiHelperChats", JSON.stringify(chats)); }
function createChat() { return { id: Date.now().toString(), title: "New conversation", messages: [] }; }

function renderHistory() {
  historyElement.innerHTML = "";
  getChats().slice().reverse().forEach(chat => {
    const button = document.createElement("button");
    button.className = "history-item";
    button.textContent = chat.title || "New conversation";
    button.onclick = () => loadChat(chat.id);
    historyElement.appendChild(button);
  });
}

function loadChat(id) {
  const chat = getChats().find(c => c.id === id);
  if (!chat) return;
  currentChatId = chat.id;
  conversation = chat.messages || [];
  messagesElement.innerHTML = "";
  if (!conversation.length) welcomeElement.style.display = "block";
  else {
    welcomeElement.style.display = "none";
    conversation.forEach(m => addMessageToScreen(m.role, m.content, false));
  }
  sidebar.classList.remove("open");
}

function newChat() {
  const chat = createChat();
  currentChatId = chat.id;
  conversation = [];
  messagesElement.innerHTML = "";
  welcomeElement.style.display = "block";
  const chats = getChats();
  chats.push(chat);
  saveChats(chats);
  renderHistory();
  input.focus();
}
newChatButton.addEventListener("click", newChat);

function saveCurrentChat() {
  if (!currentChatId) currentChatId = createChat().id;
  const chats = getChats();
  let chat = chats.find(c => c.id === currentChatId);
  if (!chat) { chat = { id: currentChatId, title: "New conversation", messages: [] }; chats.push(chat); }
  chat.messages = conversation;
  const first = conversation.find(m => m.role === "user");
  if (first) chat.title = first.content.slice(0, 35) + (first.content.length > 35 ? "..." : "");
  saveChats(chats);
  renderHistory();
}

function addMessageToScreen(role, text, save = true) {
  if (role === "assistant") welcomeElement.style.display = "none";
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.innerHTML = `
    <div class="avatar">${role === "assistant" ? "J" : "You"}</div>
    <div class="message-content">
      <div class="message-text">${role === "assistant" ? formatLocalAnswer(text) : escapeHTML(text)}</div>
      ${role === "assistant" ? '<button class="listen-button" style="margin-top:10px;border:0;background:#f2f2f2;padding:7px 10px;border-radius:8px;cursor:pointer">🔊 Listen</button>' : ""}
    </div>`;
  messagesElement.appendChild(message);
  if (role === "assistant") message.querySelector(".listen-button").onclick = () => speak(text);
  scrollChat();
  if (save) saveCurrentChat();
}

function formatLocalAnswer(text) {
  // Basic Markdown-like formatting without allowing HTML injection.
  let html = escapeHTML(text);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\n/g, "<br>");
  return html;
}

function scrollChat() {
  const chat = document.getElementById("chat");
  chat.scrollTop = chat.scrollHeight;
}

async function loadModel() {
  if (generator) return generator;
  if (modelLoading) {
    while (modelLoading) await new Promise(r => setTimeout(r, 100));
    return generator;
  }

  modelLoading = true;
  setStatus("Downloading AI model… (~272 MB first time)", "loading");
  sendButton.disabled = true;

  try {
    const useWebGPU = !!navigator.gpu;
    generator = await pipeline("text-generation", MODEL_ID, {
      device: useWebGPU ? "webgpu" : "wasm",
      dtype: useWebGPU ? "q4f16" : "q4"
    });
    modelReady = true;
    setStatus(useWebGPU ? "JARVIS • WebGPU ready" : "JARVIS • CPU ready", "online");
    return generator;
  } catch (error) {
    console.error("MODEL ERROR:", error);
    generator = null;
    setStatus("AI could not load — tap refresh", "error");
    throw new Error("The local AI model could not be loaded. Try refreshing the page or using a browser with WebGPU support.");
  } finally {
    modelLoading = false;
    sendButton.disabled = false;
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || sendButton.disabled) return;
  input.value = "";
  resizeTextarea();
  welcomeElement.style.display = "none";

  conversation.push({ role: "user", content: text });
  addMessageToScreen("user", text);
  saveCurrentChat();
  sendButton.disabled = true;

  setStatus("JARVIS is thinking…", "loading");
  const thinking = document.createElement("div");
  thinking.className = "message assistant";
  thinking.id = "thinking";
  thinking.innerHTML = '<div class="avatar">AI</div><div class="message-content"><div class="message-text">Loading local AI…</div></div>';
  messagesElement.appendChild(thinking);
  scrollChat();

  try {
    const ai = await loadModel();
    thinking.querySelector(".message-text").textContent = "Thinking…";

    const recent = conversation.slice(-10).map(m => ({ role: m.role, content: m.content }));
    const messages = [
      { role: "system", content: "You are JARVIS, a calm, highly capable futuristic personal AI assistant. Be concise, confident, helpful and polite. Give accurate answers. You are running locally in the user's browser and do not have live web access. If the user asks for current or live information, clearly say that you cannot verify it live rather than inventing facts." },
      ...recent
    ];

    let answer = "";
    const streamer = new TextStreamer(ai.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: token => {
        answer += token;
        const target = thinking.querySelector(".message-text");
        if (target) target.innerHTML = formatLocalAnswer(answer);
        scrollChat();
      }
    });

    const output = await ai(messages, {
      max_new_tokens: 384,
      do_sample: false,
      streamer
    });

    const generated = output?.[0]?.generated_text;
    if (Array.isArray(generated)) answer = generated.at(-1)?.content || answer;
    else if (typeof generated === "string") answer = generated.slice(-12000) || answer;
    answer = answer || "I couldn't generate a response.";
    thinking.remove();
    conversation.push({ role: "assistant", content: answer });
    addMessageToScreen("assistant", answer);
    saveCurrentChat();
  } catch (error) {
    thinking.remove();
    const message = error?.message || "Unknown error.";
    const answer = "I couldn't start the local AI.\n\n" + message;
    conversation.push({ role: "assistant", content: answer });
    addMessageToScreen("assistant", answer);
    saveCurrentChat();
  } finally {
    sendButton.disabled = false;
    if (modelReady) setStatus(navigator.gpu ? "JARVIS • WebGPU ready" : "JARVIS • CPU ready", "online");
    input.focus();
  }
}

function setStatus(text, state) {
  const status = document.querySelector(".status");
  if (!status) return;
  status.innerHTML = `<span class="status-${state}"></span> ${escapeHTML(text)}`;
}

sendButton.addEventListener("click", sendMessage);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
function resizeTextarea() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 130) + "px"; }
input.addEventListener("input", resizeTextarea);

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  const recognition = new SpeechRecognition();
  recognition.lang = "en-IN";
  recognition.continuous = false;
  recognition.interimResults = false;
  micButton.addEventListener("click", () => { try { recognition.start(); micButton.textContent = "🔴"; } catch {} });
  recognition.onresult = e => { input.value = e.results[0][0].transcript; resizeTextarea(); sendMessage(); };
  recognition.onend = () => micButton.textContent = "🎤";
  recognition.onerror = () => micButton.textContent = "🎤";
} else {
  micButton.disabled = true;
  micButton.title = "Voice input is not supported by this browser";
}

function speak(text) {
  if (!("speechSynthesis" in window) || isSpeaking) return;
  isSpeaking = true;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text.slice(0, 5000));
  // Prefer a calm British-English male system voice when the device provides one.
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v => /Google UK English Male|Microsoft George|Microsoft Ryan|Daniel|Arthur/i.test(v.name));
  if (preferred) utterance.voice = preferred;
  utterance.lang = preferred?.lang || "en-GB";
  utterance.rate = 0.92;
  utterance.pitch = 0.82;
  utterance.onend = () => { isSpeaking = false; };
  utterance.onerror = () => { isSpeaking = false; };
  window.speechSynthesis.speak(utterance);
}

document.querySelectorAll(".suggestions button").forEach(button => button.addEventListener("click", () => {
  input.value = button.dataset.prompt;
  resizeTextarea();
  sendMessage();
}));
mobileMenu.addEventListener("click", () => sidebar.classList.toggle("open"));

document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    input.focus();
  }
});

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}

renderHistory();
const existingChats = getChats();
if (existingChats.length) loadChat(existingChats[existingChats.length - 1].id);
else newChat();

// Load after the interface appears. The first download is cached by the browser.
setTimeout(() => loadModel().catch(() => {}), 1200);
