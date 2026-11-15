// Küçük arayüz yardımcıları (çatı kullanılmıyor).

import { CONFIG } from "../config.js";

/** h("div", {class: "x", onclick: fn}, çocuklar...) */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "class") el.className = value;
    else if (key === "html") el.innerHTML = value;
    else if (key in el && key !== "list" && typeof value !== "string") el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const TEMPO_TR = { classical: "klasik", rapid: "rapid", blitz: "blitz", bullet: "bullet" };
export const pct = (x, digits = 0) => (x === null || x === undefined || Number.isNaN(x) ? "-" : `%${(100 * x).toFixed(digits)}`);

/** İngilizce SAN -> Türkçe taş harfleri (Ş, V, K, F, A). */
const SAN_TR = { K: "Ş", Q: "V", R: "K", B: "F", N: "A" };
export const sanTr = (san) => (san || "").replace(/[KQRBN]/g, (c) => SAN_TR[c]);

/** Hamle dizisini "1. e4 e5 2. Af3" biçiminde yazar. */
export function lineTr(sans, startPly = 0) {
  const parts = [];
  sans.forEach((san, i) => {
    const ply = startPly + i;
    if (ply % 2 === 0) parts.push(`${ply / 2 + 1}.`);
    else if (i === 0) parts.push(`${(ply + 1) / 2}...`);
    parts.push(sanTr(san));
  });
  return parts.join(" ");
}

export function bar(value, cls = "") {
  const v = Math.max(0, Math.min(100, value || 0));
  return h("div", { class: "bar" }, h("span", { class: cls, style: `width:${v}%` }));
}

let chessPromise = null;
export const loadChess = () => (chessPromise ||= import(CONFIG.cdn.chess).then((m) => m.Chess));
let cgPromise = null;
export const loadChessground = () => (cgPromise ||= import(CONFIG.cdn.chessground).then((m) => m.Chessground));

/** Salt görüntülenen küçük tahta. */
export async function miniBoard(el, fen, { orientation = "white", lastMove = null, shapes = [] } = {}) {
  const Chessground = await loadChessground();
  // Tahta, sayfaya eklenip boyut kazanmadan çizilirse oklar hesaplanamaz.
  for (let i = 0; i < 200 && !(el.isConnected && el.offsetWidth > 0); i++) {
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
  if (!el.isConnected) return null;
  const cg = Chessground(el, {
    fen, orientation, viewOnly: true, coordinates: false, lastMove: lastMove || undefined,
    drawable: { enabled: false, visible: true, autoShapes: shapes },
  });
  return cg;
}

/** Çocukları (iç içe diziler ve null dahil) güvenle ekler. */
export function mount(root, ...children) {
  root.append(h("div", null, ...children));
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function errorBox(error) {
  return h("div", { class: "panel" }, h("p", { class: "error" }, error?.message || String(error)));
}

export function formatClock(ms) {
  const total = Math.max(0, ms);
  const s = Math.floor(total / 1000);
  const m = Math.floor(s / 60);
  if (total < 10000) return `${m}:${String(s % 60).padStart(2, "0")}.${Math.floor((total % 1000) / 100)}`;
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function notebookLinks() {
  const match = CONFIG.githubRepo.match(/github\.com\/([^/]+)\/([^/]+?)(?:\.git)?\/?$/);
  if (!match || match[1] === "KULLANICI_ADIN") return null;
  const [, user, repo] = match;
  const raw = `https://raw.githubusercontent.com/${user}/${repo}/main/${CONFIG.notebookPath}`;
  return {
    github: `${CONFIG.githubRepo}/blob/main/${CONFIG.notebookPath}`,
    kaggle: `https://kaggle.com/kernels/welcome?src=${encodeURIComponent(raw)}`,
    colab: `https://colab.research.google.com/github/${user}/${repo}/blob/main/${CONFIG.notebookPath}`,
  };
}
