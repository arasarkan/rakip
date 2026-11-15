// Hugging Face Hub erişimi: rakip depolarını listeleme, dosya okuma, model önbelleği.
// Sunucu yoktur; her şey tarayıcıdan doğrudan Hub'a gider.

import { CONFIG } from "../config.js";

const HF = "https://huggingface.co";
const MODEL_CACHE = "rakip-models-v1";
const LS_REPOS = "rakip.repos";
const LS_TOKEN = "rakip.hfToken";

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* özel pencere vb. */
    }
  },
};

export const getToken = () => store.get(LS_TOKEN, "");
export const setToken = (token) => store.set(LS_TOKEN, (token || "").trim());
export const savedRepos = () => store.get(LS_REPOS, []);

export function normalizeRepo(input) {
  const text = (input || "").trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/+$/, "");
  const parts = text.split("/").filter(Boolean);
  return parts.length >= 2 && /^[\w.-]+$/.test(parts[0]) && /^[\w.-]+$/.test(parts[1]) ? `${parts[0]}/${parts[1]}` : null;
}

export function addRepo(repo) {
  const repos = savedRepos();
  if (!repos.includes(repo)) store.set(LS_REPOS, [...repos, repo]);
}

export function removeRepo(repo) {
  store.set(LS_REPOS, savedRepos().filter((r) => r !== repo));
}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// Geliştirme için örnek rakip: web/sample/ klasöründeki sahte veriler. Yalnızca
// yerelde (localhost) ya da adrese ?ornek eklenince listelenir.
export const SAMPLE_REPO = "ornek/ornek";
export const sampleEnabled = () =>
  ["localhost", "127.0.0.1"].includes(location.hostname) || new URLSearchParams(location.search).has("ornek");

export const fileUrl = (repo, file) =>
  repo === SAMPLE_REPO ? new URL(`./sample/${file}`, document.baseURI).href : `${HF}/${repo}/resolve/main/${file}`;

export async function fetchJson(repo, file) {
  const response = await fetch(fileUrl(repo, file), { headers: authHeaders(), cache: "no-cache" });
  if (!response.ok) {
    const error = new Error(
      response.status === 401 || response.status === 403
        ? "Bu depo özel. Ayarlardan Hugging Face token'ı gir."
        : response.status === 404
          ? `Depoda ${file} bulunamadı. Eğitim tamamlanmamış olabilir.`
          : `Hub yanıtı: ${response.status}`,
    );
    error.status = response.status;
    throw error;
  }
  return response.json();
}

/** Hub'da etiketli herkese açık depoları bulur; ağ hatasında boş liste döner. */
async function searchTagged() {
  const authors = CONFIG.hubAuthors.length ? CONFIG.hubAuthors : [null];
  const found = [];
  for (const author of authors) {
    const params = new URLSearchParams({ filter: CONFIG.hubTag, limit: "100" });
    if (author) params.set("author", author);
    try {
      const response = await fetch(`${HF}/api/models?${params}`, { headers: authHeaders() });
      if (response.ok) for (const model of await response.json()) found.push(model.id || model.modelId);
    } catch {
      /* çevrim dışı: yalnızca kayıtlı depolar gösterilir */
    }
  }
  return found;
}

/** Listelenecek tüm depolar ve her birinin meta.json içeriği. */
export async function listOpponents() {
  const tagged = await searchTagged();
  const repos = [...new Set([...(sampleEnabled() ? [SAMPLE_REPO] : []), ...CONFIG.extraRepos, ...savedRepos(), ...tagged])];
  const items = await Promise.all(
    repos.map(async (repo) => {
      try {
        return { repo, meta: await fetchJson(repo, "meta.json"), saved: savedRepos().includes(repo) };
      } catch (error) {
        return { repo, error: error.message, saved: savedRepos().includes(repo) };
      }
    }),
  );
  // Etiketli ama meta.json'u olmayan (yarım kalmış) depolar yalnızca kullanıcı eklediyse gösterilir.
  return items.filter((item) => item.meta || item.saved);
}

export async function loadOpponent(repo) {
  const [meta, scores, report] = await Promise.all([
    fetchJson(repo, "meta.json"),
    fetchJson(repo, "scores.json"),
    fetchJson(repo, "report.json"),
  ]);
  return { repo, meta, scores, report };
}

export const loadBook = (repo) => fetchJson(repo, "book.json");

/**
 * Model dosyasını indirir; tarayıcı önbelleğinde varsa oradan okur.
 * `version` (sha256) değişince dosya yeniden iner, aksi halde her ziyarette inmez.
 */
export async function fetchModel(url, version, onProgress = () => {}, useAuth = false) {
  const key = `${url}${url.includes("?") ? "&" : "?"}v=${version || "1"}`;
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(key);
    if (hit) {
      onProgress(1, true);
      return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true };
    }
  } catch {
    cache = null; // Cache API yoksa (ör. özel pencere) doğrudan indir
  }
  const response = await fetch(url, { headers: useAuth ? authHeaders() : {} });
  if (!response.ok) throw new Error(`Model indirilemedi (${response.status}).`);
  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(total ? received / total : 0, false, received);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  if (cache) {
    try {
      // Aynı modelin eski sürümlerini sil, yenisini yaz.
      for (const request of await cache.keys()) {
        if (request.url.split(/[?&]v=/)[0] === new URL(url).href.split(/[?&]v=/)[0]) await cache.delete(request);
      }
      await cache.put(key, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
    } catch {
      /* kota dolu: önbelleksiz devam */
    }
  }
  return { bytes, cached: false };
}
