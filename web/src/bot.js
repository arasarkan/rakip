// Tarayıcıda çalışan bot: ONNX Runtime Web ile hamle olasılıkları + açılış kitabı.

import { CONFIG } from "../config.js";
import { boardCodes, historyTensor, moveIndex, positionKey, tokensFromCodes, uciOf } from "./encoding.js";
import { fetchModel, fileUrl, loadBook } from "./hub.js";

let ortPromise = null;
function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(`${CONFIG.cdn.ort}ort.wasm.bundle.min.mjs`).then((ort) => {
      ort.env.wasm.wasmPaths = CONFIG.cdn.ort;
      ort.env.wasm.numThreads = 1; // çok iş parçacığı özel sunucu başlıkları ister; tek iş parçacığı her yerde çalışır
      return ort;
    });
  }
  return ortPromise;
}

/** Oyunun pozisyon geçmişi: model son 8 pozisyonu görür. */
export class History {
  constructor(chess) {
    this.codes = [boardCodes(chess)];
  }
  push(chess) {
    this.codes.push(boardCodes(chess));
  }
  pop() {
    if (this.codes.length > 1) this.codes.pop();
  }
  truncate(length) {
    this.codes.length = Math.max(1, length);
  }
}

export class Bot {
  constructor({ ort, session, meta, book, name, isDemo }) {
    this.ort = ort;
    this.session = session;
    this.meta = meta;
    this.book = book; // { max_ply, min_count, positions } ya da null
    this.name = name;
    this.isDemo = isDemo;
    this.lastMs = 0;
  }

  /** Eğitilmiş bir rakibi (Hub deposu) ya da demo modelini yükler. */
  static async load(opponent, onProgress = () => {}) {
    const ort = await loadOrt();
    let url, version, meta, book = null, useAuth = false;
    if (opponent.isDemo) {
      url = CONFIG.demo.modelUrl;
      version = "demo-1";
      meta = { player: CONFIG.demo.name, elo_self: CONFIG.demo.elo, elo_oppo_default: CONFIG.demo.elo, temperature: 1, model: { precision: "fp16" } };
    } else {
      meta = opponent.meta;
      if (meta.sample) {
        // Örnek rakibin kendi modeli yoktur: genel model + örnek açılış kitabı kullanılır.
        url = CONFIG.demo.modelUrl;
        version = "demo-1";
      } else {
        url = fileUrl(opponent.repo, meta.model.file);
        version = meta.model.sha256;
        useAuth = true;
      }
      try {
        book = await loadBook(opponent.repo);
      } catch {
        book = null;
      }
    }
    const { bytes, cached } = await fetchModel(url, version, onProgress, useAuth);
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      // fp16 grafiklerde agresif birleştirmeler bilinen bir hataya takılıyor; "basic" her sürümde güvenli.
      graphOptimizationLevel: meta.model?.precision === "fp16" ? "basic" : "all",
    });
    const bot = new Bot({ ort, session, meta, book, name: meta.player, isDemo: !!opponent.isDemo });
    bot.cached = cached;
    bot.modelBytes = bytes.length;
    return bot;
  }

  /** Kitaptaki gerçek oynanma oranları: [{uci, n, p}] ya da null. */
  bookMoves(chess, ply) {
    if (!this.book || ply >= this.book.max_ply) return null;
    const entry = this.book.positions[positionKey(chess.fen())];
    if (!entry || entry.n < this.book.min_count) return null;
    const legal = new Set(chess.moves({ verbose: true }).map(uciOf));
    const moves = Object.entries(entry.moves)
      .filter(([uci]) => legal.has(uci))
      .map(([uci, n]) => ({ uci, n, p: n / entry.n }));
    return moves.length ? { total: entry.n, moves } : null;
  }

  /** Modelin yasal hamleler üzerindeki olasılık dağılımı (sıcaklık 1): [{uci, san, p}] büyükten küçüğe. */
  async policy(chess, history, oppElo) {
    const { ort } = this;
    const turn = chess.turn();
    const legal = chess.moves({ verbose: true });
    if (!legal.length) return [];
    const feeds = {};
    const names = this.session.inputNames;
    if (names.includes("history")) feeds.history = new ort.Tensor("float32", historyTensor(history.codes), [1, 64, 96]);
    if (names.includes("tokens")) {
      feeds.tokens = new ort.Tensor("float32", tokensFromCodes(history.codes[history.codes.length - 1]), [1, 64, 12]);
    }
    feeds.elo_self = new ort.Tensor("float32", Float32Array.of(this.meta.elo_self), [1]);
    feeds.elo_oppo = new ort.Tensor("float32", Float32Array.of(oppElo || this.meta.elo_oppo_default), [1]);
    const started = performance.now();
    const output = await this.session.run(feeds);
    this.lastMs = performance.now() - started;
    const logits = output.logits_move.data;
    const scored = legal.map((m) => ({ uci: uciOf(m), san: m.san, logit: logits[moveIndex(m, turn)] }));
    const max = Math.max(...scored.map((s) => s.logit));
    let sum = 0;
    for (const s of scored) sum += s.p = Math.exp(s.logit - max);
    for (const s of scored) s.p /= sum;
    return scored.sort((a, b) => b.p - a.p);
  }

  /**
   * Botun hamlesi. Kitaptaysa rakibin gerçek tercih oranlarıyla, değilse modelin
   * dağılımından sıcaklıkla örneklenir (argmax değil): bot rakibin tutarsızlıklarını da taşır.
   */
  async choose(chess, history, ply, { temperature = 1, oppElo, useBook = true, random = Math.random } = {}) {
    const book = useBook ? this.bookMoves(chess, ply) : null;
    if (book) {
      const pick = sample(book.moves.map((m) => m.p), random);
      const move = book.moves[pick];
      const legal = chess.moves({ verbose: true }).find((m) => uciOf(m) === move.uci);
      return {
        uci: move.uci, san: legal.san, source: "book", p: move.p, games: book.total,
        alternatives: book.moves.slice().sort((a, b) => b.p - a.p).slice(0, 4).map((m) => ({ ...m, san: sanOf(chess, m.uci) })),
      };
    }
    const policy = await this.policy(chess, history, oppElo);
    const t = Math.max(0.05, temperature);
    const weights = policy.map((m) => Math.pow(m.p, 1 / t));
    const pick = sample(weights, random);
    return {
      uci: policy[pick].uci, san: policy[pick].san, source: "model", p: policy[pick].p,
      alternatives: policy.slice(0, 4).map(({ uci, san, p }) => ({ uci, san, p })),
      ms: this.lastMs,
    };
  }
}

function sanOf(chess, uci) {
  const move = chess.moves({ verbose: true }).find((m) => uciOf(m) === uci);
  return move ? move.san : uci;
}

function sample(weights, random) {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = random() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}
