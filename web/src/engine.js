// Tarayıcıda Stockfish (wasm): oyun sonu analizi ve karşılaştırma için.
// Tek iş parçacıklı "lite" sürüm kullanılır; özel sunucu başlığı gerektirmez.

import { CONFIG } from "../config.js";

/** Santipiyon -> kazanma şansı (0-100). rakip/similarity.py içindeki formülle aynı. */
export function winPercent(cp) {
  const c = Math.max(-2000, Math.min(2000, cp));
  return 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * c)) - 1);
}

const scoreToCp = (score) => (score.mate !== undefined ? (score.mate > 0 ? 10000 : -10000) - score.mate : score.cp);

export class Stockfish {
  constructor(worker) {
    this.worker = worker;
    this.queue = Promise.resolve();
    this.listener = null;
    worker.onmessage = (event) => {
      const line = typeof event.data === "string" ? event.data : String(event.data);
      if (this.listener) this.listener(line);
    };
  }

  static async create() {
    // Başka kaynaktaki betik doğrudan Worker yapılamaz; metni indirip Blob üzerinden başlatılır.
    // Stockfish wasm dosyasının adresini URL'nin "#" sonrasından okur.
    const source = await (await fetch(CONFIG.cdn.stockfishJs)).text();
    const blob = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
    const worker = new Worker(`${blob}#${encodeURIComponent(CONFIG.cdn.stockfishWasm)}`);
    const engine = new Stockfish(worker);
    await engine._command("uci", (line) => line === "uciok", 60000);
    await engine._command("isready", (line) => line === "readyok", 60000);
    return engine;
  }

  _command(text, until, timeoutMs = 120000, onLine = null) {
    const run = () =>
      new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          this.listener = null;
          reject(new Error("Stockfish yanıt vermedi."));
        }, timeoutMs);
        this.listener = (line) => {
          if (onLine) onLine(line);
          if (until(line)) {
            clearTimeout(timer);
            this.listener = null;
            resolve(line);
          }
        };
        this.worker.postMessage(text);
      });
    const result = this.queue.then(run, run);
    this.queue = result.catch(() => {});
    return result;
  }

  /** Yanıt beklemeyen komut (setoption, position); sıradaki işlerin arkasına eklenir. */
  _post(text) {
    const run = () => this.worker.postMessage(text);
    this.queue = this.queue.then(run, run);
  }

  /**
   * Pozisyonu değerlendirir. Dönen `cp`, hamle sırası olan tarafın gözündendir.
   * Sabit düğüm sayısı kullanılır; sonuç cihazın hızına göre değişmez.
   */
  async analyse(fen, { nodes = 120000, multipv = 1 } = {}) {
    const lines = new Map();
    this._post(`setoption name MultiPV value ${multipv}`);
    this._post(`position fen ${fen}`);
    const best = await this._command(
      `go nodes ${nodes}`,
      (line) => line.startsWith("bestmove"),
      120000,
      (line) => {
        if (!line.startsWith("info") || !line.includes(" pv ")) return;
        const pvIndex = Number((line.match(/ multipv (\d+)/) || [])[1] || 1);
        const cp = line.match(/ score cp (-?\d+)/);
        const mate = line.match(/ score mate (-?\d+)/);
        const pv = line.split(" pv ")[1].trim().split(" ");
        if (!cp && !mate) return;
        lines.set(pvIndex, { uci: pv[0], pv, cp: scoreToCp(cp ? { cp: Number(cp[1]) } : { mate: Number(mate[1]) }) });
      },
    );
    const bestMove = best.split(" ")[1];
    const ordered = [...lines.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
    return { best: bestMove && bestMove !== "(none)" ? bestMove : null, lines: ordered, cp: ordered.length ? ordered[0].cp : null };
  }

  quit() {
    try {
      this.worker.postMessage("quit");
      this.worker.terminate();
    } catch {
      /* zaten kapalı */
    }
  }
}
