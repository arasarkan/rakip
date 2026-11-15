// Antrenman modu: botla oyna, oyun sonunda geri bildirim ve kritik anlar.

import { CONFIG } from "../../config.js";
import { Bot, History } from "../bot.js";
import { uciOf } from "../encoding.js";
import { Stockfish, winPercent } from "../engine.js";
import { loadOpponent } from "../hub.js";
import { clear, formatClock, h, lineTr, loadChess, loadChessground, pct, sanTr, TEMPO_TR, mount } from "../ui.js";

const TIME_CONTROLS = [
  { id: "1+0", tempo: "bullet" }, { id: "2+1", tempo: "bullet" },
  { id: "3+0", tempo: "blitz" }, { id: "3+2", tempo: "blitz" }, { id: "5+0", tempo: "blitz" }, { id: "5+3", tempo: "blitz" },
  { id: "10+0", tempo: "rapid" }, { id: "10+5", tempo: "rapid" }, { id: "15+10", tempo: "rapid" },
  { id: "30+0", tempo: "classical" }, { id: "30+20", tempo: "classical" }, { id: "90+30", tempo: "classical" },
];
const DEFAULT_TC = { bullet: "1+0", blitz: "3+2", rapid: "10+5", classical: "30+20" };
const CRITICAL_LOSS = 10; // kazanma şansı puanı
const ANALYSIS_NODES = 100000;

const plyOfFen = (fen) => {
  const parts = fen.split(" ");
  return (Number(parts[5] || 1) - 1) * 2 + (parts[1] === "b" ? 1 : 0);
};
const colorName = (c) => (c === "w" ? "white" : "black");

function destsOf(chess) {
  const dests = new Map();
  for (const m of chess.moves({ verbose: true })) {
    if (!dests.has(m.from)) dests.set(m.from, []);
    dests.get(m.from).push(m.to);
  }
  return dests;
}

function resultText(chess, userColor, flagged, resigned) {
  if (resigned) return "Terk ettin. Bot kazandı.";
  if (flagged) return flagged === userColor ? "Süren bitti. Bot kazandı." : "Botun süresi bitti. Kazandın.";
  if (chess.isCheckmate()) return chess.turn() === userColor ? "Mat oldun. Bot kazandı." : "Mat ettin. Kazandın.";
  if (chess.isStalemate()) return "Pat. Berabere.";
  if (chess.isThreefoldRepetition()) return "Üç kez tekrar. Berabere.";
  if (chess.isInsufficientMaterial()) return "Yetersiz materyal. Berabere.";
  if (chess.isDraw()) return "50 hamle kuralı. Berabere.";
  return "Oyun bitti.";
}

export async function render(root, a, b) {
  const isDemo = a === "demo" && !b;
  const repo = isDemo ? null : `${a}/${b}`;
  const opponent = isDemo ? { isDemo: true, meta: null, report: null } : await loadOpponent(repo);
  const Chess = await loadChess();
  const name = isDemo ? CONFIG.demo.name : opponent.meta.player;
  const tempo = opponent.meta?.tempo || "blitz";
  const state = { destroyed: false, timer: null, engine: null };

  const body = h("div");
  mount(root, 
    h("p", { class: "small" }, h("a", { href: isDemo ? "#/" : `#/rakip/${repo}` }, isDemo ? "← Rakipler" : `← ${name}`)),
    h("h1", null, `${name} ile oyna`),
    body,
  );

  // ---------------------------------------------------------------- kurulum
  function showSetup(message) {
    const colorSel = h("select", null,
      h("option", { value: "w" }, "Beyaz"), h("option", { value: "b" }, "Siyah"), h("option", { value: "r" }, "Rastgele"));
    const tcSel = h("select", null,
      TIME_CONTROLS.map((tc) => h("option", { value: tc.id, selected: tc.id === DEFAULT_TC[tempo] }, `${tc.id} (${TEMPO_TR[tc.tempo]})`)),
      h("option", { value: "none" }, "Süresiz"));
    const startSel = h("select", null,
      h("option", { value: "start" }, "Başlangıç pozisyonu"),
      opponent.report ? h("option", { value: "line" }, "Rakibin repertuvarından bir açılış") : null,
      h("option", { value: "fen" }, "Kendi pozisyonum (FEN)"),
      h("option", { value: "pgn" }, "Bir oyunun devamı (PGN)"));
    const lines = opponent.report
      ? [...opponent.report.openings.white.map((l) => ({ ...l, botColor: "w" })), ...opponent.report.openings.black.map((l) => ({ ...l, botColor: "b" }))]
      : [];
    const lineSel = h("select", { style: "width:100%" }, lines.map((l, i) =>
      h("option", { value: i }, `${l.botColor === "w" ? "Rakip beyaz" : "Rakip siyah"}: ${lineTr(l.moves_san)} (${l.games} oyun)`)));
    const fenInput = h("input", { type: "text", style: "width:100%", placeholder: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" });
    const pgnInput = h("textarea", { placeholder: "1. e4 e5 2. Nf3 Nc6 ..." });
    const extra = h("div");
    const tempInput = h("input", { type: "range", min: "0.3", max: "1.5", step: "0.1", value: String(opponent.meta?.temperature ?? 1), style: "flex:1" });
    const tempLabel = h("span", { class: "num small" });
    const eloInput = h("input", { type: "number", min: "400", max: "3400", step: "50", value: String(opponent.meta?.elo_oppo_default ?? CONFIG.demo.elo), style: "width:110px" });
    const error = h("p", { class: "error small" }, message || "");
    const updateTemp = () => {
      const t = Number(tempInput.value);
      tempLabel.textContent = `${t.toFixed(1)} · ${t < 0.8 ? "daha tutarlı" : t > 1.1 ? "daha değişken" : "rakibin doğal dağılımı"}`;
    };
    const updateExtra = () => {
      clear(extra);
      if (startSel.value === "line") {
        extra.append(h("div", { class: "field" }, h("label", null, "Açılış"), lineSel,
          h("p", { class: "small muted", style: "margin-top:4px" }, "Rengin açılışa göre ayarlanır: rakip kendi renginde oynar.")));
      }
      if (startSel.value === "fen") extra.append(h("div", { class: "field" }, h("label", null, "FEN"), fenInput));
      if (startSel.value === "pgn") extra.append(h("div", { class: "field" }, h("label", null, "PGN (oyun son pozisyondan devam eder)"), pgnInput));
      colorSel.disabled = startSel.value === "line";
    };
    startSel.addEventListener("change", updateExtra);
    tempInput.addEventListener("input", updateTemp);
    updateTemp();
    updateExtra();

    clear(body).append(h("div", { class: "panel", style: "max-width:560px" },
      isDemo ? h("div", { class: "notice" }, "Bu, kimseye özel eğitilmemiş genel modeldir. Belirli bir rakibe benzemez; yalnızca denemek içindir.") : null,
      h("div", { class: "field" }, h("label", null, "Nereden başlansın?"), startSel),
      extra,
      h("div", { class: "inline" },
        h("div", { class: "field" }, h("label", null, "Rengin"), colorSel),
        h("div", { class: "field" }, h("label", null, "Tempo"), tcSel),
        h("div", { class: "field" }, h("label", null, "Senin reytingin"), eloInput),
      ),
      h("div", { class: "field" },
        h("label", null, "Botun hamle çeşitliliği"),
        h("div", { class: "inline" }, tempInput, tempLabel),
        h("p", { class: "small muted", style: "margin-top:4px" },
          "Bot her zaman en olası hamleyi oynamaz; rakibin gerçek tercih oranlarına göre seçer. Böylece onun tutarsızlıklarını da taşır."),
      ),
      error,
      h("button", {
        class: "primary",
        onclick: () => {
          try {
            const chess = new Chess();
            const setupSans = [];
            let userColor = colorSel.value === "r" ? (Math.random() < 0.5 ? "w" : "b") : colorSel.value;
            if (startSel.value === "line") {
              const line = lines[Number(lineSel.value)];
              for (const uci of line.moves_uci) setupSans.push(chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] }).san);
              userColor = line.botColor === "w" ? "b" : "w";
            } else if (startSel.value === "fen") {
              chess.load(fenInput.value.trim());
            } else if (startSel.value === "pgn") {
              chess.loadPgn(pgnInput.value.trim());
            }
            if (chess.isGameOver()) throw new Error("Bu pozisyonda oyun zaten bitmiş.");
            startGame({
              chess, userColor, tc: tcSel.value, temperature: Number(tempInput.value),
              oppElo: Math.max(400, Math.min(3400, Number(eloInput.value) || CONFIG.demo.elo)),
            });
          } catch (e) {
            error.textContent = `Başlatılamadı: ${e.message}`;
          }
        },
      }, "Oyunu başlat"),
    ));
  }

  // ------------------------------------------------------------------- oyun
  async function startGame({ chess, userColor, tc, temperature, oppElo }) {
    const status = h("p", { class: "muted" }, "Bot yükleniyor...");
    const progress = h("div", { class: "progress" }, h("span"));
    clear(body).append(h("div", { class: "panel" }, status, progress));
    let bot;
    try {
      bot = await Bot.load(opponent, (fraction, cached, received) => {
        progress.firstChild.style.width = `${Math.round(fraction * 100)}%`;
        status.textContent = cached ? "Bot önbellekten yüklendi." : `Bot indiriliyor... ${received ? (received / 1e6).toFixed(1) + " MB" : ""}`;
      });
    } catch (e) {
      showSetup(`Bot yüklenemedi: ${e.message}`);
      return;
    }
    if (state.destroyed) return;
    const Chessground = await loadChessground();

    // Geçmiş: model son 8 pozisyonu görür. Kurulum hamleleri (açılış/PGN) de sayılır.
    const prior = chess.history({ verbose: true });
    const replay = new Chess(prior.length ? prior[0].before : chess.fen());
    const history = new History(replay);
    const log = []; // {ply, fenBefore, uci, san, by, info}
    for (const m of prior) {
      log.push({ ply: plyOfFen(m.before), fenBefore: m.before, uci: uciOf(m), san: m.san, by: "setup" });
      replay.move(m.san);
      history.push(replay);
    }
    const setupCount = log.length;

    const timed = tc !== "none";
    const [baseMin, incSec] = timed ? tc.split("+").map(Number) : [0, 0];
    const clocks = { w: baseMin * 60000, b: baseMin * 60000 };
    let lastTick = performance.now();
    let over = false, flagged = null, resigned = false, thinking = false, viewing = false;

    const boardEl = h("div");
    const boardWrap = h("div", { class: "board-wrap" }, boardEl);
    const topClock = h("div", { class: "clock" }, h("span", null, name), h("span", { class: "time" }));
    const bottomClock = h("div", { class: "clock" }, h("span", null, "Sen"), h("span", { class: "time" }));
    const statusEl = h("p", { style: "min-height:1.5em;margin:8px 0" });
    const moveList = h("div", { class: "movelist" });
    const side = h("div");
    const takebackBtn = h("button", { onclick: takeback }, "Hamleyi geri al");
    const resignBtn = h("button", { onclick: () => { resigned = true; finish(); } }, "Terk et");
    const backBtn = h("button", { style: "display:none", onclick: () => showLive() }, "Son pozisyona dön");
    const controls = h("div", { class: "inline" }, takebackBtn, resignBtn, backBtn,
      h("button", { onclick: () => { stop(); showSetup(); } }, "Yeni oyun"));

    clear(body).append(h("div", { class: "play" },
      h("div", null, topClock, boardWrap, bottomClock, statusEl, controls),
      h("div", null, h("div", { class: "panel" }, h("h3", null, "Hamleler"), moveList), side),
    ));

    const cg = Chessground(boardEl, {
      orientation: colorName(userColor),
      movable: { free: false, showDests: true, events: { after: onUserMove } },
      premovable: { enabled: false },
      draggable: { showGhost: true },
      animation: { duration: 180 },
    });

    function syncBoard() {
      const turn = chess.turn();
      const last = log.length ? log[log.length - 1] : null;
      cg.set({
        fen: chess.fen(),
        turnColor: colorName(turn),
        check: chess.inCheck(),
        lastMove: last ? [last.uci.slice(0, 2), last.uci.slice(2, 4)] : undefined,
        viewOnly: false,
        drawable: { autoShapes: [] },
        movable: { color: !over && turn === userColor && !thinking ? colorName(userColor) : undefined, dests: !over && turn === userColor ? destsOf(chess) : new Map() },
      });
      viewing = false;
      backBtn.style.display = "none";
    }

    function renderMoves() {
      clear(moveList);
      log.forEach((entry, i) => {
        if (entry.ply % 2 === 0 || i === 0) moveList.append(h("span", { class: "no" }, entry.ply % 2 === 0 ? `${entry.ply / 2 + 1}.` : `${(entry.ply + 1) / 2}...`));
        moveList.append(h("span", { class: "mv", style: entry.by === "setup" ? "opacity:.6" : "" }, sanTr(entry.san)));
      });
      moveList.scrollTop = moveList.scrollHeight;
    }

    function renderClocks() {
      const botColor = userColor === "w" ? "b" : "w";
      for (const [el, color] of [[topClock, botColor], [bottomClock, userColor]]) {
        el.lastChild.textContent = timed ? formatClock(clocks[color]) : "∞";
        el.classList.toggle("active", !over && chess.turn() === color);
        el.classList.toggle("low", timed && clocks[color] < Math.min(30000, baseMin * 6000));
      }
    }

    function tick() {
      if (over || !timed) return;
      const now = performance.now();
      const turn = chess.turn();
      // Saat, ilk gerçek hamle yapılana kadar işlemez.
      if (log.length > setupCount) clocks[turn] -= now - lastTick;
      lastTick = now;
      if (clocks[turn] <= 0) {
        clocks[turn] = 0;
        flagged = turn;
        finish();
      }
      renderClocks();
    }

    function applyMove(move, by, info) {
      const fenBefore = chess.fen();
      const mover = chess.turn();
      const made = chess.move(move);
      history.push(chess);
      log.push({ ply: plyOfFen(fenBefore), fenBefore, uci: uciOf(made), san: made.san, by, info });
      if (timed && log.length > setupCount + 0) clocks[mover] += incSec * 1000;
      lastTick = performance.now();
      renderMoves();
      if (chess.isGameOver()) {
        finish();
        return false;
      }
      return true;
    }

    function onUserMove(orig, dest) {
      if (over || chess.turn() !== userColor) return syncBoard();
      const candidates = chess.moves({ verbose: true }).filter((m) => m.from === orig && m.to === dest);
      if (!candidates.length) return syncBoard();
      const play = (promotion) => {
        const going = applyMove({ from: orig, to: dest, promotion }, "user");
        syncBoard();
        renderClocks();
        if (going) botTurn();
      };
      if (candidates.some((m) => m.promotion)) {
        const overlay = h("div", { class: "promo" }, ["q", "r", "b", "n"].map((p) =>
          h("button", { title: { q: "Vezir", r: "Kale", b: "Fil", n: "At" }[p], onclick: () => { overlay.remove(); play(p); } }, { q: "V", r: "K", b: "F", n: "A" }[p])));
        boardWrap.append(overlay);
      } else {
        play(undefined);
      }
    }

    async function botTurn() {
      if (over || state.destroyed || chess.turn() === userColor) return;
      thinking = true;
      statusEl.textContent = `${name} düşünüyor...`;
      syncBoard();
      const started = performance.now();
      let choice;
      try {
        choice = await bot.choose(chess, history, plyOfFen(chess.fen()), { temperature, oppElo });
      } catch (e) {
        statusEl.textContent = `Bot hamle üretemedi: ${e.message}`;
        thinking = false;
        return;
      }
      // İnsan gibi beklet: kitap hamleleri hızlı, diğerleri kalan süreye göre.
      const botColor = chess.turn();
      const target = choice.source === "book"
        ? 300 + Math.random() * 600
        : timed ? Math.min(4500, Math.max(500, clocks[botColor] / 60)) * (0.5 + Math.random()) : 500 + Math.random() * 500;
      const wait = Math.max(0, Math.min(target, timed ? clocks[botColor] - 300 : target) - (performance.now() - started));
      await new Promise((resolve) => setTimeout(resolve, wait));
      if (over || state.destroyed) return;
      thinking = false;
      applyMove({ from: choice.uci.slice(0, 2), to: choice.uci.slice(2, 4), promotion: choice.uci[4] }, "bot", choice);
      if (!over) statusEl.textContent = "Sıra sende.";
      if (!viewing) syncBoard();
      renderClocks();
    }

    function takeback() {
      if (over || thinking || chess.turn() !== userColor) return;
      let undone = 0;
      while (undone < 2 && log.length > setupCount) {
        chess.undo();
        history.pop();
        log.pop();
        undone++;
      }
      renderMoves();
      syncBoard();
    }

    function stop() {
      over = true;
      if (state.timer) clearInterval(state.timer);
      state.timer = null;
    }

    function showPosition(fen, shapes, lastMove) {
      viewing = true;
      cg.set({ fen, viewOnly: true, check: false, lastMove, turnColor: fen.split(" ")[1] === "w" ? "white" : "black", drawable: { autoShapes: shapes || [] }, movable: { color: undefined, dests: new Map() } });
      backBtn.style.display = "";
      boardWrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function showLive() {
      syncBoard();
      if (over) cg.set({ viewOnly: true });
    }

    function finish() {
      if (over) return;
      stop();
      thinking = false;
      const text = resultText(chess, userColor, flagged, resigned);
      statusEl.textContent = text;
      takebackBtn.disabled = resignBtn.disabled = true;
      syncBoard();
      cg.set({ viewOnly: true });
      renderClocks();
      showReview(text);
    }

    // ----------------------------------------------------- oyun sonu analizi
    function feedbackText(entry) {
      const c = entry.info;
      const others = c.alternatives.filter((alt) => alt.uci !== c.uci).slice(0, 2).map((alt) => `${sanTr(alt.san)} ${pct(alt.p)}`).join(", ");
      const no = `${Math.floor(entry.ply / 2) + 1}. hamle`;
      if (c.source === "book") {
        return `${no}: Bu pozisyonda ${name} gerçek oyunlarında ${pct(c.p)} ihtimalle ${sanTr(c.san)} oynuyor (${c.games} oyunda bu pozisyona gelmiş).` +
          (others ? ` Diğer tercihleri: ${others}.` : "");
      }
      return `${no}: Bu pozisyon ${isDemo ? "bir açılış kitabında" : `${name} oyunlarında`} yok; bot ${sanTr(c.san)} hamlesine ${pct(c.p)} olasılık verdi (model tahmini).` +
        (others ? ` Diğer adaylar: ${others}.` : "");
    }

    function showReview(resultLine) {
      const botMoves = log.filter((e) => e.by === "bot");
      const criticalBox = h("div", null, h("p", { class: "muted small" }, "Stockfish analizi başlatılıyor..."));
      const bar = h("div", { class: "progress" }, h("span"));
      clear(side).append(
        h("div", { class: "panel" },
          h("h3", null, "Sonuç"), h("p", null, resultLine),
          h("h3", null, "Kritik anlar"), bar, criticalBox),
        h("div", { class: "panel" },
          h("h3", null, `${name} bu oyunda nasıl seçti?`),
          botMoves.length
            ? h("ul", { class: "feedback small", style: "padding-left:18px;max-height:340px;overflow-y:auto" }, botMoves.map((entry) =>
                h("li", null, feedbackText(entry), " ",
                  h("button", { class: "link", onclick: () => showPosition(entry.fenBefore, [{ orig: entry.uci.slice(0, 2), dest: entry.uci.slice(2, 4), brush: "blue" }]) }, "pozisyonu göster"))))
            : h("p", { class: "muted small" }, "Bot bu oyunda hamle yapmadı."),
        ),
      );
      analyse(criticalBox, bar.firstChild).catch((e) => {
        clear(criticalBox).append(h("p", { class: "error small" }, `Analiz yapılamadı: ${e.message}`));
      });
    }

    async function analyse(box, fill) {
      const played = log.slice(setupCount);
      if (!played.length) {
        clear(box).append(h("p", { class: "muted small" }, "Analiz edilecek hamle yok."));
        return;
      }
      const engine = (state.engine ||= await Stockfish.create());
      const fens = played.map((e) => e.fenBefore);
      const evals = []; // her pozisyon için: hamle sırası olan tarafın gözünden {cp, best}
      for (let i = 0; i < fens.length; i++) {
        if (state.destroyed) return;
        evals.push(await engine.analyse(fens[i], { nodes: ANALYSIS_NODES }));
        fill.style.width = `${Math.round((100 * (i + 1)) / (fens.length + 1))}%`;
      }
      // Son pozisyon: oyun bittiyse sonucu, bitmediyse değerlendirmeyi kullan.
      let finalCp;
      if (chess.isCheckmate()) finalCp = -10000;
      else if (chess.isGameOver()) finalCp = 0;
      else finalCp = (await engine.analyse(chess.fen(), { nodes: ANALYSIS_NODES })).cp ?? 0;
      fill.style.width = "100%";

      const moments = [];
      played.forEach((entry, i) => {
        const before = evals[i];
        if (before.cp === null) return;
        const afterCp = i + 1 < evals.length ? evals[i + 1].cp : finalCp;
        if (afterCp === null) return;
        const loss = Math.max(0, winPercent(before.cp) - (100 - winPercent(afterCp)));
        if (loss >= CRITICAL_LOSS && before.best && before.best !== entry.uci) moments.push({ entry, loss, best: before.best });
      });
      moments.sort((x, y) => y.loss - x.loss);
      clear(box);
      if (!moments.length) {
        box.append(h("p", { class: "small" }, "Bu oyunda iki taraf da kazanma şansını 10 puandan fazla düşüren bir hamle yapmadı."));
        return;
      }
      box.append(h("ul", { class: "feedback small", style: "padding-left:18px" }, moments.slice(0, 10).map(({ entry, loss, best }) => {
        const probe = new Chess(entry.fenBefore);
        const bestSan = probe.move({ from: best.slice(0, 2), to: best.slice(2, 4), promotion: best[4] }).san;
        const who = entry.by === "user" ? "Sen" : name;
        const verb = entry.by === "user" ? "oynadın" : "oynadı";
        return h("li", null,
          `${Math.floor(entry.ply / 2) + 1}. hamle, ${who}: ${sanTr(entry.san)} ${verb}, en iyisi ${sanTr(bestSan)} idi. Kazanma şansı ${Math.round(loss)} puan düştü. `,
          h("button", {
            class: "link",
            onclick: () => showPosition(entry.fenBefore, [
              { orig: entry.uci.slice(0, 2), dest: entry.uci.slice(2, 4), brush: "red" },
              { orig: best.slice(0, 2), dest: best.slice(2, 4), brush: "green" },
            ]),
          }, "pozisyonu göster"));
      })));
      box.append(h("p", { class: "small muted" }, "Kırmızı ok: oynanan hamle. Yeşil ok: Stockfish'in önerdiği hamle."));
    }

    // ------------------------------------------------------------ başlangıç
    renderMoves();
    renderClocks();
    syncBoard();
    clear(side).append(h("div", { class: "panel small muted" },
      h("p", null, bot.cached ? "Bot tarayıcı önbelleğinden yüklendi." : `Bot indirildi (${(bot.modelBytes / 1e6).toFixed(1)} MB) ve önbelleğe alındı; bir dahaki sefere yeniden inmez.`),
      h("p", { style: "margin:0" }, bot.book ? `Açılış kitabı: ${Object.keys(bot.book.positions).length} pozisyon (ilk ${bot.book.max_ply / 2} hamle).` : "Açılış kitabı yok; bot ilk hamleden itibaren modelle oynar."),
    ));
    state.timer = setInterval(tick, 100);
    if (chess.turn() === userColor) statusEl.textContent = "Sıra sende.";
    else botTurn();
  }

  showSetup();
  return () => {
    state.destroyed = true;
    if (state.timer) clearInterval(state.timer);
    if (state.engine) state.engine.quit();
  };
}
