// Rakip sayfası: benzerlik skoru + rakip raporu.

import { loadOpponent, SAMPLE_REPO } from "../hub.js";
import { bar, h, lineTr, miniBoard, pct, sanTr, TEMPO_TR, mount } from "../ui.js";

const PHASE_TR = { opening: "Açılış", middlegame: "Orta oyun", endgame: "Oyun sonu" };

/** Ana skor + dört alt skor. Belirsizse gri gösterilir. */
export function scoreCard(scores, repo) {
  const { main, sub, references: refs } = scores;
  const uncertain = main.state !== "ok";
  const row = (label, value, valueText, text, cls = "") =>
    h("div", { class: "sub" },
      h("div", null, label),
      bar(value, cls),
      h("div", { class: "value" }, valueText),
      h("div", { class: "text" }, text),
    );
  const compareRow = (label, value, cls) =>
    value === null || value === undefined ? null
      : h("div", { class: "rowc" }, h("span", null, label), bar(100 * value, cls), h("span", { class: "num", style: "text-align:right" }, pct(value, 1)));

  return h("section", { class: `panel score ${uncertain ? "uncertain" : ""}` },
    h("div", { class: "score-head" },
      h("span", { class: "label" }, "Rakip benzerlik skoru"),
      uncertain ? h("span", { class: "chip" }, "belirsiz") : null,
    ),
    h("div", { class: "score-main" }, main.score, h("small", null, " / 100")),
    h("blockquote", null, main.sentence),
    uncertain ? h("div", { class: "notice" }, main.verdict) : h("p", { class: "small muted" }, main.verdict),
    h("div", { class: "subscores" },
      row("Açılış uyumu", sub.opening.score, sub.opening.score, sub.opening.text),
      row("Hamle uyumu", sub.move.score, sub.move.score, sub.move.text),
      sub.error.available
        ? row("Hata profili", sub.error.score, sub.error.score, sub.error.text)
        : row("Hata profili", 0, "-", sub.error.text),
      row("Güven seviyesi", sub.confidence.bar, sub.confidence.label, sub.confidence.text),
    ),
    h("h3", { style: "margin-top:18px" }, "Karşılaştırma"),
    h("p", { class: "small muted" }, "Aynı pozisyonlarda rakiple birebir aynı hamleyi oynama oranı."),
    h("div", { class: "compare" },
      compareRow("Bu bot", refs.bot, ""),
      compareRow("Genel model", refs.generic, "ref"),
      compareRow("Stockfish", refs.stockfish, "ref"),
    ),
    h("p", { class: "small muted", style: "margin-top:10px" },
      `Aynı formülle genel modelin skoru: ${main.generic_score} / 100. `, scores.expectation),
    (scores.notes || []).map((note) => h("p", { class: "small muted" }, note)),
    h("p", { class: "small" }, h("a", { href: `#/skor/${repo}` }, "Bu skor nasıl hesaplanıyor?")),
  );
}

function openingsTable(lines, colorLabel) {
  if (!lines.length) return h("p", { class: "muted small" }, `${colorLabel} oyun yok.`);
  return h("div", { class: "scroll" },
    h("table", null,
      h("thead", null, h("tr", null, h("th", null, "Hat"), h("th", { class: "r" }, "Oyun"), h("th", { class: "r" }, "Pay"), h("th", { class: "r" }, "Skor"), h("th", { class: "r" }, "+ / = / -"))),
      h("tbody", null, lines.map((line) =>
        h("tr", null,
          h("td", null, line.name ? h("div", null, line.name) : null, h("div", { class: "moves-line muted" }, lineTr(line.moves_san))),
          h("td", { class: "r" }, line.games),
          h("td", { class: "r" }, pct(line.share)),
          h("td", { class: "r" }, line.score === null ? "-" : pct(line.score)),
          h("td", { class: "r" }, `${line.wins} / ${line.draws} / ${line.losses}`),
        ))),
    ),
  );
}

function phaseSection(phase) {
  if (!phase.available) return h("p", { class: "muted" }, "Hata analizi yapılamadı (Stockfish değerlendirmesi yok).");
  return h("div", null,
    phase.text ? h("p", null, phase.text) : null,
    h("div", { class: "scroll" }, h("table", null,
      h("thead", null, h("tr", null, h("th", null, "Aşama"), h("th", { class: "r" }, "İncelenen hamle"), h("th", { class: "r" }, "Ciddi hata"), h("th", { class: "r" }, "Çok ağır hata"))),
      h("tbody", null, Object.entries(phase.phases).filter(([, v]) => v.n).map(([key, v]) =>
        h("tr", null, h("td", null, PHASE_TR[key]), h("td", { class: "r" }, v.n), h("td", { class: "r" }, pct(v.error_rate, 1)), h("td", { class: "r" }, pct(v.blunder_rate, 1))))),
    )),
    h("p", { class: "small muted" }, "Ciddi hata: kazanma şansını en az 10 puan düşüren hamle. Çok ağır hata: en az 20 puan."),
  );
}

function timeSection(time) {
  if (!time.available) return h("p", { class: "muted" }, time.text);
  return h("div", null,
    h("p", null, time.text),
    time.error_rate_in_trouble !== undefined
      ? h("p", null, `Zaman sıkıntısında hata oranı ${pct(time.error_rate_in_trouble, 1)}, normalde ${pct(time.error_rate_normal, 1)}.`)
      : null,
    time.time_trouble_loss_rate !== null && time.time_trouble_loss_rate !== undefined
      ? h("p", { class: "muted small" }, `Zaman sıkıntısına düştüğü oyunların ${pct(time.time_trouble_loss_rate)} kadarını kaybetmiş.`)
      : null,
    h("div", { class: "scroll" }, h("table", null,
      h("thead", null, h("tr", null, h("th", null, "Hamleler"), h("th", { class: "r" }, "Hamle başına süre (toplam sürenin yüzdesi)"))),
      h("tbody", null, (time.per_move_range || []).map((r) =>
        h("tr", null, h("td", null, r.moves), h("td", { class: "r" }, pct(r.share_of_base_per_move, 1))))),
    )),
    h("p", { class: "small muted" }, `${time.games_with_clock} oyunda saat bilgisi vardı.`),
  );
}

function weaknessSection(weak, playerColorOf) {
  if (!weak.available) return h("p", { class: "muted" }, "Belirgin bir zayıf nokta bulunamadı ya da yeterli veri yok.");
  return h("div", null, weak.items.map((item) => {
    const examples = h("div", { class: "examples" });
    for (const ex of item.examples) {
      const board = h("div", { class: "mini" });
      const orientation = ex.fen.split(" ")[1] === "w" ? "white" : "black";
      examples.append(h("div", { class: "example" }, board,
        h("div", { class: "cap" },
          `${ex.move_number}. hamle: ${sanTr(ex.played_san)} oynadı, en iyisi ${sanTr(ex.best_san)} (kayıp ${ex.loss} puan)`,
          ex.date ? ` · ${ex.date}` : ""),
      ));
      miniBoard(board, ex.fen, {
        orientation,
        shapes: [
          { orig: ex.played.slice(0, 2), dest: ex.played.slice(2, 4), brush: "red" },
          { orig: ex.best.slice(0, 2), dest: ex.best.slice(2, 4), brush: "green" },
        ],
      });
    }
    return h("div", { style: "margin-bottom:18px" },
      h("h3", null, item.type),
      h("p", { class: "small muted" }, `${item.text} (${item.n} pozisyon incelendi)`),
      item.examples.length ? h("p", { class: "small muted" }, "Kırmızı ok: oynadığı hamle. Yeşil ok: en iyi hamle.") : null,
      examples,
    );
  }));
}

export async function render(root, owner, name) {
  const repo = `${owner}/${name}`;
  const { meta, scores, report } = await loadOpponent(repo);
  const s = report.summary;
  const r = report.results;

  mount(root, 
    h("p", { class: "small" }, h("a", { href: "#/" }, "← Rakipler")),
    h("div", { class: "inline", style: "justify-content:space-between;margin-bottom:12px" },
      h("div", null,
        h("h1", null, meta.player),
        h("p", { class: "muted", style: "margin:0" },
          `${TEMPO_TR[meta.tempo] || meta.tempo} · ${s.games} oyun`, s.first_game ? ` · ${s.first_game} - ${s.last_game}` : "",
          s.rating_latest ? ` · son reyting ${s.rating_latest}` : ""),
      ),
      h("a", { class: "btn primary", href: `#/oyna/${repo}` }, "Botla oyna"),
    ),
    repo === SAMPLE_REPO ? h("div", { class: "notice" }, "Bu bir örnek sayfadır: içindeki sayılar arayüzü göstermek için üretilmiştir, gerçek bir ölçüm değildir.") : null,
    (report.warnings || []).map((w) => h("div", { class: "notice" }, w)),
    scoreCard(scores, repo),
    h("div", { class: "panel" },
      h("h2", null, "Sonuçlar"),
      h("div", { class: "scroll" }, h("table", null,
        h("thead", null, h("tr", null, h("th", null, ""), h("th", { class: "r" }, "Oyun"), h("th", { class: "r" }, "Skor"), h("th", { class: "r" }, "+ / = / -"))),
        h("tbody", null, [["all", "Toplam"], ["white", "Beyazla"], ["black", "Siyahla"]].map(([key, label]) =>
          h("tr", null, h("td", null, label), h("td", { class: "r" }, r[key].games),
            h("td", { class: "r" }, r[key].score === null ? "-" : pct(r[key].score)),
            h("td", { class: "r" }, `${r[key].wins} / ${r[key].draws} / ${r[key].losses}`)))),
      )),
    ),
    h("div", { class: "panel" },
      h("h2", null, "Açılış repertuvarı"),
      h("h3", null, "Beyazla en sık 5 hat"), openingsTable(report.openings.white, "Beyazla"),
      h("h3", { style: "margin-top:16px" }, "Siyahla en sık 5 hat"), openingsTable(report.openings.black, "Siyahla"),
    ),
    h("div", { class: "two" },
      h("div", { class: "panel" }, h("h2", null, "Hangi aşamada hata yapıyor?"), phaseSection(report.phase_errors)),
      h("div", { class: "panel" }, h("h2", null, "Zaman kullanımı"), timeSection(report.time)),
    ),
    h("div", { class: "panel" },
      h("h2", null, "Taktik mi, pozisyonel mi?"),
      report.style.available
        ? h("div", null,
            h("div", { class: "sub" }, h("div", null, "Pozisyonel"), bar(report.style.complexity_index), h("div", { class: "value" }, "Taktik"),
              h("div", { class: "text" }, report.style.text)),
            h("p", { class: "small muted", style: "margin-top:8px" }, report.style.note),
          )
        : h("p", { class: "muted" }, "Yeterli veri yok."),
    ),
    h("div", { class: "panel" },
      h("h2", null, "Zayıf noktalar"),
      h("p", { class: "small muted" }, "En çok puan kaybettiği pozisyon tipleri, kendi oyunlarından örneklerle."),
      weaknessSection(report.weaknesses),
    ),
    h("p", null, h("a", { class: "btn primary", href: `#/oyna/${repo}` }, "Botla oyna")),
  );
}
