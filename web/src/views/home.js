// Rakip seç: Hub'daki eğitilmiş rakipler + depo ekleme.

import { CONFIG } from "../../config.js";
import { addRepo, getToken, listOpponents, normalizeRepo, removeRepo, SAMPLE_REPO, setToken } from "../hub.js";
import { clear, h, TEMPO_TR, mount } from "../ui.js";

function opponentCard(item, refresh) {
  if (!item.meta) {
    return h("div", { class: "card" },
      h("div", { class: "name" }, item.repo),
      h("p", { class: "error small" }, item.error),
      h("button", { onclick: () => { removeRepo(item.repo); refresh(); } }, "Listeden kaldır"),
    );
  }
  const m = item.meta;
  const uncertain = m.score_state !== "ok";
  return h("a", { class: `card ${uncertain ? "uncertain" : ""}`, href: `#/rakip/${item.repo}` },
    h("div", { class: "row" },
      h("span", { class: "name" }, m.player),
      h("span", { class: "bigscore num", title: uncertain ? "Belirsiz skor" : "Rakip benzerlik skoru" }, m.main_score ?? "-"),
    ),
    h("div", { class: "row small muted" },
      h("span", null, `${TEMPO_TR[m.tempo] || m.tempo} · ${m.games?.fit ?? "?"} oyun`),
      h("span", null, uncertain ? "belirsiz" : "benzerlik / 100"),
    ),
    h("div", { class: "row small", style: "margin-top:8px" },
      h("span", { class: "chip" }, `Güven: ${m.confidence_label || "?"}`),
      h("span", { class: "muted" }, m.last_game ? `son oyun ${m.last_game}` : ""),
    ),
    item.repo === SAMPLE_REPO ? h("p", { class: "notice small", style: "margin:10px 0 0" }, "Örnek veri: arayüzü göstermek içindir, gerçek ölçüm değildir.") : null,
  );
}

export async function render(root) {
  const list = h("div", { class: "grid" }, h("p", { class: "muted" }, "Rakipler yükleniyor..."));
  const message = h("p", { class: "small error" });

  async function refresh() {
    const items = await listOpponents();
    clear(list);
    for (const item of items) list.append(opponentCard(item, refresh));
    list.append(
      h("a", { class: "card", href: "#/oyna/demo" },
        h("div", { class: "name" }, CONFIG.demo.name),
        h("p", { class: "small muted", style: "margin:6px 0 0" },
          "Kimseye özel eğitilmemiş, ortalama bir insan gibi oynayan model. Henüz rakip eğitmediysen siteyi bununla deneyebilirsin. Benzerlik skoru yoktur."),
      ),
    );
    if (!items.some((i) => i.meta && i.repo !== SAMPLE_REPO)) {
      list.prepend(
        h("div", { class: "card" },
          h("div", { class: "name" }, "Henüz eğitilmiş rakip yok"),
          h("p", { class: "small muted", style: "margin:6px 0 10px" }, "İlk rakibini Kaggle'da ücretsiz eğitebilirsin; yaklaşık yarım saat sürer."),
          h("a", { class: "btn primary", href: "#/egit" }, "Nasıl eğitilir?"),
        ),
      );
    }
  }

  const repoInput = h("input", { type: "text", placeholder: "kullanıcı/rakip-oyuncu-blitz", style: "flex:1;min-width:220px", "aria-label": "Hugging Face depo adı" });
  const tokenInput = h("input", { type: "password", placeholder: "hf_...", value: getToken(), style: "flex:1;min-width:220px", autocomplete: "off", "aria-label": "Hugging Face token" });

  mount(root, 
    h("h1", null, "Rakibini seç"),
    h("p", { class: "muted" }, "Her rakip, o oyuncunun geçmiş oyunlarından eğitilmiş bir bottur. Raporunu oku, benzerlik skoruna bak, sonra karşısına otur."),
    list,
    h("div", { class: "panel", style: "margin-top:16px" },
      h("h2", null, "Rakip ekle"),
      h("p", { class: "small muted" }, "Not defteri bittiğinde çıktının sonunda yazan depo adını buraya gir."),
      h("form", {
        class: "inline",
        onsubmit: async (event) => {
          event.preventDefault();
          const repo = normalizeRepo(repoInput.value);
          if (!repo) {
            message.textContent = "Depo adı \"kullanıcı/depo\" biçiminde olmalı.";
            return;
          }
          message.textContent = "";
          addRepo(repo);
          repoInput.value = "";
          await refresh();
        },
      }, repoInput, h("button", { class: "primary", type: "submit" }, "Ekle")),
      message,
      h("details", { style: "margin-top:12px" },
        h("summary", { class: "small" }, "Özel (private) depo kullanıyorum"),
        h("p", { class: "small muted", style: "margin-top:8px" },
          "Özel depoları okumak için salt-okunur (read) bir Hugging Face token'ı gerekir. Token yalnızca bu tarayıcıda saklanır ve yalnızca huggingface.co'ya gönderilir. Paylaşılan bilgisayarda kullanma."),
        h("div", { class: "inline" }, tokenInput,
          h("button", { onclick: async () => { setToken(tokenInput.value); await refresh(); } }, "Kaydet"),
          h("button", { onclick: async () => { setToken(""); tokenInput.value = ""; await refresh(); } }, "Sil"),
        ),
      ),
    ),
  );
  refresh().catch((error) => {
    clear(list).append(h("p", { class: "error" }, `Rakipler yüklenemedi: ${error.message}`));
  });
}
