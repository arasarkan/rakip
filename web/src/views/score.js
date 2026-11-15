// "Bu skor nasıl hesaplanıyor?" sayfası. Formül rakip/similarity.py ile aynıdır.

import { fetchJson } from "../hub.js";
import { h, pct, mount } from "../ui.js";

export async function render(root, owner, name) {
  let scores = null;
  if (owner && name) {
    try {
      scores = await fetchJson(`${owner}/${name}`, "scores.json");
    } catch {
      scores = null;
    }
  }
  const f = scores?.formula;

  mount(root, 
    owner ? h("p", { class: "small" }, h("a", { href: `#/rakip/${owner}/${name}` }, "← Rakip sayfası")) : null,
    h("h1", null, "Bu skor nasıl hesaplanıyor?"),
    h("p", { class: "muted" }, "Kısa cevap: botu, rakibin hiç görmediği en yeni oyunlarında sınıyoruz ve kaç hamlede onunla aynı hamleyi seçtiğine bakıyoruz."),

    h("div", { class: "panel" },
      h("h2", null, "1. Sınav, botun görmediği oyunlarla yapılır"),
      h("p", null, "Rakibin oyunlarının en yeni %15-20'si eğitimden önce ayrılır. Bot bu oyunları ne eğitimde ne de açılış kitabında görür. Böylece ölçülen şey ezber değil, botun rakibin gelecekteki oyunlarını tahmin edip edemediğidir."),
      h("h2", null, "2. Skoru şişiren hamleler sayılmaz"),
      h("ul", null,
        h("li", null, "Açılış kitabındaki hamleler: bot orada modelle değil, rakibin geçmiş tercihlerinden ezberle oynar."),
        h("li", null, "Tek yasal hamlesi olan pozisyonlar: tahmin edilecek bir şey yoktur."),
      ),
      h("h2", null, "3. İki referansla karşılaştırılır"),
      h("ul", null,
        h("li", null, h("strong", null, "Genel model: "), "kimseye özel eğitilmemiş, aynı seviyedeki ortalama bir insanı taklit eden model. Bot bundan anlamlı şekilde iyi değilse kişiselleştirme işe yaramamıştır ve bunu açıkça yazarız."),
        h("li", null, h("strong", null, "Stockfish: "), "en iyi hamleyi oynayan motor. Rakibini Stockfish'ten iyi tahmin edemeyen bir bot, onun gibi değil sadece iyi oynuyordur."),
      ),
    ),

    h("div", { class: "panel" },
      h("h2", null, "Formül"),
      h("pre", { class: "formula" }, "Ana skor = 100 × ( 0,6 × min(1, m / T)  +  0,2 × Açılış  +  0,2 × Hata )"),
      h("ul", null,
        h("li", null, h("strong", null, "m: "), "botun ilk tahmininin rakibin hamlesiyle aynı olma oranı."),
        h("li", null, h("strong", null, "T (tavan): "), "ulaşılabilecek en yüksek oran. İnsanlar aynı pozisyonda bile her zaman aynı hamleyi oynamaz, bu yüzden kusursuz bir taklit bile %100'e ulaşamaz. T en az 0,75 alınır; rakibin kendi kendisiyle tutarlılığı daha yüksekse o kullanılır. 0,75 bir varsayımdır."),
        h("li", null, h("strong", null, "Açılış: "), "rakibin yeni oyunlarının ilk 8 hamlesindeki seçimlerinin ne kadarı botun repertuvarında var."),
        h("li", null, h("strong", null, "Hata: "), "yarısı hata sıklığının benzerliği (oyun aşamasına göre), yarısı rakibin hata yaptığı pozisyonlarda botun da hata yapma oranı. Hata, kazanma şansını en az 10 puan düşüren hamledir."),
      ),
      h("p", null, "Hata profili hesaplanamadıysa ağırlıklar 0,75 (hamle) ve 0,25 (açılış) olur."),
      h("p", null, "Aynı formül genel model için de hesaplanır ve yan yana gösterilir. Skorlar hep aşağı yuvarlanır."),
      f ? h("div", null,
        h("h3", null, `${scores.player} için sayılar`),
        h("div", { class: "scroll" }, h("table", null,
          h("tbody", null,
            h("tr", null, h("td", null, "m (bot)"), h("td", { class: "r" }, pct(f.m, 1))),
            h("tr", null, h("td", null, "Genel model"), h("td", { class: "r" }, pct(f.b, 1))),
            h("tr", null, h("td", null, "Stockfish"), h("td", { class: "r" }, pct(scores.references.stockfish, 1))),
            h("tr", null, h("td", null, "T (tavan)"), h("td", { class: "r" }, `${pct(f.T, 0)} (${f.ceiling.source})`)),
            h("tr", null, h("td", null, "Açılış"), h("td", { class: "r" }, pct(f.opening, 0))),
            h("tr", null, h("td", null, "Hata"), h("td", { class: "r" }, f.error === null ? "-" : `${Math.floor(f.error)} / 100`)),
            h("tr", null, h("td", null, "Ölçüme giren pozisyon"), h("td", { class: "r" }, scores.counts.measured_positions)),
            h("tr", null, h("td", null, "Kitap nedeniyle çıkarılan"), h("td", { class: "r" }, scores.counts.excluded?.in_book ?? "-")),
            h("tr", null, h("td", null, "Zorunlu hamle nedeniyle çıkarılan"), h("td", { class: "r" }, scores.counts.excluded?.forced ?? "-")),
            h("tr", null, h("td", null, "Bot ile genel model farkı (%95 aralık)"),
              h("td", { class: "r" }, `${(100 * scores.references.bot_minus_generic_ci[0]).toFixed(1)} .. ${(100 * scores.references.bot_minus_generic_ci[1]).toFixed(1)} puan`)),
            scores.quantization ? h("tr", null, h("td", null, "Model sürümü"),
              h("td", { class: "r" }, `${scores.quantization.chosen}, ${(scores.quantization.bytes / 1e6).toFixed(1)} MB`)) : null,
          ),
        )),
        scores.quantization?.variants ? h("div", null,
          h("h3", { style: "margin-top:14px" }, "Küçültme (quantize) karşılaştırması"),
          h("div", { class: "scroll" }, h("table", null,
            h("thead", null, h("tr", null, h("th", null, "Sürüm"), h("th", { class: "r" }, "Boyut"), h("th", { class: "r" }, "Hamle eşleşmesi"), h("th", null, "Durum"))),
            h("tbody", null, scores.quantization.variants.map((v) =>
              h("tr", null, h("td", null, v.name), h("td", { class: "r" }, `${(v.bytes / 1e6).toFixed(1)} MB`), h("td", { class: "r" }, pct(v.match, 2)),
                h("td", null, v.name === scores.quantization.chosen ? "kullanılıyor" : v.accepted ? "uygun" : "benzerliği düşürdü")))),
          )),
        ) : null,
      ) : null,
    ),

    h("div", { class: "panel" },
      h("h2", null, "Güven seviyesi ve gri skor"),
      h("ul", null,
        h("li", null, "200'den az oyun: Düşük. 200-1000: Orta. 1000'den fazla: Yüksek."),
        h("li", null, "Ölçüm 1000'den az pozisyona dayanıyorsa ya da belirsizliği yüksekse seviye bir kademe düşer."),
        h("li", null, "Güven düşükse ya da bot genel modelden (veya Stockfish'ten) anlamlı şekilde iyi değilse skor gri ve \"belirsiz\" olarak gösterilir."),
      ),
      h("h2", null, "Gerçekçi beklenti"),
      h("p", null, "Araştırmalardaki en iyi kişisel modeller bile bir oyuncunun 10 hamlesinden 6-7'sinden fazlasını tutturamıyor. 100 puan, \"her hamlesi aynı\" demek değil, \"ulaşılabilir tavana ulaştı\" demektir."),
      h("h2", null, "Sınırlar"),
      h("ul", null,
        h("li", null, "Bot önünü görerek hesap yapmaz; yalnızca \"bu oyuncu burada ne oynardı\" sorusunu yanıtlar. Uzun zorunlu varyantlarda gerçek oyuncudan zayıf kalabilir."),
        h("li", null, "Temel model çevrim içi oyunlarla eğitildi. Masa başı klasik oyunlarda ve çok güçlü oyuncularda benzerlik daha düşük çıkabilir."),
        h("li", null, "Rakibin test dönemindeki oyunları geçmişinden çok farklıysa (yeni repertuvar) skor düşer; bu bir hata değil, gerçeğin kendisidir."),
        h("li", null, "Tarz göstergesi ve zayıf nokta tipleri kaba sınıflandırmalardır; kesin yargı değildir."),
      ),
    ),
  );
}
