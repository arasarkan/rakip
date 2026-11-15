// "Kaggle'da nasıl eğitilir" yönlendirmesi. Eğitim sitede değil, ücretsiz Kaggle/Colab GPU'sunda yapılır.

import { CONFIG } from "../../config.js";
import { h, notebookLinks, mount } from "../ui.js";

export async function render(root) {
  const links = notebookLinks();
  mount(root, 
    h("h1", null, "Yeni rakip eğit"),
    h("p", { class: "muted" },
      "Eğitim senin bilgisayarında değil, Kaggle'ın ücretsiz GPU'sunda çalışır. 1000 oyunluk bir rakip için yaklaşık 30-40 dakika sürer. Para ödemen ya da kart girmen gerekmez."),
    links
      ? h("div", { class: "panel inline" },
          h("a", { class: "btn primary", href: links.kaggle, target: "_blank", rel: "noopener" }, "Not defterini Kaggle'da aç"),
          h("a", { class: "btn", href: links.colab, target: "_blank", rel: "noopener" }, "Colab'da aç (yedek)"),
          h("a", { class: "btn", href: links.github, target: "_blank", rel: "noopener" }, "GitHub'da gör"),
        )
      : h("div", { class: "notice" },
          "Bu sitenin ayarlarında GitHub deposu henüz tanımlı değil (web/config.js içindeki githubRepo). Not defteri depodaki ",
          h("code", null, CONFIG.notebookPath), " dosyasıdır."),
    h("div", { class: "panel" },
      h("h2", null, "Adım adım"),
      h("ol", { class: "steps" },
        h("li", null, h("strong", null, "Hesapları aç. "), "kaggle.com ve huggingface.co üzerinde ücretsiz hesap aç. Kaggle'da GPU kullanabilmek için telefon numaranı doğrulaman gerekir."),
        h("li", null, h("strong", null, "Hugging Face token'ı oluştur. "), "Hugging Face'te Settings > Access Tokens > New token, tür olarak ", h("em", null, "Write"), " seç."),
        h("li", null, h("strong", null, "Token'ı Kaggle'a gizli anahtar olarak ekle. "), "Not defterinde Add-ons > Secrets > Add secret. Ad: ", h("code", null, "HF_TOKEN"), ". Ekledikten sonra not defterine erişimi aç. Token'ı hiçbir hücreye yazma."),
        h("li", null, h("strong", null, "Ayarları yap. "), "Sağ panelde Accelerator: GPU T4 x1, Internet: On."),
        h("li", null, h("strong", null, "Parametreleri doldur. "), "İlk kod hücresinde oyuncunun adını, tempoyu ve kaynağı (Lichess / Chess.com kullanıcı adı ya da PGN) yaz."),
        h("li", null, h("strong", null, "Çalıştır. "), "Save Version > Save & Run All (Commit). Böyle çalıştırınca iş bitince oturum kendiliğinden kapanır ve haftalık GPU kotan boşa gitmez. Sekmeyi kapatabilirsin."),
        h("li", null, h("strong", null, "Rakibi siteye ekle. "), "Çıktının sonunda yazan depo adını (", h("code", null, "kullanıcı/rakip-..."), ") ",
          h("a", { href: "#/" }, "Rakipler"), " sayfasındaki \"Rakip ekle\" alanına gir. Herkese açık depolar listede kendiliğinden de görünür."),
      ),
    ),
    h("div", { class: "panel" },
      h("h2", null, "Bilmen gerekenler"),
      h("ul", null,
        h("li", null, "Seans yarıda kesilirse aynı parametrelerle yeniden çalıştır: eğitim son kayıttan devam eder."),
        h("li", null, "Seçtiğin tempoda 200'den az oyun varsa bot yine eğitilir ama skor belirsiz (gri) gösterilir."),
        h("li", null, "Masa başı oyunlarda saat bilgisi olmadığı için zaman kullanımı bölümü boş kalır."),
        h("li", null, "Kaggle haftada yaklaşık 30 saat GPU verir; bir rakip bunun küçük bir kısmını kullanır."),
      ),
    ),
  );
}
