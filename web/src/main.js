// Yönlendirici: adres çubuğundaki "#/..." kısmına göre sayfayı çizer.
//   #/                      rakip seç
//   #/egit                  Kaggle'da nasıl eğitilir
//   #/rakip/<kullanıcı>/<depo>   rapor + benzerlik skoru
//   #/oyna/<kullanıcı>/<depo>    botla oyna   (#/oyna/demo: genel model)
//   #/skor[/<kullanıcı>/<depo>]  skor nasıl hesaplanıyor

import { clear, errorBox, h } from "./ui.js";

const app = document.getElementById("app");
let cleanup = null;

const routes = [
  [/^\/?$/, () => import("./views/home.js"), "home"],
  [/^\/egit$/, () => import("./views/train.js"), "egit"],
  [/^\/skor(?:\/([\w.-]+)\/([\w.-]+))?$/, () => import("./views/score.js"), "skor"],
  [/^\/rakip\/([\w.-]+)\/([\w.-]+)$/, () => import("./views/opponent.js"), "home"],
  [/^\/oyna\/(demo)$/, () => import("./views/play.js"), "home"],
  [/^\/oyna\/([\w.-]+)\/([\w.-]+)$/, () => import("./views/play.js"), "home"],
];

async function render() {
  const path = location.hash.replace(/^#/, "") || "/";
  if (cleanup) {
    try {
      cleanup();
    } catch {
      /* sayfa zaten kapanmış */
    }
    cleanup = null;
  }
  clear(app).append(h("p", { class: "muted" }, "Yükleniyor..."));
  for (const [pattern, load, nav] of routes) {
    const match = path.match(pattern);
    if (!match) continue;
    document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav));
    try {
      const view = await load();
      const root = h("div");
      const result = await view.render(root, ...match.slice(1));
      if (path !== (location.hash.replace(/^#/, "") || "/")) return; // bu arada sayfa değişti
      clear(app).append(root);
      cleanup = typeof result === "function" ? result : null;
      window.scrollTo(0, 0);
    } catch (error) {
      console.error(error);
      clear(app).append(errorBox(error), h("p", null, h("a", { href: "#/" }, "Rakip listesine dön")));
    }
    return;
  }
  clear(app).append(h("div", { class: "panel" }, h("p", null, "Sayfa bulunamadı."), h("a", { href: "#/" }, "Rakip listesine dön")));
}

window.addEventListener("hashchange", render);
render();
