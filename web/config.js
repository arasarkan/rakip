// Site ayarları. Yayınlamadan önce `githubRepo` ve isteğe bağlı olarak `hubAuthors` alanlarını düzenle.
export const CONFIG = {
  // Eğitilmiş rakip depolarını Hub'da bulmak için kullanılan etiket (rakip/config.py ile aynı).
  hubTag: "rakip-bot",

  // Yalnızca bu Hugging Face kullanıcılarının depoları listelensin. Boş bırakılırsa
  // etiketi taşıyan herkese açık tüm depolar listelenir.
  hubAuthors: ["imawss"],

  // Her zaman listelenecek depolar ("kullanıcı/depo").
  extraRepos: [],

  // Not defterinin bulunduğu GitHub deposu ("Kaggle'da nasıl eğitilir" sayfasındaki bağlantılar için).
  githubRepo: "https://github.com/arasarkan/rakip",
  notebookPath: "notebooks/rakip_train.ipynb",

  // Kişiselleştirilmemiş genel model: siteyi ve botu eğitim yapmadan denemek için.
  demo: {
    id: "demo/genel-maia3",
    name: "Genel Maia-3 (demo)",
    modelUrl: "https://huggingface.co/bqrio/maia3-onnx/resolve/main/maia3-5m.fp16.onnx",
    elo: 1900,
  },

  cdn: {
    ort: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/",
    chess: "https://cdn.jsdelivr.net/npm/chess.js@1.4.0/dist/esm/chess.js",
    chessground: "https://cdn.jsdelivr.net/npm/chessground@9.2.1/dist/chessground.min.js",
    stockfishJs: "https://unpkg.com/stockfish@18.0.5/bin/stockfish-18-lite-single.js",
    stockfishWasm: "https://unpkg.com/stockfish@18.0.5/bin/stockfish-18-lite-single.wasm",
  },
};
