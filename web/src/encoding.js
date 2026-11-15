// Maia-3 girdi/çıktı kodlaması. rakip/encoding.py ile birebir aynıdır; ikisi birlikte değişmeli.
// web/tests/index.html, Python'ın ürettiği örneklerle bu dosyayı karşılaştırır.

export const HISTORY = 8;
export const N_SQUARE_MOVES = 64 * 64;
export const N_MOVES = N_SQUARE_MOVES + 256;
const PIECE_CODE = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
const PROMO = ["q", "r", "b", "n"];

const squareIndex = (name) => (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
const squareName = (index) => String.fromCharCode(97 + (index & 7)) + String.fromCharCode(49 + (index >> 3));
const mirror = (index) => index ^ 56; // dikey çevirme (yatay numarası 7 - r)

/** (64) kodlar: 0 boş, 1-6 sıradaki tarafın P,N,B,R,Q,K, 7-12 rakibinkiler; sıradaki tarafın gözünden. */
export function boardCodes(chess) {
  const codes = new Int8Array(64);
  const turn = chess.turn();
  const rows = chess.board(); // rows[0] = 8. yatay
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = rows[r][f];
      if (!piece) continue;
      let index = (7 - r) * 8 + f;
      if (turn === "b") index = mirror(index);
      codes[index] = PIECE_CODE[piece.type] + (piece.color === turn ? 0 : 6);
    }
  }
  return codes;
}

/** Tek pozisyonun (64 x 12) one-hot gösterimi. */
export function tokensFromCodes(codes) {
  const out = new Float32Array(64 * 12);
  for (let sq = 0; sq < 64; sq++) if (codes[sq]) out[sq * 12 + codes[sq] - 1] = 1;
  return out;
}

/**
 * Son pozisyonlardan (en eski önce, güncel sonda) model girdisi: Float32Array(64 * 96).
 * 8'den az pozisyon varsa en eskisi öne doğru tekrarlanır.
 */
export function historyTensor(codesList) {
  const recent = codesList.slice(-HISTORY);
  const out = new Float32Array(64 * HISTORY * 12);
  const pad = HISTORY - recent.length;
  for (let slot = 0; slot < HISTORY; slot++) {
    const codes = recent[Math.max(0, slot - pad)];
    for (let sq = 0; sq < 64; sq++) {
      if (codes[sq]) out[sq * 96 + slot * 12 + codes[sq] - 1] = 1;
    }
  }
  return out;
}

/** Hamlenin 4352'lik sözlükteki indeksi (sıradaki tarafın gözünden). */
export function moveIndex(move, turn) {
  let from = squareIndex(move.from);
  let to = squareIndex(move.to);
  if (turn === "b") {
    from = mirror(from);
    to = mirror(to);
  }
  if (move.promotion) {
    return N_SQUARE_MOVES + ((from & 7) * 8 + (to & 7)) * 4 + PROMO.indexOf(move.promotion);
  }
  return from * 64 + to;
}

export function indexMove(index, turn) {
  let from, to, promotion;
  if (index >= N_SQUARE_MOVES) {
    const rest = index - N_SQUARE_MOVES;
    const pair = Math.floor(rest / 4);
    from = 48 + Math.floor(pair / 8);
    to = 56 + (pair % 8);
    promotion = PROMO[rest % 4];
  } else {
    from = Math.floor(index / 64);
    to = index % 64;
  }
  if (turn === "b") {
    from = mirror(from);
    to = mirror(to);
  }
  return { from: squareName(from), to: squareName(to), promotion };
}

export const uciOf = (move) => move.from + move.to + (move.promotion || "");

/** Açılış kitabı anahtarı: taş dizilişi + sıra + rok hakları (geçerken alma karesi hariç). */
export const positionKey = (fen) => fen.split(" ").slice(0, 3).join(" ");
