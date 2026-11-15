import random

import chess
import numpy as np
import pytest

from rakip import encoding as enc
from rakip.book import build_book, opening_coverage, top_lines, OpeningBook
from rakip.features import build_samples, nominal_elo
from rakip.games import GameRecord
from rakip.split import LeakageError

from conftest import make_record, random_moves


# ---- Maia-3 kodlaması ----------------------------------------------------

def reference_vocab():
    """Resmi Maia-3 kodundaki get_all_possible_moves() ile aynı sıra."""
    moves = []
    for rank in range(8):
        for file in range(8):
            src = chess.square(file, rank)
            for trank in range(8):
                for tfile in range(8):
                    moves.append(chess.square_name(src) + chess.square_name(chess.square(tfile, trank)))
    for ff in "abcdefgh":
        for ft in "abcdefgh":
            for piece in "qrbn":
                moves.append(f"{ff}7{ft}8{piece}")
    return moves


def reference_mirror(uci):
    def sq(s):
        return s[0] + str(9 - int(s[1]))
    return sq(uci[:2]) + sq(uci[2:4]) + uci[4:]


def test_vocabulary_matches_upstream_order():
    vocab = reference_vocab()
    assert len(vocab) == enc.N_MOVES == 4352
    table = {m: i for i, m in enumerate(vocab)}
    rng = random.Random(0)
    checked = 0
    for seed in range(40):
        board = chess.Board()
        for move in random_moves(random.Random(seed), 120):
            for legal in board.legal_moves:
                uci = legal.uci() if board.turn == chess.WHITE else reference_mirror(legal.uci())
                assert enc.move_index(legal, board.turn) == table[uci], (board.fen(), legal)
                assert enc.index_move(table[uci], board.turn) == legal
                checked += 1
            board.push(move)
    assert checked > 50000


def test_promotions_including_underpromotion_and_black():
    white = chess.Board("1n5k/P7/8/8/8/8/7p/K5N1 w - - 0 1")
    black = chess.Board("1n5k/P7/8/8/8/8/7p/K5N1 b - - 0 1")
    vocab = reference_vocab()
    for board in (white, black):
        promos = [m for m in board.legal_moves if m.promotion]
        assert len(promos) == 8  # düz + alarak, 4'er taş
        for m in promos:
            idx = enc.move_index(m, board.turn)
            assert idx >= enc.N_SQUARE_MOVES
            uci = m.uci() if board.turn == chess.WHITE else reference_mirror(m.uci())
            assert vocab[idx] == uci
            assert enc.index_move(idx, board.turn) == m


def reference_tokens(board):
    """Resmi tokenize_board() ile aynı mantık."""
    tokens = np.zeros((64, 12), dtype=np.float32)
    if board.turn == chess.BLACK:
        board = board.mirror()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tokens[square][piece.piece_type + (6 if piece.color == chess.BLACK else 0) - 1] = 1
    return tokens


def test_board_tokens_match_upstream_for_both_sides():
    for seed in range(20):
        board = chess.Board()
        for move in random_moves(random.Random(seed), 80):
            got = enc.codes_to_tokens(enc.board_codes(board))
            assert np.array_equal(got, reference_tokens(board)), board.fen()
            board.push(move)


def test_history_stack_matches_upstream_deque_semantics():
    from collections import deque

    moves = random_moves(random.Random(3), 30)
    board = chess.Board()
    codes = []
    history = deque(maxlen=8)
    for ply, move in enumerate(moves):
        codes.append(enc.board_codes(board))
        history.append(reference_tokens(board))
        ref = np.concatenate(list(history), axis=1)
        if len(history) < 8:
            pad = np.tile(history[0], (1, 8 - len(history)))
            ref = np.concatenate([pad, ref], axis=1)
        got = enc.stack_history(np.stack(codes), enc.history_indices(ply)[None, :])[0]
        assert got.shape == (64, 96)
        assert np.array_equal(got, ref), ply
        board.push(move)


def test_legal_mask_roundtrip_and_position_key():
    board = chess.Board()
    mask = enc.legal_mask(board)
    assert mask.sum() == 20
    assert np.array_equal(enc.unpack_mask(enc.pack_mask(mask)), mask)
    board.push_san("e4")
    assert enc.position_key(board) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq"


# ---- açılış kitabı -------------------------------------------------------

def game_from_san(i, san_moves, color="white", result=1.0, date="2024-01-01", filler_seed=None):
    board = chess.Board()
    moves = []
    for san in san_moves:
        move = board.parse_san(san)
        moves.append(move.uci())
        board.push(move)
    rng = random.Random(filler_seed if filler_seed is not None else i)
    while len(moves) < 24 and not board.is_game_over():
        move = rng.choice(list(board.legal_moves))
        moves.append(move.uci())
        board.push(move)
    return GameRecord(
        game_id=f"b{i:04d}", source="lichess", player_color=color, tempo="blitz", date=date, result=result,
        player_rating=2000, opponent_rating=2000, opponent_name="x", time_control="300", moves=moves,
        opening="Test Açılışı", eco="C20",
    )


@pytest.fixture
def repertoire():
    games = []
    for i in range(6):
        games.append(game_from_san(i, ["e4", "e5", "Nf3", "Nc6"], result=1.0))
    for i in range(6, 9):
        games.append(game_from_san(i, ["e4", "c5", "Nf3", "d6"], result=0.0))
    games.append(game_from_san(9, ["d4", "d5", "c4", "e6"], result=0.5))
    return games


def test_book_frequencies_follow_real_choices(repertoire):
    book = build_book(repertoire, min_count=3)
    probs = book.probabilities(chess.Board(), 0)
    assert probs == {"e2e4": 0.9, "d2d4": 0.1}
    after = chess.Board()
    after.push_san("e4")
    after.push_san("e5")
    assert book.probabilities(after, 2) == {"g1f3": 1.0}
    # d4 hattına yalnızca 1 kez girilmiş: min_count altında, kitapta değil
    d4 = chess.Board()
    d4.push_san("d4")
    d4.push_san("d5")
    assert book.lookup(d4, 2) is None


def test_book_sampling_reproduces_distribution(repertoire):
    book = build_book(repertoire, min_count=3)
    rng = random.Random(1)
    picks = [book.sample(chess.Board(), rng, 0).uci() for _ in range(4000)]
    assert abs(picks.count("e2e4") / 4000 - 0.9) < 0.02
    assert abs(picks.count("d2d4") / 4000 - 0.1) < 0.02


def test_book_respects_max_ply_and_only_player_moves(repertoire):
    book = build_book(repertoire, max_ply=2, min_count=3)
    after = chess.Board()
    after.push_san("e4")
    after.push_san("e5")
    assert book.lookup(after, 2) is None
    # siyahın hamle sırası olan pozisyonlar (rakip beyazken) kitaba girmez
    black_to_move = chess.Board()
    black_to_move.push_san("e4")
    assert build_book(repertoire).lookup(black_to_move, 1) is None


def test_book_refuses_test_games(repertoire):
    with pytest.raises(LeakageError):
        build_book(repertoire, test_games=[repertoire[0]])


def test_book_json_roundtrip_drops_rare_positions(repertoire):
    book = build_book(repertoire, min_count=3)
    data = book.to_json()
    assert all(p["n"] >= 3 for p in data["positions"].values())
    again = OpeningBook.from_json(data)
    assert again.probabilities(chess.Board(), 0) == book.probabilities(chess.Board(), 0)
    start = data["positions"][enc.position_key(chess.Board())]
    assert start["score"] == pytest.approx((6 * 1.0 + 3 * 0.0 + 0.5) / 10)


def test_opening_coverage_counts_known_choices(repertoire):
    book = build_book(repertoire, min_count=3)
    known = game_from_san(100, ["e4", "e5", "Nf3", "Nc6"], filler_seed=0)   # eğitimdeki oyunla aynı devam
    novel = game_from_san(101, ["c4", "e5", "Nc3", "Nf6"], filler_seed=5)   # repertuvar dışı
    full = opening_coverage(book, [known], plies=4)
    assert full["coverage"] == 1.0 and full["moves_total"] == 2
    none = opening_coverage(book, [novel], plies=4)
    assert none["coverage"] == 0.0
    both = opening_coverage(book, [known, novel], plies=4)
    assert both["coverage"] == 0.5


def test_top_lines_shares_and_scores(repertoire):
    lines = top_lines(repertoire, "white", depth_plies=4, top=5)
    assert [l["games"] for l in lines] == [6, 3, 1]
    assert lines[0]["moves_san"] == ["e4", "e5", "Nf3", "Nc6"]
    assert lines[0]["share"] == 0.6 and lines[0]["score"] == 1.0
    assert lines[1]["score"] == 0.0 and lines[1]["losses"] == 3
    assert top_lines(repertoire, "black") == []


# ---- örnek üretimi -------------------------------------------------------

def test_samples_cover_only_player_moves_and_decode_back(repertoire):
    games = [make_record(1, "2024-01-01", color="white", plies=25), make_record(2, "2024-01-02", color="black", plies=24)]
    samples = build_samples(games)
    assert len(samples) == 13 + 12
    assert samples.codes.shape == (49, 64)
    # her hedef hamle, yasal maskenin içinde ve geri çözülünce oynanan hamleye eşit
    row = 0
    for gi, game in enumerate(games):
        board = chess.Board()
        player_turn = chess.WHITE if game.player_color == "white" else chess.BLACK
        for ply, uci in enumerate(game.moves):
            if board.turn == player_turn:
                assert samples.game_index[row] == gi and samples.ply[row] == ply
                assert samples.legal_mask(np.array([row]))[0][samples.target[row]]
                assert enc.index_move(int(samples.target[row]), board.turn).uci() == uci
                assert samples.n_legal[row] == board.legal_moves.count()
                expected = enc.stack_history(
                    np.stack([enc.board_codes(b) for b in boards_until(game, ply)]), enc.history_indices(ply)[None, :]
                )[0]
                assert np.array_equal(samples.tokens(np.array([row]))[0], expected)
                row += 1
            board.push(chess.Move.from_uci(uci))
    assert row == len(samples)


def boards_until(game, ply):
    board = chess.Board()
    out = [board.copy()]
    for uci in game.moves[:ply]:
        board.push(chess.Move.from_uci(uci))
        out.append(board.copy())
    return out


def test_samples_flag_book_and_forced_positions(repertoire):
    book = build_book(repertoire, min_count=3)
    samples = build_samples(repertoire, book=book)
    first_moves = samples.ply == 0
    assert samples.in_book[first_moves].all()
    assert not samples.in_book[samples.ply >= book.max_ply].any()
    assert (samples.measured == (~samples.in_book & (samples.n_legal > 1))).all()
    assert samples.measured.sum() < len(samples)


def test_elo_inputs_use_game_ratings_with_otb_offset():
    online = make_record(1, "2024-01-01")
    otb = make_record(2, "2024-01-02")
    otb.source, otb.player_rating, otb.opponent_rating = "twic", 2500, 2600
    unknown = make_record(3, "2024-01-03")
    unknown.player_rating = unknown.opponent_rating = None
    samples = build_samples([online, otb, unknown])
    by_game = {gi: (samples.self_elo[samples.game_index == gi][0], samples.oppo_elo[samples.game_index == gi][0]) for gi in range(3)}
    assert by_game[0] == (2100, 2050)
    assert by_game[1] == (2700, 2800)
    median = nominal_elo([online, otb, unknown])
    assert by_game[2] == (median, median) and median == 2400


def test_clock_before_move_comes_from_previous_own_move():
    game = make_record(1, "2024-01-01", plies=10)
    game.clocks = [300.0 - i for i in range(10)]
    samples = build_samples([game])
    assert np.isnan(samples.clock[0])
    assert samples.clock[1] == 300.0 and samples.clock[2] == 298.0
