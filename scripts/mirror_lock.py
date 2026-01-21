# -*- coding: utf-8 -*-
# ============================================================
# ONE-CELL COLAB SCRIPT (SLOW + RELIABLE + LOCAL RETRY ON FALSE)
# - Page 0 fixed from TEMPLATE grid
# - Other pages flexible
# ============================================================

import os, re, unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -------------------------
# 0) CONFIG
# -------------------------
TXT_PATH = "/content/quran-uthmani-min_brut.txt"
OUT_FILE = "/content/khonsari_pages.txt"

LINES_PER_PAGE = 11
PAGES_BEFORE = 0
PAGES_AFTER = 3

SPLIT_WAW_PREFIX = True
ANCHOR_TEXT = "او ترقى فى السماء"
MATCH_THRESHOLD = 0.60
ANCHOR_WINDOW = 3000
MATCH_YEH_FAMILY = True

# --- FLEX (normal) ---
FLEX_TOTAL_DELTA = 12
FLEX_MIN_W = 6
FLEX_MAX_W = 16

# --- RETRY (only when mirror=False) ---
RETRY_DELTA = 40
RETRY_MIN_W = 5
RETRY_MAX_W = 18

# -------------------------
# 1) TEMPLATE (11 lines) — page 0
# -------------------------
TEMPLATE_PAGE = """أَو تَرقىٰ فِى السَّماءِ وَلَن نُؤمِنَ لِرُقِيِّكَ حَتّىٰ تُنَزِّلَ عَلَينا
كِتٰبًا نَقرَؤُهُ قُل سُبحانَ رَبّى هَل كُنتُ إِلّا بَشَرًا رَسولًا
وَما مَنَعَ النّاسَ أَن يُؤمِنوا إِذ جاءَهُمُ الهُدىٰ إِلّا أَن قالوا
أَبَعَثَ اللَّهُ بَشَرًا رَسولًا قُل لَو كانَ فِى الأَرضِ مَلٰئِكَةٌ يَمشونَ
مُطمَئِنّينَ لَنَزَّلنا عَلَيهِم مِنَ السَّماءِ مَلَكًا رَسولًا قُل كَفىٰ بِاللَّهِ
شَهيدًا bينى وَبَينَكُم إِنَّهُ كانَ بِعِبادِهِ خَبيرًا بَصيرًا وَ
مَن يَهدِ اللَّهُ فَهُوَ المُهتَدِ وَمَن يُضلِل فَلَن تَجِدَ لَهُم
أَولِياءَ مِن دونِهِ وَنَحشُرُهُم يَومَ القِيٰمَةِ عَلىٰ وُجوهِهِم عُميًا وَبُكمًا
وَصُمًّا مَأوىٰهُم جَهَنَّمُ كُلَّما خَبَت زِدنٰهُم سَعيرًا ذٰلِكَ جَزاؤُهُم بِأَنَّهُم
كَفَروا بِـٔايٰتِنا وَقالوا أَءِذا كُنّا عِظٰمًا وَرُفٰتًا
أَءِنّا لَمَبعوثونَ خَلقًا جَديدًا أَوَلَم يَرَوا أَنَّ اللَّهَ الَّذى خَلَقَ""".strip("\n")

# ============================================================
# MODULES
# ============================================================

QURAN_PUNCT = set(list("ۖۗۘۙۚۛۜ۝۞؞،؛؟") + ["﴿","﴾","…","٬","٫",".",",",":","؛","؟","!","(",")","[","]","{","}","“","”",'"',"'"])

@dataclass
class Word:
    text: str
    norm_match: str

def is_combining(ch: str) -> bool:
    return unicodedata.category(ch) == "Mn"

def strip_harakat(s: str) -> str:
    return "".join(ch for ch in s if not is_combining(ch))

def strip_edge_harakat(s: str) -> str:
    if not s: return s
    i, j = 0, len(s)
    while i < j and is_combining(s[i]): i += 1
    while j > i and is_combining(s[j-1]): j -= 1
    return s[i:j]

def clean_word_edges(w: str) -> str:
    w = w.strip()
    if not w: return ""
    while w and (w[0] in QURAN_PUNCT): w = w[1:]
    while w and (w[-1] in QURAN_PUNCT): w = w[:-1]
    w = strip_edge_harakat(w.strip())
    w = re.sub(r"^[^\u0600-\u06FF]+", "", w)
    w = re.sub(r"[^\u0600-\u06FF]+$", "", w)
    return strip_edge_harakat(w.strip())

def split_waw(word: str) -> List[str]:
    if not SPLIT_WAW_PREFIX: return [word]
    if word.startswith("و") and len(word) > 1: return ["و", word[1:]]
    return [word]

def norm_for_match(s: str) -> str:
    s = strip_harakat(s).replace("ـ", "").replace("ٱ", "ا")
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    if MATCH_YEH_FAMILY: s = s.replace("ى", "ي")
    return re.sub(r"[^\u0600-\u06FF]+", "", s)

def tokenize_text(text: str) -> List[Word]:
    parts = [p for p in re.split(r"\s+", text) if p.strip()]
    out = []
    for p in parts:
        p = clean_word_edges(p)
        if not p: continue
        for piece in split_waw(p):
            piece = clean_word_edges(piece)
            nm = norm_for_match(piece)
            if piece and nm:
                out.append(Word(text=piece, norm_match=nm))
    return out

def load_words_from_txt(path: str) -> List[Word]:
    with open(path, "r", encoding="utf-8") as f:
        return tokenize_text(f.read())

def template_lines(template: str) -> List[List[Word]]:
    lines = [ln.strip() for ln in template.splitlines() if ln.strip()]
    if len(lines) != LINES_PER_PAGE:
        raise RuntimeError(f"Template error: {len(lines)} lines instead of {LINES_PER_PAGE}")
    return [tokenize_text(ln) for ln in lines]

def grid_counts(tpl_lines: List[List[Word]]) -> List[int]:
    return [len(x) for x in tpl_lines]

def flatten(lines: List[List[Word]]) -> List[Word]:
    return [w for ln in lines for w in ln]

def find_all_subseq(hay: List[str], needle: List[str]) -> List[int]:
    res = []
    n, m = len(hay), len(needle)
    if m == 0: return res
    for i in range(n - m + 1):
        if hay[i:i+m] == needle: res.append(i)
    return res

def best_template_location(norms: List[str], tpl_norm: List[str], anchor_norm: List[str],
                           window: int, threshold: float) -> Optional[Tuple[int,int,int]]:
    hits = find_all_subseq(norms, anchor_norm)
    if not hits: return None
    m, best_score, best_i = len(tpl_norm), -1, None
    for h in hits:
        left, right = max(0, h - window), min(len(norms) - m, h + window)
        for i in range(left, right + 1):
            score = sum(1 for a, b in zip(norms[i:i+m], tpl_norm) if a == b)
            if score > best_score:
                best_score, best_i = score, i
    if best_i is not None and best_score >= int(threshold * m):
        return best_i, best_score, len(hits)
    return None

def first_letter_for_sig(token_text: str) -> str:
    s = re.sub(r"[^\u0600-\u06FF]+", "", strip_harakat(token_text).replace("ـ",""))
    return s[0] if s else ""

def mirror_signature(lines: List[List[Word]]) -> Tuple[str, bool]:
    keys = [first_letter_for_sig(ln[0].text) if ln else "" for ln in lines]
    ok = all(keys[i] == keys[-1-i] for i in range(len(keys)//2))
    return "".join(keys), ok

def render_page_fixed(words_slice: List[Word], grid: List[int]) -> List[List[Word]]:
    out, i = [], 0
    for c in grid:
        out.append(words_slice[i:i+c])
        i += c
    return out

def render_lines_text(lines: List[List[Word]]) -> List[str]:
    return [" ".join(w.text for w in ln) for ln in lines]

# ============================================================
# FLEX SOLVER
# ============================================================

FLEX_MIRROR_BONUS = 25.0
FLEX_PAIR_MATCH_BONUS = 2.0

def _pair_letter_score(a_tok: Word, b_tok: Word) -> float:
    a, b = first_letter_for_sig(a_tok.text), first_letter_for_sig(b_tok.text)
    return 3.0 if (a and a == b) else -2.0

def _len_penalty(k: int, target: int) -> float:
    return -0.12 * abs(k - target)

def _page_quality_bonus(lines: List[List[Word]]) -> float:
    sig, ok = mirror_signature(lines)
    if ok: return FLEX_MIRROR_BONUS
    keys = [first_letter_for_sig(ln[0].text) if ln else "" for ln in lines]
    m = sum(1 for i in range(len(keys)//2) if keys[i] and keys[i] == keys[-1-i])
    return FLEX_PAIR_MATCH_BONUS * m

def _solve_page_dp(tokens: List[Word], target_grid: List[int], min_w: int, max_w: int):
    L, N = 11, len(tokens)
    pairs = L // 2
    if N <= 0: return None
    dp, back = {(0, 0): 0.0}, {}
    for p in range(pairs):
        new_dp = {}
        for (l, br), sc in dp.items():
            for k_top in range(min_w, max_w + 1):
                nl = l + k_top
                if nl >= N: continue
                for k_bot in range(min_w, max_w + 1):
                    nbr = br + k_bot
                    if nbr >= N or (N - (nl + nbr)) < min_w: continue
                    
                    bot_start = N - nbr
                    if bot_start <= nl: continue
                    
                    add = _pair_letter_score(tokens[l], tokens[bot_start])
                    add += _len_penalty(k_top, target_grid[p])
                    add += _len_penalty(k_bot, target_grid[L - 1 - p])
                    
                    cand = sc + add
                    if (nl, nbr) not in new_dp or cand > new_dp[(nl, nbr)]:
                        new_dp[(nl, nbr)] = cand
                        back[(p + 1, nl, nbr)] = (l, br, k_top, k_bot)
        dp = new_dp
    
    best_score, best_state = float("-inf"), None
    for (l, br), sc in dp.items():
        mid_len = N - (l + br)
        if min_w <= mid_len <= (max_w + 8):
            sc2 = sc + _len_penalty(mid_len, target_grid[pairs])
            if sc2 > best_score:
                best_score, best_state = sc2, (l, br)
    
    if best_state is None: return None
    l, br = best_state
    top_lens, bot_lens = [0]*pairs, [0]*pairs
    for p in range(pairs, 0, -1):
        l, br, kt, kb = back[(p, l, br)]
        top_lens[p-1], bot_lens[p-1] = kt, kb
        
    lines, idx = [], 0
    for k in top_lens:
        lines.append(tokens[idx:idx+k]); idx += k
    lines.append(tokens[idx:N-sum(bot_lens)])
    rr = N
    bottoms = []
    for k in bot_lens:
        bottoms.append(tokens[rr-k:rr]); rr -= k
    lines.extend(reversed(bottoms))
    return lines, best_score

def _best_page_at(words, start, target_grid, expected_size, delta, min_w, max_w):
    best_lines, bestN, best_score = None, None, float("-inf")
    for N in range(max(1, expected_size-delta), expected_size+delta+1):
        if start + N > len(words): break
        solved = _solve_page_dp(words[start:start+N], target_grid, min_w, max_w)
        if solved:
            lines, sc = solved
            sc += _page_quality_bonus(lines)
            if sc > best_score:
                best_score, best_lines, bestN = sc, lines, N
    return (best_lines, bestN, best_score) if best_lines else None

# ============================================================
# EXPORT
# ============================================================

def export_pages(words, start_idx, grid, out_file, pages_before, pages_after):
    expected_size, pages = sum(grid), {}
    
    # Page 0
    p0_lines = render_page_fixed(words[start_idx:start_idx+expected_size], grid)
    sig0, ok0 = mirror_signature(p0_lines)
    pages[0] = (p0_lines, start_idx, start_idx+expected_size, sig0, ok0)

    # Forward
    cur = start_idx + expected_size
    for rel in range(1, pages_after + 1):
        print(f"⏳ Page +{rel}...")
        best = _best_page_at(words, cur, grid, expected_size, FLEX_TOTAL_DELTA, FLEX_MIN_W, FLEX_MAX_W)
        if best:
            lines, usedN, _ = best
            sig, ok = mirror_signature(lines)
            if not ok:
                retry = _best_page_at(words, cur, grid, expected_size, RETRY_DELTA, RETRY_MIN_W, RETRY_MAX_W)
                if retry and mirror_signature(retry[0])[1]:
                    lines, usedN, _ = retry
                    sig, ok = mirror_signature(lines)
                    print(f"🔁 Retry fixed at {cur}")
            pages[rel] = (lines, cur, cur+usedN, sig, ok)
            cur += usedN
        else: break

    # Write
    chunks = []
    for rel in sorted(pages.keys()):
        lines, ps, pe, sig, ok = pages[rel]
        header = [f"═════ صفحة ({rel}) [word {ps}..{pe}] ═════", f"[MirrorSig] {sig} | mirror={ok}", ""]
        chunks.append("\n".join(header + render_lines_text(lines)))
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))
    print(f"✅ Saved: {out_file}")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(TXT_PATH):
        from google.colab import files
        uploaded = files.upload()
        TXT_PATH = "/content/" + next(iter(uploaded.keys()))

    words = load_words_from_txt(TXT_PATH)
    norms = [w.norm_match for w in words]
    tpl_lines = template_lines(TEMPLATE_PAGE)
    grid = grid_counts(tpl_lines)
    
    anchor_norm = [w.norm_match for w in tokenize_text(ANCHOR_TEXT)]
    located = best_template_location(norms, [w.norm_match for w in flatten(tpl_lines)], anchor_norm, ANCHOR_WINDOW, MATCH_THRESHOLD)
    
    if located:
        start_idx, score, _ = located
        export_pages(words, start_idx, grid, OUT_FILE, PAGES_BEFORE, PAGES_AFTER)
    else:
        print("❌ Template not found.")