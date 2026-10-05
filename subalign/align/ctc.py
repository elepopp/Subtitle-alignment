"""CTC forced alignment.

* :func:`viterbi_align` - numpy Viterbi over a CTC state graph with optional
  *star* (garbage) states between lines that absorb ad-libs, backing vocals and
  anything not in the script.
* :class:`HFCTCEmitter` - frame log-probabilities from any HuggingFace
  ``Wav2Vec2ForCTC`` model (needs ``torch`` + ``transformers``).  Characters
  missing from the model vocabulary are romanised (pinyin / romaji / RR /
  accent stripping) so that multilingual letter-vocab models such as MMS work
  for CJK text.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..text.tokenize import normalize_key

STAR = -1


@dataclass
class CTCUnitSpan:
    start: Optional[float]
    end: Optional[float]
    score: float           # mean log-prob of the unit's frames (higher is better)


def viterbi_align(logp: np.ndarray, unit_ids: Sequence[Sequence[int]], blank: int = 0,
                  frame_s: float = 0.02, line_breaks: Sequence[int] = (), star_penalty: float = 2.0
                  ) -> List[CTCUnitSpan]:
    """Align units (each a list of vocab ids) to CTC log-probs (T, V).

    ``line_breaks``: unit indices *before which* an optional star state is
    inserted (typically the first unit of every line).
    Units with no ids get ``start=end=None`` (interpolated later).
    """
    T, Vn = logp.shape
    star_emis = np.max(np.delete(logp, blank, axis=1), axis=1) - star_penalty
    # build state list: (emission id, unit index, can-skip-from-2-back, kind)
    emis: List[int] = []
    unit_of: List[int] = []
    allow2: List[bool] = []
    allow_star_skip: List[bool] = []
    breaks = set(line_breaks)
    emis.append(blank); unit_of.append(-1); allow2.append(False); allow_star_skip.append(False)
    last_label = None
    for ui, ids in enumerate(unit_ids):
        if ui in breaks and ui > 0:
            # ... blank, STAR, blank ...  (STAR is skippable)
            emis.append(STAR); unit_of.append(-1); allow2.append(False); allow_star_skip.append(False)
            emis.append(blank); unit_of.append(-1); allow2.append(False); allow_star_skip.append(True)
            last_label = None
        for tid in ids:
            emis.append(tid); unit_of.append(ui)
            allow2.append(last_label is not None and last_label != tid)
            allow_star_skip.append(False)
            emis.append(blank); unit_of.append(-1); allow2.append(False); allow_star_skip.append(False)
            last_label = tid
    S = len(emis)
    emis_a = np.array(emis)
    is_star = emis_a == STAR
    emis_idx = np.where(is_star, 0, emis_a)
    allow2_a = np.array(allow2)
    star_skip = np.array(allow_star_skip)   # blank after STAR may come from blank before STAR (s-2)
    allow2_a = allow2_a | star_skip
    # first label after a star-blank may also skip that blank (s-2 is STAR, need s-3? keep simple)
    NEG = -1e30
    dp = np.full(S, NEG)
    dp[0] = logp[0, blank]
    if S > 1:
        dp[1] = (star_emis[0] if is_star[1] else logp[0, emis_idx[1]])
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        e = np.where(is_star, star_emis[t], logp[t, emis_idx])
        stay = dp
        m1 = np.full(S, NEG); m1[1:] = dp[:-1]
        m2 = np.full(S, NEG); m2[2:] = np.where(allow2_a[2:], dp[:-2], NEG)
        best = stay; arg = np.zeros(S, dtype=np.int8)
        sel = m1 > best; best = np.where(sel, m1, best); arg[sel] = 1
        sel = m2 > best; best = np.where(sel, m2, best); arg[sel] = 2
        dp = best + e
        back[t] = arg
    # end in last blank or last label
    s = S - 1 if dp[S - 1] >= dp[S - 2] or S < 2 else S - 2
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
        s = max(s, 0)
    unit_of_a = np.array(unit_of)
    spans: List[CTCUnitSpan] = []
    frame_unit = unit_of_a[path]
    frame_score = logp[np.arange(T), emis_idx[path]]
    for ui in range(len(unit_ids)):
        fr = np.flatnonzero(frame_unit == ui)
        if len(fr) == 0:
            spans.append(CTCUnitSpan(None, None, -np.inf))
            continue
        spans.append(CTCUnitSpan(fr[0] * frame_s, (fr[-1] + 1) * frame_s, float(np.mean(frame_score[fr]))))
    return spans


# --------------------------------------------------------------------------
# romanisation helpers (used when the model vocab lacks the native script)
# --------------------------------------------------------------------------
_KANA = {}
_ROMAJI_ROWS = {
    "あいうえお": ["a", "i", "u", "e", "o"], "かきくけこ": ["ka", "ki", "ku", "ke", "ko"],
    "さしすせそ": ["sa", "shi", "su", "se", "so"], "たちつてと": ["ta", "chi", "tsu", "te", "to"],
    "なにぬねの": ["na", "ni", "nu", "ne", "no"], "はひふへほ": ["ha", "hi", "fu", "he", "ho"],
    "まみむめも": ["ma", "mi", "mu", "me", "mo"], "やゆよ": ["ya", "yu", "yo"],
    "らりるれろ": ["ra", "ri", "ru", "re", "ro"], "わをん": ["wa", "o", "n"],
    "がぎぐげご": ["ga", "gi", "gu", "ge", "go"], "ざじずぜぞ": ["za", "ji", "zu", "ze", "zo"],
    "だぢづでど": ["da", "ji", "zu", "de", "do"], "ばびぶべぼ": ["ba", "bi", "bu", "be", "bo"],
    "ぱぴぷぺぽ": ["pa", "pi", "pu", "pe", "po"], "ぁぃぅぇぉ": ["a", "i", "u", "e", "o"],
    "ゃゅょ": ["ya", "yu", "yo"], "っ": [""], "ー": [""],
}
for row, roms in _ROMAJI_ROWS.items():
    for ch, r in zip(row, roms):
        _KANA[ch] = r
        kata = chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch
        _KANA[kata] = r

_L = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_V = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we",
      "wi", "yu", "eu", "ui", "i"]
_T = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p",
      "t", "t", "ng", "t", "t", "k", "t", "p", "t"]


def romanize(ch: str) -> str:
    if ch in _KANA:
        return _KANA[ch]
    cp = ord(ch[0]) if ch else 0
    if 0xAC00 <= cp <= 0xD7A3:
        k = cp - 0xAC00
        return _L[k // 588] + _V[(k % 588) // 28] + _T[k % 28]
    try:
        from ..text.phonetic import pinyin_of

        py = pinyin_of(ch)
        if py:
            return py
    except Exception:  # pragma: no cover
        pass
    s = unicodedata.normalize("NFKD", ch)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def text_to_ids(key: str, vocab: Dict[str, int]) -> List[int]:
    """Map a normalised token key onto vocabulary ids (native chars first,
    romanisation as fallback)."""
    ids: List[int] = []
    lower_vocab = vocab
    for ch in key:
        if ch in lower_vocab:
            ids.append(lower_vocab[ch])
            continue
        up = ch.upper()
        if up in lower_vocab:
            ids.append(lower_vocab[up])
            continue
        for r in romanize(ch):
            if r in lower_vocab:
                ids.append(lower_vocab[r])
            elif r.upper() in lower_vocab:
                ids.append(lower_vocab[r.upper()])
    return ids


DEFAULT_CTC_MODELS = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "facebook/wav2vec2-large-960h-lv60-self",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "*": "MahmoudAshraf/mms-300m-1130-forced-aligner",
}


class HFCTCEmitter:
    """Frame-level CTC log-probabilities from a HuggingFace wav2vec2 model."""

    def __init__(self, model: Optional[str] = None, language: Optional[str] = None, device: Optional[str] = None):
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, Wav2Vec2ForCTC  # type: ignore
        except ImportError as e:  # pragma: no cover - optional
            raise ImportError("CTC alignment needs `pip install torch transformers`") from e
        import torch

        name = model or DEFAULT_CTC_MODELS.get((language or "").split("-")[0], DEFAULT_CTC_MODELS["*"])
        self.name = name
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        tok = getattr(self.processor, "tokenizer", self.processor)
        self.vocab: Dict[str, int] = dict(tok.get_vocab())
        # the CTC blank is the model's pad_token_id (what transformers' CTC loss uses); the
        # tokenizer's <pad> differs from it in some models (MMS: <blank>=0, <pad>=1)
        blank = getattr(self.model.config, "pad_token_id", None)
        if blank is None:
            blank = self.vocab.get("<blank>", self.vocab.get("<pad>", self.vocab.get("[PAD]", tok.pad_token_id or 0)))
        self.blank = int(blank)
        self.frame_s = 0.02

    def emissions(self, y: np.ndarray, sr: int = 16000, chunk_s: float = 20.0, ctx_s: float = 1.0) -> np.ndarray:
        import torch

        hop = int(chunk_s * sr)
        ctx = int(ctx_s * sr)
        outs = []
        for s in range(0, len(y), hop):
            a, b = max(0, s - ctx), min(len(y), s + hop + ctx)
            x = torch.from_numpy(np.ascontiguousarray(y[a:b])).float()[None].to(self.device)
            x = (x - x.mean()) / (x.std() + 1e-7)
            with torch.inference_mode():
                lp = torch.log_softmax(self.model(x).logits[0].float(), dim=-1).cpu().numpy()
            f_per_s = lp.shape[0] / ((b - a) / sr)
            c0 = int(round((s - a) / sr * f_per_s))
            c1 = c0 + int(round(min(hop, len(y) - s) / sr * f_per_s))
            outs.append(lp[c0:c1])
        lp = np.concatenate(outs, axis=0)
        self.frame_s = len(y) / sr / max(1, lp.shape[0])
        return lp

    def unit_ids(self, keys: Sequence[str]) -> List[List[int]]:
        return [text_to_ids(k, self.vocab) for k in keys]


def ctc_align_tokens(emitter: "HFCTCEmitter", y: np.ndarray, line_token_keys: Sequence[Sequence[str]],
                     sr: int = 16000) -> List[List[CTCUnitSpan]]:
    """Forced-align lines of token keys; returns spans per line."""
    lp = emitter.emissions(y, sr)
    flat: List[str] = []
    breaks: List[int] = []
    for keys in line_token_keys:
        breaks.append(len(flat))
        flat.extend(normalize_key(k) for k in keys)
    ids = emitter.unit_ids(flat)
    spans = viterbi_align(lp, ids, blank=emitter.blank, frame_s=emitter.frame_s, line_breaks=breaks)
    out, k = [], 0
    for keys in line_token_keys:
        out.append(spans[k:k + len(keys)])
        k += len(keys)
    return out
