/*
 * tokenizer.js — three tokenizers so you can compare how text gets chopped up.
 *
 *  char : one token per character. Tiny vocab, long sequences.
 *  word : GPT-style pre-tokens (" word", punctuation, whitespace); keeps the
 *         top-N most frequent, everything else becomes <unk>.
 *  bpe  : byte-pair encoding over characters (the GPT-2 idea, minus the byte
 *         fallback). Starts at characters and greedily merges the most frequent
 *         adjacent pair until the vocab-size target is reached.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.TV = Object.assign(root.TV || {}, { tokenizer: mod });
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const UNK = '<unk>';
  // GPT-2 style pre-tokenizer: optional leading space + letters / digits / punctuation, or whitespace runs.
  const PRETOKEN_RE = /\s?[A-Za-zÀ-ɏ]+|\s?\d+|\s?[^\sA-Za-zÀ-ɏ\d]+|\s+(?!\S)|\s+/gu;

  function pretokenize(text) {
    return text.match(PRETOKEN_RE) || [];
  }

  class BaseTokenizer {
    constructor(type) {
      this.type = type;
      this.vocab = [UNK]; // id -> string
      this.index = new Map([[UNK, 0]]);
      this.unkId = 0;
    }
    get vocabSize() { return this.vocab.length; }
    addToken(s) {
      if (!this.index.has(s)) {
        this.index.set(s, this.vocab.length);
        this.vocab.push(s);
      }
      return this.index.get(s);
    }
    idOf(s) { return this.index.has(s) ? this.index.get(s) : this.unkId; }
    decode(ids) { return ids.map((i) => (i === this.unkId ? '�' : this.vocab[i] ?? '�')).join(''); }
    tokenString(id) { return this.vocab[id] ?? UNK; }
  }

  class CharTokenizer extends BaseTokenizer {
    constructor(corpus) {
      super('char');
      [...new Set(corpus)].sort().forEach((c) => this.addToken(c));
    }
    encode(text) { return [...text].map((c) => this.idOf(c)); }
  }

  class WordTokenizer extends BaseTokenizer {
    constructor(corpus, maxVocab = 512) {
      super('word');
      const counts = new Map();
      for (const w of pretokenize(corpus)) counts.set(w, (counts.get(w) || 0) + 1);
      const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
      for (const [w] of sorted.slice(0, Math.max(1, maxVocab - 1))) this.addToken(w);
    }
    encode(text) { return pretokenize(text).map((w) => this.idOf(w)); }
  }

  class BPETokenizer extends BaseTokenizer {
    constructor(corpus, targetVocab = 256) {
      super('bpe');
      this.merges = []; // [left, right] in rank order
      this.ranks = new Map(); // "left\u0000right" -> rank
      this.cache = new Map();
      [...new Set(corpus)].sort().forEach((c) => this.addToken(c));
      this._train(corpus, targetVocab);
    }

    _train(corpus, targetVocab) {
      const wordFreq = new Map();
      for (const w of pretokenize(corpus)) wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
      const words = [...wordFreq.entries()].map(([w, f]) => ({ syms: [...w], f }));
      while (this.vocab.length < targetVocab) {
        const pairCounts = new Map();
        for (const { syms, f } of words) {
          for (let i = 0; i < syms.length - 1; i++) {
            const key = syms[i] + '\u0000' + syms[i + 1];
            pairCounts.set(key, (pairCounts.get(key) || 0) + f);
          }
        }
        let best = null, bestCount = 1; // a pair seen only once isn't worth a merge
        for (const [k, c] of pairCounts) {
          if (c > bestCount || (c === bestCount && best !== null && k < best)) { best = k; bestCount = c; }
        }
        if (best === null) break;
        const [l, r] = best.split('\u0000');
        const merged = l + r;
        this.ranks.set(best, this.merges.length);
        this.merges.push([l, r, bestCount]);
        this.addToken(merged);
        for (const w of words) {
          const s = w.syms;
          if (s.length < 2) continue;
          const out = [];
          for (let i = 0; i < s.length; i++) {
            if (i < s.length - 1 && s[i] === l && s[i + 1] === r) { out.push(merged); i++; }
            else out.push(s[i]);
          }
          w.syms = out;
        }
      }
    }

    _encodeWord(word) {
      if (this.cache.has(word)) return this.cache.get(word);
      let syms = [...word];
      while (syms.length > 1) {
        let bestRank = Infinity, bestIdx = -1;
        for (let i = 0; i < syms.length - 1; i++) {
          const r = this.ranks.get(syms[i] + '\u0000' + syms[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
        }
        if (bestIdx < 0) break;
        syms = [...syms.slice(0, bestIdx), syms[bestIdx] + syms[bestIdx + 1], ...syms.slice(bestIdx + 2)];
      }
      const ids = syms.map((s) => this.idOf(s));
      if (this.cache.size < 50000) this.cache.set(word, ids);
      return ids;
    }

    encode(text) {
      const out = [];
      for (const w of pretokenize(text)) for (const id of this._encodeWord(w)) out.push(id);
      return out;
    }
  }

  function createTokenizer(type, corpus, vocabSize) {
    if (typeof corpus !== 'string' || corpus.length === 0) throw new Error('Corpus must be a non-empty string');
    switch (type) {
      case 'char': return new CharTokenizer(corpus);
      case 'word': return new WordTokenizer(corpus, vocabSize);
      case 'bpe': return new BPETokenizer(corpus, vocabSize);
      default: throw new Error(`Unknown tokenizer type: ${type}`);
    }
  }

  return { createTokenizer, pretokenize, CharTokenizer, WordTokenizer, BPETokenizer, UNK };
});
