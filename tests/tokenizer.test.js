'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { createTokenizer, pretokenize } = require('../js/tokenizer.js');

const corpus = 'the cat sat on the mat. the cat ate the rat! 123 cats, 45 mats.\nnew line  double space';

test('pretokenize is lossless', () => {
  assert.equal(pretokenize(corpus).join(''), corpus);
});

for (const type of ['char', 'word', 'bpe']) {
  test(`${type} tokenizer round-trips in-vocabulary text`, () => {
    const tok = createTokenizer(type, corpus, 1000);
    assert.equal(tok.decode(tok.encode(corpus)), corpus);
    for (const id of tok.encode(corpus)) assert.ok(id >= 0 && id < tok.vocabSize);
  });
}

test('unknown characters map to <unk>', () => {
  const tok = createTokenizer('char', 'abc');
  assert.deepEqual(tok.encode('abz'), [tok.idOf('a'), tok.idOf('b'), tok.unkId]);
});

test('BPE: bigger vocab => fewer tokens, and respects the target size', () => {
  const small = createTokenizer('bpe', corpus, 30);
  const big = createTokenizer('bpe', corpus, 60);
  assert.ok(small.vocabSize <= 30);
  assert.ok(big.encode(corpus).length < small.encode(corpus).length);
  assert.ok(big.merges.length > 0);
});

test('word tokenizer caps vocab and falls back to <unk>', () => {
  const tok = createTokenizer('word', corpus, 5);
  assert.equal(tok.vocabSize, 5);
  assert.ok(tok.encode(corpus).includes(tok.unkId));
});

test('rejects empty corpus / unknown type', () => {
  assert.throws(() => createTokenizer('char', ''));
  assert.throws(() => createTokenizer('nope', 'abc'));
});
