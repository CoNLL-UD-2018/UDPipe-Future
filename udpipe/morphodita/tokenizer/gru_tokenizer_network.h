// This file is part of MorphoDiTa <http://github.com/ufal/morphodita/>.
//
// Copyright 2016 Institute of Formal and Applied Linguistics, Faculty of
// Mathematics and Physics, Charles University in Prague, Czech Republic.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "common.h"
#include "unilib/unicode.h"
#include "utils/binary_decoder.h"

namespace ufal {
namespace udpipe {
namespace morphodita {

// Declarations

class gru_tokenizer_network {
 public:
  virtual ~gru_tokenizer_network() {}

  template <int R, int C> struct matrix {
    float w[R][C];
    float b[R];

    void clear();
    void load(binary_decoder& data);
  };

  enum { NO_SPLIT, END_OF_TOKEN, END_OF_SENTENCE, OUTCOMES };
  struct outcome_t {
    int outcome;
    float w[3];
    const float* embedding;
  };
  struct char_info {
    char32_t chr;
    unilib::unicode::category_t cat;

    char_info() {}
    char_info(char32_t chr, unilib::unicode::category_t cat) : chr(chr), cat(cat) {}
  };

  virtual void classify(const vector<char_info>& chars, vector<outcome_t>& outcomes) const = 0;

  static gru_tokenizer_network* load(binary_decoder& data);
};

template <int D>
class gru_tokenizer_network_implementation : public gru_tokenizer_network {
 public:
  virtual void classify(const vector<char_info>& chars, vector<outcome_t>& outcomes) const override;

  static gru_tokenizer_network_implementation<D>* load(binary_decoder& data);

 protected:
  void cache_embeddings();

  struct cached_embedding {
    matrix<1, D> e;
    matrix<6, D> cache;
  };

  struct gru {
    matrix<D,D> X, X_r, X_z;
    matrix<D,D> H, H_r, H_z;

    void load(binary_decoder& data);
  };

  unordered_map<char32_t, cached_embedding> embeddings;
  cached_embedding empty_embedding;
  gru gru_fwd, gru_bwd;
  matrix<3, D> projection_fwd, projection_bwd;
  unordered_map<unilib::unicode::category_t, char32_t> unknown_chars;
};

// Definitions

template <int R, int C>
void gru_tokenizer_network::matrix<R, C>::clear() {
  for (int i = 0; i < R; i++) fill_n(w[i], C, 0.f);
  fill_n(b, R, 0.f);
}

template <int R, int C>
void gru_tokenizer_network::matrix<R, C>::load(binary_decoder& data) {
  for (int i = 0; i < R; i++) memcpy(w[i], data.next<float>(C), sizeof(float) * C);
  memcpy(b, data.next<float>(R), sizeof(float) * R);
}

template <int D>
void gru_tokenizer_network_implementation<D>::gru::load(binary_decoder& data) {
  X.load(data);
  X_r.load(data);
  X_z.load(data);
  H.load(data);
  H_r.load(data);
  H_z.load(data);
}

template <int D>
void gru_tokenizer_network_implementation<D>::classify(const vector<char_info>& chars, vector<outcome_t>& outcomes) const {
  if (chars.empty()) return;

  // Resolve embeddings, possibly with unknown_chars or empty_embedding
  for (size_t i = 0; i < chars.size(); i++) {
    auto embedding = embeddings.find(chars[i].chr);
    if (embedding != embeddings.end()) {
      outcomes[i].embedding = embedding->second.cache.w[0];
    } else {
      auto unknown_char = unknown_chars.find(chars[i].cat);
      if (unknown_char != unknown_chars.end()) embedding = embeddings.find(unknown_char->second);
      outcomes[i].embedding = embedding != embeddings.end() ? embedding->second.cache.w[0] : empty_embedding.cache.w[0];
    }
  }

  // Clear outcome probabilities
  for (auto&& outcome : outcomes)
    for (int i = 0; i < 3; i++)
      outcome.w[i] = projection_fwd.b[i];

  // Perform forward & backward GRU
  matrix<1, D> state, update, reset, candidate;
  for (int dir = 0; dir < 2; dir++) {
    auto& gru = dir == 0 ? gru_fwd : gru_bwd;
    auto& projection = dir == 0 ? projection_fwd : projection_bwd;

    state.clear();
    for (size_t i = 0; i < outcomes.size(); i++) {
      auto& outcome = outcomes[dir == 0 ? i : outcomes.size() - 1 - i];
      auto* embedding_cache = outcome.embedding + (dir == 1) * 3 * D;

      for (int j = 0; j < D; j++) {
        update.w[0][j] = gru.X_z.b[j] + embedding_cache[2*D + j];
        reset.w[0][j] = gru.X_r.b[j] + embedding_cache[D + j];
        for (int k = 0; k < D; k++) {
          update.w[0][j] += state.w[0][k] * gru.H_z.w[j][k];
          reset.w[0][j] += state.w[0][k] * gru.H_r.w[j][k];
        }
        update.w[0][j] = 1.f / (1.f + exp(-update.w[0][j]));
        reset.w[0][j] = 1.f / (1.f + exp(-reset.w[0][j]));
        reset.w[0][j] *= state.w[0][j];
      }
      for (int j = 0; j < D; j++) {
        candidate.w[0][j] = gru.X.b[j] + embedding_cache[j];
        for (int k = 0; k < D; k++)
          candidate.w[0][j] += reset.w[0][k] * gru.H.w[j][k];
        candidate.w[0][j] = tanh(candidate.w[0][j]);
        state.w[0][j] = update.w[0][j] * state.w[0][j] + (1.f - update.w[0][j]) * candidate.w[0][j];
      }

      for (int j = 0; j < 3; j++)
        for (int k = 0; k < D; k++)
          outcome.w[j] += projection.w[j][k] * state.w[0][k];
    }
  }

  // Choose the outcome with the highest weight
  for (auto&& outcome : outcomes) {
    outcome.outcome = outcome.w[1] > outcome.w[0];
    if (outcome.w[2] > outcome.w[outcome.outcome]) outcome.outcome = 2;
  }
}

template <int D>
gru_tokenizer_network_implementation<D>* gru_tokenizer_network_implementation<D>::load(binary_decoder& data) {
  unique_ptr<gru_tokenizer_network_implementation<D>> network(new gru_tokenizer_network_implementation<D>());

  for (unsigned chars = data.next_4B(); chars; chars--) {
    auto& embedding = network->embeddings[data.next_4B()];
    copy_n(data.next<float>(D), D, embedding.e.w[0]);
  }
  fill_n(network->empty_embedding.e.w[0], D, 0.f);

  network->gru_fwd.load(data);
  network->gru_bwd.load(data);
  network->projection_fwd.load(data);
  network->projection_bwd.load(data);

  network->unknown_chars.clear();
  for (unsigned unknown_chars_len = data.next_1B(); unknown_chars_len; unknown_chars_len--) {
    unilib::unicode::category_t cat = data.next_4B();
    network->unknown_chars[cat] = data.next_4B();
  }

  network->cache_embeddings();

  return network.release();
}

template <int D>
void gru_tokenizer_network_implementation<D>::cache_embeddings() {
  for (auto&& embedding : embeddings) {
    auto& e = embedding.second.e;
    auto& cache = embedding.second.cache;

    for (int i = 0; i < 6; i++) fill_n(cache.w[i], D, 0.f);
    for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) cache.w[0][i] += e.w[0][j] * gru_fwd.X.w[i][j];
    for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) cache.w[1][i] += e.w[0][j] * gru_fwd.X_r.w[i][j];
    for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) cache.w[2][i] += e.w[0][j] * gru_fwd.X_z.w[i][j];
    for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) cache.w[3][i] += e.w[0][j] * gru_bwd.X.w[i][j];
    for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) cache.w[4][i] += e.w[0][j] * gru_bwd.X_r.w[i][j];
    for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) cache.w[5][i] += e.w[0][j] * gru_bwd.X_z.w[i][j];
  }
  for (int i = 0; i < 6; i++) fill_n(empty_embedding.cache.w[i], D, 0.f);
}

} // namespace morphodita
} // namespace udpipe
} // namespace ufal
