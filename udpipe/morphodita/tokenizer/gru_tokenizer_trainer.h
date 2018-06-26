// This file is part of MorphoDiTa <http://github.com/ufal/morphodita/>.
//
// Copyright 2016 Institute of Formal and Applied Linguistics, Faculty of
// Mathematics and Physics, Charles University in Prague, Czech Republic.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "common.h"
#include "unicode_tokenizer.h"

namespace ufal {
namespace udpipe {
namespace morphodita {

struct tokenized_sentence {
  u32string sentence;
  vector<token_range> tokens;
};

class gru_tokenizer_trainer {
 public:
  enum { URL_EMAIL_LATEST = unicode_tokenizer::URL_EMAIL_LATEST };

  static bool train(unsigned url_email_tokenizer, unsigned segment, bool allow_spaces, unsigned dimension, unsigned epochs,
                    unsigned batch_size, float learning_rate, float learning_rate_final, float beta_2, float dropout,
                    float initialization_range, bool early_stopping, const vector<tokenized_sentence>& data,
                    const vector<tokenized_sentence>& heldout, ostream& os, string& error);
};

} // namespace morphodita
} // namespace udpipe
} // namespace ufal
