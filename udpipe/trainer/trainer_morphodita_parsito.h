// This file is part of UDPipe <http://github.com/ufal/udpipe/>.
//
// Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
// Mathematics and Physics, Charles University in Prague, Czech Republic.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <unordered_set>

#include "common.h"
#include "model/model_morphodita_parsito.h"
#include "sentence/sentence.h"
#include "trainer.h"
#include "utils/named_values.h"

namespace ufal {
namespace udpipe {

class trainer_morphodita_parsito : public trainer {
 public:
  static bool train(const vector<sentence>& training, const vector<sentence>& heldout,
                    const string& tokenizer, const string& tagger, const string& parser, ostream& os, string& error);

 private:
  static bool train_tokenizer(const vector<sentence>& training, const vector<sentence>& heldout,
                              const string& options, ostream& os, string& error);
  static bool train_tagger(const vector<sentence>& training, const vector<sentence>& heldout,
                           const string& options, ostream& os, string& error);
  static bool train_parser(const vector<sentence>& training, const vector<sentence>& heldout,
                           const string& options, const string& tagger_model, ostream& os, string& error);

  // Generic model methods
  enum model_type { TOKENIZER_MODEL, TAGGER_MODEL, PARSER_MODEL };
  static bool load_model(const string& data, model_type model, string_piece& range);
  static const string& model_normalize_form(string_piece form, string& output);
  static const string& model_normalize_lemma(string_piece lemma, string& output);
  static void model_fill_word_analysis(const morphodita::tagged_lemma& analysis, bool upostag, int lemma, bool xpostag, bool feats, word& word);

  // Tagger-specific model methods
  static bool train_tagger_model(const vector<sentence>& training, const vector<sentence>& heldout,
                                 unsigned model, unsigned models, const named_values::map& tagger, ostream& os, string& error);
  static bool can_combine_tag(const word& w, string& error);
  static const string& combine_tag(const word& w, bool xpostag, bool feats, string& combined_tag);
  static const string& most_frequent_tag(const vector<sentence>& data, const string& upostag, bool xpostag, bool feats, string& combined_tag);
  static const string& combine_lemma(const word& w, int use_lemma, string& combined_lemma, const unordered_set<string>& flat_lemmas = unordered_set<string>());

  // Generic options handling
  static const string& option_str(const named_values::map& options, const string& name, int model = -1);
  static bool option_int(const named_values::map& options, const string& name, int& value, string& error, int model = -1);
  static bool option_bool(const named_values::map& options, const string& name, bool& value, string& error, int model = -1);
  static bool option_double(const named_values::map& options, const string& name, double& value, string& error, int model = -1);

  // Various string data
  static const string empty_string;
  static const string tag_separators;
  static const string tagger_features_tagger;
  static const string tagger_features_lemmatizer;
  static const string parser_nodes;
};

} // namespace udpipe
} // namespace ufal
