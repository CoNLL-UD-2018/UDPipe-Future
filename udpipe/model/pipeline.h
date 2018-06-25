// This file is part of UDPipe <http://github.com/ufal/udpipe/>.
//
// Copyright 2015 Institute of Formal and Applied Linguistics, Faculty of
// Mathematics and Physics, Charles University in Prague, Czech Republic.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <utility>

#include "common.h"
#include "model.h"

namespace ufal {
namespace udpipe {

class pipeline {
 public:
  pipeline(const model* m, const string& input, const string& tagger, const string& parser, const string& output);

  void set_model(const model* m);
  void set_input(const string& input);
  void set_tagger(const string& tagger);
  void set_parser(const string& parser);
  void set_output(const string& output);

  void set_immediate(bool immediate);
  void set_document_id(const string& document_id);

  bool process(istream& is, ostream& os, string& error) const;

  static const string DEFAULT;
  static const string NONE;

 private:
  const model* m;
  string input, tokenizer, tagger, parser, output;
  string document_id;
  bool immediate;
};

} // namespace udpipe
} // namespace ufal
