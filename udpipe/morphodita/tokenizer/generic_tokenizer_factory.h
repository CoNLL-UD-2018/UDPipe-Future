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
#include "tokenizer_factory.h"

namespace ufal {
namespace udpipe {
namespace morphodita {

class generic_tokenizer_factory : public tokenizer_factory {
 public:
  // Construct a new tokenizer instance.
  virtual tokenizer* new_tokenizer() const override;

  bool load(istream& is);
 private:
  unsigned version;
};

} // namespace morphodita
} // namespace udpipe
} // namespace ufal
