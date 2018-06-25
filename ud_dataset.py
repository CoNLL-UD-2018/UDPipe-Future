import re

import numpy as np

def _min_edit_script(source, target, allow_copy):
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, "")
            else:
                if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i-1][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "→")
                if i and a[i-1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j][0] + 1, a[i-1][j][1] + "-")
                if j and a[i][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j-1][0] + 1, a[i][j-1][1] + "+" + target[j - 1])
    return a[-1][-1][1]

def _gen_lemma_rule(form, lemma, allow_copy):
    form = form.lower()

    previous_case = -1
    lemma_casing = ""
    for i, c in enumerate(lemma):
        case = "↑" if c.lower() != c else "↓"
        if case != previous_case:
            lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case
    lemma = lemma.lower()

    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
            if cpl > best:
                best = cpl
                best_form = f
                best_lemma = l

    rule = lemma_casing + ";"
    if not best:
        rule += "a" + lemma
    else:
        rule += "d{}¦{}".format(
            _min_edit_script(form[:best_form], lemma[:best_lemma], allow_copy),
            _min_edit_script(form[best_form + best:], lemma[best_lemma + best:], allow_copy),
        )
    return rule

def _apply_lemma_rule(form, lemma_rule):
    casing, rule = lemma_rule.split(";", 1)
    if rule.startswith("a"):
        lemma = rule[1:]
    else:
        form = form.lower()
        rules, rule_sources = rule[1:].split("¦"), []
        assert len(rules) == 2
        for rule in rules:
            source, i = 0, 0
            while i < len(rule):
                if rule[i] == "→" or rule[i] == "-":
                    source += 1
                else:
                    assert rule[i] == "+"
                    i += 1
                i += 1
            rule_sources.append(source)

        try:
            lemma, form_offset = "", 0
            for i in range(2):
                j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                while j < len(rules[i]):
                    if rules[i][j] == "→":
                        lemma += form[offset]
                        offset += 1
                    elif rules[i][j] == "-":
                        offset += 1
                    else:
                        assert(rules[i][j] == "+")
                        lemma += rules[i][j + 1]
                        j += 1
                    j += 1
                if i == 0:
                    lemma += form[rule_sources[0] : len(form) - rule_sources[1]]
        except:
            lemma = form

    for rule in casing.split("¦"):
        if rule == "↓0": continue # The lemma is lowercased initially
        case, offset = rule[0], int(rule[1:])
        lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

    return lemma

class UDDataset:
    FORMS = 0
    LEMMAS = 1
    UPOS = 2
    XPOS = 3
    FEATS = 4
    HEAD = 5
    DEPREL = 6
    DEPS = 7
    MISC = 8
    FACTORS = 9

    FACTORS_MAP = {"FORMS": FORMS, "LEMMAS": LEMMAS, "UPOS": UPOS, "XPOS": XPOS, "FEATS": FEATS,
                   "HEAD": HEAD, "DEPREL": DEPREL, "DEPS": DEPS, "MISC": MISC}

    UNIVERSAL_FEATURES = {
        "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Gender",
        "Animacy", "Number", "Case", "Definite", "Degree", "VerbForm", "Mood",
        "Tense", "Aspect", "Voice", "Evident", "Polarity", "Person", "Polite"
    }

    re_extras = re.compile(r"^#|^\d+-|^\d+\.")

    class _Factor:
        PAD = 0
        UNK = 1
        ROOT = 2
        def __init__(self, with_root, characters, train=None):
            self.words_map = train.words_map if train else {'<pad>': 0, '<unk>': 1, '<root>': 2}
            self.words = train.words if train else ['<pad>', '<unk>', '<root>']
            self.word_ids = []
            self.strings = []
            self.with_root = with_root
            self.characters = characters
            if characters:
                self.alphabet_map = train.alphabet_map if train else {'<pad>': 0, '<unk>': 1, '<root>': 2}
                self.alphabet = train.alphabet if train else ['<pad>', '<unk>', '<root>']
                self.charseqs_map = {'<pad>': 0, '<unk>': 2, '<root>': 1}
                self.charseqs = [[self.PAD], [self.UNK], [self.ROOT]]
                self.charseq_ids = []

    def __init__(self, filename, lr_allow_copy, root_factors=[], train=None, shuffle_batches=True, max_sentences=None):
        # Create factors
        self._factors = []
        for f in range(self.FACTORS):
            self._factors.append(self._Factor(f in root_factors, f == self.FORMS, train._factors[f] if train else None))
        self._extras = []


        # Load the sentences
        with open(filename, "r", encoding="utf-8") as file:
            in_sentence = False
            for line in file:
                line = line.rstrip("\r\n")

                if line:
                    if self.re_extras.match(line):
                        if in_sentence:
                            while len(self._extras) < len(self._factors[0].word_ids): self._extras.append([])
                            while len(self._extras[-1]) <= len(self._factors[0].word_ids[-1]) - self._factors[0].with_root:
                                self._extras[-1].append("")
                        else:
                            while len(self._extras) <= len(self._factors[0].word_ids): self._extras.append([])
                            if not len(self._extras[-1]): self._extras[-1].append("")
                        self._extras[-1][-1] += ("\n" if self._extras[-1][-1] else "") + line
                        continue

                    columns = line.split("\t")[1:]
                    for f in range(self.FACTORS):
                        factor = self._factors[f]
                        if not in_sentence:
                            if len(factor.word_ids): factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)
                            factor.word_ids.append([])
                            factor.strings.append([])
                            if factor.characters: factor.charseq_ids.append([])
                            if factor.with_root:
                                factor.word_ids[-1].append(factor.ROOT)
                                factor.strings[-1].append(factor.words[factor.ROOT])
                                if factor.characters: factor.charseq_ids[-1].append(factor.ROOT)

                        word = columns[f]
                        factor.strings[-1].append(word)

                        # Preprocess word
                        if f == self.LEMMAS:
                            word = _gen_lemma_rule(columns[self.FORMS], columns[self.LEMMAS], lr_allow_copy)
                            # print(columns[self.FORMS], columns[self.LEMMAS], word, _apply_lemma_rule(columns[self.FORMS], word))
                            # assert(_apply_lemma_rule(columns[self.FORMS], word) == columns[self.LEMMAS])

                        if f == self.FEATS:
                            word = "|".join(sorted(feat for feat in word.split("|")
                                                   if feat.split("=", 1)[0] in self.UNIVERSAL_FEATURES))
                        if f == self.DEPREL:
                            word = word.split(":")[0]

                        # Character-level information
                        if factor.characters:
                            if word not in factor.charseqs_map:
                                factor.charseqs_map[word] = len(factor.charseqs)
                                factor.charseqs.append([])
                                for c in word:
                                    if c not in factor.alphabet_map:
                                        if train:
                                            c = '<unk>'
                                        else:
                                            factor.alphabet_map[c] = len(factor.alphabet)
                                            factor.alphabet.append(c)
                                    factor.charseqs[-1].append(factor.alphabet_map[c])
                            factor.charseq_ids[-1].append(factor.charseqs_map[word])

                        # Word-level information
                        if f == self.HEAD:
                            factor.word_ids[-1].append(int(word) if word != "_" else -1)
                        else:
                            if word not in factor.words_map:
                                if train:
                                    word = '<unk>'
                                else:
                                    factor.words_map[word] = len(factor.words)
                                    factor.words.append(word)
                            factor.word_ids[-1].append(factor.words_map[word])
                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and len(self._factors[self.FORMS].word_ids) >= max_sentences:
                        break

        # Compute sentence lengths
        sentences = len(self._factors[self.FORMS].word_ids)
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(len(self._factors[self.FORMS].word_ids)):
            self._sentence_lens[i] = len(self._factors[self.FORMS].word_ids[i]) - self._factors[self.FORMS].with_root

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def factors(self):
        return self._factors

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))
            return True
        return False

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        batch_word_ids = []
        for factor in self._factors:
            batch_word_ids.append(np.zeros([batch_size, max_sentence_len + factor.with_root], np.int32))
            for i in range(batch_size):
                batch_word_ids[-1][i, 0:batch_sentence_lens[i] + factor.with_root] = factor.word_ids[batch_perm[i]]

        # Character-level data
        batch_charseq_ids, batch_charseqs, batch_charseq_lens = [], [], []
        for factor in self._factors:
            if not factor.characters:
                batch_charseq_ids.append([])
                batch_charseqs.append([])
                batch_charseq_lens.append([])
                continue

            batch_charseq_ids.append(np.zeros([batch_size, max_sentence_len + factor.with_root], np.int32))
            charseqs_map = {}
            charseqs = []
            charseq_lens = []
            for i in range(batch_size):
                for j, charseq_id in enumerate(factor.charseq_ids[batch_perm[i]]):
                    if charseq_id not in charseqs_map:
                        charseqs_map[charseq_id] = len(charseqs)
                        charseqs.append(factor.charseqs[charseq_id])
                    batch_charseq_ids[-1][i, j] = charseqs_map[charseq_id]

            batch_charseq_lens.append(np.array([len(charseq) for charseq in charseqs], np.int32))
            batch_charseqs.append(np.zeros([len(charseqs), np.max(batch_charseq_lens[-1])], np.int32))
            for i in range(len(charseqs)):
                batch_charseqs[-1][i, 0:len(charseqs[i])] = charseqs[i]

        return self._sentence_lens[batch_perm], batch_word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens

    def write_sentence(self, output, index, overrides):
        for i in range(self._sentence_lens[index] + 1):
            # Start by writing extras
            if index < len(self._extras) and i < len(self._extras[index]) and self._extras[index][i]:
                print(self._extras[index][i], file=output)
            if i == self._sentence_lens[index]: break

            fields = []
            fields.append(str(i + 1))
            for f in range(self.FACTORS):
                factor = self._factors[f]
                offset = i + factor.with_root

                field = factor.strings[index][offset]

                # Overrides
                if overrides is not None and f < len(overrides) and overrides[f] is not None:
                    if f == self.HEAD:
                        field = str(overrides[f][offset]) if overrides[f][offset] >= 0 else "_"
                    else:
                        field = factor.words[overrides[f][offset]]
                    if f == self.LEMMAS:
                        field = _apply_lemma_rule(fields[-1], field)

                fields.append(field)

            print("\t".join(fields), file=output)
        print(file=output)
