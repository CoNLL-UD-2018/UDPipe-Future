#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import dependency_decoding

import conll18_ud_eval
import ud_dataset

class Network:
    METRICS = ["UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads,
                                                                       allow_soft_placement=True))

    def construct(self, args, num_words, num_chars, num_tags, num_deprels, predict_only):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.charseqs = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.charseq_ids = tf.placeholder(tf.int32, [None, None])
            if args.embeddings: self.embeddings = tf.placeholder(tf.float32, [None, None, args.embeddings_size])
            if args.elmo_size: self.elmo = tf.placeholder(tf.float32, [None, None, args.elmo_size])
            self.tags = dict((tag, tf.placeholder(tf.int32, [None, None])) for tag in args.tags)
            self.heads = tf.placeholder(tf.int32, [None, None])
            self.deprels = tf.placeholder(tf.int32, [None, None])
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [])

            # RNN Cell
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            # Word embeddings
            inputs = []
            if args.we_dim:
                word_embeddings = tf.get_variable("word_embeddings", shape=[num_words, args.we_dim], dtype=tf.float32)
                inputs.append(tf.nn.embedding_lookup(word_embeddings, self.word_ids))

            # Character-level embeddings
            character_embeddings = tf.get_variable("character_embeddings", shape=[num_chars, args.cle_dim], dtype=tf.float32)
            characters_embedded = tf.nn.embedding_lookup(character_embeddings, self.charseqs)
            characters_embedded = tf.layers.dropout(characters_embedded, rate=args.dropout, training=self.is_training)
            _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.cle_dim), tf.nn.rnn_cell.GRUCell(args.cle_dim),
                characters_embedded, sequence_length=self.charseq_lens, dtype=tf.float32)
            cle = tf.concat([state_fwd, state_bwd], axis=1)
            cle_inputs = tf.nn.embedding_lookup(cle, self.charseq_ids)
            # If CLE dim is half WE dim, we add them together, which gives
            # better results; otherwise we concatenate CLE and WE.
            if 2 * args.cle_dim == args.we_dim:
                inputs[-1] += cle_inputs
            else:
                inputs.append(cle_inputs)

            # Pretrained embeddings
            if args.embeddings:
                inputs.append(self.embeddings)

            # Contextualized embeddings
            if args.elmo_size:
                inputs.append(self.elmo)

            # All inputs done
            inputs = tf.concat(inputs, axis=2)

            # Shared RNN layers
            hidden_layer = tf.layers.dropout(inputs, rate=args.dropout, training=self.is_training)
            for i in range(args.rnn_layers):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                    hidden_layer, sequence_length=self.sentence_lens + 1, dtype=tf.float32,
                    scope="word-level-rnn-{}".format(i))
                previous = hidden_layer
                hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                if i: hidden_layer += previous

            # Tagger
            loss = 0
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            weights_sum = tf.reduce_sum(weights)
            self.predictions = {}
            tag_hidden_layer = hidden_layer[:, 1:]
            for i in range(args.rnn_layers_tagger):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                    tag_hidden_layer, sequence_length=self.sentence_lens, dtype=tf.float32,
                    scope="word-level-rnn-tag-{}".format(i))
                previous = tag_hidden_layer
                tag_hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                if i: tag_hidden_layer += previous
            for tag in args.tags:
                tag_layer = tag_hidden_layer
                for _ in range(args.tag_layers):
                    tag_layer += tf.layers.dropout(tf.layers.dense(tag_layer, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                if tag == "LEMMAS": tag_layer = tf.concat([tag_layer, cle_inputs[:, 1:]], axis=2)
                output_layer = tf.layers.dense(tag_layer, num_tags[tag])
                self.predictions[tag] = tf.argmax(output_layer, axis=2, output_type=tf.int32)

                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.tags[tag], num_tags[tag]) * (1 - args.label_smoothing) + args.label_smoothing / num_tags[tag]
                    loss += tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                else:
                    loss += tf.losses.sparse_softmax_cross_entropy(self.tags[tag], output_layer, weights=weights)

            # Parsing
            if args.parse:
                max_words = tf.reduce_max(self.sentence_lens)

                if args.rnn_layers == 0:
                    parser_inputs = [inputs]
                    for tag in ["UPOS", "XPOS", "FEATS"]:
                        parser_inputs.append(tf.nn.embedding_lookup(tf.get_variable(tag + "_embeddings", shape=[num_tags[tag], 128], dtype=tf.float32),
                                                                    tf.pad(self.predictions[tag], ((0, 0),(1, 0)), constant_values=2)))
                    parser_inputs = tf.concat(parser_inputs, axis=2)
                    hidden_layer = tf.layers.dropout(parser_inputs, rate=args.dropout, training=self.is_training)

                for i in range(args.rnn_layers_parser):
                    (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                        rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                        hidden_layer, sequence_length=self.sentence_lens + 1, dtype=tf.float32,
                        scope="word-level-rnn-parser-{}".format(i))
                    previous = hidden_layer
                    hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                    if i: hidden_layer += previous

                # Heads
                head_deps = hidden_layer[:, 1:]
                for _ in range(args.parser_layers):
                    head_deps += tf.layers.dropout(tf.layers.dense(head_deps, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                head_roots = hidden_layer
                for _ in range(args.parser_layers):
                    head_roots += tf.layers.dropout(tf.layers.dense(head_roots, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)

                head_deps_bias = tf.get_variable("head_deps_bias", [args.rnn_cell_dim], dtype=tf.float32, initializer=tf.zeros_initializer)
                head_roots_bias = tf.get_variable("head_roots_bias", [args.rnn_cell_dim], dtype=tf.float32, initializer=tf.zeros_initializer)
                head_biaffine = tf.get_variable("head_biaffine", [args.rnn_cell_dim, args.rnn_cell_dim], dtype=tf.float32, initializer=tf.zeros_initializer)

                heads = tf.reshape(tf.matmul(tf.reshape(head_deps, [-1, args.rnn_cell_dim]) + head_deps_bias, head_biaffine),
                                   [tf.shape(hidden_layer)[0], -1, args.rnn_cell_dim])
                heads = tf.matmul(heads, head_roots + head_roots_bias, transpose_b=True)
                self.heads_logs = tf.nn.log_softmax(heads)
                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.heads, max_words + 1) * (1 - args.label_smoothing)
                    gold_labels += args.label_smoothing / tf.to_float(max_words + 1)
                    loss += tf.losses.softmax_cross_entropy(gold_labels, heads, weights=weights)
                else:
                    loss += tf.losses.sparse_softmax_cross_entropy(self.heads, heads, weights=weights)

                # Deprels
                self.deprel_hidden_layer = tf.identity(hidden_layer)
                self.deprel_heads = tf.identity(self.heads)

                deprel_deps = tf.layers.dropout(tf.layers.dense(self.deprel_hidden_layer[:, 1:], args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                for _ in range(args.parser_layers - 1):
                    deprel_deps += tf.layers.dropout(tf.layers.dense(deprel_deps, args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)

                deprel_indices = tf.stack([
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.deprel_heads)[0]), axis=1), multiples=[1, tf.shape(self.deprel_heads)[1]]),
                    self.deprel_heads], axis=2)
                deprel_roots = tf.gather_nd(self.deprel_hidden_layer, deprel_indices, )
                deprel_roots = tf.layers.dropout(tf.layers.dense(deprel_roots, args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                for _ in range(args.parser_layers - 1):
                    deprel_roots += tf.layers.dropout(tf.layers.dense(deprel_roots, args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)

                deprel_deps_bias = tf.get_variable("deprel_deps_bias", [args.parser_deprel_dim], dtype=tf.float32, initializer=tf.zeros_initializer)
                deprel_roots_bias = tf.get_variable("deprel_roots_bias", [args.parser_deprel_dim], dtype=tf.float32, initializer=tf.zeros_initializer)
                deprel_biaffine = tf.get_variable("deprel_biaffine", [args.parser_deprel_dim, num_deprels * args.parser_deprel_dim], dtype=tf.float32, initializer=tf.zeros_initializer)

                deprels = tf.reshape(tf.matmul(tf.reshape(deprel_deps, [-1, args.parser_deprel_dim]) + deprel_deps_bias, deprel_biaffine),
                                     [tf.shape(self.deprel_hidden_layer)[0], -1, num_deprels, args.parser_deprel_dim])
                deprels = tf.squeeze(tf.matmul(deprels, tf.expand_dims(deprel_roots + deprel_roots_bias, axis=3)), axis=3)
                self.predictions_deprel = tf.argmax(deprels, axis=2, output_type=tf.int32)
                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.deprels, num_deprels) * (1 - args.label_smoothing)
                    gold_labels += args.label_smoothing / num_deprels
                    loss += tf.losses.softmax_cross_entropy(gold_labels, deprels, weights=weights)
                else:
                    loss += tf.losses.sparse_softmax_cross_entropy(self.deprels, deprels, weights=weights)

            # Pretrain saver
            self.saver_inference = tf.train.Saver(max_to_keep=1)
            if predict_only: return

            # Training
            self.global_step = tf.train.create_global_step()
            self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=self.global_step)

            # Train saver
            self.saver_train = tf.train.Saver(max_to_keep=2)

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [
                    tf.contrib.summary.scalar("train/loss", loss),
                    tf.contrib.summary.scalar("train/lr", self.learning_rate)]
                for tag in args.tags:
                    self.summaries["train"].append(tf.contrib.summary.scalar(
                        "train/{}".format(tag),
                        tf.reduce_sum(tf.cast(tf.equal(self.tags[tag], self.predictions[tag]), tf.float32) * weights) /
                        weights_sum))
                if args.parse:
                    heads_acc = tf.reduce_sum(tf.cast(tf.equal(self.heads, tf.argmax(heads, axis=-1, output_type=tf.int32)),
                                                      tf.float32) * weights) / weights_sum
                    self.summaries["train"].extend([tf.contrib.summary.scalar("train/heads_acc", heads_acc)])
                    deprels_acc = tf.reduce_sum(tf.cast(tf.equal(self.deprels, tf.argmax(deprels, axis=-1, output_type=tf.int32)),
                                                        tf.float32) * weights) / weights_sum
                    self.summaries["train"].extend([tf.contrib.summary.scalar("train/deprels_acc", deprels_acc)])

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=weights_sum)
                self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
                self.metrics = dict((metric, tf.placeholder(tf.float32, [])) for metric in self.METRICS)
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss)]
                    for metric in self.METRICS:
                        self.summaries[dataset].append(tf.contrib.summary.scalar("{}/{}".format(dataset, metric),
                                                                                 self.metrics[metric]))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, learning_rate, args):
        batches, at_least_one_epoch = 0, False
        while batches < args.min_epoch_batches:
            while not train.epoch_finished():
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(args.batch_size)
                if args.word_dropout:
                    mask = np.random.binomial(n=1, p=args.word_dropout, size=word_ids[train.FORMS].shape)
                    word_ids[train.FORMS] = (1 - mask) * word_ids[train.FORMS] + mask * train.factors[train.FORMS].words_map["<unk>"]
                if args.char_dropout:
                    mask = np.random.binomial(n=1, p=args.char_dropout, size=charseqs[train.FORMS].shape)
                    charseqs[train.FORMS] = (1 - mask) * charseqs[train.FORMS] + mask * train.factors[train.FORMS].alphabet_map["<unk>"]

                feeds = {self.is_training: True, self.learning_rate: learning_rate, self.sentence_lens: sentence_lens,
                         self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                         self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS]}
                if args.embeddings:
                    if args.word_dropout:
                        mask = np.random.binomial(n=1, p=args.word_dropout, size=word_ids[train.EMBEDDINGS].shape)
                        word_ids[train.EMBEDDINGS] = (1 - mask) * word_ids[train.EMBEDDINGS]
                    embeddings = np.zeros([word_ids[train.EMBEDDINGS].shape[0], word_ids[train.EMBEDDINGS].shape[1], args.embeddings_size])
                    for i in range(embeddings.shape[0]):
                        for j in range(embeddings.shape[1]):
                            if word_ids[train.EMBEDDINGS][i, j]:
                                embeddings[i, j] = args.embeddings_data[word_ids[train.EMBEDDINGS][i, j] - 1]
                    feeds[self.embeddings] = embeddings
                if args.elmo_size:
                    feeds[self.elmo] = word_ids[train.ELMO]
                for tag in args.tags: feeds[self.tags[tag]] = word_ids[train.FACTORS_MAP[tag]]
                if args.parse:
                    feeds[self.heads] = word_ids[train.HEAD]
                    feeds[self.deprels] = word_ids[train.DEPREL]
                self.session.run([self.training, self.summaries["train"]], feeds)
                batches += 1
                if at_least_one_epoch: break
            at_least_one_epoch = True

    def predict(self, dataset, evaluating, args):
        import io
        conllu, sentences = io.StringIO(), 0

        if evaluating: self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(args.batch_size)

            feeds = {self.is_training: False, self.sentence_lens: sentence_lens,
                     self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                     self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS]}
            if args.embeddings:
                embeddings = np.zeros([word_ids[train.EMBEDDINGS].shape[0], word_ids[train.EMBEDDINGS].shape[1], args.embeddings_size])
                for i in range(embeddings.shape[0]):
                    for j in range(embeddings.shape[1]):
                        if word_ids[train.EMBEDDINGS][i, j]:
                            embeddings[i, j] = args.embeddings_data[word_ids[train.EMBEDDINGS][i, j] - 1]
                feeds[self.embeddings] = embeddings
            if args.elmo_size:
                feeds[self.elmo] = word_ids[train.ELMO]
            if evaluating:
                for tag in args.tags: feeds[self.tags[tag]] = word_ids[train.FACTORS_MAP[tag]]
                if args.parse:
                    feeds[self.heads] = word_ids[train.HEAD]
                    feeds[self.deprels] = word_ids[train.DEPREL]

            targets = [self.predictions]
            if args.parse: targets.extend([self.heads_logs, self.deprel_hidden_layer])
            if evaluating: targets.append(self.update_loss)
            predictions, *other_values = self.session.run(targets, feeds)
            if args.parse: prior_heads, deprel_hidden_layer, *_ = other_values

            if args.parse:
                heads = np.zeros(prior_heads.shape[:2], dtype=np.int32)
                for i in range(len(sentence_lens)):
                    padded_heads = np.pad(prior_heads[i][:sentence_lens[i], :sentence_lens[i] + 1].astype(np.float),
                                          ((1, 0), (0, 0)), mode="constant")
                    padded_heads[:, 0] = np.nan
                    padded_heads[1 + np.argmax(prior_heads[i][:sentence_lens[i], 0]), 0] = 0
                    chosen_heads, _ = dependency_decoding.chu_liu_edmonds(padded_heads)
                    heads[i, :sentence_lens[i]] = chosen_heads[1:]
                deprels = self.session.run(self.predictions_deprel,
                                           {self.is_training: False, self.deprel_hidden_layer: deprel_hidden_layer, self.deprel_heads: heads})

            for i in range(len(sentence_lens)):
                overrides = [None] * dataset.FACTORS
                for tag in args.tags: overrides[dataset.FACTORS_MAP[tag]] = predictions[tag][i]
                if args.parse:
                    overrides[dataset.HEAD] = heads[i]
                    overrides[dataset.DEPREL] = deprels[i]
                dataset.write_sentence(conllu, sentences, overrides)
                sentences += 1

        return conllu.getvalue()

    def evaluate(self, dataset_name, dataset, dataset_conllu, args):
        import io

        conllu = self.predict(dataset, True, args)
        metrics = conll18_ud_eval.evaluate(dataset_conllu, conll18_ud_eval.load_conllu(io.StringIO(conllu)))
        self.session.run(self.summaries[dataset_name],
                         dict((self.metrics[metric], metrics[metric].f1) for metric in self.METRICS))

        if args.parse:
            return (metrics["LAS"].f1 + metrics["MLAS"].f1 + metrics["BLEX"].f1) / 3., metrics
        else:
            return metrics["AllTags"].f1, metrics


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import sys
    import re

    # Fix random seed
    np.random.seed(42)

    command_line = " ".join(sys.argv[1:])

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", type=str, help="Base data name")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2")
    parser.add_argument("--char_dropout", default=0, type=float, help="Character dropout")
    parser.add_argument("--checkpoint", default="", type=str, help="Checkpoint.")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    parser.add_argument("--elmo", default=None, type=str, help="External contextualized embeddings to use.")
    parser.add_argument("--embeddings", default=None, type=str, help="External embeddings to use.")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4", type=str, help="Epochs and learning rates.")
    parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
    parser.add_argument("--min_epoch_batches", default=300, type=int, help="Minimum number of batches per epoch.")
    parser.add_argument("--parse", default=1, type=int, help="Parse.")
    parser.add_argument("--parser_layers", default=1, type=int, help="Parser layers.")
    parser.add_argument("--parser_deprel_dim", default=128, type=int, help="Parser deprel dim.")
    parser.add_argument("--predict", default=False, action="store_true", help="Only predict.")
    parser.add_argument("--predict_input", default=None, type=str, help="Input to prediction.")
    parser.add_argument("--predict_output", default=None, type=str, help="Output to prediction.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=2, type=int, help="RNN layers.")
    parser.add_argument("--rnn_layers_parser", default=1, type=int, help="Parser RNN layers.")
    parser.add_argument("--rnn_layers_tagger", default=1, type=int, help="Tagger RNN layers.")
    parser.add_argument("--tags", default="UPOS,XPOS,FEATS,LEMMAS", type=str, help="Tags.")
    parser.add_argument("--tag_layers", default=1, type=int, help="Additional tag layers.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout")
    args = parser.parse_args()

    if args.exp is None:
        args.exp = "{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    # Create logdir name
    do_not_log = {"exp", "predict", "predict_input", "predict_output", "tags", "threads"}
    args.logdir = "logs/{}-{}".format(
        args.exp,
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value)
                  for key, value in sorted(vars(args).items()) if key not in do_not_log))
    )
    if not args.predict and not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Postprocess args
    args.tags = args.tags.split(",")
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]

    # Load the data
    if args.embeddings:
        with np.load(args.embeddings) as embeddings_npz:
            args.embeddings_words = embeddings_npz["words"]
            args.embeddings_data = embeddings_npz["embeddings"]
            args.embeddings_size = args.embeddings_data.shape[1]

    root_factors = [ud_dataset.UDDataset.FORMS]
    train = ud_dataset.UDDataset("{}-ud-train.conllu".format(args.basename), root_factors,
                                 embeddings=args.embeddings_words if args.embeddings else None,
                                 elmo=args.elmo + "-train.npz" if args.elmo else None)
    dev = ud_dataset.UDDataset("{}-ud-dev.conllu".format(args.basename), root_factors, train=train, shuffle_batches=False,
                               elmo=args.elmo + "-dev.npz" if args.elmo else None)
    test = ud_dataset.UDDataset("{}-ud-test.conllu".format(args.basename), root_factors, train=train, shuffle_batches=False,
                                elmo=args.elmo + "-test.npz" if args.elmo else None)
    args.elmo_size = train.elmo_size

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      dict((tag, len(train.factors[train.FACTORS_MAP[tag]].words)) for tag in args.tags),
                      len(train.factors[train.DEPREL].words), predict_only=args.predict)

    if args.checkpoint:
        saver = network.saver_train if args.predict is None else network.saver_inference
        saver.restore(network.session, args.checkpoint)

    if args.predict:
        test = ud_dataset.UDDataset(args.predict_input, root_factors, train=train, shuffle_batches=False)
        conllu = network.predict(test, False, args)
        print(conllu, end="", file=open(args.predict_output, "w", encoding="utf-8") if args.predict_output else sys.stdout)
        exit(0)

    with open("{}/cmd".format(args.logdir), "w") as cmd_file:
        cmd_file.write(command_line)
    log_file = open("{}/log".format(args.logdir), "w")
    for tag in args.tags + ["DEPREL"]:
        print("{}: {}".format(tag, len(train.factors[train.FACTORS_MAP[tag]].words)), file=log_file, flush=True)
    print("Parsing with args:", "\n".join(("{}: {}".format(key, value) for key, value in sorted(vars(args).items())
                                           if key not in ["embeddings_data", "embeddings_words"])), flush=True)

    dev_conllu = conll18_ud_eval.load_conllu_file("{}-ud-dev.conllu".format(args.basename))
    test_conllu = conll18_ud_eval.load_conllu_file("{}-ud-test.conllu".format(args.basename))
    for i, (epochs, learning_rate) in enumerate(args.epochs):
        for epoch in range(epochs):
            network.train_epoch(train, learning_rate, args)

            dev_accuracy, metrics = network.evaluate("dev", dev, dev_conllu, args)
            metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric].f1) for metric in Network.METRICS))
            print("Dev, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=log_file, flush=True)

            test_accuracy, metrics = network.evaluate("test", test, test_conllu, args)
            metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric].f1) for metric in Network.METRICS))
            print("Test, epoch {}, lr {}, {}".format(epoch + 1, learning_rate, metrics_log), file=log_file, flush=True)

            #network.saver_train.save(network.session, "{}/checkpoint".format(args.logdir), global_step=network.global_step, write_meta_graph=False)
    network.saver_inference.save(network.session, "{}/checkpoint-inference-last".format(args.logdir), write_meta_graph=False)
