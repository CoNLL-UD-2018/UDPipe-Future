#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import conll18_ud_eval
import ud_dataset

class Network:
    METRICS = ["UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_tags):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.charseqs = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.charseq_ids = tf.placeholder(tf.int32, [None, None])
            self.tags = dict((tag, tf.placeholder(tf.int32, [None, None])) for tag in args.tags)
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [])

            # RNN Cell
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            # Word embeddings
            inputs = 0
            if args.we_dim:
                word_embeddings = tf.get_variable("word_embeddings", shape=[num_words, args.we_dim], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            # Character-level embeddings
            character_embeddings = tf.get_variable("character_embeddings", shape=[num_chars, args.cle_dim], dtype=tf.float32)
            characters_embedded = tf.nn.embedding_lookup(character_embeddings, self.charseqs)
            characters_embedded = tf.layers.dropout(characters_embedded, rate=args.dropout, training=self.is_training)
            _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.cle_dim), tf.nn.rnn_cell.GRUCell(args.cle_dim),
                characters_embedded, sequence_length=self.charseq_lens, dtype=tf.float32)
            cle = tf.concat([state_fwd, state_bwd], axis=1)
            inputs += tf.nn.embedding_lookup(cle, self.charseq_ids)

            # Computation
            hidden_layer = tf.layers.dropout(inputs, rate=args.dropout, training=self.is_training)
            for i in range(args.rnn_layers):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                    hidden_layer, sequence_length=self.sentence_lens, dtype=tf.float32,
                    scope="word-level-rnn-{}".format(i))
                hidden_layer += tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)

            loss = 0
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            self.predictions = {}
            for tag in args.tags:
                tag_layer = hidden_layer
                for _ in range(args.tag_layers):
                    tag_layer += tf.layers.dropout(tf.layers.dense(tag_layer, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                output_layer = tf.layers.dense(tag_layer, num_tags[tag])
                self.predictions[tag] = tf.argmax(output_layer, axis=2, output_type=tf.int32)

                # Training
                if args.label_smoothing:
                    gold_labels = tf.one_hot(self.tags[tag], num_tags[tag]) * (1 - args.label_smoothing) + args.label_smoothing / num_tags[tag]
                    loss += tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                else:
                    loss += tf.losses.sparse_softmax_cross_entropy(self.tags[tag], output_layer, weights=weights)

            # Pretrain saver
            self.saver_inference = tf.train.Saver(max_to_keep=2)

            # Training
            self.global_step = tf.train.create_global_step()
            self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=self.global_step)

            # Train saver
            self.saver_train = tf.train.Saver(max_to_keep=2)

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/lr", self.learning_rate)]
                for tag in args.tags:
                    self.summaries["train"].append(tf.contrib.summary.scalar(
                        "train/{}".format(tag),
                        tf.reduce_sum(tf.cast(tf.equal(self.tags[tag], self.predictions[tag]), tf.float32) * weights) /
                        tf.reduce_sum(weights)))

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
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
        while batches < 150:
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
                for tag in args.tags: feeds[self.tags[tag]] = word_ids[train.FACTORS_MAP[tag]]
                self.session.run([self.training, self.summaries["train"]], feeds)
                batches += 1
                if at_least_one_epoch: break
            at_least_one_epoch = True

    def evaluate(self, dataset_name, dataset, dataset_conllu, args):
        import io

        conllu, sentences = io.StringIO(), 0

        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(args.batch_size)


            feeds = {self.is_training: False, self.sentence_lens: sentence_lens,
                     self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                     self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS]}
            for tag in args.tags: feeds[self.tags[tag]] = word_ids[train.FACTORS_MAP[tag]]
            predictions, _ = self.session.run([self.predictions, self.update_loss], feeds)

            for i in range(len(sentence_lens)):
                overrides = [None] * dataset.FACTORS
                for tag in args.tags: overrides[dataset.FACTORS_MAP[tag]] = predictions[tag][i]
                dataset.write_sentence(conllu, sentences, overrides)
                sentences += 1

        metrics = conll18_ud_eval.evaluate(dataset_conllu, conll18_ud_eval.load_conllu(io.StringIO(conllu.getvalue())))
        self.session.run(self.summaries[dataset_name],
                         dict((self.metrics[metric], metrics[metric].f1) for metric in self.METRICS))

        return metrics["LAS"].f1 if metrics["LAS"].f1 < 1 else metrics["AllTags"].f1, metrics


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
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2")
    parser.add_argument("--char_dropout", default=0, type=float, help="Character dropout")
    parser.add_argument("--checkpoint", default="", type=str, help="Checkpoint.")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    parser.add_argument("--epochs", default="40:1e-3,20:1e-4", type=str, help="Epochs and learning rates.")
    parser.add_argument("--label_smoothing", default=0.03, type=float, help="Label smoothing.")
    parser.add_argument("--lr_allow_copy", default=0, type=int, help="Allow_copy in lemma rule.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=2, type=int, help="RNN layers.")
    parser.add_argument("--tags", default="UPOS,XPOS,FEATS,LEMMAS", type=str, help="Tags.")
    parser.add_argument("--tag_layers", default=0, type=int, help="Additional tag layers.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout")
    # Load defaults
    args, defaults = parser.parse_args(), []
    with open("ud_parser.args", "r") as args_file:
        for line in args_file:
            columns = line.rstrip("\n").split()
            if re.search(columns[0], args.basename): defaults.extend(columns[1:])
    args = parser.parse_args(args=defaults + sys.argv[1:])

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Postprocess args
    args.tags = args.tags.split(",")
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]

    # Load the data
    train = ud_dataset.UDDataset("{}-ud-train.conllu".format(args.basename), args.lr_allow_copy)
    dev = ud_dataset.UDDataset("{}-ud-dev.conllu".format(args.basename), args.lr_allow_copy, train=train, shuffle_batches=False)
    dev_conllu = conll18_ud_eval.load_conllu_file("{}-ud-dev.conllu".format(args.basename))

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      dict((tag, len(train.factors[train.FACTORS_MAP[tag]].words)) for tag in args.tags))

    if args.checkpoint:
        network.saver_train.restore(network.session, args.checkpoint)

    with open("{}/cmd".format(args.logdir), "w") as cmd_file:
        cmd_file.write(command_line)
    log_file = open("{}/log".format(args.logdir), "w")
    for tag in args.tags:
        print("{}: {}".format(tag, len(train.factors[train.FACTORS_MAP[tag]].words)), file=log_file, flush=True)

    # Train
    dev_best = 0
    for i, (epochs, learning_rate) in enumerate(args.epochs):
        for epoch in range(epochs):
            network.train_epoch(train, learning_rate, args)

            dev_accuracy, metrics = network.evaluate("dev", dev, dev_conllu, args)
            metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric].f1) for metric in Network.METRICS))
            print("Epoch {}, lr {}, dev {}".format(epoch + 1, learning_rate, metrics_log), file=log_file, flush=True)

            if dev_accuracy > dev_best:
                network.saver_train.save(network.session, "{}/checkpoint-best".format(args.logdir), global_step=network.global_step, write_meta_graph=False)
            dev_best = max(dev_best, dev_accuracy)
    network.saver_train.save(network.session, "{}/checkpoint-last".format(args.logdir), global_step=network.global_step, write_meta_graph=False)
    network.saver_inference.save(network.session, "{}/checkpoint-inference".format(args.logdir), global_step=network.global_step, write_meta_graph=False)
