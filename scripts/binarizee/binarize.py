#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads,
                                                                       allow_soft_placement=True))

    def construct(self, args, input_dimension):
        with self.session.graph.as_default():
            # Inputs
            self.input_embeddings = tf.placeholder(tf.float32, [None, None])
            self.learning_rate = tf.placeholder(tf.float32, [])

            # Autoencoder
            weights = tf.get_variable("weights", shape=[input_dimension, args.dimension], dtype=tf.float32)
            self.binarized_embeddings = tf.to_float(tf.less(tf.matmul(self.input_embeddings, weights), 0.))
            biases = tf.get_variable("biases", shape=[input_dimension], dtype=tf.float32, initializer=tf.zeros_initializer)
            self.reconstructed_embeddings = tf.tanh(tf.matmul(self.binarized_embeddings, weights, transpose_b=True) + biases)

            # Training
            loss_rec = tf.losses.mean_squared_error(
                self.input_embeddings, self.reconstructed_embeddings, reduction=tf.losses.Reduction.MEAN)
            loss_reg = tf.losses.mean_squared_error(
                tf.eye(args.dimension), tf.matmul(weights, weights, transpose_a=True), reduction=tf.losses.Reduction.MEAN)
            loss = loss_rec + args.regularization * loss_reg

            # Training
            self.global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=self.global_step)

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = [
                    tf.contrib.summary.scalar("train/loss", loss),
                    tf.contrib.summary.scalar("train/loss_rec", loss_rec),
                    tf.contrib.summary.scalar("train/loss_reg", loss_reg),
                    tf.contrib.summary.scalar("train/lr", self.learning_rate),
                ]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, embeddings, learning_rate, args):
        permutation = np.random.permutation(len(embeddings))
        while len(permutation):
            self.session.run([self.training, self.summaries], {self.input_embeddings: embeddings[permutation[:args.batch_size]],
                                                               self.learning_rate: learning_rate})
            permutation = permutation[args.batch_size:]

    def predict(self, embeddings, args):
        predictions = np.zeros((embeddings.shape[0], args.dimension if args.binarized else embeddings.shape[1]), dtype=np.float32)
        for i in range(0, len(embeddings), args.batch_size):
            batch = embeddings[i:i + args.batch_size]
            output = self.session.run(self.binarized_embeddings if args.binarized else self.reconstructed_embeddings,
                                      {self.input_embeddings: batch})
            predictions[i : i + len(batch)] = output

        return predictions if args.binarized else predictions * 2.


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import pickle
    import sys
    import re

    # Fix random seed
    np.random.seed(42)

    command_line = " ".join(sys.argv[1:])

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_basename", type=str, help="Input basename")
    parser.add_argument("output_basename", type=str, help="Output basename")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2")
    parser.add_argument("--binarized", default=1, type=int, help="Binarized.")
    parser.add_argument("--epochs", default="5:1e-3", type=str, help="Epochs and learning rates.")
    parser.add_argument("--dimension", default=256, type=int, help="Size of binarized embeddings.")
    parser.add_argument("--regularization", default=1.0, type=float, help="Regularization weight.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value)
                  for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Postprocess args
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]

    # Load the data
    with open("{}.words".format(args.input_basename), "rb") as words_file:
        words = pickle.load(words_file)
    embeddings = np.load("{}.embeddings.npy".format(args.input_basename))
    embeddings *= 0.5
    print("Out-of-range form {:.6f}% of embedding values.".format(
        100 * (np.sum(embeddings < -1.) + np.sum(embeddings > 1.)) / np.size(embeddings)), file=sys.stderr)
    embeddings = np.clip(embeddings, -1, 1)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(embeddings[0]))

    with open("{}/cmd".format(args.logdir), "w") as cmd_file:
        cmd_file.write(command_line)

    # Train
    for i, (epochs, learning_rate) in enumerate(args.epochs):
        for epoch in range(epochs):
            network.train_epoch(embeddings, learning_rate, args)

    # Predict and save
    predicted = network.predict(embeddings, args)
    with open("{}.words".format(args.output_basename), "wb") as words_file:
        pickle.dump(words, words_file)
    np.save("{}.embeddings.npy".format(args.output_basename), predicted)
