import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os
# to plot training acc evolution
import matplotlib.pyplot as plt

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

"""

File with the train() and test() functions that are called from the main.py file.


"""

from model import Model
#from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset





def train(config):
    """ Training the network. """
    
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    # get total number of entries in dev
    dev_total = meta["total"]
    print("Building model...")

    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        
        # get the datasets in the form of batches:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)

        handle = tf.placeholder(tf.string, shape=[])

        print(train_dataset.output_shapes)
        #print(train_dataset.shape)

        # create the iterators for the training:
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        # create an instance of the model to be trained:
        model = Model(config, iterator, word_mat, char_mat, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        loss_save = 100.0
        patience = 0
        best_f1 = 0.
        best_em = 0.

        list_losses = []
        list_acc = []

        with tf.Session(config=sess_config) as sess:

            writer = tf.summary.FileWriter(config.log_dir)
            # initialize all variables:
            sess.run(tf.global_variables_initializer())
            # instance of the saver to save the model at the end of training:
            saver = tf.train.Saver()

            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())


            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
                print("loaded from save.",config.save_dir)

            elif os.path.exists(os.path.join(config.load_dir, "checkpoint")):
                # define the pretrained variable scopes
                scopes_to_be_loaded = ['Input_Embedding_Layer','Embedding_Encoder_Layer','Context_to_Query_Attention_Layer','Model_Encoder_Layer','Output_Layer']
                
                for scope in scopes_to_be_loaded:
                    variables_can_be_restored = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

                    temp_saver = tf.train.Saver(variables_can_be_restored)

                    temp_saver.restore(sess, tf.train.latest_checkpoint(config.load_dir))
                print("from: ",config.load_dir," loaded: ",scopes_to_be_loaded)

            else:
                print("training from scratch")

            global_step_1 = max(sess.run(model.global_step), 1)

            for _ in tqdm(range(global_step_1, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1


                # one training run:
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle, model.dropout: config.dropout})


                # if in periode to save batch loss
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)

                    metrics, summ = evaluate_batch(model, 2, train_eval_file, sess, "train", handle, train_handle)


                # if at a checkpoint to save and evaluate the model
                if global_step % config.checkpoint == 0:

                    print("Evaluating batches...")
                    metrics, summ = evaluate_batch(model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)
                    print("Perf on train sample: " + str(metrics))

                    metrics, summ = evaluate_batch(
                        model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                    print("Perf on dev sample: " + str(metrics))

                    list_losses.append(metrics["loss"])
                    list_acc.append(metrics["true_acc"])
    
                    print("Loss list: ",list_losses)
                    print("True acc list: ",list_acc)

                    # early stoping method
                    dev_f1 = metrics["f1"]
                    dev_em = metrics["exact_match"]
                    if dev_f1 < best_f1 and dev_em < best_em:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_em = max(best_em, dev_em)
                        best_f1 = max(best_f1, dev_f1)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()

                    # save the model
                    try:
                        filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                        saver.save(sess, filename)
                    except:
                        pass
    



def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    """
    Evaluate a 
    """

    all_yp3 = []
    conter_high = 0

    answer_dict = {}
    losses = []
    for numb_b in (range(1, num_batches + 1)):

        qa_id, loss, yp1, yp2, yp3, y1, y2, y3, logging, logging2, q = sess.run([model.qa_id, model.loss, model.yp1, model.yp2, model.yp3, model.y1, model.y2, model.y3, model.logging, model.logging2, model.q], feed_dict={handle: str_handle})

        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist(), yp3.tolist())

        answer_dict.update(answer_dict_)
        losses.append(loss)

    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    print(metrics)
    metrics["loss"] = loss

    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])

    return metrics, [loss_sum, f1_sum, em_sum]



def demo(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)
    demo = Demo(model, config)



def test(config):
    """ Testing the model on the dev set (the test set of SQuAD2 is not available.) """

    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)


    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")

    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
                print("loaded from save.",config.save_dir)

            elif os.path.exists(os.path.join(config.load_dir, "checkpoint")):
                # define the pretrained variable scopes
                scopes_to_be_loaded = ['Input_Embedding_Layer','Embedding_Encoder_Layer','Context_to_Query_Attention_Layer','Model_Encoder_Layer','Output_Layer']
                
                for scope in scopes_to_be_loaded:
                    variables_can_be_restored = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

                    temp_saver = tf.train.Saver(variables_can_be_restored)

                    temp_saver.restore(sess, tf.train.latest_checkpoint(config.load_dir))
                print("from: ",config.load_dir," loaded: ",scopes_to_be_loaded)
                
                
            if config.decay < 1.0:
                sess.run(model.assign_vars)
                
            losses = []
            answer_dict = {}
            remapped_dict = {}

            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, loss, yp1, yp2, yp3 = sess.run([model.qa_id, model.loss, model.yp1, model.yp2, model.yp3])

                answer_dict_, remapped_dict_ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist(), yp3.tolist())

                answer_dict.update(answer_dict_)
                remapped_dict.update(remapped_dict_)
                losses.append(loss)

            loss = np.mean(losses)
            metrics = evaluate(eval_file, answer_dict)
            print(metrics)
            with open(config.answer_file, "w") as fh:
                json.dump(remapped_dict, fh)

            print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))


