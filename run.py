from seq2seq.model import *

train_graph = tf.Graph()
with train_graph.as_default():
    # define the global step of the graph
    global_step = tf.train.create_global_step(train_graph)
    # 获得模型输入
    input_data, targets = get_inputs()
    # 获得seq2seq模型的输出
    training_decoder_output, predicting_decoder_output, bm_decoder_output = seq2seq_model(input_data,targets)
