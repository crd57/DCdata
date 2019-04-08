import tensorflow as tf

## Parameters
learning_rate = 0.01
##
batch_size = 64
##
target_sequence_length = 3
##
input_sequence_length = 12


def get_inputs():

    enc_inp = tf.placeholder(tf.float32, shape=(None, 12,1200,1200,1))
    # Decoder: target outputs
    target_seq = tf.placeholder(tf.float32, shape=(None, 12,1200,1200,1))
    return enc_inp,target_seq
def get_cell(input_shape):
    return tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                       input_shape=input_shape,
                                       output_channels=1,
                                       kernel_shape=[3, 3])


def get_encoder_layer(inputs):
    cell = tf.contrib.rnn.MultiRNNCell([get_cell([1200, 1200, 1]), get_cell([1200, 1200, 1])])
    initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major=False,
                                            initial_state=initial_state)
    return output, final_state


def get_decoder_layer(target_seq, encoder_state):
    # 创建一个常量tensor并复制为batch_size的大小
    start_tokens = tf.fill([batch_size, 1200, 1200, 1], 2, name="GO")
    end_tokens = tf.fill([batch_size, 1200, 1200, 1], 3,name="EOS")
    decoder_embeddings = [ start_tokens ] + target_seq[:-1]
    cell = tf.contrib.rnn.MultiRNNCell([get_cell([1200, 1200, 1]), get_cell([1200, 1200, 1])])
    output_layer = tf.layers.dense(1200*1200,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name="dense_layer")
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embeddings,
                                                            sequence_length=target_sequence_length,
                                                            time_major = False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=target_sequence_length)
    with tf.variable_scope("decoder", reuse=True):

        # "GreedyEmbeddingHelper"：预测阶段最常使用的Helper，下一时刻输入是上一时刻概率最大的单词通过embedding之后的向量
        # 用于inference阶段的helper，将output输出后的logits使用argmax获得id再经过embedding layer来获取下一时刻的输入。
        # start_tokens：起始
        # target_letter_to_int：结束
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     end_tokens)

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            maximum_iterations=target_sequence_length)
        # # dynamic_decode返回(final_outputs, final_state, final_sequence_lengths)。其中：final_outputs是tf.contrib.seq2seq.BasicDecoderOutput类型，包括两个字段：rnn_output，sample_id
        # tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, 1)
        # #将encoder_state输出的状态向量复制beam_width边其中beam_width为想要输出的句子的个数。
        # # 树搜索
        # bm_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, decoder_embeddings, start_tokens,
        #                                                   end_tokens, tiled_encoder_state,
        #                                                   1, output_layer)
        #
        # # impute_finished must be set to false when using beam search decoder
        # bm_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(bm_decoder,
        #                                                             maximum_iterations=target_sequence_length)
    return training_decoder_output, predicting_decoder_output

def seq2seq_model(inputs,target):
    _,encode_state = get_encoder_layer(inputs)
    training_decoder_output, predicting_decoder_output = get_decoder_layer(target,encode_state)
    return training_decoder_output, predicting_decoder_output
