import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import tensorflow as tf

from datasets import polsar

from model_new import test_model, ClassificationHead



def compute_test_accuracy(data, f_net, c_net):
    batch_size = 500
    num_batches = data.num_test_images // batch_size
    print(data.num_test_images)

    num_correct_predictions = 0
    for batch_id in range(num_batches):
        x, y = data.get_batch_testing(batch_id, batch_size)
        h = f_net(x, training=False)
        y_pred_logits = c_net(h)
        y_pred_labels = tf.argmax(y_pred_logits, axis=1, output_type=tf.int32)

        num_correct_predictions += tf.reduce_sum(tf.cast(tf.equal(y_pred_labels, y), tf.int32))

    print(num_correct_predictions)
    return tf.cast(num_correct_predictions / data.num_test_images, tf.float32)


def main():

    # Load dataset
    data = polsar()

    num_epochs = 100
    batch_size = 64

    # Instantiate networks
    f_net = test_model()
    c_net = ClassificationHead()

    # Initialize the weights
    # x, y = data.get_batch_finetuning(batch_id=0, batch_size=batch_size)
    # print(x.shape, y.shape)
    x = tf.random.normal((64, 16, 16, 6))
    h = f_net(x, training=False)
    print('Shape of h:', h.shape)
    s = c_net(h)
    print('Shape of s:', s.shape)

    # Load the weights of f from pretraining
    encoder_weights = 'pretrain_weights.h5'
    f_net.load_weights(encoder_weights)
 
    print('Weights of f loaded.')


    # Define optimizer
    batches_per_epoch = data.num_eval_images // batch_size
    total_update_steps = num_epochs * batches_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(5e-2, total_update_steps, 5e-4, power=2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    
    @tf.function
    def train_step_evaluation(x, y):  # (bs, 32, 32, 3), (bs)

        # Forward pass
        with tf.GradientTape() as tape:
            h = f_net(x, training=False)  # (bs, 512)
            y_pred_logits = c_net(h)  # (bs, 10)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred_logits))
        
        # Backward pass
        grads = tape.gradient(loss, c_net.trainable_variables)
        opt.apply_gradients(zip(grads, c_net.trainable_variables))

        return loss


    log_every = 10  # batches
    for epoch_id in range(num_epochs):
        data.shuffle_finetuning_data()
        
        for batch_id in range(batches_per_epoch):
            x, y = data.get_batch_finetuning(batch_id, batch_size)
            loss = train_step_evaluation(x, y)
            if (batch_id + 1) % log_every == 0:
                print('[Epoch {}/{} Batch {}/{}] Loss: {:.4f}'.format(epoch_id+1, num_epochs, batch_id+1, batches_per_epoch, loss))
    
    # Compute classification accuracy on test set
    test_accuracy = compute_test_accuracy(data, f_net, c_net)
    print('Test Accuracy: {:.4f}'.format(test_accuracy))
    


if __name__ == '__main__':
    main()