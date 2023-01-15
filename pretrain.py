
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, datasets

from tqdm import tqdm

from datasets import polsar

from model_new import test_model, ProjectionHead, Prediction



def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)


def mix_loss(p, z, mix):
    max_rep = tf.maximum(p, z)
    
    max_rep = tf.math.l2_normalize(max_rep, axis=1)  # (bs, 128)
    mix = tf.math.l2_normalize(mix, axis=1)
    
    similarities = tf.reduce_sum(tf.multiply(max_rep, mix), axis=1)
    
    loss = 2 - 2 * tf.reduce_mean(similarities)
    
    return loss

def main():

    # Load dataset
    data = polsar()

    # Instantiate networks
    f_online = test_model()
    g_online = ProjectionHead()
    q_online = Prediction()

    f_target = test_model()
    g_target = ProjectionHead()
    
   


    # Initialize the weights of the networks
    x = tf.random.normal((512, 16, 16, 6))
    h = f_online(x, training=False)
    print('Initializing online networks...')
    print('Shape of h:', h.shape)
    z = g_online(h, training=False)
    print('Shape of z:', z.shape)
    p = q_online(z, training=False)
    print('Shape of p:', p.shape)

    h = f_target(x, training=False)
    print('Initializing target networks...')
    print('Shape of h:', h.shape)
    z = g_target(h, training=False)
    print('Shape of z:', z.shape)
    
    num_params_f = tf.reduce_sum([tf.reduce_prod(var.shape) for var in f_online.trainable_variables])    
    print('The encoders have {} trainable parameters each.'.format(num_params_f))

    f_online.save_weights('random.h5')
    #f_target.save_weights('zbn7w5_target_random.h5')
    
    @tf.function
    def train_step_pretraining(x1, x2, x3):  # (bs, 32, 32, 3), (bs, 32, 32, 3)
        # Forward pass
        
        h_target_1 = f_target(x1, training=True)
        z_target_1 = g_target(h_target_1, training=True)

        h_target_2 = f_target(x2, training=True)
        z_target_2 = g_target(h_target_2, training=True)
        
        #h_target_3 = f_target(x3, training=True)
        #z_target_3 = g_target(h_target_3, training=True)

        with tf.GradientTape(persistent=True) as tape:
            h_online_1 = f_online(x1, training=True)
            z_online_1 = g_online(h_online_1, training=True)
            p_online_1 = q_online(z_online_1, training=True)
            
            h_online_2 = f_online(x2, training=True)
            z_online_2 = g_online(h_online_2, training=True)
            p_online_2 = q_online(z_online_2, training=True)
            
            h_online_3 = f_online(x3, training=True)
            z_online_3 = g_online(h_online_3, training=True)
            p_online_3 = q_online(z_online_3, training=True)
            
            
            p_online = tf.concat([p_online_1, p_online_2], axis=0)
            z_target = tf.concat([z_target_2, z_target_1], axis=0)
            
            
            loss = byol_loss(p_online, z_target) + mix_loss(z_target_1, z_target_2, p_online_3)

        # Backward pass (update online networks)
        grads = tape.gradient(loss, f_online.trainable_variables)
        opt.apply_gradients(zip(grads, f_online.trainable_variables))
        grads = tape.gradient(loss, g_online.trainable_variables)
        opt.apply_gradients(zip(grads, g_online.trainable_variables))
        grads = tape.gradient(loss, q_online.trainable_variables)
        opt.apply_gradients(zip(grads, q_online.trainable_variables))
        del tape

        return loss
    
    num_epochs = 300
    batch_size = 512
    
    x1, _, mix = data.get_batch_pretraining(3, batch_size)
    print(x1.shape, _.shape, mix.shape)
    f_online.summary()

    batches_per_epoch = data.num_train_images // batch_size
      
    # Define optimizer
    #lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, num_epochs * batches_per_epoch, 1e-5, power=2)
    #lr = tf.keras.experimental.CosineDecayRestarts(1e-3, 100)
    lr = 1e-3 * batch_size / 512
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    print('Using Adam optimizer with learning rate {}.'.format(lr))

    log_every = 40  # batches
    save_every = 20 # epochs

    losses = []
    for epoch_id in tqdm(range(num_epochs)):
        data.shuffle_training_data()
        
        for batch_id in range(batches_per_epoch):
            x1, x2, x3 = data.get_batch_pretraining(batch_id, batch_size)
            loss = train_step_pretraining(x1, x2, x3)
            losses.append(float(loss))

            # Update target networks (exponential moving average of online networks)
            beta = 0.999

            f_target_weights = f_target.get_weights()
            f_online_weights = f_online.get_weights()
            for i in range(len(f_online_weights)):
                f_target_weights[i] = beta * f_target_weights[i] + (1 - beta) * f_online_weights[i]
            f_target.set_weights(f_target_weights)
            
            g_target_weights = g_target.get_weights()
            g_online_weights = g_online.get_weights()
            for i in range(len(g_online_weights)):
                g_target_weights[i] = beta * g_target_weights[i] + (1 - beta) * g_online_weights[i]
            g_target.set_weights(g_target_weights)

            if (batch_id + 1) % log_every == 0:
                print('[Epoch {}/{} Batch {}/{}] Loss={:.5f}.'.format(epoch_id+1, num_epochs, batch_id+1, batches_per_epoch, loss))

        if (epoch_id + 1) % save_every == 0:
            f_online.save_weights('pretrain_weights.h5')
            #f_target.save_weights('zbn4w_target_{}.h5'.format(epoch_id + 1))
            print('Weights of f saved.')
    
    np.savetxt('losses_f_online.txt', tf.stack(losses).numpy())


if __name__ == '__main__':
    main()
    
