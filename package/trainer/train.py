from matplotlib import pyplot
import tensorflow as tf
import keras as keras
import numpy as np
import argparse
import pickle


initialWeights = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)


def define_discriminator(dim):
    in_shape = (dim, dim, 3)

    model = keras.models.Sequential()
    # normal
    model.add(keras.layers.Conv2D(
        64, (3, 3), padding='same', input_shape=in_shape, kernel_initializer=initialWeights))
    model.add(keras.layers.LeakyReLU(alpha=0.2))

    while (dim > 4):
        # downsample
        model.add(keras.layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=initialWeights))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        dim = dim / 2

    # classifier
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model


def define_generator(latent_dim, dim):
    model = keras.models.Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(keras.layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Reshape((4, 4, 256)))

    while (dim > 4):
        # upsample
        model.add(keras.layers.Conv2DTranspose(
            128, (4, 4), strides=(2, 2), padding='same', activation='relu', kernel_initializer=initialWeights))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        dim = dim / 2

    model.add(keras.layers.Conv2D(
        3, (3, 3), activation='tanh', padding='same'))
    return model


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = keras.models.Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = np.zeros((n_samples, 1))
    return X, y


def train(g_model, d_model, gan_model, dataset, latent_dim, job_dir, n_epochs=10, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # print('%d/%d' % (i+1, n_epochs))

        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(
                g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 25 == 0:
            summarize_performance(i, g_model, d_model,
                                  dataset, latent_dim, job_dir)


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, job_dir, n_samples=49):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' %
          (acc_real*100, acc_fake*100))

    logs_path = job_dir + 'plots'
    model_path = job_dir + 'models'

    # save plot
    save_plot(x_fake, epoch, logs_path)
    # save the generator model tile file
    filename = 'model_2_%03d.h5' % (epoch+1)
    g_model.save(filename)

    with tf.io.gfile.GFile(filename, mode='rb') as input_f:
        with tf.io.gfile.GFile(model_path + '/' + filename, mode='wb+') as output_f:
            output_f.write(input_f.read())


def save_plot(examples, epoch, plots_path, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = 'plot_2_e%03d.png' % (epoch+1)
    pyplot.savefig(filename, dpi=300)
    pyplot.close()

    with tf.io.gfile.GFile(filename, mode='rb') as input_f:
        with tf.io.gfile.GFile(plots_path + '/' + filename, mode='wb+') as output_f:
            output_f.write(input_f.read())


def load_dataset(job_dir):
    dataset_parts = []

    pickle_files = [
        job_dir + 'MAIVb.pickle'
    ]

    for pickle_file in pickle_files:
        with tf.io.gfile.GFile(pickle_file, mode='rb') as f:
            dataset_parts.append(pickle.load(f))
    
    dataset = np.concatenate(dataset_parts)    
    #dataset = np.array_split(dataset, 2)[0]

    return dataset


def main(job_dir, **args):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    latent_dim = 100
    dim = 128

    dataset = load_dataset(job_dir)

    with strategy.scope():
        d_model = define_discriminator(dim)
        g_model = define_generator(latent_dim, dim)
        gan_model = define_gan(g_model, d_model)
    
    train(g_model, d_model, gan_model, dataset, latent_dim, job_dir, 999999, 256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='Location to write checkpoints and export models',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
