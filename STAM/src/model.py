"""
LSTM Autoencoder model definition
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import config


def build_lstm_ae():
    """Build LSTM Autoencoder for anomaly detection"""
    
    # Encoder
    encoder_input = layers.Input(shape=(config.WINDOW_SIZE, config.N_FEATURES), name='encoder_input')
    
    x = layers.LSTM(config.ENCODE_UNITS[0], return_sequences=True, 
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='encoder_lstm_1')(encoder_input)
    x = layers.Dropout(0.2)(x)
    
    x = layers.LSTM(config.ENCODE_UNITS[1], return_sequences=False,
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='encoder_lstm_2')(x)
    x = layers.Dropout(0.2)(x)
    
    # Latent space
    latent = layers.Dense(config.LATENT_DIM, activation='relu', name='latent')(x)
    
    # Decoder
    x = layers.RepeatVector(config.WINDOW_SIZE, name='repeat_vector')(latent)
    
    x = layers.LSTM(config.DECODE_UNITS[0], return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='decoder_lstm_1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.LSTM(config.DECODE_UNITS[1], return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='decoder_lstm_2')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    decoder_output = layers.TimeDistributed(
        layers.Dense(config.N_FEATURES, activation='linear'),
        name='decoder_output'
    )(x)
    
    # Model
    lstm_ae = Model(encoder_input, decoder_output, name='lstm_autoencoder')
    
    # Use Huber loss (delta=1.0)
    lstm_ae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.Huber(delta=1.0)
    )
    
    return lstm_ae