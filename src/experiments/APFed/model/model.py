from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.regularizers import l2


def get_model():
    inputs = Input(shape=(39, 1))

    # Conv Layer with L2 regularization
    x = Conv1D(8, 3, padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LayerNormalization()(x)

    x = Conv1D(24, 2, padding='same', kernel_regularizer=l2(0.001))(x)
    x = Conv1D(192, 2, padding='same', kernel_regularizer=l2(0.001))(x)
    x = LayerNormalization()(x)

    x = Flatten()(x)

    # Fully Connected Layers with Dropout and L2 regularization
    x = Dense(160, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(80, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[
        'accuracy',
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(curve='ROC', name='auc_roc')
    ])
    
    return model