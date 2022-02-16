"""
COPYRIGHT © BSH HOME APPLIANCES GROUP  2022

ALLE RECHTE VORBEHALTEN. ALL RIGHTS RESERVED.

The reproduction, transmission or use of this document or its contents is not permitted without express
written authority. Offenders will be liable for damages. All rights, including rights created by  patent
grant or registration of a utility model or design, are reserved.
"""
import argparse
import numpy as np
import os
import tensorflow as tf


class MyArgParser:

    def __init__(self):
        # Instantiate the parser
        parser = argparse.ArgumentParser(description='Edge Impulse argument parser')
        parser.add_argument("--epochs", help="number of epochs to train (e.g. 50)", nargs='?', type=int, const=1, default=1)
        parser.add_argument("--learning-rate", help="learning rate (e.g. 0.001).", nargs='?', type=float, const=0.05, default=0.05)
        parser.add_argument("--validation-set-size", help="size of the validation set (e.g. 0.2 for 20% of total training set).", nargs='?', type=float, const=0.2, default=0.2)
        parser.add_argument("--input-shape", help="shape of the training data (e.g. (320, 320, 3)) for a 320x320 RGB image", nargs='?', type=str, const=(1, 1, 1), default=(1, 1, 1))
        self.parser = parser

    def retrieve_args(self):
        return self.parser.parse_args()


class DummyCustomModel:

    def __init__(self):
        arguments = MyArgParser().retrieve_args()
        self.epochs = arguments.epochs
        self.learning_rate = arguments.learning_rate
        self.validation_set_size = arguments.validation_set_size
        self.input_shape = arguments.input_shape.replace("(", "[").replace(")", "]")
        self.input_shape = list(filter(lambda x: x.isnumeric(), self.input_shape))
        self.input_shape = list(map(lambda x: int(x), self.input_shape))
        self.__init_train_datasets()

    def __init_train_datasets(self):
        self.x_train = np.load("/home/X_train_features.npy")
        self.y_train = np.load("/home/y_train.npy")[:,0]
        # convert Y to a categorical vector
        self.classes = np.max(self.y_train)
        self.y_train = tf.keras.utils.to_categorical(self.y_train - 1, self.classes)

    def __create_model(self):
        # Create a model using high-level tf.keras.* APIs
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=self.classes, activation='softmax', name='y_pred')
        ])
        return model

    def __compile_and_train(self, model):
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='mean_squared_error')
        model.fit(x=self.x_train, y=self.y_train, epochs=self.epochs, validation_split=self.validation_set_size)
        return model

    def __representative_dataset(self):
        for i in range(self.x_train.shape[0]):
            observation = np.expand_dims(self.x_train[i], axis=0).astype('float32')
            yield [observation]

    def __convert_model(self, model):
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        # Save the model.
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.__representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_quant_model = converter.convert()
        with open('model_quantized_int8_io.tflite', 'wb') as f:
            f.write(tflite_quant_model)
        os.system("mv model.tflite /home/model.tflite")
        os.system("mv model_quantized_int8_io.tflite /home/model_quantized_int8_io.tflite")

    def train_and_convert(self):
        model = self.__create_model()
        model = self.__compile_and_train(model)
        self.__convert_model(model)


if __name__ == '__main__':
    # DummyCustomModel().train_and_convert()
    # np.save("X_train_features.npy", np.random.rand(3, 2))
    # np.save("y_train.npy", np.random.rand(3))
    DummyCustomModel().train_and_convert()
