from dataset import KittiDataset
from channelshuffle import ChannelShuffle
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    DepthwiseConv2D,
    ReLU,
    Add,
    Concatenate,
)
from keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List
import numpy as np
from tqdm import tqdm
from CBAM import CBAM


class ShuffleNetV2:
    def __init__(self, CBAM_status: bool, train_enhanced: bool):
        self.num_output_class: int = 3
        self.loss_function: str = "mse"
        self.optimizer: str = "adam"
        self.eval_metrics: List[str] = ["mae"]
        self.dataset: KittiDataset = KittiDataset()

        # Block architecture
        self.depthwise_kernel_size: int = 3
        self.conv_kernel_size: int = 1
        self.padding: str = "same"
        self.activation_function = "relu"

        # Model layers
        self.layer1_filters: int = 24
        self.layer1_kernel_size: int = 3
        self.layer1_strides: int = 2
        self.block2_filters: int = 116
        self.block2_strides: int = 1
        self.block3_filters: int = 232
        self.block3_strides: int = 2
        self.dense4_units: int = 128
        self.CBAM_status: bool = CBAM_status

        self.model = self.initialize_model()

        # Training/Evaluation metrics
        self.num_epochs: int = 15
        self.trained_model = None
        self.train_enhanced = train_enhanced

    def initialize_shufflenet_block(self, inputs, filters, stride):
        """
        Returns a single ShuffleNetV2 block.
        """
        x = DepthwiseConv2D(
            kernel_size=self.depthwise_kernel_size, strides=stride, padding=self.padding
        )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(
            filters=filters // 2,
            kernel_size=self.conv_kernel_size,
            padding=self.padding,
        )(x)
        x = BatchNormalization()(x)

        x = ChannelShuffle()(x)

        if stride == 1 and inputs.shape[-1] == filters:
            if self.CBAM_status:
                shortcut = CBAM()(inputs)
            x = Add()([x, inputs])

        else:
            shortcut = DepthwiseConv2D(
                kernel_size=self.depthwise_kernel_size,
                strides=stride,
                padding=self.padding,
            )(inputs)
            shortcut = BatchNormalization()(shortcut)
            shortcut = Conv2D(
                filters=filters // 2,
                kernel_size=self.conv_kernel_size,
                padding=self.padding,
            )(shortcut)
            shortcut = BatchNormalization()(shortcut)
            if self.CBAM_status:
                shortcut = CBAM()(shortcut)
            x = Concatenate()([x, shortcut])

        x = ReLU()(x)

        return x

    def initialize_model(self):
        """
        Initializes the ShuffleNetV2 regression model.
        """
        inputs = Input(shape=self.dataset.input_img_shape)

        # Layer 1
        x = Conv2D(
            self.layer1_filters,
            kernel_size=self.layer1_kernel_size,
            strides=self.layer1_strides,
            padding=self.padding,
        )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Block 2
        x = self.initialize_shufflenet_block(
            x, filters=self.block2_filters, stride=self.block2_strides
        )

        # Block 3
        x = self.initialize_shufflenet_block(
            x, filters=self.block3_filters, stride=self.block2_strides
        )

        # Output layer
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.dense4_units, activation=self.activation_function)(x)
        outputs = Dense(self.num_output_class)(x)

        model = Model(inputs, outputs)

        model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=self.eval_metrics
        )

        print("Model architecture:")
        model.summary()

        return model

    def train_model(self) -> None:
        """
        Trains model using training dataset and training parameters.
        """
        if self.train_enhanced:
            self.trained_model = self.model.fit(
                self.dataset.enhanced_training_dataset,
                epochs=self.num_epochs,
                validation_data=self.dataset.validation_dataset,
            )
        else:
            self.trained_model = self.model.fit(
                self.dataset.training_dataset,
                epochs=self.num_epochs,
                validation_data=self.dataset.validation_dataset,
            )

    def mean_absolute_percentage_error(self, y_true, y_pred):
        """
        Calculates MAPE, handling zero values in y_true to avoid division by zero.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            return (
                np.mean(
                    np.abs(
                        (y_true[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            return np.inf

    def evaluate_model_metrics(self):
        """
        Returns MAE, MSE, MAPE and R-squared values of output classes.
        """
        true_vals = []
        pred_vals = []

        # Extract predictions
        print("Collecting predictions...")
        for imgs, labels in tqdm(self.dataset.testing_dataset):
            preds = self.trained_model.predict(imgs, verbose=0)
            true_vals.append(labels.numpy())
            pred_vals.append(preds)

        true_vals = np.concatenate(true_vals, axis=0)
        pred_vals = np.concatenate(pred_vals, axis=0)

        metrics = {"MAE": [], "MSE": [], "MAPE": [], "R2": []}

        for i in range(self.num_output_class):
            print(f"calculating class {i+1}")
            true_dim = true_vals[:, i]
            pred_dim = pred_vals[:, i]

            # Calculate metrics
            mae = mean_absolute_error(true_dim, pred_dim)
            mse = mean_squared_error(true_dim, pred_dim)
            mape = self.mean_absolute_percentage_error(true_dim, pred_dim)
            r2 = r2_score(true_dim, pred_dim)

            # Store metrics
            metrics["MAE"].append(mae)
            metrics["MSE"].append(mse)
            metrics["MAPE"].append(mape)
            metrics["R2"].append(r2)

        return metrics
