from numpy import np
from typing import Dict, List
import cv2
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.data.experimental import cardinality


class KittiDataset:
    def __init__(self):
        self.input_img_shape: List[int, int] = (224, 224, 3)
        self.img_dtype = tf.int32
        self.batch_size: int = 64
        self.shuffle_seed: int = 15

        self.training_dataset: List[Dict] = None
        self.validation_dataset: List[Dict] = None
        self.testing_dataset: List[Dict] = None
        self.enhanced_training_dataset: List[Dict] = None

        self.load_dataset()
        self.process_datasets()

    def load_dataset(self) -> None:
        """
        Loads Falling Things training and testing datasets.
        """
        self.training_dataset = tfds.load("kitti", split="train", as_supervised=False)
        self.validation_dataset = tfds.load(
            "kitti", split="validation", as_supervised=False
        )
        self.testing_dataset = tfds.load("kitti", split="test", as_supervised=False)

    def image_enhance(img):
        """
        Denoises and applies filter to enhance edges in image.
        """
        deblurred_img = cv2.fastNlMeansDenoisingColored(img.numpy(), None, 11, 11, 5, 7)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen_img = cv2.filter2D(deblurred_img, -1, sharpen_kernel)

        return sharpen_img

    def process_img(self, img: np.ndarray, enhanced: bool):
        """
        Crops and normalizes img to single objects, and returns object size annotations.
        """
        # Extract first object if more than one in img
        ori_height = tf.cast(tf.shape(img["image"])[0], tf.float32)
        ori_width = tf.cast(tf.shape(img["image"])[1], tf.float32)

        # Convert pixel coordinates to img sizes for first annotated obj
        obj_bboxes = img["objects"]["bbox"]
        bounding_boxes = (
            obj_bboxes[0][0] * ori_height,
            obj_bboxes[0][1] * ori_width,
            obj_bboxes[0][2] * ori_height,
            obj_bboxes[0][3] * ori_width,
        )

        # Extract object dimesnions for first annotated obj
        obj_sizes = img["objects"]["dimensions"]
        sizes_3d = tf.stack([obj_sizes[0][0], obj_sizes[0][1], obj_sizes[0][2]])

        # Crop img to single object, and resize/normalize
        cropped_img = tf.image.crop_to_bounding_box(
            img["image"],
            offset_height=tf.cast(bounding_boxes[0], self.img_dtype),
            offset_width=tf.cast(bounding_boxes[1], self.img_dtype),
            target_height=tf.cast(
                tf.maximum(bounding_boxes[2] - bounding_boxes[0], 1.0), self.img_dtype
            ),
            target_width=tf.cast(
                tf.maximum(bounding_boxes[1] - bounding_boxes[3], 1.0), self.img_dtype
            ),
        )

        if enhanced:
            # Apply enhancements
            img_enhanced = tf.py_function(
                func=self.image_enhance, inp=[cropped_img], Tout=tf.uint8
            )
            img_enhanced.set_shape(cropped_img.shape)
            cropped_img = img_enhanced

        norm_img = tf.image.resize(cropped_img, list(self.input_img_shape[:-1])) / 255.0
        if enhanced:
            # Set image shape
            norm_img.set_shape(self.input_img_shape)

        return norm_img, sizes_3d

    def process_datasets(self) -> None:
        """
        Processes training, validation and testing datasets for object size estimations.
        """
        # Process training dataset
        training_dataset = self.training_dataset.map(
            lambda x: self.process_img(x, False)
        )
        training_dataset = training_dataset.cache()
        total_samples = cardinality(training_dataset).numpy()
        training_dataset = training_dataset.shuffle(
            buffer_size=total_samples, seed=self.shuffle_seed
        )
        training_dataset = training_dataset.batch(self.batch_size)

        # Process validation dataset
        validation_dataset = self.validation_dataset.map(
            lambda x: self.process_img(x, False)
        )
        validation_dataset = validation_dataset.cache()
        total_samples = cardinality(validation_dataset).numpy()
        validation_dataset = validation_dataset.shuffle(
            buffer_size=total_samples, seed=self.shuffle_seed
        )
        validation_dataset = validation_dataset.batch(self.batch_size)

        # Process testing dataset
        testing_dataset = self.testing_dataset.map(lambda x: self.process_img(x, False))
        testing_dataset = testing_dataset.cache()
        total_samples = cardinality(testing_dataset).numpy()
        testing_dataset = testing_dataset.shuffle(
            buffer_size=total_samples, seed=self.shuffle_seed
        )
        testing_dataset = testing_dataset.batch(self.batch_size)

        # Process enhanced training dataset
        enhanced_training_dataset = self.training_dataset.map(
            lambda x: self.process_img(x, True)
        )
        enhanced_training_dataset = enhanced_training_dataset.cache()
        total_samples = cardinality(enhanced_training_dataset).numpy()
        enhanced_training_dataset = enhanced_training_dataset.shuffle(
            buffer_size=total_samples, seed=self.shuffle_seed
        )
        enhanced_training_dataset = enhanced_training_dataset.batch(self.batch_size)

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.testing_dataset = testing_dataset
        self.enhanced_training_dataset = enhanced_training_dataset
