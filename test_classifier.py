import unittest
from classifier import IMG_SIZE, ImageIngestion


class TestImageIngestion(unittest.TestCase):
    def test_image_pre_process_values(self):
        self.assertRaises(TypeError, 10)

    def test_image_pre_process_output(self):
        out = ImageIngestion.image_pre_process(
            self, r"Labels//CSV Format//train_labels.csv"
        )
        self.assertEqual(len(out[0][0]), IMG_SIZE)
