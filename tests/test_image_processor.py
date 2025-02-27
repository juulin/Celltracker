import unittest
from src.image_processor import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = ImageProcessor()

    def test_load_tif_stack(self):
        # Test loading a TIF stack
        tif_stack = self.processor.load_tif_stack('path/to/tif_stack.tif')
        self.assertIsNotNone(tif_stack)
        self.assertGreater(len(tif_stack), 0)

    def test_preprocess_image(self):
        # Test preprocessing of a single image
        image = self.processor.load_tif_stack('path/to/tif_stack.tif')[0]
        preprocessed_image = self.processor.preprocess_image(image)
        self.assertIsNotNone(preprocessed_image)

    def test_extract_features(self):
        # Test feature extraction from an image
        image = self.processor.load_tif_stack('path/to/tif_stack.tif')[0]
        features = self.processor.extract_features(image)
        self.assertIn('intensity', features)
        self.assertIn('size', features)

if __name__ == '__main__':
    unittest.main()