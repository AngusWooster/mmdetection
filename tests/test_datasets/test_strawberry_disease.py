# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import cv2
import numpy as np

from mmdet.datasets import StrawberryDiseaseDataset


class TestStrawberryDiseaseDataset(unittest.TestCase):

    # def setUp(self) -> None:
    #     img_path = 'tests/data/strawberry_disease_dataset/2023_v1/images/gray_mold262.jpg'  # noqa: E501
    #     dummy_img = np.zeros((683, 1024, 3), dtype=np.uint8)
    #     cv2.imwrite(img_path, dummy_img)

    def test_strawberry_disease_dataset(self):
        train_dataset = StrawberryDiseaseDataset(data_root='tests/data/strawberry_disease_dataset/2023_v1/train',
                                                 ann_file = 'annotation.txt',
                                                 pipeline=[])
        train_dataset.full_init()
        self.assertEqual(len(train_dataset), 1)

        data_list = train_dataset.load_data_list()
        print(f"len = {len(data_list)}")
        print(f"data_list: {data_list}")
        self.assertEqual(len(data_list), 1)
        self.assertEqual(len(data_list[0]['instances']), 1)

test = TestStrawberryDiseaseDataset()
test.test_strawberry_disease_dataset()
print(f"end test")