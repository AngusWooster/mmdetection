import os.path as osp
from typing import List, Optional, Union
from mmengine.fileio import get_local_path, list_from_file

from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class StrawberryDiseaseDataset(XMLDataset):
    """Reader for the StrawberryDisease dataset in PASCAL VOC format."""

    METAINFO = {
        'classes':
        ('Angular Leafspot', 'Anthracnose Fruit Rot', 'Blossom Blight', 'Gray Mold',
         'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255),
                    (0, 60, 100), (0, 0, 142), (255, 77, 255)]
    }

    def __init__(self, **kwargs):
        super().__init__(img_subdir = 'images',
                         ann_subdir = 'images',
                         **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = osp.join(self.data_root, self.img_subdir, f'{img_id}.jpg')
            xml_path = osp.join(self.data_root, self.ann_subdir, f'{img_id}.xml')

            ## debug
            # print(f"file: {file_name}\nxml: {xml_path}")
            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list