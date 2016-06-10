from datasets.image_sample import ImageSample
from datasets.loaders import load_bboxes_dataset_with_json_marking
from datasets.loaders import load_images_from_directory_without_marking


class ImagesCollection(object):
    """Коллекция изображений, поддерживает различные форматы:
        BBOX_JSON_MARKING - изображения с разметкой в json
                (см. loaders.load_bboxes_dataset_with_json_marking)
        IMAGES_DIR - директория с изображениями в форматах
                jpg, png, jpeg без разметки

    """

    def __init__(self, params: dict):
        """
        Args:
            params: Параметры коллекции, имеют формат:
                {
                    TYPE: тип датасета
                    PATH: путь к датасету
                    MARKING_NAME: (опционально) только для 'BBOX_JSON_MARKING'
                    SCALES: [500, 600]
                    MAX_SIZE: 1000
                }

        Returns:

        """
        assert params['TYPE'] in ['BBOX_JSON_MARKING', 'IMAGES_DIR']

        self._params = params
        self._samples = None
        self._max_size = params['MAX_SIZE']
        self._scales = params['SCALES']

        if params['TYPE'] == 'BBOX_JSON_MARKING':
            if 'MARKING_NAME' in params:
                self._samples = \
                    load_bboxes_dataset_with_json_marking(
                        params['PATH'], params['MARKING_NAME'],
                        self._max_size, self._scales)
            else:
                self._samples = \
                    load_bboxes_dataset_with_json_marking(
                        params['PATH'], 'marking.json', self._max_size, self._scales)

        elif params['TYPE'] == 'IMAGES_DIR':
            self._samples = \
                load_images_from_directory_without_marking(
                    params['PATH'], self._max_size, self._scales)

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def scales(self) -> list:
        return self._scales

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, key: int) -> ImageSample:
        return self._samples[key]
