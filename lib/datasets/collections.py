from abc import abstractmethod
from abc import ABCMeta
from datasets.image_sample import ImageSample
from datasets.loaders import load_bboxes_dataset_with_json_marking
from datasets.loaders import load_images_from_directory_without_marking
import random


class AbstractImagesCollection(metaclass=ABCMeta):
    """Интерфейс коллекции изображений.
    Данный класс позволяет итерировать по загруженной коллекции изображений.
    """

    @abstractmethod
    def next(self):
        """
        Сдвинуть указатель к следующему изображению в коллекции. Порядок обхода
        должен быть специфицирован в производных классах.

        Returns:
            None
        """
        pass

    @property
    @abstractmethod
    def current_sample(self) -> ImageSample:
        """
        Получить доступ к изображению, на которое указывает текущий
        указатель.

        Returns:
            ImageSample: обёртка над изображением с разметкой
        """
        pass

    @property
    @abstractmethod
    def current_max_size(self) -> int:
        """Возвращает максимальный MAX_SIZE, который может быть
        применён к текущему изображению.

        Returns:

        """
        pass

    @property
    @abstractmethod
    def current_scales(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class SimpleImageCollection(AbstractImagesCollection):
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
                    SHUFFLE: (опционально, True) Перемешивать коллекцию на каждой эпохе
                }

        Returns:

        """
        assert params['TYPE'] in ['BBOX_JSON_MARKING', 'IMAGES_DIR']

        if 'SHUFFLE' not in params:
            params['SHUFFLE'] = True

        self._params = params
        self._samples = None
        self._shuffle = params['SHUFFLE']
        self._indx = 0

        if params['TYPE'] == 'BBOX_JSON_MARKING':
            if 'MARKING_NAME' in params:
                self._samples = \
                    load_bboxes_dataset_with_json_marking(
                        params['PATH'], params['MARKING_NAME'])
            else:
                self._samples = \
                    load_bboxes_dataset_with_json_marking(params['PATH'])

        elif params['TYPE'] == 'IMAGES_DIR':
            self._samples = \
                load_images_from_directory_without_marking(params['PATH'])

        if self._shuffle:
            random.shuffle(self._samples)

    def next(self):
        self._indx += 1

        if self._indx == len(self._samples):
            self._indx = 0

            if self._shuffle:
                random.shuffle(self._samples)

    @property
    def current_sample(self) -> ImageSample:
        return self._samples[self._indx]

    @property
    def current_scales(self) -> list:
        return self._params['SCALES']

    @property
    def current_max_size(self):
        return self._params['MAX_SIZE']

    def __len__(self):
        return len(self._samples)


class CombinedImageCollection(AbstractImagesCollection):
    """Комбинированная коллеция изображений, которая может состоять
    сразу из нескольких SimpleImageCollection
    """

    def __init__(self, datasets: list):
        """

        Args:
            datasets: список параметров датасетов, формат
                параметров см. в SimpleImageCollection.__init__

        Returns:

        """
        self._datasets = [SimpleImageCollection(params) for params in datasets]

        self._indx = 0

        # Создаем множество индексов для итерирования датасетов
        # Вероятность выбрать какой-то датасет пропорциональна его размеру
        self._indicies = []
        for indx, dataset in enumerate(self._datasets):
            self._indicies += [indx] * len(dataset)

        random.shuffle(self._indicies)

    def next(self):
        # Сдвигаем вперед указатель текущего датасета
        self.current_dataset.next()

        # Переходим к следующему датасету
        self._indx += 1
        if self._indx == len(self._indicies):
            self._indx = 0
            random.shuffle(self._indicies)

    @property
    def current_dataset(self):
        return self._datasets[self._indicies[self._indx]]

    @property
    def current_sample(self) -> ImageSample:
        return self.current_dataset.current_sample

    @property
    def current_scales(self) -> list:
        return self.current_dataset.current_scales

    @property
    def current_max_size(self):
        return self.current_dataset.current_max_size

    def __len__(self):
        return len(self._indicies)