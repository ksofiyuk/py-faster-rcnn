from abc import abstractmethod
from abc import ABCMeta
import cv2

__author__ = 'Konstantin Sofiyuk'


class ImageSample(metaclass=ABCMeta):
    """Интерфейс класса для доступа к изображению"""

    @property
    @abstractmethod
    def bgr_data(self):
        pass

    @property
    @abstractmethod
    def marking(self):
        pass

    @property
    @abstractmethod
    def id(self):
        pass


class ImageFileSampleCV(ImageSample):
    """Изображение изначально не хранится в оперативной памяти,
       при необходимости каждый раз загружается с жёсткого диска
    """
    _prev_id=None
    _prev_bgr_data=None

    def __init__(self, image_path, marking):
        self._image_path = image_path
        self._marking = marking

    @property
    def bgr_data(self):
        """Загрузка изображения с жёсткого диска, если предыдущее обращение
        было к этому же классу, то изображение не загружается повторно

        Returns:
            numpy array с dtype=np.uint8, содержащий пиксели изображения
        """
        if ImageFileSampleCV._prev_id == self.id:
            data = ImageFileSampleCV._prev_bgr_data
        else:
            data = cv2.imread(self._image_path)
            ImageFileSampleCV._prev_bgr_data = data
            ImageFileSampleCV._prev_id = self.id

        return data.copy()

    @property
    def marking(self):
        return self._marking

    @property
    def id(self):
        return self._image_path
