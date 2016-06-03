import json
import os
from datasets.image_sample import ImageFileSampleCV


def load_bboxes_dataset_with_json_marking(dataset_path: str,
                                          marking_filename='marking.json') -> list:
    """Данная функция загружает набор данных, которые представлены совокупностью
    изображений и размеченными объектами.

    Структура директории с разметкой:
        dataset_path/marking_filename: разметка всех изображений
        dataset_path/imgs: директория, содержащая файлы изображений

    Структура файла разметки:
        {
            "image01.jpg": [
                {
                    x: координата левого верхнего угла по горизонтали,
                    y: координата левого верхнего угла по вертикали,
                    w: ширина объекта,
                    h: высота объекта,
                    object_class: номер класса объекта, может отсутствовать (default: None),
                        0 - класс фона
                    ignore: игнорировать или нет объект (Boolean),
                        этот параметр может отсутствовать (default: False)
                }
            ], ...
        }
    "image01.jpg" - относительный путь к изображению по отношению к dataset_path/imgs.

    Args:
        dataset_path (str): путь к директории с разметкой и изображениями
        marking_filename (str): имя файла разметки

    Returns:
        list: Список объектов типа ImageFileSampleCV
    """

    marking_path = os.path.join(dataset_path, marking_filename)

    with open(marking_path, 'r') as f:
        marking = json.load(f)

    samples = []
    for image_name, image_marking in marking.items():
        image_path = os.path.join(dataset_path, 'imgs', image_name)

        if 'ignore' not in image_marking:
            image_marking['ignore'] = False
        if 'object_class' not in image_marking:
            image_marking['object_class'] = None

        image_sample = ImageFileSampleCV(image_path, image_marking)
        samples.append(image_sample)

    return samples


def load_images_from_directory_without_marking(images_path: str) -> list:
    """Загружает все изображения в форматах *.jpg,*.jpeg,*.png
    из указанной директории без разметки.

    Данная функция полезна для подготовки данных для тестирования на них детектора.

    Args:
        images_path: путь к папке, содержащей изображения
    Returns:
        list: Список объектов типа ImageFileSampleCV
    """

    files = filter(os.path.isfile, os.listdir(images_path))
    images_files = filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')), files)

    return [ImageFileSampleCV(image_path, []) for image_path in images_files]