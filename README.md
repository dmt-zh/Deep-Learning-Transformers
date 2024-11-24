# Обзор механизмов работы Transformer на примере фреймворка OpenNMT-tf

![](https://lena-voita.github.io/resources/lectures/seq2seq/transformer/model-min.png)

# Содержание

- ## [Создание тренировочного датасета](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset)
  - ### [Механизм формирования тренировочного датасета](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset#механизм-формирования-тренировочного-датасета)
  - ### [Механизм преобразования тренировочного датасета](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset#механизм-преобразования-тренировочного-датасета)
  - ### [Механизм формирования матрицы алайнмента](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset#механизм-формирования-матрицы-алайнмента)

<br>

- ## [Механизм тренировочного процесса](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md)
  - ### [Инициализация](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#инициализация)
  - ### [Тренировочный процесс](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#тренировочный-процесс)
     - #### [Энкодер](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#в-энкодере-векторное-представлнение-токенов-source-языка)
     - #### [Декодер](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#в-декодере-векторное-представлнение-токенов-target-языка)
     - #### [Расчет функции потерь](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#рассчет-функции-потерь)
     - #### [Механизм алайнмента](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#механизм-алайнмента)
     - #### [Механизм расчета и применения градиентов](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#механизм-расчета-и-применения-градиентов)
     - #### [Применение экспоненциального скользящего среднего](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#применение-экспоненциального-скользящего-среднего)
     - #### [Механизм затухания коэффициента скорости обучения](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#meханизм-затухания-коэффициента-скорости-обучения)
     - #### [Механизм усреднения чекпойнтов](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#meханизм-усреднения-чекпойнтов)

- ## [Механизм инференса](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/inference#инференс-модели-с-помощью-фреймворка-opennmt-tf)
