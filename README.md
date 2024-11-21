# Обзор механизмов работы фреймворка OpenNMT-tf

![](https://lena-voita.github.io/resources/lectures/seq2seq/transformer/model-min.png)

# Содержание

- ## [Создание тренировочного датасета](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset)
  - ### [Механизм формирования тренировочного датасета](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset#механизм-формирования-тренировочного-датасета)
  - ### [Механизм преобразования тренировочного датасета](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset#механизм-преобразования-тренировочного-датасета)
  - ### [Механизм формирования матрицы алайнмента](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset#механизм-формирования-матрицы-алайнмента)

<br>

- ## [Механизм тренировочного процесса](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/training)
  - ### [Инициализация](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/training#инициализация)
  - ### [Тренировочный процесс]()
     - #### [Энкодер]()
     - #### [Декодер]()
     - #### [Расчет функции потерь]()
     - #### [Механизм алайнмента]()
     - #### [Механизм расчета и применения градиентов]()
     - #### [Применение экспоненциального скользящего среднего]()
     - #### [Механизм затухания коэффициента скорости обучения]()
     - #### [Механизм усреднения чекпойнтов]()

- ## [Механизм инференса]()
