<br>

# Инициализация

### При запуске тренировки последовательно происходят следующие шаги:
- ##### считывается и финализируется конфигурационный файл (объединяются auto_config и пользовательский config файлы);
  >>>
  ├── [config = self._finalize_config()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L214) модуль `runner.py`\
  ├── [def _finalize_config()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L116) модуль `runner.py`\
  ├── [def auto_config()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/transformer.py#L149) модуль `transformer.py`
  >>>

- ##### устанавливается тип вычислений tensorflow;
  >>>
  При тренировке со смешанной точностью (FP16) в tensorflow передается значение `set_global_policy("mixed_float16")`\
  ├── [mixed_precision = self._mixed_precision and misc.enable_mixed_precision()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L218C9-L218C82) модуль `runner.py`\
  ├── [def enable_mixed_precision()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/utils/misc.py#L53) модуль `misc.py`
  >>>

- ##### инициализируется модель трансформера (или другой тип модели, определенный пользователем);
  >>>
   Инициализация модели происходит только после установки типа вычислений, т.к. в приватных атрибутах модели фиксируется тип вычисления.\
 <span style="color:red">**Если проинициализировать модель до инициализации типа вычислений, то значение в атрибутах модели не измениться и будет дефолтным float32**</span>:\
 `_dtype_policy: <Policy "float32">`\
 `_compute_dtype_object: <dtype: 'float32'`>

  ├── [model = self._init_model(config)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L219) модуль `runner.py`\
  ├── [def _init_model()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L169) `runner.py`
  >>>

- ##### определяется и инициализируется оптимизатор модели;
  >>>

   * из [автоконфига](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/transformer.py#L157) модели трансформер берется наименования оптимизатора, который по дефолту задан `LazyAdam`. Этот оптимизатор реализован в библиотеке дополнений [TensorFlow Addons](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/LazyAdam), **(развитие и поддержка которой остановлена)**.

  <br>

   * инициализируется функция затухания `NoamDecay` со следующими параметрами:\
    ━ 1) `scale` - устанавливается значение из параметра `Learning rate` конфигурационного файла тренировки (в нашем примере значение равно 2)\
    ━ 2) `model_dim` - устанавливается значение из параметра `num_units` конфигурационного файла тренировки (в нашем примере значение равно 4)\
    ━ 3) `warmup_steps`- устанавливается значение из параметра `Warmup steps` конфигурационного файла тренировки (в нашем примере значение равно 8000)

  <br>

   * инициализируется класс-обертка `ScheduleWrapper` для дополнения поведения планировщика скорости обучения со следующими параметрами:\
    ━ 1) `schedule` - проинициализированная выше функции затухания `NoamDecay`\
    ━ 2) `step_start` - устанавливается значение из параметра `start_decay_steps` конфигурационного файла тренировки (дефолтное значение 0)\
    ━ 3) `step_duration`- устанавливается значение из параметра `decay_step_duration` конфигурационного файла тренировки (дефолтное значение 1)\
    ━ 4) `minimum_learning_rate`- устанавливается значение из параметра `Minimum learning rate` конфигурационного файла тренировки (в нашем примере значение равно 0.0001)
  <br>

   * инициализируется класс оптимизатора [LazyAdam](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/LazyAdam) со следующими параметрами:\
    ━ 1) `learning_rate` - проинициализированный выше класс `ScheduleWrapper`\
    ━ 2) `kwargs` - словарь коэффициентов `beta` из параметра `optimizer_params` конфигурационного файла тренировки (в нашем примере {'beta_1': 0.9, 'beta_2': 0.998})\
    ━ 3) при тренировке со смешанной точностью (FP16) класс `LazyAdam`, в свою очередь наследуется от класса [tf.keras.mixed_precision.LossScaleOptimizer](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer) у которого инициализируются следующие параметры:\
           `initial_scale = 32 768` - значение, на которое будет корректироваться значение полученное из `loss` функции\
           `dynamic_growth_steps = 2 000` - как часто обновлять значение, на которое будет корректироваться величина `loss` функции
  <br>

  ├── [optimizer = model.get_optimizer()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L220) модуль `runner.py`\
  ├── [def get_optimizer()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/model.py#L353) модуль `model.py`\
  ├── [def make_learning_rate_schedule()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/schedules/lr_schedules.py#L39C5-L39C32) модуль `schedules/lr_schedules.py`\
  ├── [def __init__() class NoamDecay(tf.keras.optimizers.schedules.LearningRateSchedule)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/schedules/lr_schedules.py#L119) модуль `schedules/lr_schedules.py`\
  ├── [def __init__() class ScheduleWrapper(tf.keras.optimizers.schedules.LearningRateSchedule)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/schedules/lr_schedules.py#L88) модуль `schedules/lr_schedules.py`\
  ├── [def make_optimizer()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/optimizers/utils.py#L51C5-L51C19) модуль `optimizers/utils.py`
  >>>

- ##### устанавливается значение `batch_size_multiple`;
  >>>
  Если активирована функция `Mixed precision` или `Jit compile`, то `batch_size_multiple` будет равен **8**, иначе 1.\
  ├── [batch_size_multiple = (...)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L228) модуль `runner.py`
  >>>

- ##### создается функция создания и преобразования датасета;
  >>>
  Подробно механизм преобразования рассмотрен в разделе [Создание тренировочного датасета](https://github.com/dmt-zh/Deep-Learning-Transformers/tree/main/dataset)\
  ├── [dataset_fn = (...)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L241) модуль `runner.py`
  >>>

- ##### если передано значение `effective_batch_size`, то рассчитывается значение, при достижение которого будет происходит обновление градиента;
  >>>
  ├── [accum_steps = _count_batch_accum()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L281) модуль `runner.py`\
  ├── [def _count_batch_accum()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L543) модуль `runner.py`
  >>>

-  ##### инициализируются эмбеддинги для **$`\textcolor{blue}{\text{source}}`$** и для **$`\textcolor{green}{\text{target}}`$**;

  >>>
   размерность эмбеддингов будет определяться размером словаря и размером параметра `num_units`\
   ![matrix_m_n](https://github.com/user-attachments/assets/5e603122-c223-4d00-a680-dbdaa07e6972)\
   где m - количество токенов в словаре, n - значение num_units

  Эмбеддинги инициализируются через объект tf.keras.Layer, функцию [add_weight](https://www.tensorflow.org/api_docs/python/tf/keras/Layer#add_weight), которая вызывается через функцию [def build()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L497) класса WordEmbedder.

  ├── [source_inputter = inputters.WordEmbedder(embedding_size=num_units)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/transformer.py#L89C41-L89C53)\
  ├── [def build()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L484C5-L484C15)\
  ├── [target_inputter = inputters.WordEmbedder(embedding_size=num_units)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/transformer.py#L89C41-L89C53)\
  ├── [def build()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L484C5-L484C15)

   Например, для словаря размером 700 токенов и `num_units` со значением 8, будет сформирована матрица, содержащая 700 строк по 8 чисел каждая. Эмбеддинги будут выглядеть примерно так:
  >>>
```python
<tf.Variable 'custom_model_1/embedding:0' shape=(700, 8) dtype=float32, numpy=
array([[ 0.08776996, -0.0866429 , -0.04787251, ..., -0.00401733, 0.07676391, -0.07758132],
       [ 0.04529808, -0.07747891, -0.04506141, ..., -0.06822421, 0.03881324, -0.0873639 ],
       [-0.01676839,  0.05199892, -0.07963458, ...,  0.04423855, 0.09198435,  0.00941346],
       ...,
       [ 0.07298765, -0.05240389,  0.0363823 , ...,  0.02232499, -0.02742455, -0.0536664 ],
       [ 0.02921638, -0.0334079 ,  0.08479016, ..., -0.07649383, 0.05127611, -0.00408895],
       [-0.06463868, -0.0779385 , -0.02059536, ...,  0.0702227 , -0.07077053,  0.06921662]],
    dtype=float32)>
```

Значения, которые представлены в матрице эмбеддингов формируются по определенному механизму. Например, для матрицы размерностью 700х8:
```python
scale = 1
fan_in, fan_out = (700, 8) # где fan_in - размерность входа, fan_out - размерность выхода
scale /= max(1.0, (fan_in + fan_out) / 2.0) ==> max(1.0, (700 + 8 / 2.0)) = 1 / 354 = 0.0028
limit = math.sqrt(3.0 * scale) ==> math.sqrt(3.0 * 0.0028) = 0.0920
random_uniform(shape, -limit, limit) ==> random_uniform((700, 8), -0.0920, 0.0920)
```
Т.е. у нас будет сформирована матрица размерностью 700 х 8, где значения будут от -0.0920 до 0.0920, взятые рандомно из равномерного распределения. Описанный выше механизм называется "Инициализация Ксавье". Реализация в tensorflow:\
  ├── [class GlorotNormal().__init__()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/initializers/initializers_v2.py#L713)\
  ├── [class VarianceScaling().__call__()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/initializers/initializers_v2.py#L472)

-  ##### инициализируются веса (слои) модели;
  >>> 
  ├── [encoder =  SelfAttentionEncoder()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/transformer.py#L98) модуль `transformer.py`\
  ├── [self.layer_norm = common.LayerNorm()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/encoders/self_attention_encoder.py#L61C9-L61C45) модуль `self_attention_encoder.py`\
  ├── [LayerNorm()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L91) модуль `common.py`\
  ├── [SelfAttentionEncoderLayer()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/encoders/self_attention_encoder.py#L63) модуль `self_attention_encoder.py`\
  ├── [self.self_attention = MultiHeadAttention()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L205) модуль `layers/transformer.py`\
  ├── [def build()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L272) модуль `layers/transformer.py`\
  ├── [TransformerLayerWrapper(self.self_attention)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L390) модуль `layers/transformer.py`\
  ├── [self.ffn = FeedForwardNetwork()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L120) модуль `layers/transformer.py`\
  ├── [TransformerLayerWrapper(self.ffn)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L390) модуль `layers/transformer.py`


  ├── [decoder =  SelfAttentionDecoder()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/transformer.py#L122) модуль `transformer.py`\
  ├── [self.layer_norm = common.LayerNorm()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/decoders/self_attention_decoder.py#L68) модуль `self_attention_decoder.py`\
  ├── [LayerNorm()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L91) модуль `common.py`\
  ├── [SelfAttentionDecoderLayer()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/decoders/self_attention_decoder.py#L70) модуль `self_attention_decoder.py`\
  ├── [self.self_attention = MultiHeadAttention()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L501) модуль `layers/transformer.py`\
  ├── [def build()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L272) модуль `layers/transformer.py`\
  ├── [TransformerLayerWrapper(self.self_attention)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L390) модуль `layers/transformer.py`\
  ├── [attention = MultiHeadAttention()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L553) модуль `layers/transformer.py`\
  ├── [TransformerLayerWrapper(attention)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L390) модуль `layers/transformer.py`\
  ├── [self.ffn = FeedForwardNetwork()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L120) модуль `layers/transformer.py`\
  ├── [TransformerLayerWrapper(self.ffn)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L390) модуль `layers/transformer.py`

  как выглядит изначально **[модель](https://github.com/dmt-zh/Deep-Learning-Transformers/blob/main/training/model.md)**, веса и их значения показаны на примере небольшой размерности сети:\
  vocab: 26\
  num_units: 4\
  num_layers: 1\
  num_heads: 2\
  ffn_inner_dim: 1\
  maximum_relative_position: 8

  Схематично модель можно отобразить следующим образом\
![model_scheme](https://github.com/user-attachments/assets/353893a5-9f8e-47af-85bb-78f933a96af0)

**Инициализация kernel (т.е. основных) весов модели происходит с помощью описанного выше механизма "Инициализация Ксавье".** Слои `queries`, `keys`, `values` и `output` инициализируются через [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) с добавлением вектора смещения `'use_bias': True`

Веса нейросети прямого распространения (FeedForwardNetwork) также инициализируются через [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) с активацией линейного слоя [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)

Слой нормализации инициализируется через [tf.keras.layers.LayerNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization), где `beta` размерность будет представлена нулями, а `gamma` размерность - единицами.

*Нормализация слоев (Layer normalization, LN) - это метод глубокого обучения, используемый для стабилизации процесса обучения и повышения производительности нейронных сетей. Она решает проблему внутреннего ковариационного сдвига (ICS), когда распределение активаций внутри слоя меняется в процессе обучения, что затрудняет эффективное обучение сети.*
*Работа в которой было представлена техника нормализации [Layer Normalization](https://arxiv.org/pdf/1607.06450)*
  >>>

<br>

**При инициализации модели необходимо чтобы выполнялось следующее условие**: [num_units % num_heads](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L233C12-L233C33), **т.е. размерность эмбеддингов (и соответственно `queries`, `keys` и `values`) должна быть кратна количеству голов в MultiHeadAttention.**\
В действительности, матрицы `queries`, `keys` и `values` делятся на число указанное в параметре `num_heads` и формируются более маленькие матрицы, количество которых равно `num_units // num_heads`. Алгоритм реализован в функции [split_heads](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L47)  модуля `transformer.py`.
![split_heads](https://github.com/user-attachments/assets/bf2adee9-2cc1-4d61-aa3f-d03a54543ab4)


##### Рассмотрим алгоритм разбиения матриц `queries`, `keys` и `values` на количество голов.
Для простоты восприятия, размерность `num_units` возьмем равную 4 и количество голов `num_heads` равное 2. Например, у нас есть батч токенов:
```
batch = [
    ["▁she" "▁knows" "▁it"]
    ["▁he" "▁knows" "▁it"]
]
```
Для каждого токена из матрицы эмбеддингов извлекается векторное представление токена (механизм описан ниже) и формируется матрица векторов. Рассчитывается размерность батча: `input_shapes = [2, 3, 4]`. В батче 2 последовательности, в каждой последовательности 3 токена, каждый токен представлен вектором размерностью 4.
Следующий шаг - изменение размерности матрицы `inputs`:
>>>
outputs = [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape)(inputs, [shape[0], shape[1], num_heads, shape[2] // num_heads]) → tf.reshape(inputs, [2, 3, 2, 4 // 2])\
outputs = [tf.transpose](https://www.tensorflow.org/api_docs/python/tf/transpose)(outputs, perm=[0, 2, 1, 3])
>>>
После переформирования размерности, получаем матрицы меньшей размерности.
![split_heads2](https://github.com/user-attachments/assets/19e25d5d-2585-4910-9df2-bdf820ae3248)



<br>

# Тренировочный процесс

### После инициализации:
- ##### финализируется датасет, т.е. приводится в действие [весь пайплайн](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/dataset/README.md#создание-тренировочного-датасета) подготовки данных к тренировке;
     ├── [dataset = self._finalize_dataset()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L219) модуль `training.py`
<hr>

- ##### запускается цикл с количеством шагов, указанном в параметре `Train steps`
<hr>

- ##### на каждом шаге тренировочного цикла из сгруппированного в батчи тренировочного датасета извлекаются [группы](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/dataset/README.md#после-фильтрации-применяется-функция-группировки-датасета-batch_sequence_dataset-датасет-группируется-в-бакетыпартии-одинаковой-длины-чтобы-оптимизировать-эффективность-обучения-и-преобразуется-в-следующий-тип). 

   * количество групп будет равно `effective batch size` // `batch size`. Например, если у нас задан параметр `effective batch size = 200 000`, a `batch size = 6 250`, то количество групп будет равно `200 000 // 6 250 = 32`
   * в каждой из 32 групп будет 6250 токенов, таким образом суммарный объем токенов, который будет обработан за один шаг тренировки будет равен размеру `effective batch size`, т.е. 200 000 токенов.

<hr>

- ##### из группы извлекается батч, и для каждого токена в батче извлекается векторное представление токена из матрицы эмбеддингов с помощью функции [tf.nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
   ├── [call() | class WordEmbedder(TextInputter)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L513) модуль `text_inputter.py`

На примере нашей модели с размерность `vocab_size = 26` и `num_units = 4`, схематически, для `source` языка это можно представить следующим образом:
![lookup](https://github.com/user-attachments/assets/cbc57eb8-92ef-421a-9750-b197c3e3cfa2)

<hr>

Предположим, что наш батч состоит 3 обучающих элементов.\
**Source:**\
![source_batch](https://github.com/user-attachments/assets/4f845df0-520c-420d-8580-f9b2b9e9449f)

**Target:**\
![target_batch](https://github.com/user-attachments/assets/4d5c04c2-d70a-4e32-af33-a34c4b6dfefb)

<hr>


##### В ЭНКОДЕРЕ ВЕКТОРНОЕ ПРЕДСТАВЛНЕНИЕ ТОКЕНОВ `source` ЯЗЫКА:

   * умножается на квадратный корень размерности - `inputs` = inputs *  $\sqrt{num  units}$, для примера `num_units = 4`
![encoder_sqrt](https://github.com/user-attachments/assets/b5e442fd-dee6-47a1-bbae-84abcedcded3)

   * применяется  `dropout` (параметр `dropout` из конфигурационного файла), т.е. случайным образом значения заменяются на ноль, с помощью функции [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout). При этом все остальные значения (кроме замененных на ноль) корректируются умножением на `1/(1 - p)`, где `p` - вероятность дропаута. Это делается для того чтобы привести значения к одному масштабу что позволяет использовать одну и ту же сеть для обучения (при probability < 1.0) и инференса (при probability = 1.0). [Why input is scaled in tf.nn.dropout in tensorflow?](https://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow)
![encoder_dropout](https://github.com/user-attachments/assets/ad05fd84-c859-4982-b2bb-36e98f58277b)

   * по размерности батча, с помощью функции [tf.sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) строится тензор маски; для нашего примера размерность батча будет `[3 3 3]` и функция возвращает маску\
![encoder_mask](https://github.com/user-attachments/assets/92b488bc-9701-4b6b-a807-3bf63e9184fe)

   * в цикле, для каждого слоя `layer`, равное количеству параметра `Layers` векторное представление батча (которое хранится в переменной `inputs`) и тензор маски `mask` передается в слой `layer`, результат, возвращаемый `layer` передается в следующий `layer`. Например, если у нас 6 слоев, то результат из первого слоя будет входным результатом для второго слоя и т.д.\
![encoder_loop](https://github.com/user-attachments/assets/420bde81-1a6d-45ae-bb62-7df80c986469)

   * каждый слой `layer` представляет собой объект класса [SelfAttentionEncoderLayer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L205), в котором и реализован механизм внимания с помощью класса [MultiHeadAttention](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L205). Ниже описан механизм преобразований происходящих в каждом слое `SelfAttentionEncoderLayer`.
  <hr>

  - ##### слой нормализации, класс LayerNorm()
   * с параметром `Pre norm = True`, к батчу после операции `dropout` и маскирования, перед расчетом весов attention применяется слой нормализации - для каждой стоки бачта с `k` значениям мы вычисляем среднее значение и дисперсию:\
  `mean_i = sum(x_i[j] for j in range(k))/k`\
  `var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k))/k`\
   затем вычисляется нормализованное значение `x_i_normalized`, включая небольшой коэффициент эпсилон (0.001) для численной стабильности:\
   `x_i_normalized = (x_i — mean_i) / sqrt(var_i + epsilon)`\
   и, наконец, `x_i_normalized` линейно преобразуется с помощью gamma и beta, которые являются тренируемыми параметрами (при инициализации `gamma = [1, ..., 1]`, `beta = [0, ..., 0]`):\
  `output_i = x_i_normalized * gamma + beta`\
![encoder_layer_norm](https://github.com/user-attachments/assets/c8d5c9b2-2378-46a5-93ba-2f7e1000cd82)
  <hr>

   * рассчитываются значения матрицы `queries` - нормализованные значения матрицы `inputs`, полученные на предыдущем шаге проходят через линейное преобразование слоя [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense):
  <hr>
  
  - ##### линейное преобразование, класс Dense()
    ━ 1) рассчитывается размерность батча → `shape = [3, 3, 4]`\
    ━ 2) изменяется размерность батча [tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape)(inputs, [-1, shape[-1]]) → `tf.reshape(inputs, [-1, 4])`\
![dence_reshape](https://github.com/user-attachments/assets/1b0c3079-d61b-47a5-af32-535eafb41dac)
    ━ 3) **при mixed_precision и num_units кратное 8**  рассчитывается размер паддинга и его формирование\
           `padding_size = 8 - num_units % 8`  →  `padding_size = 8 - 4 % 8 = 4`\
           `paddings = [[0, 0], [0, padding_size]]`  → `paddings = [[0, 0], [0, 4]]`\
            веса слоя `kernel`(которые были сформированы при инициализации) матрицы `queries` дополняются паддингом [tf.pad(kernel, paddings)](https://www.tensorflow.org/api_docs/python/tf/pad)\
![dense_3](https://github.com/user-attachments/assets/5636c4f9-11a2-4a14-bf7d-f33fb02b8926)
    ━ 4) перемножаются матрицы `kernel` и матрица батча (`inputs`) [tf.matmul(inputs, kernel)](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul)\
![dense_4](https://github.com/user-attachments/assets/75815d5b-e9b6-4ffe-896e-7fe9f0745e2b)
    ━ 5) из образовавшейся матрицы берется срез по размерности слоя `num_units` и формируется матрица `outputs`\
![dense_5](https://github.com/user-attachments/assets/a4dcd085-a1f8-45db-a09d-07a17bebc926)
    ━ 6) к полученной на прошлом шаге матрице `outputs` добавляется вектор смещения `bias` (исходные значения `bias` инициализируются со слоем `kernel` и изначально равны нулю) [tf.nn.bias_add(outputs, bias)](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add)\
![dense_6](https://github.com/user-attachments/assets/e1e1473c-5ca7-4996-bdaa-9629942e5017)
    ━ 7) после добавления `bias` матрица `outputs` проходит активацию линейного слоя [activation(outputs)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense). Активация линейного слоя - это применение функции к матрице. Поскольку функция для слоя `tf.keras.layers.Dense` не задается, по дефолту, функция активации равна `a(x) = x`, т.е. матрица остается без изменений\
![dense_7](https://github.com/user-attachments/assets/49b56436-f50d-471d-98b0-5e5e5f328def)\
    ━ 8) после активации линейного слоя матрица `outputs` переформируется `tf.reshape(outputs, shape[:-1] + [num_units])`  → `tf.reshape(outputs, [3, 3, 4])`. После этого шага мы получаем матрицу `queries`
![dense_8](https://github.com/user-attachments/assets/17848faf-19c0-487c-8cff-6ca021eb1a1b)
  <hr>

   * полученная на прошлом шаге матрица `queries` делится на количество голов (в нашей архитектуре количество голов равно 2, механизм разделения описан выше [⬆️](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#рассмотрим-алгоритм-разбиения-матриц-queries-keys-и-values-на-количество-голов)
![split_heads](https://github.com/user-attachments/assets/f79a505c-461f-4a2a-b7bb-40f831e21056)

   * разделенная на количество голов матрица `queries` делится на квадратный корень `num_units_per_head`:\
   `num_units_per_head = num_units // num_heads`\
   `num_units_per_head = 4 // 2 = 2`\
![n_u_per_head](https://github.com/user-attachments/assets/f532fe12-cbc8-4b5a-b24c-9f68564cb5bf)

   * по описанным шагам выше 1-8, рассчитывается матрица `keys` и делиться на количество голов\
![keys](https://github.com/user-attachments/assets/485d9157-161d-48c6-a736-de3e0e748a54)

   * по описанным шагам выше 1-8, рассчитывается матрица `values` и делиться на количество голов\
![values](https://github.com/user-attachments/assets/a8ca5bfd-8cdf-497b-a83e-94b1b8fc4a5a)

   * поскольку в рассматриваемом примере используется относительное позиционное кодирование (`maximum_relative_position = 8`), то следующий шаг - **относительное кодирование**:\
    ━ 1) рассчитывается размерность матрицы `keys`\
           `keys_length = tf.shape(keys)[2]`\
           `keys_length = [3 2 3 2][2]`\
           `keys_length = 3`\
    ━ 2) формируется массив целочисленных элементов длины `keys_length`\
           `arange = tf.range(length)`\
           `arange = [0 1 2]`\
    ━ 3) формируются две матрицы по оси 0 и 1 с помощью функции [tf.expand_dims(input, axis)](https://www.tensorflow.org/api_docs/python/tf/expand_dims) и из полученных матриц рассчитывается расстояние к диагонали `distance = tf.expand_dims(arange, 0) - tf.expand_dims(arange, 1)`\
![rp_3](https://github.com/user-attachments/assets/5a4372a4-37cc-4a44-9ada-4ddf38925c65)\
    ━ 4) матрицы расстояний к диагонали `distance` обрезается по значению `maximum_relative_position` [tf.clip_by_value(distance, -maximum_position, maximum_position)](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)\
![rp_4](https://github.com/user-attachments/assets/f2407b0c-1817-46a1-b2ab-aa3b811786df)\
    ━ 5) к полученной на прошлом шаге матрице `distance` добавляется значение `maximum_relative_position`\
![rp_5](https://github.com/user-attachments/assets/90b247ba-6696-42c8-b27b-11cb4c47461e)\
    ━ 6) полученная на прошлом шаге матрица `relative_pos` используется для извлечения из матрицы `relative_position_keys`, сформированной при инициализации модели, соответствующих элементов по индексам с помощью функции `tf.nn.embedding_lookup` и формируется матрица `relative_repr_keys`\
![rp_6](https://github.com/user-attachments/assets/e82d9262-2b4e-4dbc-b38a-e29e0432d12e)\
    ━ 7) таким же образом формируется матрица `relative_repr_values` → `embedding_lookup(relative_position_values, relative_pos)`\
![rp_7](https://github.com/user-attachments/assets/3ca1bcfd-74b0-486a-b1dd-b95317942857)

   * следующий шаг - скалярное произведение матриц `queries` и `keys`  → `dot = tf.matmul(queries, keys, transpose_b=True)`\
![dot_product](https://github.com/user-attachments/assets/4be0fd6a-1343-4dad-885e-5651757c9873)

   * матрица `queries` перемножается с матрицей `relative_repr_keys`  → `matmul_with_relative_representations(queries, relative_repr_keys, transpose_b=True)`\
![queries_rpk](https://github.com/user-attachments/assets/6b6ff6d0-37f8-4745-a033-6ba75154c1da)

   * к матрице `dot` прибавляется матрица полученная на шаге `matmul_with_relative_representations`\
![sum_dot_matmul](https://github.com/user-attachments/assets/ff01fa60-268f-4a50-96bf-8c7a3b0a4524)

   * матрица `dot` преобразуется с помощью матрицы `mask`, полученной по батчу токенов:\
    ━ 1) изменяется размерность матрицы `mask`\
           `mask` = [tf.expand_dims(mask, 1)](https://www.tensorflow.org/api_docs/python/tf/expand_dims)\
    ━ 2) матрица `dot` преобразуется следующим образом\
           `dot = (dot * mask) + (1.0 - mask) * dot.dtype.min`\
![dot_mask](https://github.com/user-attachments/assets/9391c7af-cc1a-447e-89be-78a826bb4c20)

   * к матрице `dot` применяется функция активация `softmax` и получаем матрицу `attn` → `attn` = [tf.nn.softmax(dot)](https://www.tensorflow.org/api_docs/python/tf/nn/softmax). Функция `softmax`  используется для преобразования вектора значений в вероятностное распределение, которое суммируется до 1\
![attn](https://github.com/user-attachments/assets/f9148433-704b-4a3d-806d-346976d3b524)

   * к матрице `attn` применяется `dropout` (параметр `attention_dropout` из конфигурационного файла)\
![attn_drop](https://github.com/user-attachments/assets/914d9c3a-15d7-461f-85fd-fd6d475d8776)

   * после дропаута матрица `drop_attn` перемножается с матрицей `values` - образуется матрица `heads`\
![heads](https://github.com/user-attachments/assets/aab5a871-8613-45cf-9286-35b2eab589bd)

   * матрица `drop_attn` перемножается с матрицей `relative_repr_values`\
![matmul_rpv](https://github.com/user-attachments/assets/a9aae29a-887b-46e5-a9e1-e3c511e894e1)

   * к матрице `heads` прибавляется матрица полученная на шаге `matmul_with_relative_representations`\
![sum_heads_matmul](https://github.com/user-attachments/assets/e19780b8-7ef2-4000-8770-159fc5d6f1eb)

   * матрица `heads` преобразуется в размерность исходного батча через функцию объединения голов [combine_heads](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L63) - т.е. выполняем обратные операции `split_heads` - получаем матрицу `combined`\
![combine_heads](https://github.com/user-attachments/assets/7c9ea205-f5bc-4985-b8ac-96f07eae9219)

   * матрица `combined` проходит линейное преобразование, после которого получаем матрицу `outputs` → `outputs = linear_output(combined)`. Линейной преобразование проходит полностью идентично с 1-го по 8-й шаг описанный в классе Dense()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#линейное-преобразование-класс-dense)

   * к матрице `outputs` применяется  `dropout` (параметр `dropout` из конфигурационного файла)\
![outputs_drop](https://github.com/user-attachments/assets/71152b2b-43a9-4fab-b852-e1b3717c9139)

   * применяется механизм `residual connection` - к матрице `outputs` прибавляется матрица `inputs`. Residual connection - это механизм, используемый для решения проблемы исчезающего градиента в глубоких нейронных сетях и улучшения обучения и сходимости модели\
![residual](https://github.com/user-attachments/assets/f4a086a1-093c-4547-8d42-4f0be886a7be)

   * матрица `outputs` передается в нейросеть прямого распространения (Feed Forward Network):\
    ━ 1) применяется слой нормализации LayerNorm()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#слой-нормализации-класс-layernorm)\
    ━ 2) линейное преобразование класс Dense()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#линейное-преобразование-класс-dense). Линейное преобразование с функцией активации ReLU [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)\
    ━ 3) применяется  `dropout` (параметр `ffn_dropout` из конфигурационного файла)\
    ━ 4) линейное преобразование класс Dense()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#линейное-преобразование-класс-dense)\
    ━ 5) применяется  `dropout` (параметр `dropout` из конфигурационного файла)\
    ━ 6) применяется механизм `residual connection`\
![ffn](https://github.com/user-attachments/assets/7550aa01-1462-458a-8e4e-583d055bd023)

   * если значение `Layers` больше одного, то после преобразования матрицы `outputs` с помощью Feed Forward Network, полученная матрица отправляется на вход следующему слою, пока не пройдет через все слои

   * полученная матрица `outputs` **из последнего слоя**, проходит слой нормализации LayerNorm()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#слой-нормализации-класс-layernorm) - завершающая операция в энкодере. Полученная на этом шаге матрица передается в декодер\
![ffn_layer_norm](https://github.com/user-attachments/assets/3e1a5f57-918d-4746-b006-cff4f99cb17e)

  <hr>

   * вышеописанный механизм преобразования батча токенов `source` языка в энкодере можно отобразить следующим образом
![encoder](https://github.com/user-attachments/assets/9d134443-f7dc-4a0c-b500-104beae29d35)

  <hr>

   **Энкодер. Упрощенная последовательность вызова:**\
   ├── [def __call__() | class Model(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/model.py#L92C5-L92C17) модуль `model.py`\
   ├── [def call() | class SequenceToSequence(model.SequenceGenerator)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/sequence_to_sequence.py#L165) модуль `sequence_to_sequence.py`\
   ├── [def call() | class WordEmbedder(TextInputter)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L505) модуль `text_inputter.py`\
   ├── [def call() | class SelfAttentionEncoder(Encoder)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/encoders/self_attention_encoder.py#L78C5-L78C13) модуль `self_attention_encoder.py`\
   ├── [def build_mask() | class Encoder(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/encoders/encoder.py#L14) модуль `encoder.py`\
   ├── [def call() | class LayerWrapper(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L133) модуль `layers
/common.py`\
   ├── [def call() | class SelfAttentionEncoderLayer(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L488C5-L488C13) модуль `layers/transformer.py`\
   ├── [def call() | class MultiHeadAttention(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L284) модуль `layers/transformer.py`\
   ├── [def call() | class Dense(tf.keras.layers.Dense)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L59) модуль `layers
/common.py`\
   ├── [def call() | class LayerWrapper(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L133) модуль `layers
/common.py`\
   ├── [def call() | class FeedForwardNetwork(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L146) модуль `layers/transformer.py`\
   ├── [def call() | class LayerNorm(tf.keras.layers.LayerNormalization)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L91) модуль `layers/common.py`

<hr>

##### В ДЕКОДЕРЕ ВЕКТОРНОЕ ПРЕДСТАВЛНЕНИЕ ТОКЕНОВ `target` ЯЗЫКА:

   * сформированная матрица `inputs`с помощью функции `tf.nn.embedding_lookup`умножается на квадратный корень `num_units` (в нашем примере num_units=4)  → `inputs` = inputs *  $\sqrt{num  units}$
   ![decoder_1](https://github.com/user-attachments/assets/1a823916-6b3b-4206-83ab-933c153a43c6)

   * применяется  `dropout` (параметр `dropout` из конфигурационного файла)
   ![decoder_2](https://github.com/user-attachments/assets/3326414f-380f-4693-a8c1-fa09b499afee)

   * по размерности батча матрицы `inputs`, с помощью функции [future_mask](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L21) строится тензор маски `future_mask`. При обучении декодера будущие токены последовательности будут скрыты, декодер имеет доступ только к текущему токену и предыдущим
  ![decoder_3](https://github.com/user-attachments/assets/2a4835d0-29af-4dd8-bdc0-07088347b6b9)

   * по матрице, полученной в энкодере, с помощью функции [tf.sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) формируется тензор маски `memory_mask`
  ![decoder_4](https://github.com/user-attachments/assets/ebafaa91-c986-4e10-8996-8d80954ec838)


   * в цикле, для каждого слоя `layer`, равное количеству параметра `Layers` матрица `inputs`, тензор маски `future_mask`, тензор маски `memory_mask` и матрица полученная после энкодера `encoder_outputs` передаются в слой `layer`, результат, возвращаемый `layer` - `inputs` передается в следующий `layer`. Например, если у нас 6 слоев, то результат из первого слоя будет входным результатом для второго слоя и т.д. Из каждого слоя возвращается и сохраняется в список матрица `attention`, образованная в слое `cross_attention` (или `encoder-decoder attention`). В каждом слое происходят ниже описанные преобразования\
![decoder_5](https://github.com/user-attachments/assets/6c5916ac-6b4b-4000-9b5a-4ef1684a54a2)


  <hr>

   * матрица `inputs` и матрица маски `future_mask` преобразуются с помощью слоя `self attention`. Полный механизм матричных преобразований описан в разделе энкодера. Здесь же перечислим основные шаги:\
    ━ 1) матрица `inputs` проходит через слой нормализации\
    ━ 2) рассчитывается матрица `queries` и делится на количество голов\
    ━ 3) разделенная на количество голов матрица `queries` делится на квадратный корень `num_units_per_head`\
    ━ 4) рассчитывается матрица `keys` и делиться на количество голов\
    ━ 5) рассчитывается матрица `values` и делиться на количество голов\
    ━ 6) рассчитывается матрицы `relative_repr_keys` и `relative_repr_values`\
    ━ 7) скалярное произведение матриц `queries` и `keys`\
    ━ 8) матрица `queries` перемножается с матрицей `relative_repr_key`\
    ━ 9) к скалярному произведению матриц прибавляется матрица `matmul_with_relative_representations`\
    ━ 10) преобразование с помощью матрицы `future_mask`\
    ━ 11) применяется функция активация `softmax`\
    ━ 12) применяется `dropout` (параметр `attention_dropout` из конфигурационного файла)\
    ━ 13) перемножение с матрицей `values` с образованием матрицы `heads`\
    ━ 14) перемножение с матрицей `relative_repr_values`\
    ━ 15) сложение матрицы `heads` с матрицей `matmul_with_relative_representations`\
    ━ 16) объединения голов (матриц `heads`) в общую матрицу\
    ━ 17) линейное преобразование\
    ━ 18) применяется `dropout` (параметр `dropout` из конфигурационного файла)\
    ━ 19) применяется механизм `residual connection`

      После вышеперечисленных преобразований получаем матрицу `outputs`
![decoder_sa](https://github.com/user-attachments/assets/fafe4571-f913-4cef-9f72-378bb680f29e)

  <hr>

   * матрица после слоя `self attention`, матрица маски `memory_mask`, и матрица полученная в энкодере `encoder_outputs` подаются на вход слоя `cross attention`. Полный механизм матричных преобразований описан в разделе энкодера. Здесь же перечислим основные шаги:\
    ━ 1) матрица `inputs` проходит через слой нормализации\
    ━ 2) рассчитывается матрица `queries` и делится на количество голов\
    ━ 3) разделенная на количество голов матрица `queries` делится на квадратный корень `num_units_per_head`\
    ━ 4) рассчитывается матрица `keys`  **по матрице энкодера** и делиться на количество голов\
    ━ 5) рассчитывается матрица `values` **по матрице энкодера** и делиться на количество голов\
    ━ 6) рассчитывается матрицы `relative_repr_keys` и `relative_repr_values`\
    ━ 7) скалярное произведение матриц `queries` и `keys`\
    ━ 8) матрица `queries` перемножается с матрицей `relative_repr_key`\
    ━ 9) к скалярному произведению матриц прибавляется матрица `matmul_with_relative_representations`\
    ━ 10) преобразование с помощью матрицы `memory_mask`\
    ━ 11) применяется функция активация `softmax`, получаем матрицу `attention`, которая **возвращается из этого слоя**\
    ━ 12) применяется `dropout` (параметр `attention_dropout` из конфигурационного файла)\
    ━ 13) перемножение с матрицей `values` с образованием матрицы `heads`\
    ━ 14) перемножение с матрицей `relative_repr_values`\
    ━ 15) сложение матрицы `heads` с матрицей `matmul_with_relative_representations`\
    ━ 16) объединения голов (матриц `heads`) в общую матрицу\
    ━ 17) линейное преобразование\
    ━ 18) применяется `dropout` (параметр `dropout` из конфигурационного файла)\
    ━ 19) применяется механизм `residual connection`

      После вышеперечисленных преобразований получаем матрицу `outputs` и матрицу `attention`
![decoder_cra](https://github.com/user-attachments/assets/39f9e844-a4e1-49ac-9cdf-35cd1068e528)

  <hr>

   * матрица `outputs` передается в нейросеть прямого распространения (Feed Forward Network):\
    ━ 1) применяется слой нормализации LayerNorm()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#слой-нормализации-класс-layernorm)\
    ━ 2) линейное преобразование класс Dense()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#линейное-преобразование-класс-dense). Линейное преобразование с функцией активации ReLU [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)\
    ━ 3) применяется  `dropout` (параметр `ffn_dropout` из конфигурационного файла)\
    ━ 4) линейное преобразование класс Dense()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#линейное-преобразование-класс-dense)\
    ━ 5) применяется  `dropout` (параметр `dropout` из конфигурационного файла)\
    ━ 6) применяется механизм `residual connection`\
![decoder_ffn](https://github.com/user-attachments/assets/608b53ea-568a-44bd-b0a9-861fedf47a93)

  <hr>

   * после преобразования матрицы `outputs` с помощью Feed Forward Network, полученная матрица `outputs` проходит слой нормализации LayerNorm()[⬆️](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/training#слой-нормализации-класс-layernorm)\
![decoder_ffn_ln](https://github.com/user-attachments/assets/ae758d4e-cbc8-4b8a-842d-048a0bfb0535)

  <hr>
   * массив матриц `attention` из каждого слоя преобразуется с помощью заданной стратегии обработки. По дефолту определена стратегия `FIRST_HEAD_LAST_LAYER`, т.е. будет взята матрица `attention` полученная на последнем слое, и у этой матрице будет взята первая голова\
![mha_reduction](https://github.com/user-attachments/assets/2da1c812-acee-4eab-b121-09dc28f98753)

  <hr>

   * матрица `outputs` после слоя нормализации преобразуется с помощью выходного линейного слоя декодера, размерность которого равна `vocab_size` х `num_units` (в нашем примере 26 х 4) с образованием матрицы `logits`. Это преобразование в декодере - финальная операция\
![logits](https://github.com/user-attachments/assets/9cd0fd8a-b524-46be-a130-97eec354d8ee)

  <hr>

   * таким образом, после преобразований в декодере батча токенов `target` языка на выходе из декодера у нас две матрицы - матрица `logits` и матрица значений `attention` из последнего слоя `cross attention`. Схематически, полный цикл преобразований в декодере, можно представить следующим образом\
![docoder_full](https://github.com/user-attachments/assets/2cdb2f97-6811-4fb0-b5e9-f9117bba0ccd)

  <hr>

   **Decoder. Упрощенная последовательность вызова:**\
   ├── [def __call__() | class Model(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/model.py#L92C5-L92C17) модуль `model.py`\
   ├── [def call() | class SequenceToSequence(model.SequenceGenerator)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/sequence_to_sequence.py#L165) модуль `sequence_to_sequence.py`\
   ├── [def _decode_target() | class SequenceToSequence(model.SequenceGenerator)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/sequence_to_sequence.py#L219) модуль `sequence_to_sequence.py`\
   ├── [def call() | class WordEmbedder(TextInputter)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L505) модуль `text_inputter.py`\
   ├── [def call() | class Decoder(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/decoders/decoder.py#L205) модуль `decoder.py`\
   ├── [def forward() | class SelfAttentionDecoder(decoder.Decoder)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/decoders/self_attention_decoder.py#L177) модуль `self_attention_decoder.py`\
   ├── [def _run() | class SelfAttentionDecoder(decoder.Decoder)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/decoders/self_attention_decoder.py#L105C5-L105C14) модуль `self_attention_decoder.py`\
   ├── [def call() | class SelfAttentionDecoderLayer(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L574) модуль `layers/transformer.py`\
   ├── [def call() | class LayerWrapper(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L133) модуль `layers
/common.py`\
   ├── [def call() | class MultiHeadAttention(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L284) модуль `layers/transformer.py`\
   ├── [def call() | class Dense(tf.keras.layers.Dense)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L59) модуль `layers
/common.py`\
   ├── [def call() | class LayerWrapper(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L133) модуль `layers
/common.py`\
   ├── [def call() | class MultiHeadAttention(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L284) модуль `layers/transformer.py`\
   ├── [def call() | class Dense(tf.keras.layers.Dense)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L59) модуль `layers
/common.py`\
   ├── [def call() | class LayerWrapper(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L133) модуль `layers
/common.py`\
   ├── [def call() | class FeedForwardNetwork(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L146) модуль `layers/transformer.py`\
   ├── [def call() | class LayerNorm(tf.keras.layers.LayerNormalization)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L91) модуль `layers/common.py`\
   ├── [def reduce() | class MultiHeadAttentionReduction](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L179) модуль `layers/transformer.py`\
   ├── [def call() | class Dense(tf.keras.layers.Dense)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/common.py#L59) модуль `layers
/common.py`

<hr>

- ##### РАССЧЕТ ФУНКЦИИ ПОТЕРЬ:

- матрица `logits` полученная после преобразований в декодере и батч `target` языка передаются в функцию [cross_entropy_sequence_loss](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/utils/losses.py#L28). Внутри функции происходят следующие преобразования:\
![cross_entropy_loss](https://github.com/user-attachments/assets/ef7a366f-c7ec-4a21-9954-cc187a54ea99)

   * рассчитывается матрица `cross_entropy` с помощью функции [_softmax_cross_entropy](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/utils/losses.py#L18C5-L18C27)\
    ━ 1) значения матрицы `logits` c помощью функции [tf.cast](https://www.tensorflow.org/api_docs/python/tf/cast) приводится к типу float32 → `logits = tf.cast(logits, tf.float32)`\
    ━ 2) рассчитывается значение переменной `num_classes` по размерности матрицы `logits` → `num_classes = logits.shape[-1]`; поскольку размерность матриц `logits` равна [3, 4, 26], то значение переменной будет равно 26\
    ━ 3) рассчитывается значение переменной `on_value` → `1.0 - label_smoothing` (параметр `label_smoothing` берется из тренировочного конфига); значение переменной будет равно 0,9\
    ━ 4) рассчитывается значение переменной `off_value` → `label_smoothing / (num_classes - 1)`; 1/(26 - 1) = 0,004\
    ━ 5) с помощью функции [tf.one_hot](https://www.tensorflow.org/api_docs/python/tf/one_hot) рассчитывается матрица `smoothed_labels` → `tf.one_hot(labels, 26, 0.9, 0.004)`; это преобразование работает следующим образом - из батча токенов `target` языка извлекаются индексы выходящих токенов `ids_out`, строится матрица глубиной 26 элементов и если индекс элемента матрицы совпадает с индексом из матрицы `ids_out`, то такой индекс заполняется значением из `on_value` все остальные элементы заполняются значением из `off_value`  \
![smothed_labels](https://github.com/user-attachments/assets/369eb9b8-5fa0-4226-8738-6c15ad009b17)\
    ━ 6) с помощью функции [tf.nn.softmax_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits) вычисляется softmax кросс-энтропия между матрицами `smoothed_labels` и `logits`. Кросс-энтропия измеряет расхождение между двумя вероятностными распределениями. Алгоритм этой функции следующий:\
            - вычисляется экспонента матрицы `logits` - эквивалентно numpy.exp(logits)\
            - суммируются элементы матрицы построчно - эквивалентно numpy.sum(numpy.exp(logits), axis=-1)\
            - берется десятичный логарифм полученной матрицы и транспонируется по размерности матрицы 
 `logits` (в нашем примере 3 х 4 х 1) эквивалентно numpy.log(numpy.sum(numpy.exp(logits), axis=-1)).reshape(3, 4, 1)\
            - из матрицы `logits` вычитается матрица полученная на прошлом шаге с образованием матрицы `logsoftmax`  →  `logsoftmax = logits - numpy.log(numpy.sum(numpy.exp(logits), axis=-1)).reshape(3, 4, 1)`\
            - считается матрица `cross_entropy` - матрица `logsoftmax` умножается на отрицательную матрицу `logits` и произведение суммируется построчно  →  `cross_entropy = numpy.sum(logsoftmax * -labels, axis=-1)`\
![cross_entropy](https://github.com/user-attachments/assets/6f0a0d03-ad69-460e-aa9d-161c0af9b498)

   * с помощью функции [tf.sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) рассчитывается матрица весов `weight` по переменной `sequence_length` (в этой переменной находятся значения длин предложений в токенах сгруппированных в батч) и размерности матрицы `logits.shape[1]`  [3, **4**, 26]\
![weight](https://github.com/user-attachments/assets/769b53a2-8936-4353-a767-abd52ffa8aad)

   * с помощью функции [tf.math.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum) по произведению матриц `cross_entropy` и `weight` рассчитывается переменная `loss`  →  `loss = tf.reduce_sum(cross_entropy * weight) = 39.6399841`\
![loss](https://github.com/user-attachments/assets/8326120f-972e-4030-930d-e7dd6587fdb1)

   * с помощью функции [tf.math.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum) по матрицe `weight` рассчитывается переменная `loss_token_normalizer`, которая будет равна количеству токенов в батче  →  `loss_token_normalizer = tf.reduce_sum(weight) = 12`

   * в результате возвращаются две переменные `loss = 39.6399841` и `loss_token_normalizer = 12`

   **Упрощенная последовательность вызова**:\
   ├── [def _accumulate_gradients(self, batch)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L287) модуль `training.py`\
   ├── [def compute_gradients(features, labels, optimizer) class Model(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/model.py#L223) модуль `model.py`\
   ├── [def compute_training_loss(features, labels) class Model(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/model.py#L274) модуль `model.py`\
   ├── [def compute_loss(outputs, labels) class SequenceToSequence(model.SequenceGenerator)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/sequence_to_sequence.py#L424) модуль `sequence_to_sequence.py`\
   ├── [def cross_entropy_sequence_loss()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/utils/losses.py#L28C1-L28C32) модуль `utils/losses.py`

<hr>

- ##### МЕХАНИЗМ АЛАЙНМЕНТА:

- при тренировке модели с алайнментом, значение в переменной `loss`, полученное на прошлом шаге корректируется с помощью функции [guided_alignment_cost](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/utils/losses.py#L126C5-L126C26). Внутри функции происходят следующие преобразования:

   * в зависимости от установленного в конфигурационном файле параметра `Guided alignment type` определяется функция преобразования:\
    ━ для значения `ce` - [tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) - дефолтное значение\
    ━ для значения `mse` - [tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError)

   * по батчу токенов `target` языка рассчитывается длина предложений в токенах:\
![labels_len](https://github.com/user-attachments/assets/6127fe83-5fd7-4a2a-8208-106fa896706e)

   * с помощью функции [tf.sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) по полученным длинам и размерности матрицы `attention` tf.shape(attention)[1] строится тензор весов `sample_weight`; для нашего примера длина предложений в токенах будет [3 3 3] и размерность матрицы `attention` [3 3 3]\
![sample_weight](https://github.com/user-attachments/assets/a0e27803-b0c6-47a5-9f46-1f4365e4f5c3)

   * с помощью функции [tf.expand_dims(input, axis)](https://www.tensorflow.org/api_docs/python/tf/expand_dims) матрица `sample_weight` изменяется по размерности  → `sample_weight = tf.expand_dims(sample_weight, -1)`\
![sample_weight_expand](https://github.com/user-attachments/assets/2dc6f717-c115-44a7-97fa-99f501f30675)

   * с помощью функции [tf.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/math/reduce_sums) по массиву длин предложений батча рассчитывается значение `normalizer` → `normalizer = tf.reduce_sum([3 3 3]) = 9`

   * с помощью функции `tf.keras.losses.CategoricalCrossentropy(alignment, attention)` по матрице алайнмента ([механизм формирования матрицы](https://github.com/dmt-zh/Transformers-Full-Review/tree/main/dataset#механизм-формирования-матрицы-алайнмента)) матрице `attention` (в матрице `attention` в каждом векторном представлении токена предварительно удален последний элемент `attention[:, :-1]`, что бы размерности матриц совпадали, т.к. размерность исходной матрицы `attention` 3 x 4 x 3) и матрице `sample_weight` рассчитывается значение переменной `cost`
![caterorical_cross_entropy](https://github.com/user-attachments/assets/14fc5da1-395f-46f4-b7dc-db1940f85627)

   * переменная `cost` делится на переменную `normalizer` → `cost = cost / normalizer = 9.00836372 / 9 = 1.00092936`

   * переменная `cost` умножается на значение переменной `weight` (параметр из конфигурационного файла `Guided alignment weight`) →  `cost = cost * weight = 1.00092936 * 1 = 1.00092936`

   * значение из переменной `loss`, полученное в функции `cross_entropy_sequence_loss` корректируется значением переменной `cost` путем сложения `loss = loss + cost = 39.6399841 + 1.00092936 = 40.6409149`

   **Упрощенная последовательность вызова**:\
   ├── [def _accumulate_gradients(self, batch)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L287) модуль `training.py`\
   ├── [def compute_gradients(features, labels, optimizer) class Model(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/model.py#L223) модуль `model.py`\
   ├── [def compute_training_loss(features, labels) class Model(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/model.py#L274) модуль `model.py`\
   ├── [def compute_loss(outputs, labels) class SequenceToSequence(model.SequenceGenerator)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/sequence_to_sequence.py#L424) модуль `sequence_to_sequence.py`\
   ├── [def guided_alignment_cost()](https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/utils/losses.py#L126) модуль `utils/losses.py`

- Таким образом, итеративно, в процессе обучения модели с алайнментом мы корректируем значение лосс функции (увеличивая ее значение) и таким образом будем "принуждать" оптимизатор минимизировать функцию потерь с учетом влияния алайнмента. На картинке ниже показаны распределения вероятностей матриц `attention`   токенов `target` языка к токенам `source` полностью обученных моделей без алайнмента и с алайнментом. По распределениям видно, что по вероятностям матрицы `attention` модели с алайнментом токены `[▁П, ровер, ьте]` можно корректно сопоставить с токеном `▁Check`, а по матрице `attention` без алайнмента нельзя.\
![alignment](https://github.com/user-attachments/assets/6352826b-c3e9-419a-be02-5eb03ff6839c)

<hr>

- ##### МЕХАНИЗМ РАСЧЕТА И ПРИМЕНЕНИЯ ГРАДИЕНТОВ:

   * полученное значение `loss` корректируется на значение `loss_scale` (изначальное это значение равно [32 768](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer)) класса оптимизатора `LazyAdam`:\
    `scaled_loss = optimizer.get_scaled_loss(loss)`  → `scaled_loss = 40.640914 * 32 768 = 1 331 721.5`

   * по полученному выше значению `scaled_loss` и весам модели [trainable_weights](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/model.md), с помощью функции `gradient` класса [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) рассчитываются градиенты. Градиенты - это производные по весам модели. Расчет градиентов достаточно сложен и основан на алгоритме обратного распространения ошибки (backpropagation).
   
   * Суть этого метода заключается в том, что на основании полученного значения `scaled_loss` и матрицы весов модели `trainable_weights` расcчитвываются производные весов модели слева направо, по всему графу вычисления. Т.е. мы берем значение лосс функции `scaled_loss` и находим производные для значений полученных на выходе из декора, потом для значений полученных в энкодере, и так до момента исходных значений модели. Цель - найти значение производных таким образом, чтобы минимизировать функцию потерь. Концептуально, желаемая схема обучения нейросети выглядит примерно так: функция потерь принимает минимальное значение → находим соответствующие этому значению веса → ошибка минимальна → предсказание нейросети точное.

   * Визуально схема расчета вектора градиентов можно отобразить следующим образом:\
![backprop](https://github.com/user-attachments/assets/ff9c5aaf-e012-47c1-b148-7a1e16b6fd58)

   * Визуализация механизма метода обратного распространения (на простом примере):\
![image](https://habrastorage.org/files/627/6e1/d36/6276e1d365ba4f8497cd41fb110d7619.gif)

   * при тренировке со смешанной точностью (FP16), с помощью функции оптимизатора `optimizer.get_unscaled_gradients(gradients)` градиенты делятся на значение `loss_scale`; ниже представлен срез весов модели и ее градиентов (начало и конец матрицы)\
![scale_grads](https://github.com/user-attachments/assets/fcde665e-916d-4290-a3d5-b902df0e2275)

   * таким образом, получаем матрицу градиентов, размер этой матрицы будет равен размеру матрицы весов модели, т.е. если у нас в модели 1 млн. параметров, то и матрица градиентов будет содержать 1 млн. значений

   * после расчета матрицы градиентов для всех групп (батчей) **градиенты аккумулируются**. Например, при `effective batch size = 200 000` и `batch size = 6 250`, количество групп будет равно `32`, т.е. после расчета градиентов, у нас будет 32 матрицы градиентов - для каждого батча, своя матрица градиентов. Аккумулирование градиентов происходит путем сложения матриц друг с другом. Помимо аккумулирования матриц градиентов, так же **аккумулируются путем сложения значения полученные в лосс функции** `loss` и `loss_token_normalizer` (общее количество `target` токенов в батче) с формирование переменных:
    ━ `loss = all_reduce_sum(loss)`\
    ━ `sample_size = all_reduce_sum(loss_token_normalizer)`

   * после того как будут суммированы матрицы градиентов, `loss` и `loss_token_normalizer` значение градиентов делиться на общее количество токенов в группе `sample_size`. Для примера, ниже представлен механизм для трех батчей с общим количеством токенов 146\
![acc_grads](https://github.com/user-attachments/assets/257d19db-ed62-4533-8120-3fa4aa3af74d)

   * с помощью функции [apply_gradients](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer#apply_gradients) класса оптимизатора применяются градиенты к весам модели, т.е. происходит обновление весов модели. Реализация механизма обновления весов оптимизатора [Adam](https://github.com/keras-team/keras/blob/v3.3.3/keras/src/optimizers/adam.py#L115). Алгоритм обновления весов на примере одного значения из нашего примера:\
    ━ `momentums` и `velocities` - изначально инициализируются нулями. Они содержат значения импульсов для каждого веса модели и с каждым шагом будут корректироваться и обновляться\
    ━ `alpha` - адаптивное значение параметра `learning rate`\
![apply_grads](https://github.com/user-attachments/assets/7ffb102a-02d2-4628-b9f3-9edd4c2cdfe3)

   * при использовании другого типа оптимизатора механизм расчета и применения градиентов будет отличаться.

   **Упрощенная последовательность вызова**:\
   ├── [def __call__() class Trainer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L57) модуль `training.py`\
   ├── [def _steps() class Trainer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L209C4-L209C45) модуль `training.py`\
   ├── [def _accumulate_gradients(batch) class Trainer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L287) модуль `training.py`\
   ├── [def compute_gradients() class Model(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/models/model.py#L223) модуль `model.py`\
   ├── [def _accumulate_loss() class Trainer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L264) модуль `training.py`\
   ├── [def __call__() class GradientAccumulator](https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py#L116) модуль `optimizers/utils.py`\
   ├── [def _apply_gradientss() class Trainer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L300) модуль `training.py`

<hr>

- ##### после применения градиентов, значение лосс функции делится на общее количество токенов, `loss = float(loss) / float(sample_size)` → `40.6409149 / 12 = 3.38674291`.
    Именно это значение и будет отображено у нас в лога тренировки: Step = 1 ; Loss = 3.386743. И по этому значению будет строиться график функции потерь тренировки, отображаемый в Tensorboard

<hr>

- ##### ПРИМЕНЕНИЕ ЭКСПОНЕНЦИАЛЬНОГО СКОЛЬЗЯЩЕГО СРЕДНЕГО

   * в дефолтном конфигурационном файле модели Transformer параметр `moving_average_decay` не задан

   * задав параметр `moving_average_decay` значением близким к единице (согласно [документации tensorflow](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)), следующий шаг - расчет экспоненциального скользящего среднего для весов модели. Согласно документации, применение `moving_average_decay` для весов модели может существенно улучшить результаты модели

   * алгоритм `moving_average_decay` следующий:\
    ━ на каждом шаге тренировки, после расчета и применения градиентов, инициализируется класс [MovingAverage](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L435C7-L435C20)\
    ━ после инициализации MovingAverage вызывается функция обновления весов модели\
    ━ веса модели обновляются следующим образом:\
             - вычисляется коэффициент затухания `decay`: `decay = 1 - min(0.9999, (1.0 + training_step) / (10.0 + training_step))`\
             - для каждого веса модели применяется следующий алгоритм: `shadow_weigth = previous_weight - (previous_weight - current_weight) * decay` (на первом шаге тренировки previous_weight = current_weight)\
    ━ сглаженные веса после каждого шага тренировки хранятся в классе `MovingAverage`, **{- замена натренированных весов сглаженными весами происходит только при сохранении чекпоинта -}** модели\
![exponentional_mva](https://github.com/user-attachments/assets/89fb2e44-0509-4d25-972d-ee4affd813e0)

   **Упрощенная последовательность вызова**:\
   ├── [def __call__() class Trainer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L57) модуль `training.py`\
   ├── [def __init__() class MovingAverage](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L438) модуль `training.py`\
   ├── [def _update_moving_average() class Trainer](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L316) модуль `training.py`\
   ├── [def update() class MovingAverage](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/training.py#L464) модуль `training.py`

<hr>

- ##### MEХАНИЗМ ЗАТУХАНИЯ КОЭФФИЦИЕНТА СКОРОСТИ ОБУЧЕНИЯ

   * в механизме затухания коэффициента скорости обучения (`learning rate`) используется переменные проинициализированные в классах `NoamDecay` и `ScheduleWrapper`

   * после каждого шага тренировки, в классе `ScheduleWrapper` происходят следующие преобразования:\
    ━ с помощью функции [tf.maximum](https://www.tensorflow.org/api_docs/python/tf/math/maximum) рассчитывается переменная `step` → `tf.maximum(step - step_start, 0) = 1`\
    ━ переменная `step` корректируется на значение `step_duration` путем целочисленного деления → `step //= step_duration = 1 // 1 = 1`\
    ━ скорректированная на прошлом шаге переменная `step` передается в класс `NoamDecay`

   * в классе `NoamDecay` происходят следующие преобразования:\
    ━ рассчитывается переменная `step` → `step = step + 1 = 2`\
    ━ промежуточное значение `a`: с помощью функции [tf.pow](https://www.tensorflow.org/api_docs/python/tf/math/pow) значение `model_dim` возводится в степень `-0.5`, что эквивалентно единица деленная на корень квадратный `model_dim`  → `1 / sqrt(4) = 0.5`\
    ━ промежуточное значение `b`: с помощью функции `tf.pow` значение `step` полученное выше возводится в степень `-0.5`, что эквивалентно единица деленная на корень квадратный `step`  → `1 / sqrt(2) = 0.7071`\
    ━ промежуточное значение `с`: с помощью функции `tf.pow` значение `warmup_steps` возводится в степень `-1.5` и умножается на значение `step`  → `(1 / 8000^1.5) * 2 = 0.000001397 * 2 = 0.000002795`\
    ━ с помощью функции [tf.minimum](https://www.tensorflow.org/api_docs/python/tf/math/minimum) определяется минимальное значение из двух промежуточных значений b и с  → `min(b, c) → 0.000002795`\
    ━ полученное минимальное значение умножается на промежуточное значение `a` и `scale` → `0.0000027951 * 0.5 * 2 = 0.000002795`\
    ━ полный цикл промежуточных преобразований выглядит следующим образом `(scale * tf.pow(model_dim, -0.5) * tf.minimum(tf.pow(step, -0.5), step * tf.pow(warmup_steps, -1.5)))`\
    ━ полученное выше значение `0.000002795` возвращается обратно в класс `ScheduleWrapper`

   * в классе `ScheduleWrapper` определяется финальное значение коэффициента `learning rate = tf.maximum(learning_rate, minimum_learning_rate)` →  `learning rate = max(0.000002795, 0.0001) = 0.000002795 = 0.0001`. Именно это значение и выводится в лог тренировки: ` Step = 1 ; Learning rate = 0.000100 ; Loss = 3.386743`

   * по описанному выше алгоритму построим график изменений значения `learning rate` оптимизатора для модели размерностью 768 и заданным параметром `Learning rate = 2` в конфигурационном файле тренировщика
![lr_1](https://github.com/user-attachments/assets/f74461f0-59d7-47d8-8a60-ed80cf8cf2e3)

   * а теперь построим график изменений значения `learning rate` оптимизатора для модели размерностью 768 и заданным параметром `Learning rate = 6` в конфигурационном файле тренировщика
![lr_2](https://github.com/user-attachments/assets/d1e02cde-e85a-47cf-9860-d57456c85081)

   * по графикам можно сделать вывод, что с понижением значения `warmup_steps` стремительно увеличивается значение `learning rate` оптимизатора, при этом с увеличением `Learning rate` в конфиге позволяет достичь более высоких значений `learning rate` оптимизатора что может способствовать более быстрому обучения при большой размерности модели.

   * влиять на изменение значения `learning rate` можно так же с помощью параметра `start_decay_steps`, т.е. можем указать через сколько шагов после начала обучения применятся механизм `warmup_steps` и последующее затухание. На графике ниже видно, что при `start_decay_steps = 10 000`, первые 10 тыс. шагов модель обучается при фиксированном значении `learning rate`, которое равно минимуму, а после 10 тыс. шагов начинает работать механизм `warmup_steps` с затуханием
![lr_3](https://github.com/user-attachments/assets/bc5d4bdb-ad2e-42d9-a08f-34347d96c921)

   * с помощью параметра `decay_step_duration` можно увеличить длительность действия механизма `warmup_steps` и замедлить скорость затухания
![lr_4](https://github.com/user-attachments/assets/502f4a39-eb92-4441-a61b-5cce82dd1f29)

   **Упрощенная последовательность вызова**:\
   ├── [def __call__() class ScheduleWrapper](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/schedules/lr_schedules.py#L107) модуль `schedules/lr_schedules.py`\
   ├── [def __call__() class class NoamDecay](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/schedules/lr_schedules.py#L131) модуль `schedules/lr_schedules.py`\
   ├── [def __call__() class ScheduleWrapper](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/schedules/lr_schedules.py#L107) модуль `schedules/lr_schedules.py`

<hr>

- ##### MEХАНИЗМ УСРЕДНЕНИЯ ЧЕКПОЙНТОВ

   * чекпойнт (checkpoint) -  это состояние модели на определенный шаг тренировки. Чекпойнт натренированной модели хранит измененные в процессе тренировки веса модели, переменные оптимизатора по каждому слою (состоянием оптимизатора на определенный шаг тренировки), а также граф вычисления. Что представляет собой чекпойнт на примере рассматриваемой модели можно посмотреть [здесь](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/checkpoint.md) (чекпойнт сохранен на 10-м шаге тренировки). Пример небольшого графа вычисления для простой сети представлен на скрине ниже\
![graph](https://github.com/user-attachments/assets/0aadeb3c-a359-48a5-97fc-84fee2104bbf)\
Оптимизатор выделен красным, обычные переменные - синим, а переменные слота оптимизатора - оранжевым. Другие узлы, выделены черным цветом. Переменные слота являются частью состояния оптимизатора, но создаются для конкретной переменной. Например, ребра 'm' выше соответствуют импульсу, который оптимизатор Адама отслеживает для каждой переменной.

   * в конце тренировки, из директории модели считываются и восстанавливаются **последние** чекпойнты модели в количестве равном параметру `average_last_checkpoints`

   * по архитектуре натренированной модели инициализируются веса со значением ноль для всех слоев модели

   * далее в цикле, для каждого восстановленного чекпойнта считываются веса; веса каждого слоя делятся на количество чекпойнтов указанное в параметре `average_last_checkpoints` и полученные значения с помощью функции [variable.assign_add(value / num_checkpoints)](https://www.tensorflow.org/api_docs/python/tf/Variable#assign_add) прибавляются к проинициализированным выше весам (слой `embeddings` суммируется только со слоем `embeddings` и т.д.)

   * механизм усреднения небольшого слоя из модели нашего примера, с усреднением двух последних чекпойнтов показан ниже\
![avg_chkpt](https://github.com/user-attachments/assets/af56442e-8b10-4fae-98ce-ca2a9e2346eb)

