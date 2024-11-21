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

   * из [автоконфига](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/models/transformer.py#L157) модели трансформер берется наименования оптимизатора, который по дефолту задан `LazyAdam`. Этот оптимизатор реализован в библиотеке дополнений [TensorFlow Addons](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/LazyAdam), **{- развитие и поддержка которой остановлена -}**.

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
  Если активирована функция `Mixed precision` или `Jit compile`, то `batch_size_multiple` будет равен {+ 8 +}, иначе 1.\
  ├── [batch_size_multiple = (...)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/runner.py#L228) модуль `runner.py`
  >>>

- ##### создается функция создания и преобразования датасета;
  >>>
  Подробно механизм преобразования рассмотрен в разделе [Создание тренировочного датасета](https://git.nordicwise.com/infra/machine-translate-utils/-/wikis/Создание-тренировочного-датасета)\
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

  как выглядит изначально **[модель]()
**, веса и их значения показаны на примере небольшой размерности сети:\
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

**{- При инициализации модели необходимо чтобы выполнялось следующее условие-}**: [num_units % num_heads](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/layers/transformer.py#L233C12-L233C33), **т.е. размерность эмбеддингов (и соответственно `queries`, `keys` и `values`) должна быть кратна количеству голов в MultiHeadAttention.**\
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

