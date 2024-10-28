# Создание тренировочного датасета

Механизм создания и преобразования тренировочного датасета рассмотрим на следующем примере, где:\
$`- \textcolor{blue}{\text{Features file}}`$ - токенизированный файл, содержащий строки исходного языка\
$`- \textcolor{green}{\text{Labels file}}`$ - токенизированный файл, содержащий строки целевого языка

| Features file | Labels file |
|---------------|-------------|
| ▁a | ▁a |
| ▁ 2 1 | ▁ 2 1 |
| ▁yes | ▁да |
| ▁no | ▁нет |
| ▁cat | ▁кот |
| ▁he ▁can | ▁он ▁может |
| ▁door ▁open | ▁дверь ▁открыта |
| ▁I ▁like ▁it | ▁Мне ▁нравится ▁это |
| ▁bat ▁in ▁bad | ▁лету чая ▁мы шь ▁в ▁крова ти |
| ▁good ▁day ▁today | ▁хороший ▁день ▁сегодня |

<br>

# Механизм формирования тренировочного датасета

<br>

Каждый отдельный файл передается в класс библиотеки tensorfolow [TextLineDataset](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset).

```python
tf.data.TextLineDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None,
    name=None
)
```

Каждая строка переданного txt файла преобразуется в байтовое представление и возвращается объект `TextLineDatasetV2 element_spec=TensorSpec(shape=(), dtype=tf.string, name=None)`.

| Features TextLineDatasetV2 | Labels TextLineDatasetV2 |
|----------------------------|--------------------------|
| b'\\xe2\\x96\\x81a' | b'\\xe2\\x96\\x81a' |
| b'\\xe2\\x96\\x81 2 1' | b'\\xe2\\x96\\x81 2 1' |
| b'\\xe2\\x96\\x81yes' | b'\\xe2\\x96\\x81\\xd0\\xb4\\xd0\\xb0' |
| b'\\xe2\\x96\\x81no' | b'\\xe2\\x96\\x81\\xd0\\xbd\\xd0\\xb5\\xd1\\x82' |
| b'\\xe2\\x96\\x81cat' | b'\\xe2\\x96\\x81\\xd0\\xba\\xd0\\xbe\\xd1\\x82' |
| b'\\xe2\\x96\\x81he \\xe2\\x96\\x81can' | b'\\xe2\\x96\\x81\\xd0\\xbe\\xd0\\xbd \\xe2\\x96\\x81\\xd0\\xbc\\xd0\\xbe\\xd0\\xb6\\xd0\\xb5\\xd1\\x82' |
| b'\\xe2\\x96\\x81door \\xe2\\x96\\x81open' | b'\\xe2\\x96\\x81\\xd0\\xb4\\xd0\\xb2\\xd0\\xb5\\xd1\\x80\\xd1\\x8c \\xe2\\x96\\x81\\xd0\\xbe\\xd1\\x82\\xd0\\xba\\xd1\\x80\\xd1\\x8b\\xd1\\x82\\xd0\\xb0' |
| b'\\xe2\\x96\\x81I \\xe2\\x96\\x81like \\xe2\\x96\\x81it' | b'\\xe2\\x96\\x81\\xd0\\x9c\\xd0\\xbd\\xd0\\xb5 \\xe2\\x96\\x81\\xd0\\xbd\\xd1\\x80\\xd0\\xb0\\xd0\\xb2\\xd0\\xb8\\xd1\\x82\\xd1\\x81\\xd1\\x8f \\xe2\\x96\\x81\\xd1\\x8d\\xd1\\x82\\xd0\\xbe' |
| b'\\xe2\\x96\\x81bat \\xe2\\x96\\x81in \\xe2\\x96\\x81bad' | b'\\xe2\\x96\\x81\\xd0\\xbb\\xd0\\xb5\\xd1\\x82\\xd1\\x83 \\xd1\\x87\\xd0\\xb0\\xd1\\x8f \\xe2\\x96\\x81\\xd0\\xbc\\xd1\\x8b \\xd1\\x88\\xd1\\x8c \\xe2\\x96\\x81\\xd0\\xb2 \\xe2\\x96\\x81\\xd0\\xba\\xd1\\x80\\xd0\\xbe\\xd0\\xb2\\xd0\\xb0 \\xd1\\x82\\xd0\\xb8' |
| b'\\xe2\\x96\\x81watch' | b'\\xe2\\x96\\x81\\xd1\\x87\\xd0\\xb0\\xd1\\x81\\xd1\\x8b' |
| b'\\xe2\\x96\\x81good \\xe2\\x96\\x81day \\xe2\\x96\\x81today' | b'\\xe2\\x96\\x81\\xd1\\x85\\xd0\\xbe\\xd1\\x80\\xd0\\xbe\\xd1\\x88\\xd0\\xb8\\xd0\\xb9 \\xe2\\x96\\x81\\xd0\\xb4\\xd0\\xb5\\xd0\\xbd\\xd1\\x8c \\xe2\\x96\\x81\\xd1\\x81\\xd0\\xb5\\xd0\\xb3\\xd0\\xbe\\xd0\\xb4\\xd0\\xbd\\xd1\\x8f' |

Преобразованные по отдельности txt файлы в формате `TextLineDatasetV2` собираются в параллельный датасет с помощью метода zip класса [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#zip).

```python
parallel_datasets = [
    tf.data.Dataset.zip(tuple(parallel_dataset))
    for parallel_dataset in zip(*datasets)
]
```
Bозвращается объект `ZipDataset element_spec=(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.string, name=None))`. Каждый элемент данного объекта представлен в виде кортежа (src_line, trg_line).

```
(b'\xe2\x96\x81a', b'\xe2\x96\x81a')
(b'\xe2\x96\x81 2 1', b'\xe2\x96\x81 2 1')
(b'\xe2\x96\x81yes', b'\xe2\x96\x81\xd0\xb4\xd0\xb0')
(b'\xe2\x96\x81no', b'\xe2\x96\x81\xd0\xbd\xd0\xb5\xd1\x82')
...
```

**Упрощенная последовательность вызова при формировании тренировочного датасета**:\
├── [dataset_fn()](https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/runner.py#L241C9-L241C19) модуль `runner.py`\
├── [\_finalize_dataset()](https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/training.py#L195C9-L195C26) модуль `training.py`\
├── [make_training_dataset()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L694) модуль `inputter.py`\
├── [make_dataset()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L370) модуль `inputter.py`\
├── [make_datasets()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L291) модуль `text_inputter.py`\
├── [make_datasets()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L9) модуль `dataset.py`

----
<br>

# Механизм преобразования тренировочного датасета

<br>

Перед началом тренировочного процесса, созданный датасет будет преобразован в соответствии с заданными параметрами, переданными через конфигурационный файл.

При этом, все преобразования будут осуществляться на объектах Tensor, с помощью механизмов tensorflow на объектах графа. Это значит, что к тензорам последовательно добавляются функции, которые необходимо будет применить в будущем, т.е. строиться граф последовательного вызова функций преобразования. В единый момент времени, у нас есть только тензоры, которые могут ничего не содержать (никаких структур данных). Например, параллельный датасет представляет собой следующий объект:
```ruby
ZipDataset element_spec=(
    TensorSpec(shape=(), dtype=tf.string, name=None),
    TensorSpec(shape=(), dtype=tf.string, name=None)
)
```
По элементам `ZipDataset` можно проитерироваться:
```ruby
for elem in dataset.as_numpy_iterator():
    print(elem) ==> (b'\xe2\x96\x81a', b'\xe2\x96\x81a')
```

<br>

## Рассмотрим функции преобразования и последовательность их вызова.

После создания датасета у нас есть датасет, в виде объекта tensorflow `ZipDataset`. Напомню, что каждый элемент этого датасета содержит кортеж вида `(b'\xe2\x96\x81a', b'\xe2\x96\x81a')`, где первый элемент это байтовое представление токенов исходного языка, а второй элемент, это байтовое представление токенов целевого языка.

Преобразование тренировочного датасета осуществляется с помощью функции `make_training_dataset` класса `ExampleInputterAdapter`: [def make_training_dataset()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L694)

Cоздается функция [filter_fn](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L777) для последующий фильтрации датасета. Основная цель этой функции - удаление из тренировочного корпуса таких строк, длина в токенах которых будет больше указанных значений в параметрах `maximum_features_length` и `maximum_labels_length`.

**Упрощенная последовательность вызова при формировании функции фильтрации датасета**:\
├── [def keep_for_training() | class Inputter(tf.keras.layers.Layer)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L227)  модуль `inputter.py`\
├── [def keep_for_training() | class ParallelInputter(MultiInputter)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L512) модуль `inputter.py`

Упрощенно, механизм применения следующий:\
├── is_valid = [tf.greater](https://www.tensorflow.org/api_docs/python/tf/math/greater)(length, 0) где length - длина предложения в токенах\
├── is_valid = [tf.logical_and](https://www.tensorflow.org/api_docs/python/tf/math/logical_and)(is_valid, [tf.less_equal](https://www.tensorflow.org/api_docs/python/tf/math/less_equal)(length, maximum_length))\
├── constraints = [is_valid_src, is_valid_trg]\
├── [tf.reduce_all](https://www.tensorflow.org/api_docs/python/tf/math/reduce_all)(constraints)

Функция `tf.reduce_all` возвращает объект тензора `Tensor("All:0", shape=(), dtype=bool)`


> $`\textcolor{#134925}{\text{Пример}}`$ Пусть у нас длина для `source` предложения будет 250 токенов, `maximum_features_length = 512`. Длина для `target` предложения будет 280 токенов, `maximum_labels_length = 512`

```ruby
is_valid_src = tf.greater(250, 0)
is_valid_src = tf.logical_and(is_valid_src, tf.less_equal(250, [512]))

is_valid_trg = tf.greater(280, 0)
is_valid_trg = tf.logical_and(is_valid_trg, tf.less_equal(280, [512]))

tf.math.reduce_all([is_valid_src, is_valid_trg]) ==> tf.Tensor(True, shape=(), dtype=bool)
```

Далее создается перечень функций преобразования - [transform_fns](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L783). 

В список преобразований тренировочного датасета добавляется функция создания свойств датасета ([features](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L1009)).
Свойства датесета это:
* `length` - длина строки в токенах
* `tokens` - строка (последовательность токенов)
* `ids` - позиция токена в $`\textcolor{blue}{\text{source}}`$ | $`\textcolor{green}{\text{target}}`$ словаре. Последовательность `ids` формируется от нулевого индекса, до предпоследнего → [features["ids"] = ids[:-1]](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L479C17-L479C43)
* `ids_out` - позиция токена в $`\textcolor{green}{\text{target}}`$ словаре. Последовательность `ids_out` формируется от первого индекса до последнего → [features["ids_out"] = ids[1:]](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L480)

Объект свойств датасета выглядит следующим образом:
```ruby
features = [
    {
        'length': <tf.Tensor 'strided_slice:0' shape=() dtype=int32>,
        'tokens': <tf.Tensor 'StringSplit/RaggedGetItem/strided_slice_5:0' shape=(None,) dtype=string>,
        'ids': <tf.Tensor 'None_Lookup/SelectV2:0' shape=(None,) dtype=int64>
    },
    {
        'length': <tf.Tensor 'sub:0' shape=() dtype=int32>,
        'tokens': <tf.Tensor 'StringSplit_1/RaggedGetItem/strided_slice_5:0' shape=(None,) dtype=string>,
        'ids': <tf.Tensor 'strided_slice_2:0' shape=(None,) dtype=int64>,
        'ids_out': <tf.Tensor 'strided_slice_3:0' shape=(None,) dtype=int64>
    }
]
```

**Упрощенная последовательность вызова при формировании функции свойств датасета**:\
├── [lambda *arg: inputter.make_features() | def _get_dataset_transforms()](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L1029) модуль `inputter.py`\
├── [def make_features() | class TextInputter(Inputter)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L309) модуль `text_inputter.py`\
├── [def make_features() | class WordEmbedder(TextInputter)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/text_inputter.py#L453) модуль `text_inputter.py`\
├── [def make_features() | class ParallelInputter(MultiInputter)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L481) модуль `inputter.py`


<br>

После формирования списка применяемых функций трансформации датасета [filter_fn](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/inputters/inputter.py#L777) датасет передается в пайплайн для преобразования.

#### {+ В пайплайне подготовки датасета к тренировочному процессу происходят следующие преобразования: +}
* ##### на вход функции приходит сформированный датасет
```ruby
ZipDataset element_spec=(
    TensorSpec(shape=(), dtype=tf.string, name=None),
    TensorSpec(shape=(), dtype=tf.string, name=None)
)
```
* ##### датасет шафлится (перемешивается), преобразуется в следующий тип:
```ruby
ShuffleDataset element_spec=(
    TensorSpec(shape=(), dtype=tf.string, name=None),
    TensorSpec(shape=(), dtype=tf.string, name=None)
)
```

> Перемешивание датасета происходит по разному, в зависимости от значения параметра `sample_buffer_size`.
> - При `sample_buffer_size = -1` датасет шафлится с помощью функции [tf.data.Dataset.shuffle](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)
> - При `0 < sample_buffer_size < total_size` датасет перемешивается иначе. Логика реализована в функции [def random_shard(shard_size, dataset_size)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L263). Рассмотрим механизм при следующих параметрах:\
    _total_size (размер тренировочного датасета, строк) = 250_\
    _sample_buffer_size = 50_\
    1) рассчитывается целое число `num_shards = -(-250/50) ==> 5`\
    2) формируется последовательность от ноля до 250 с шагом 50 `==> [0, 50, 100, 150, 200]`\
    3) полученная последовательность шафлится `==> [100, 0, 150, 50, 200]`\
    4) датасет перемешивается следующим образом lambda offset: dataset.[skip](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#skip)(offset).[take](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take)(shard_size). Это означает, что из датасета `ZipDataset` берется кусок размером 50 строк с 100-й по 149-ю строку, далее берется кусок с 1-й строки (нулевой индекс) по 49-ю строку и т.д. Полученные срезы `[[100, 101, 102, 103, 104], [...], ...]` сплющиваются в одномерный массив с помощью функции [flat_map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map) 


* ##### применяется функция создания свойств датасета, датасет приобретает свойства, преобразуется в следующий тип:
```ruby
ParallelMapDataset element_spec=(
    {
        'length': TensorSpec(shape=(), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None,), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None,), dtype=tf.int64, name=None)
    },
    {
        'length': TensorSpec(shape=(), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None,), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None,), dtype=tf.int64, name=None),
        'ids_out': TensorSpec(shape=(None,), dtype=tf.int64, name=None)
    }
)
```
* ##### применяется функция фильтрации датасета, **{- из датасета удалены все строки -}**, длина которых больше `maximum_features_length` и `maximum_labels_length`, преобразуется в следующий тип:
```ruby
FilterDataset element_spec=(
    {
        'length': TensorSpec(shape=(), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None,), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None,), dtype=tf.int64, name=None)
    },
    {
        'length': TensorSpec(shape=(), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None,), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None,), dtype=tf.int64, name=None),
        'ids_out': TensorSpec(shape=(None,), dtype=tf.int64, name=None)
    }
)
```
> Каждый элемент датасета будет представлять собой следующий вид (для лучшего восприятия токены декодированы, в действительности они представлены в байтовом виде):
> - `length`: [ 5 ] - длина последовательности токенов в $`\textcolor{blue}{\text{source}}`$ строке;
> - `ids`: [ 39, 369, 155, 23, 370 ] - позиция каждого токена $`\textcolor{blue}{\text{source}}`$ в словаре, токен `▁just` - 39-й номер в словаре и т.д.;
> - `tokens`: ['▁just', '▁wonder', 'ing', '▁what', '▁happened'] - последовательность $`\textcolor{blue}{\text{source}}`$ токенов;
> 
> ---
> - `length`: [ 10 ] - длина последовательности токенов в $`\textcolor{green}{\text{target}}`$ строке;
> - `ids`: [ 1, 29, 202, 498, 7, 13, 50, 203, 499, 500 ] - позиция каждого токена в $`\textcolor{green}{\text{target}}`$ словаре. В `ids` $`\textcolor{green}{\text{target}}`$ токенов в начало списка добавляется токен `<s> - начало последовательности`, с индексом 1 в словаре. Токен `▁просто` - 29-й номер в словаре и т.д.;
> - `ids_out`: [ 29, 202, 498, 7, 13, 50, 203, 499, 500, 2 ]  - позиция каждого токена в $`\textcolor{green}{\text{target}}`$ словаре. В `ids_out` $`\textcolor{green}{\text{target}}`$ токенов в конец списка добавляется токен `</s> - конец последовательности`, с индексом 2 в словаре. Токен `▁просто` - 29-й номер в словаре и т.д.;
> - `tokens`: [ ▁просто ▁интерес но , ▁что ▁же ▁слу чи лось ] - последовательность $`\textcolor{green}{\text{target}}`$ токенов;

* ##### после фильтрации применяется функция группировки датасета [batch_sequence_dataset](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L440). Датасет группируется в бакеты/партии одинаковой длины, чтобы оптимизировать эффективность обучения и преобразуется в следующий тип:
```ruby
_GroupByWindowDataset element_spec=(
    {
        'length': TensorSpec(shape=(None,), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None, None), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None, None), dtype=tf.int64, name=None)
    },
    {
        'length': TensorSpec(shape=(None,), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None, None), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None, None), dtype=tf.int64, name=None),
        'ids_out': TensorSpec(shape=(None, None), dtype=tf.int64, name=None)
    }
)
```
>>>
**Рассмотрим механизм группировки на примере `batch_size = 100`, `length_bucket_width = 1`.**\
Датасет группируется с помощью функции tensorflow [group_by_window](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#group_by_window), которая сопоставляет каждый последовательный элемент в датасете с ключом с помощью `key_func` и группирует элементы по ключу. Затем применяется `reduce_func` к не более чем `window_size_func(key)` элементам, соответствующим одному и тому же ключу. Функции передаются:
- `key_func` - [def _key_func(*args)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L552)
- `window_size_func` - [def _window_size_func(key)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L563)
- `reduce_func` - [def _reduce_func(unused_key, dataset)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L560)

Прежде чем будет вызвана функция `group_by_window` в функции [batch_sequence_dataset](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L511) корректируется значение `batch_size`:
- `batch_size = batch_size * batch_multiplier`
- значение {+ batch_multiplier всегда равно единице, этот параметр нигде в тренировочном процессе не переопределяется +}
- таким образом значение `batch_size` = 100 * 1

Далее формируется функция получения длин элементов [element_length_func](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L521C5-L521C25)

Вызывается функция [group_by_window](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#group_by_window), которая в свою очередь вызывает:
> - [_key_func(*args)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L552). На вход функции приходит элемент датасета. Из этого элемента с помощью функции определения длины [get_element_length_func(length_fns)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L403) определяется максимальная длина `max([src_tokens_len, trg_tokens_len])`; например, `length` = {+ 5 +}\
> рассчитывается `bucket_id` - [tf.math.ceil](https://www.tensorflow.org/api_docs/python/tf/math/ceil)(length / length_bucket_width) - 1); `bucket_id` = {+ 5 / 1 - 1 =  4 +}
> ---
> - [_window_size_func(key)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L403). На вход функции приходит объект `key` в виде `bucket_id`, рассчитанном на предыдущем шаге; в функции рассчитывается размер группирующего окна:\
> `bucket_max_length = (key + 1) * length_bucket_width`; `bucket_max_length` = {+ (4 + 1) * 1 → 5 +}\
> `size = batch_size // bucket_max_length`; `size` = {+ 100 // 5 → 20 +}\
> корректируется значение `size = (size // batch_size_multiple) * batch_size_multiple`; `size` = {+ (20 // 1) * 1 = 20 +}\
> **{- при тренировке FP16 batch_size_multiple = 8 -}** `size` = (20 // 8) * 8 = **{- 16 -}**\
> [tf.math.maximum](https://www.tensorflow.org/api_docs/python/tf/math/maximum)(`size, batch_size_multiple`); {+ max(20 , 1) → 20 +}
> ---
> - [_reduce_func(unused_key, dataset)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L560). Датасет группируется с помощью функции [dataset.padded_batch(batch_size)](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch). Значение `batch_size` было рассчитано функцией `_window_size_func`. В нашем примере это 20. Т.е. будет сформирован бакет размером 20 элементов с `bucket_id` = 4
---
Пример группировки. Небольшой срез датасета, состоящего и 370 строк, длинной до 12 токенов. Параметры группировки: `batch_size = 100`, `length_bucket_width = 1`

![image](![image](https://github.com/user-attachments/assets/678da17a-4674-495b-a3f7-2dea040310ab)

Файл с исходным датасетом и  группировкой датасета `batch_size = 100` и листами `length_bucket_width = 1`, `length_bucket_width = 1_FP16` и `length_bucket_width = 2`

[openmnt_tf_grouped_df.xlsx](https://github.com/user-attachments/files/17547764/openmnt_tf_grouped_df.xlsx)

>>>

* ##### после группировки применяется функция удаления неравномерных бакетов [filter_irregular_batches](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L749)
>>>
- функция на вход принимает параметр `batch_multiplier`, который равен единице
- поскольку `batch_multiplier = 1`, датасет остается без изменений [def filter_irregular_batches(multiple)](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L137)
>>>

* ##### к сгруппированному датасету применяется функция бесконечного повторения [dataset](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L752C17-L752C43) = dataset.[repeat()](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#repeat)
> - это означает, что сгруппированный датасет будет генерироваться бесконечное количество раз

* ##### финальная стадия пайплайна преобразования - в бесконечно повторяющемся датасете вызывается функция извлечения предварительной выборки [dataset](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/dataset.py#L755C9-L755C57) = dataset.[prefetch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch)(prefetch_buffer_size). Датасет преобразуется в следующий тип:
```ruby
PrefetchDataset element_spec=(
    {
        'length': TensorSpec(shape=(None,), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None, None), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None, None), dtype=tf.int64, name=None)
    },
    {
        'length': TensorSpec(shape=(None,), dtype=tf.int32, name=None),
        'tokens': TensorSpec(shape=(None, None), dtype=tf.string, name=None),
        'ids': TensorSpec(shape=(None, None), dtype=tf.int64, name=None),
        'ids_out': TensorSpec(shape=(None, None), dtype=tf.int64, name=None)
    }
)
```

### Таким образом, пайплай преобразования датасета следующий:
├── шафлинг (перемешивание) датасета\
├── создание свойств датасета\
├── фильтрация датасета по максимальной длине предложений\
├── группировка предложений датасета в бакеты\
├── применение функции генерации сгруппированного датасета бесконечное количество раз\
├── применение функции извлечения выборки из датасета

----
<br>

# Механизм формирования матрицы алайнмента

Механизм формирования тензоров на основании матрицы алайнмента рассмотрим на небольшом примере: 
```python
0-0 1-1 2-2
```
Токены `source`:
```
▁you ▁know ▁it
```
Токены `target`:
```
▁вы ▁знаете ▁это
```

* ##### преобразование строк из файла с матрицей алайнмента осуществляется с помощью функции [alignment_matrix_from_pharaoh](https://github.com/OpenNMT/OpenNMT-tf/blob/6f3b952ebb973dec31250a806bf0f56ff730d0b5/opennmt/data/text.py#L60C5-L60C34), которая на вход принимает строку с индексами алайнера, длину токенов из `source` и длину токенов из `target` файлов

* из строки индексов `"0-0 1-1 2-2"` формируется массив `["0-0" "1-1" "2-2"]`

* из строки индексов `"0-0 1-1 2-2"` формируется массив `["0-0" "1-1" "2-2"]`

* массив `["0-0" "1-1" "2-2"]` преобразуется в массив чисел `[0 0 1 1 2 2]`

* массив `[0 0 1 1 2 2]` преобразуется в матрицу `sparse_indices`:
  ```
  [[0 0]
   [1 1]
   [2 2]]
  ```
* по нулевой размерности матрицы (по количеству строк, в нашем примере 3 строки)  `sparse_indices` формируется матрица `sparse_values` из единиц длинной 3 элемента `[1 1 1]`

* с помощью функции [tf.sparse.SparseTensor](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor) по матрицам `sparse_indices` и `sparse_values` и массиву длин `[source_length, target_length]` строится разреженный тензор → `alignment_matrix_sparse = tf.sparse.SparseTensor(sparse_indices, sparse_values, [source_length, target_length])`:

  ```python
     SparseTensor(
        indices=[[0 0]
                 [1 1]
                 [2 2]],
        values=[1 1 1],
        shape=[3 3]
     )
  ```

* с помощью функции [tf.sparse.to_dense](https://www.tensorflow.org/api_docs/python/tf/sparse/to_dense) разреженный тензор преобразуется в сжатый тензор `alignment_matrix = tf.sparse.to_dense(alignment_matrix_sparse)`:

  ```python
     [[1 0 0]
      [0 1 0]
      [0 0 1]]
  ```

* полученная на прошлом шаге матрица `alignment_matrix` с помощью функции [tf.transpose](https://www.tensorflow.org/api_docs/python/tf/transpose) преобразуется в финальную матрицу вида:

  ```python
     tf.Tensor(
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]],
      shape=(3, 3),
      dtype=float32
      )
  ```

* при тренировке с алайнментом, финальная строка тренировочного датасета будет иметь следующий вид:
  ```python
    features:
    {
        'ids': [[7 21 10]],
        'length': [3],
        'tokens': [["▁he" "▁knows" "▁it"]]
    }

    labels:
    {
        'alignment':[
            [[1 0 0]
            [0 1 0]
            [0 0 1]]
        ],
        'ids': [[1 9 17 8]],
        'ids_out': [[9 17 8 2]],
        'length': [4],
        'tokens': [["▁он" "▁знает" "▁это"]]
    }
  ```
