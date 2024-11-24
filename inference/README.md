<br>

# Инференс модели с помощью фреймворка OpenNMT-tf

Механизм инференса натренированной модели рассмотрим на примере следующей строки `▁he ▁knows ▁it`. Архитектура модели, для наглядности, будет примерно такая же как и в рассматриваемом примере механизма тренировочного процесса:
* vocab: 26
* num_units: 6
* num_layers: 1
* num_heads: 2
* ffn_inner_dim: 12
* maximum_relative_position: 8

Для каждого токена извлекается векторное представление токена из матрицы `source` эмбеддингов натренированной модели и подается на вход в энкодер. В энкодере, происходят абсолютно все те же преобразования, как и при тренировке модели. Отличие - к матрицам **не применяется** механизм дропаута. Подробно механизм преобразования описан [здесь](https://git.nordicwise.com/infra/machine-translate-utils/-/wikis/Механизм-тренировочного-процесса#в-энкодере-векторное-представлнение-токенов-source-языка). После преобразований в энкодере получаем матрицу `encoder_outputs`\
![encoder](https://github.com/user-attachments/assets/264fafd5-f0fa-4948-a9ae-6fbf4c9905a7)

#### Рассмотрим пример для инференса со следующими параметрами:
 - beam_size = 2
 - length_penalty = 0.2
 - coverage_penalty = 0.2
 - sampling_topk = 5
 - sampling_temperature = 0.5
<br>

   * c помощью функции [tfa.seq2seq.tile_batch(encoder_outputs, beam_size)](https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/tile_batch) матрица значений полученная после энкодера дублируется на количество `beam_size`\
![beam_size](https://github.com/user-attachments/assets/5659ca93-f7ca-41a8-adbb-7b3c5acc2488)

   * инициализируются переменные:\
    ━ 1) размер бачта для перевода, в нашем примере batch_size = `1`\
    ━ 2) матрица `start_ids`, содержащая индексы токенов начала последовательности `<s>` по размерности `beam_size` → start_ids = tfa.seq2seq.tile_batch(start_ids, beam_size) = `[1 1]`\
    ━ 3) матрица `finished`, заполненная нулями, c булевым типом данных и размерностью `batch_size * beam_size` → [tf.zeros](https://www.tensorflow.org/api_docs/python/tf/zeros)([batch_size * beam_size], dtype=tf.bool)] = `[0 0]`\
    ━ 4) матрица `initial_log_probs` → [tf.tile](https://www.tensorflow.org/api_docs/python/tf/tile)([0.0] + [-float("inf")] * (beam_size - 1), [batch_size]) = `[0 -inf]`\
    ━ 5) инициализируется словарь с дополнительными переменными `extra_vars`, который содержит следующие переменные:
        - `parent_ids`: [tf.TensorArray(tf.int32, size=0, dynamic_size=True)](https://www.tensorflow.org/api_docs/python/tf/TensorArray) → `[]`
        - `sequence_lengths`: [tf.zeros([batch_size * beam_size], dtype=tf.int32)](https://www.tensorflow.org/api_docs/python/tf/zeros) → `[0 0]`
        - `accumulated_attention`: [tf.zeros([batch_size * beam_size, attention_size])](https://www.tensorflow.org/api_docs/python/tf/zeros), где attention_size это размерность матрицы encoder_outputs (tf.shape(encoder_outputs)[1] = 3)  → `[[0 0 0] [0 0 0]]`


Далее, в цикле, итерируемся, пока не достигнем максимального значения (максимальное значение по дефолту равно `maximum_decoding_length = 250`) или пока не будет сгенерирован токен окончания последовательности:
<hr>

   * #### step = 0

   * по матрице `start_ids`, извлекаются векторные представления токенов из матрицы `target` эмбеддингов натренированной модели и вместе с матрицей `encoder_outputs`, продублированной на количество `beam_size` подается на вход в декодер. В декодере, происходят абсолютно все те же преобразования, как и при тренировке модели. Отличие -  **{- не формирует тензор маски future_mask а так же не применяется -}** механизм дропаута. Подробно механизм преобразования описан [здесь](https://git.nordicwise.com/infra/machine-translate-utils/-/wikis/Механизм-тренировочного-процесса#в-декодере-векторное-представлнение-токенов-target-языка)\
![step0_decoder](https://github.com/user-attachments/assets/08550791-c15d-414e-aee5-aebeaf0b95eb)

   * по размерности матрицы `logits`, полученной из декодера, формируются переменные `batch_size = 1` и `vocab_size = 26`

   * с помощью функции [tf.one_hot(tf.fill([batch_size], end_id), vocab_size, on_value=logits.dtype.max, off_value=logits.dtype.min)](https://www.tensorflow.org/api_docs/python/tf/one_hot) формируется матрица `eos_max_prob`, где:
        - [tf.fill([batch_size], end_id)](https://www.tensorflow.org/api_docs/python/tf/fill); `end_id` - индекс токена окончания последовательности `</s>`  →  [2 2]
        - `logits.dtype.max` - максимальное значение типа данных tf.float32, которое равно 3.4028235e+38 или 34028235000000000000000000000000000000
        - `logits.dtype.min` - максимальное значение типа данных tf.float32, которое равно -3.4028235e+38 или -34028235000000000000000000000000000000
![logits](https://github.com/user-attachments/assets/b9c517e2-3dd3-499b-9bae-19954f1aeb63)\
        таким образом матрица `eos_max_prob` будет размерностью 2 x 26, где элементы по индексом 2 будут заполнены максимальным значением, а все остальные элементы будут заполнены минимальным значением

   * с помощью функции [tf.where(tf.broadcast_to(tf.expand_dims(finished, -1), tf.shape(logits)), x=eos_max_prob, y=logits)](https://www.tensorflow.org/api_docs/python/tf/where), где:
        - матрица `finished` изменяется по размерности: [tf.expand_dims([0, 0], -1)](https://www.tensorflow.org/api_docs/python/tf/fill) → `[[0][0]]`
        - получаем массив значений по размерности матрицы logits: tf.shape(logits) → `[2 26]`
        - увеличиваем размерность: tf.broadcast_to([[0], [0]], [2, 26]) → `[[0 0 0 ... 0 0 0], [0 0 0 ... 0 0 0]]`
        - поскольку матрица `finished` содержит нули, и расширенная матрица содержит нули, значения финальной матрицы заполняются значениями из матрицы `logits`
![logits_where](https://github.com/user-attachments/assets/eea18536-884d-4b37-bcfd-2a9e0345a432)

   * по матрице `logits` с помощью функции [tf.nn.log_softmax(logits)](https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax) рассчитывается матрица `log_probs`\
![logits_softmax](https://github.com/user-attachments/assets/02bff33e-0706-4b5d-aec8-3ff4200c71f6)

   * если параметр `coverage_penalty != 0`, то дополнительно производятся следующие действие:
        - матрица `finished` изменяется преобразуется с помощью функции [tf.math.logical_not([0, 0])](https://www.tensorflow.org/api_docs/python/tf/math/logical_not) с образованием матрицы `not_finished` → `[1 1]`
        - полученная матрица изменяется по размерности tf.expand_dims(tf.cast(not_finished, attention.dtype), 1) → `[[1], [1]]`
        - полученная матрица перемножается с матрицей `attention`
![step0_attn](https://github.com/user-attachments/assets/3c13bcaf-6e04-4dc2-b61a-7fe274c45490)
        - формируется переменная `accumulated_attention`
![step0_acc_attn](https://github.com/user-attachments/assets/7bd76dbf-df0f-426d-b4f7-47e684cfe3d8)

   * формируется матрица `total_probs` путем сложения матрицы `log_probs` и перевернутой матрицы `cum_log_probs` → log_probs + tf.expand_dims(cum_log_probs, 1)\
![log_probs](https://github.com/user-attachments/assets/56dce430-099c-4b4e-8855-df02484aeef1)
