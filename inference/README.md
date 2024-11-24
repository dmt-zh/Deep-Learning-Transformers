<br>

# Инференс модели с помощью фреймворка OpenNMT-tf

Механизм инференса натренированной модели рассмотрим на примере следующей строки `▁he ▁knows ▁it`. Архитектура модели, для наглядности, будет примерно такая же как и в рассматриваемом примере механизма тренировочного процесса:
* vocab: 26
* num_units: 6
* num_layers: 1
* num_heads: 2
* ffn_inner_dim: 12
* maximum_relative_position: 8

Для каждого токена извлекается векторное представление токена из матрицы `source` эмбеддингов натренированной модели и подается на вход в энкодер. В энкодере, происходят абсолютно все те же преобразования, как и при тренировке модели. Отличие - к матрицам **не применяется** механизм дропаута. Подробно механизм преобразования описан [здесь](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#в-энкодере-векторное-представлнение-токенов-source-языка). После преобразований в энкодере получаем матрицу `encoder_outputs`\
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

   * по матрице `start_ids`, извлекаются векторные представления токенов из матрицы `target` эмбеддингов натренированной модели и вместе с матрицей `encoder_outputs`, продублированной на количество `beam_size` подается на вход в декодер. В декодере, происходят абсолютно все те же преобразования, как и при тренировке модели. Отличие -  **не формирует тензор маски future_mask а так же не применяется** механизм дропаута. Подробно механизм преобразования описан [здесь](https://github.com/dmt-zh/Transformers-Full-Review/blob/main/training/README.md#в-декодере-векторное-представлнение-токенов-target-языка)

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

   * по матрицам `total_probs`, `sequence_lengths`, `finished` и `accumulated_attention` рассчитываются оценки **scores**:
        - по матрице `total_probs` инициализируется исходная матрица `scores` → `scores = total_probs`\
        ![scores](https://github.com/user-attachments/assets/d649efd3-a8bc-44ac-97f8-39141768d9db)
        - если `length_penalty != 0`, тогда, выполняем следующие действия:
           - формируется матрица `expand_sequence_lengths` - матрица `sequence_lengths` изменяется по размерности tf.expand_dims(sequence_lengths, 1) → `[[0], [0]]`
           - к матрице `expand_sequence_lengths` прибавляется единица и матрица приводится к типу значений `log_probs` - tf.cast(expand_sequence_lengths + 1, log_probs.dtype) → `[[0], [0]] + 1 = [[1], [1]]`
            - формируется матрица `sized_expand_sequence_lengths` - к полученной выше матрице прибавляется константа 5 и делится на константу 6: (5.0 + expand_sequence_lengths) / 6.0) → `(5 + [[1], [1]]) / 6 = [[1], [1]]`
           - формируется матрица `penalized_sequence_lengths` - матрица `sized_expand_sequence_lengths` возводится в степень, указанную в параметре `length_penalty`: tf.pow(sized_expand_sequence_lengths, length_penalty) → `[[1]**0.2, [1]**0.2] = [[1], [1]]`
           - значения матрицы корректируются целочисленным делением на матрицу `penalized_sequence_lengths` → `scores /= penalized_sequence_lengths`
       ![penalized_sequence_lengths](https://github.com/user-attachments/assets/81e2ee87-9573-4089-905c-1f25eb5f71ee)

        - если `coverage_penalty!= 0`, тогда, выполняем следующие действия:
           - по матрице `accumulated_attention` формируется матрица `equal` с помощью функции [tf.equal(accumulated_attention, 0.0)](https://www.tensorflow.org/api_docs/python/tf/math/equal)  tf.expand_dims(sequence_lengths, 1), т.е. проверяем элементы матрицы на равенство нулю\
        ![acc_attn2](https://github.com/user-attachments/assets/98b35ba1-e2c3-42b9-9192-889b357f561a)
           - по матрице `accumulated_attention` формируется матрица единиц `ones_like` с помощью функции [tf.ones_like(accumulated_attention)](https://www.tensorflow.org/api_docs/python/tf/ones_like)\
        ![acc_attn3](https://github.com/user-attachments/assets/21c3a862-e254-40d7-aab6-1d17bc68efe3)
           - с помощью функции [tf.where(equal, x=ones_like, y=accumulated_attention)](https://www.tensorflow.org/api_docs/python/tf/where) переопределяется матрица `accumulated_attention`, в которой значения будут браться из `x`, если элемент из `equal` равен единице, иначе из `y`. Поскольку все элементы матрицы `equal` равны нулю, то все значения будут взяты из `y`\
        ![acc_attn4](https://github.com/user-attachments/assets/05b91319-e915-4e9a-9260-2149f8f38b10)
           - формируется матрица `coverage_penalty` - по матрице `accumulated_attention` и единице берется минимальное значение и рассчитывается логарифм, а затем элементы суммируются `tf.reduce_sum(tf.math.log(tf.minimum(accumulated_attention, 1.0)), axis=1)`
        ![cov_penalty](https://github.com/user-attachments/assets/2e21ce08-74e8-41ab-9688-1a314d6196d5)
           - полученная на прошлом шаге матрица `coverage_penalty` перемножается с матрицей `finished` `coverage_penalty *= finished`\
        ![cov_penalty2](https://github.com/user-attachments/assets/5aa72733-2d9d-4bd2-9d26-46d2693f9356)
           - матрица `scores ` корректируется на заданное значение `coverage_penalty` из параметров и рассчитанную матрицу `coverage_penalty` - `scores += self.coverage_penalty * tf.expand_dims(coverage_penalty, 1)`
        ![scores_cov_penalty](https://github.com/user-attachments/assets/7002d5f5-a890-4662-af88-d45961aeb560)

   * матрица `scores` изменяется по размерности `tf.reshape(scores, [-1, beam_size * vocab_size])`\
   ![scores_flat](https://github.com/user-attachments/assets/7c974c90-e3d9-4e8d-89f0-74b36190fdd2)

   * матрица `total_probs` изменяется по размерности `tf.reshape(scores, [-1, beam_size * vocab_size])`\
   ![total_probs](https://github.com/user-attachments/assets/75c86961-0884-4fa4-9b3f-797be4e6bc72)


   * рассчитываются `id` target токенов `sample_ids` и оценки для этих токенов `sample_scores`:
        - по матрице `scores` с помощью функции [tf.nn.top_k](https://www.tensorflow.org/api_docs/python/tf/math/top_k) находятся максимальные значения и их индексы `top_scores, top_ids = tf.nn.top_k(scores, k=sampling_topk)` (если параметр `sampling_topk` не задан, то `k` будет равно `beam_size`)\
        ![top_scores](https://github.com/user-attachments/assets/fb838e6f-397f-4280-92b7-babc6d845cb3)
        - матрица `top_scores` делится на значение параметра `sampling_temperature`\
        ![temperature](https://github.com/user-attachments/assets/b259e7a8-dce6-43e9-8b5f-70a6a008cd7e)
        - из скорректированной матрицы `top_scores` с помощью функции [tf.random.categorical](https://www.tensorflow.org/api_docs/python/tf/random/categorical) извлекаются индексы элементов в количестве `beam_size`\
        ![sample_ids](https://github.com/user-attachments/assets/24d0072d-0afe-4030-9cd7-2af0dc876f39)
        - по индексам элементов, с помощью функции [tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather) из матрицы `top_ids` извлекаются индексы токенов
        ![top_ids](https://github.com/user-attachments/assets/30cfb90c-40a5-4c2f-b2a9-d37db647f3db)
        - по индексам элементов, с помощью функции [tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather) из матрицы `scores` извлекаются оценки для этих токенов
        ![sample_scores](https://github.com/user-attachments/assets/593db4a6-5c58-42b2-8dda-d64f97409bb0)

   * по полученным значениям `sample_ids` из матрицы `total_probs` формируется матрица `cum_log_probs`
   ![cum_log_probs](https://github.com/user-attachments/assets/1cf1350c-7557-4f62-892f-2cc90d9b0df6)

   * по полученным значениям `sample_ids`, путем деления с остатком на значение `vocab_size` формируется матрица `word_ids` → word_ids = sample_ids % vocab_size = [9 12] % 26 = `[9 12]`

   * по полученным значениям `sample_ids`, путем целочисленного деления на значение `vocab_size` формируется матрица `beam_ids` → beam_ids = sample_ids // vocab_size = [9 12] // 26 = `[0 0]`

   * по полученным матрицам `word_ids` и `beam_ids`, а так же значению `beam_size` формируется матрица `beam_indices` → beam_indices = (tf.range(tf.shape(word_ids)[0]) // beam_size) * beam_size + beam_ids = ([0 1] // 2) * 2 + [0 0] = `[0 0]`

   * переопределяется матрица `sequence_lengths` → sequence_lengths = tf.where(finished, x=sequence_lengths, y=sequence_lengths + 1)\
   ![seq_length](https://github.com/user-attachments/assets/84ccdff8-78cc-4928-93de-2d11fa64f72d)

   * переопределяется матрица `finished` → finished = tf.gather(finished, beam_indices) → tf.gather([0 0], [0 0]) = `[0 0]`

   * переопределяется матрица `sequence_lengths` → finished = tf.gather(sequence_lengths, beam_indices) → tf.gather([1 1], [0 0]) = `[1 1]`

   * в словарь `extra_vars` по ключу `sequence_lengths` сохраняется матрица `sequence_lengths` → 
 extra_vars = {"sequence_lengths": sequence_lengths}

   * в словарь `extra_vars` по ключу `parent_ids` записываются значения `beam_ids` и значения текущего шага → extra_vars = {"parent_ids": parent_ids.write(step, beam_ids)}

   * в словарь `extra_vars` по ключу `accumulated_attention` сохраняется матрица `accumulated_attention` → 
 tf.gather(accumulated_attention, beam_indices)\
   ![extra_vars_acc_attn](https://github.com/user-attachments/assets/30c02486-d743-4a69-ba5a-e5cd7e60dac6)

   * полученные значения `word_ids` проверяются на равенство токену окончания последовательности `</s>` и матрица `finished` переопределяется → `finished = tf.logical_or(finished, tf.equal(word_ids, end_id))`

   * ##### на этом step 0 завершается, матрицы `word_ids`, `cum_log_probs`, `finished`, и словарь `extra_vars` передаются в начало цикла и весь описанных выше процесс
   ![step0_finish](https://github.com/user-attachments/assets/f6af2443-3bd2-4acc-800f-edeae168a096)

<hr>

   * #### step = 1

   * по матрице `word_ids`, полученной на шаге `step = 0` извлекаются векторные представления токенов из матрицы `target` эмбеддингов натренированной модели и вместе с матрицей `encoder_outputs`, продублированной на количество `beam_size` подается на вход в декодер. Из декодера получаем матрицы `logits` и `attention`\
![decoder_step1](https://github.com/user-attachments/assets/cd30427e-c8c8-4cbc-bd4b-8df5a0f3c34d)

   * проходим вышеописанные операции в step = 0 и получаем после шага 1 матрицы `word_ids`, `cum_log_probs`, `finished`, и словарь `extra_vars`\
![log_probs_step1](https://github.com/user-attachments/assets/71409a87-4366-490f-a228-5988e69815a8)\
![total_probs_step1](https://github.com/user-attachments/assets/73f078f6-889b-4256-95d8-14a5c2cb4804)\
![top_ids_step1](https://github.com/user-attachments/assets/fce3d6d0-65cc-403d-a021-97ea73fff4e2)\
![finished_stpe1](https://github.com/user-attachments/assets/44e53475-0187-4766-a357-142fc9807670)

<hr>

   * #### step = 2

   * по матрице `word_ids`, полученной на шаге `step = 1` извлекаются векторные представления токенов из матрицы `target` эмбеддингов натренированной модели и вместе с матрицей `encoder_outputs`, продублированной на количество `beam_size` подается на вход в декодер. Из декодера получаем матрицы `logits` и `attention`\
![docoder_step2](https://github.com/user-attachments/assets/bd763139-35a1-4253-8136-2266e385430c)

   * проходим вышеописанные операции и получаем после шага 2 матрицы `word_ids`, `cum_log_probs`, `finished`, и словарь `extra_vars`\
![log_probs_step2](https://github.com/user-attachments/assets/3e43f9b0-3b5d-4c9a-ab06-ee4ff664962b)\
![total_probs_step2](https://github.com/user-attachments/assets/f418b551-d4eb-4d3e-8921-a88d24a928e2)\
![top_ids_stpe2](https://github.com/user-attachments/assets/5638d751-a1a0-472a-bb34-36e0eacda636)\
![finished_step2](https://github.com/user-attachments/assets/ba010d1c-0d06-4f34-bc20-9720a4a89d42)

<hr>

   * #### step = 3

   * по матрице `word_ids`, полученной на шаге `step = 2` извлекаются векторные представления токенов из матрицы `target` эмбеддингов натренированной модели и вместе с матрицей `encoder_outputs`, продублированной на количество `beam_size` подается на вход в декодер. Из декодера получаем матрицы `logits` и `attention`\
![decoder_step3](https://github.com/user-attachments/assets/612680f7-9830-4767-9a7b-b7545190e3cd)

   * проходим вышеописанные операции и получаем после шага 3 матрицы `word_ids`, `cum_log_probs`, `finished`, и словарь `extra_vars`\
![log_probs_step3](https://github.com/user-attachments/assets/879fa585-034e-4e02-b689-bf7efe9700d5)\
![total_probs_step3](https://github.com/user-attachments/assets/8d7ac731-cb60-468e-b5db-7cb0e820f688)\
![top_ids_step3](https://github.com/user-attachments/assets/9d1d977f-3792-4638-b544-40c70ef4d253)\
![finished_step3](https://github.com/user-attachments/assets/83c737bf-155e-4f3f-9109-196428168808)


   * на этом шаге цикл прерывается, т.к. декодером были сгенерированы токены окончания последовательности `</s>` с id = 2 
  
<hr>

Полученные последовательности id токенов декодируются и получаются гипотезы перевода `source` предложения. **Количество гипотез не может быть больше beam_size, т.e. если мы хотим получить 3 альтернативных варианта перевода, то нам необходимо установить beam_size=3**. Фактически, в качестве результата у нас 2 гипотезы, 1-я гипотеза будет содержать последовательности наивероятнейших токенов, получаемых и нескольких распределений.\
![decode_tokens](https://github.com/user-attachments/assets/36d8198b-d119-40e6-ad7a-fd2bc60a27bb)

