# 05 — Признаки и оценка интересности

## 5.1 Полный реестр признаков

Каждая строка — то, что кладётся в `scenes.features` как `{"key": {"v": float, "c": float}}`.
Столбец «Плагин» = значение `FeatureAnalyzer.key`.

### Аудио (`audio/`)

| Ключ признака | Плагин | Как считается |
|---|---|---|
| `audio.loudness.mean` | `audio.loudness` | Интегральная громкость EBU R128 (LUFS) по сцене |
| `audio.loudness.peak` | `audio.loudness` | True peak, dBTP |
| `audio.loudness.delta` | `audio.loudness` | Скачок громкости относительно медианы всего видео |
| `audio.loudness.dynamic_range` | `audio.loudness` | LRA — разброс громкости внутри сцены |
| `audio.pause.total` | `audio.pauses` | Суммарная длительность тишины (VAD) |
| `audio.pause.max` | `audio.pauses` | Самая длинная пауза — маркер «драматической паузы» |
| `audio.pause.ratio` | `audio.pauses` | Доля тишины в сцене |
| `audio.event.laughter` | `audio.events` | Вероятность смеха (AudioSet-классификатор, AST) |
| `audio.event.scream` | `audio.events` | Вероятность крика |
| `audio.event.applause` | `audio.events` | Аплодисменты |
| `audio.event.music` | `audio.events` | Музыка — маркер монтажной вставки / титров |
| `audio.event.impact` | `audio.events` | Удары, взрывы, выстрелы |

### Речь (`speech/`)

| Ключ | Плагин | Как считается |
|---|---|---|
| `speech.rate.wpm` | `speech.rate` | Слов в минуту по word-level таймингам Whisper |
| `speech.rate.delta` | `speech.rate` | Отклонение темпа от базового для этого видео |
| `speech.density` | `speech.rate` | Доля сцены, занятая речью |
| `speech.dialogue_turns` | `speech.rate` | Оценка числа смен реплик (по паузам + смене темпа) |
| `speech.emotion.arousal` | `speech.emotion` | Возбуждение, 0..1 (проекция классов SER) |
| `speech.emotion.valence` | `speech.emotion` | Позитив/негатив, −1..1 (проекция классов SER) |
| `speech.emotion.neu` / `.hap` / `.ang` / `.sad` | `speech.emotion` | Вероятности классов wav2vec2-SER; argmax — та самая метка |
| `speech.text.sentiment` | `speech.emotion` | Тональность текста (лексикон) |
| `speech.text.exclamations` | `speech.emotion` | Плотность `!`, `?`, капса |

> **Почему вместо `speech.emotion.dominant` четыре вероятности.** Признак
> в базе — число, а не строка (`{"value": float, "confidence": float}`),
> и ради одной метки менять домен, схему и скоринг не стоит. Вероятности
> классов несут больше информации: «0.51 злости против 0.49 радости»
> и «0.98 злости» — разные сцены, а метка у них одна. Подробности —
> в разборе Этапа 5.

### Видео (`vision/`)

| Ключ | Плагин | Как считается |
|---|---|---|
| `vision.cuts.count` | `vision.cuts` | Число монтажных склеек внутри сцены |
| `vision.cuts.rate` | `vision.cuts` | Склеек в секунду — маркер экшена |
| `vision.motion.mean` | `vision.motion` | Средняя магнитуда оптического потока (Farnebäck, на 1/4 разрешении) |
| `vision.motion.peak` | `vision.motion` | Пик потока |
| `vision.camera.pan` | `vision.motion` | Глобальное панорамирование (аффинная оценка по ORB) |
| `vision.camera.zoom` | `vision.motion` | Наезд/отъезд |
| `vision.camera.shake` | `vision.motion` | Дрожание — ручная камера, погоня |
| `vision.faces.count_max` | `vision.faces` | Максимум лиц в кадре (YuNet из OpenCV) |
| `vision.faces.count_mean` | `vision.faces` | Среднее число лиц |
| `vision.faces.area_ratio` | `vision.faces` | Доля кадра, занятая крупнейшим лицом (крупный план = эмоция) |
| `vision.faces.closeup` | `vision.faces` | Флаг крупного плана |
| `vision.people.count` | `vision.faces` | Число людей (SSDLite/COCO — шире, чем лица: человек со спины тоже считается) |
| `vision.dynamics.hist_delta` | `vision.dynamics` | Изменение цветовой гистограммы кадр-к-кадру |
| `vision.dynamics.entropy` | `vision.dynamics` | Энтропия кадра — визуальная насыщенность |
| `vision.dynamics.brightness_delta` | `vision.dynamics` | Резкие изменения яркости (вспышки, взрывы) |

### Структурные (`highlight_detector/`, считаются без ML)

| Ключ | Как считается |
|---|---|
| `struct.position` | Нормализованная позиция сцены в видео (начало/середина/конец) |
| `struct.duration` | Длительность сцены |
| `struct.novelty` | Насколько сцена не похожа на соседние (косинус по вектору признаков) |
| `struct.build_up` | Рост «энергии» на протяжении предшествующих N сцен → кульминация |

## 5.2 Составные сигналы из ТЗ

Заказчик перечислил признаки высокого уровня. Они не измеряются напрямую — это
**линейные комбинации** базовых фич, вычисляемые в `highlight_detector/signals.py`.
Веса вынесены в YAML-профиль, чтобы правились без релиза.

| Сигнал | Формула (концептуально) |
|---|---|
| `emotional_speech` | `0.45·emotion.arousal + 0.25·|valence| + 0.20·loudness.delta + 0.10·rate.delta` |
| `laughter` | `0.80·event.laughter + 0.20·(valence > 0)` |
| `screaming` | `0.55·event.scream + 0.30·loudness.peak + 0.15·emotion.arousal` |
| `fight` | `0.35·motion.peak + 0.25·cuts.rate + 0.20·event.impact + 0.20·camera.shake` |
| `chase` | `0.40·camera.pan + 0.25·motion.mean + 0.20·cuts.rate + 0.15·camera.shake` |
| `action` | `max(fight, chase)` усиленный `0.2·loudness.dynamic_range` |
| `plot_twist` | `0.40·novelty + 0.25·pause.max + 0.20·emotion.surprise + 0.15·brightness_delta` |
| `long_dialogue` | `0.50·speech.density + 0.30·dialogue_turns + 0.20·faces.count_mean` |
| `tension` | `0.35·build_up + 0.25·pause.ratio + 0.20·(low loudness, high arousal) + 0.20·music` |
| `romantic` | `0.35·faces.closeup + 0.25·(valence>0 & arousal low) + 0.20·music + 0.20·(faces.count == 2)` |

Каждый сигнал даёт значение 0..1 и попадает в `scenes.features` под ключом `signal.<name>`.
LLM получает именно эти сигналы, а не 40 сырых чисел — так промпт короче и понятнее модели.

## 5.3 Нормализация

Сырые признаки несопоставимы (LUFS ≈ −23, поток ≈ 0.4, лиц ≈ 3). Нормализуем
**внутри одного видео**, а не глобально — «громко» в фильме ужасов и в лекции разное.

```python
def robust_normalize(values: NDArray) -> NDArray:
    """Робастная нормализация: медиана + межквартильный размах, клип по 3 IQR,
    затем min-max в [0,1]. Устойчива к выбросам (одна вспышка не съедает шкалу)."""
```

Для признаков, где важна именно аномальность (`loudness.delta`, `rate.delta`),
дополнительно считается модифицированный z-score через MAD.

Видео короче 3 минут или с < 8 сценами: статистики ненадёжны — переключаемся на
преднастроенные глобальные шкалы из профиля.

## 5.4 Эвристический скор

```
heuristic = 100 · Σ(w_g · s_g) / Σ(w_g)    по группам g, где s_g определён
```

Группы и базовые веса (профиль `default`):

| Группа | Вес | Состав |
|---|---|---|
| `emotion` | 0.25 | `emotional_speech`, `laughter`, `screaming` |
| `action` | 0.25 | `action`, `fight`, `chase` |
| `narrative` | 0.20 | `plot_twist`, `tension`, `build_up` |
| `dialogue` | 0.15 | `long_dialogue`, `speech.density` |
| `visual` | 0.15 | `dynamics`, `faces.closeup`, `cuts.rate` |

Знаменатель `Σ(w_g)` берётся только по **доступным** группам — это и есть механизм
деградации из док 04, §4.5: если `vision` упал, оставшиеся группы автоматически
масштабируются до 100.

`scenes.score_breakdown` хранит вклад каждой группы — критично для отладки
(«почему эта сцена набрала 90?») и для UI.

## 5.5 Профили скоринга

`config/profiles/*.yaml`, выбираются параметром `profile` в `POST /analyze`.

| Профиль | Смещение весов |
|---|---|
| `default` | как в таблице выше |
| `movie` | `action` 0.35, `narrative` 0.25, `dialogue` 0.10 |
| `podcast` / `interview` | `dialogue` 0.35, `emotion` 0.30, `action` 0.05 |
| `lecture` | `dialogue` 0.40, `narrative` 0.30, `visual` 0.05 |
| `gaming` | `action` 0.40, `emotion` 0.30 (задел под стримы) |

Профиль также задаёт целевые длительности клипов и минимальный порог скора.

## 5.6 Слияние с LLM

```
final = α · heuristic_norm + β · llm_score + γ · agreement_bonus
```

По умолчанию `α = 0.4`, `β = 0.5`, `γ = 0.1`.

- `heuristic_norm` — эвристика, 0..100.
- `llm_score` — от 0 до 100, приходит из JSON (док 06).
- `agreement_bonus` — `100 − |heuristic_norm − llm_score|`, вознаграждает согласие
  двух независимых оценок. Момент, который считают удачным и сигналы, и модель,
  почти всегда действительно удачный.

Если LLM недоступна (`ExternalServiceError` после ретраев) — `β = 0`, `γ = 0`,
джоб помечается `partial`, клипы всё равно генерируются по эвристике.

## 5.7 Отбор финальных моментов

Алгоритм в `highlight_detector/selector.py`:

1. **Расширение.** Сцена редко = хайлайт. Соседние сцены со скором ≥ 70 % от пика
   склеиваются в кандидата, пока длительность ≤ `max_duration`.
2. **Снап границ.** Начало/конец сдвигаются к ближайшей паузе в речи (± 1.5 с),
   чтобы клип не начинался с середины слова. Если паузы нет — к границе сцены.
   Финальный снап — к ближайшему ключевому кадру для точной резки без перекодирования.
3. **Фильтр длительности.** Отбрасываем всё вне `[min_duration, max_duration]`
   (по умолчанию 15–90 с; для вертикали — 15–60 с).
4. **NMS.** Кандидаты сортируются по `final_score`; кандидат отбрасывается,
   если IoU с уже выбранным > 0.3.
5. **Разнообразие (MMR).** Из оставшихся выбираем `top_k`, максимизируя
   `λ·score − (1−λ)·max_similarity_to_selected`, где similarity — косинус по
   вектору сигналов + штраф за ту же категорию. `λ = 0.7`. Так в выдаче не окажется
   пять почти одинаковых сцен драки.
6. **Ограничение по бюджету.** Если задан `total_duration`, набираем жадно, пока влезает.

Результат — упорядоченный список `Highlight`, из которого рождаются записи `clips`.

## 5.8 Оценка качества

Без измерения скоринг превращается в набор магических чисел.

- **Golden-датасет.** 10 видео разных жанров × 10 экспертно размеченных «лучших» моментов.
  Лежит в `tests/fixtures/golden/` как JSON с тайм-кодами (сами видео — по ссылкам/чек-суммам).
- **Метрики.** `Precision@5`, `Precision@10`, `Recall@10`, `mAP`, средний IoU
  предложенных границ с эталонными.
- **Регрессионный тест.** `pytest -m quality` прогоняет весь конвейер на golden-датасете
  и падает, если `Precision@10` упал более чем на 5 п.п. относительно зафиксированного baseline.
- **Ablation.** Скрипт `scripts/ablation.py` отключает по одной группе признаков
  и печатает дельту метрик — так видно, какие анализаторы реально работают, а какие шум.
