# nlp_project

# Корпус русских народных сказок
Корпус состоит из 120 русских народных сказок. Сказки взяты с сайта https://skazki-pesni.ru.

## База данных
В файле 'fairy_tales_razdel_full.csv' хранятся данные всех сказок. Сказки были разделены по предложениям. Столбцы: 
- 'sents' - предложения из сказок,
- 'name' - название сказки,
- 'href' - ссылка на сказку,
- 'sent_without' - предложения без знаков препинания.

В файле 'corpus.csv' добавляется информация о словах, входящих в предложение.
- 'tokens' - токены,
- 'pos_tags' - падежи слов,
- 'lemmas' - леммы слов
  
## Основные используемые библиотеки 
- fake_useragent;
- nltk - токенезация;
- pymorphy2 - лемматизация;
- Natasha - POS-теггинг и лемматизация;
- pandas - создание DataFrame;
- re - написание регулярных выражений.

## Работа корпуса

Программа разделена на несколько блоков.
1. Парсинг
   - Для обхода защиты сайта использовался fake_useragent.
   - Функция get_nth_page обрабатывает одну страницу сайта (вычленяет текст, название, ссылку на сказку)
   - Функция run_all принимает количество страниц с сайта и обрабатывает их по-странично.
  
2. Создание базы данных
   - Разбиваем все тексты на предложения, создаем колонки для всех предложений, названий сказок, ссылок на сказки, предложений без пунктуации.
   - Записываем все в файл 'fairy_tales_razdel_full.csv'.
     
3. Обработка запроса и поиск предложений

   3.1. Лемматизация, токенизация, POS-tagging
   - Выполняем POS-теггинг, токенезацию, лемматизацию с помощью Natasha и pymorphy2.
   - Резултаты сохраняем в файл 'corpus.csv'.
   
   3.2. Проверка конкретного слова на соответствие запросу
   - Функция check_word_form - функция для проверки всех форм слова, игнорируя регистр.
   - Функция check_exact_form - функция для проверки точной формы слова без учета регистра.
   - Функция check_lemma_and_pos - функция для проверки по лемме и тегу части речи.
   - Функция check_pos - функция для проверки соответствия частеречного тега.
   - Функция word_fits_request_part - функция для вызова нужной из четырех функций выше в зависимости от запроса.

   3.3. Основная функция поиска
   - Функция search - основная функция, которая может принимать слово, слово в кавычках, лемма+POS-tag, POS-tag.
   - Возвращает предложения, подходящие под запрос, вместе с источником и ссылкой на источник.

4. Поиск коллокаций
   - Функция get_collocations - функция для посика коллокаций, принимает слово, возвраащет коллокации с этим словом и их количество.
  
5. Итоговая программа

   5.1. Вывод найденных примеров
   - Функция pretty_line - функция для выравнивая контекстов определенного примера.
   - Функция get_all_pretty_lines - функция, которая выводит все найденный примеры (выровнянные).
     
   5.2. Взаимодействие с программой (см.Взаимодействие пользователя с корпусом)
      

## Взаимодействие пользователя с корпусом

1. Пользователь вводит интересующее его слово, слово в кавычках, лемма+POS-tag, POS-tag.
2. Вывод программы состоит из:
   * предложений, которые содержат все предложения с этим запросом с разделением на левый, центральный, парвый контексты и с указанием на источник и название сказки.
   * коллокаций с их частотностью.

   Пример: ввод пользователя - 'лисичка'

     *Забралась | лисичка   | в теремок. || Предложение из Теремок https://skazki-pesni.ru/teremok/*
     
     *Жили-были в лесу | лисичка   | и зайка. || Предложение из Заюшкина избушка https://skazki-pesni.ru/zayushkina-izbushka/*

     
     *с лисичкой | 3*
     
     *лисичкой близко | 3*
     
     *пришла лисичка | 3*
   

## Поддержка
Если у Вас возникли вопросы относительно работы корпуса, Вы можете их задать авторам проекта, написав в телеграмме. Мы всегда будем рады Вам ответить!

- Майя Горшенина - @latentgypsy
- Заряна Дамашова - @Mr_ushanovich
- Софья Черноусенко - @carasinaa
- София Герен - @ssofiyk


