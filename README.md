# nlp_project

# Корпус русских народных сказок
Корпус состоит из 120 русских народных сказок. Сказки взяты с сайта https://skazki-pesni.ru.

## База данных
В файле 'fairy_tales_razdel_full.csv' хранятся данные всех сказок. Сказки были разделены по предложениям. Столбцы: 
- 'sents' - предложения из сказок,
- 'name' - название сказки,
- 'href' - ссылка на сказку,
- 'sent_without' - предложения без знаков препинания.

## Основные используемые библиотеки 
- nltk -
- pymorphy2 - лемматизация;
- Natasha - POS-теггинг и лемматизация;
- pandas - создание DataFrame;
- re - написание регулярных выражений.

## Работа корпуса

## Взаимодействие пользователя с корпусом

1. Пользователь вводит интересующее его слово, например 'теремок'.
2. Вывод программы состоит из:
   * предложений, которые содержат все предложения с этим словом с указанием на источник и название сказки.
     *'Стоит в поле теремок.' Теремок https://skazki-pesni.ru/teremok/*
     *'Лягушка прыгнула в теремок.' Теремок https://skazki-pesni.ru/teremok/*
   * коллокаций с их частотностью. 
   

## Поддержка
Если у Вас возникли вопросы относительно работы корпуса, Вы можете их задать авторам проекта, написав в телеграмме. Мы всегда будем рады Вам ответить!

- Майя Горшенина - @latentgypsy
- Заряна Дамашова - @Mr_ushanovich
- Софья Черноусенко - @carasinaa
- София Герен - @ssofiyk


