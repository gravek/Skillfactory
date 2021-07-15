# MASTER без random. Угадывает число от 1 до 100  за 5 попыток.

def game_core_v2_5_tries(number):
    # Создаем переменные с границами загаданного числа 
    # (от них отталкнемся в угадывании)
    min_edge = 1
    max_edge = 101

    # Создаем переменную для попыток угадать,
    # В ней же первая попытка угадать:
    count = 1
    predict = int((max_edge - min_edge) / 2)    

    while number != predict:        
        count += 1
        if number > predict:
            min_edge = predict
            predict = int(min_edge + (max_edge - min_edge) / 2)

        elif number < predict:
            max_edge = predict
            predict = int(max_edge - (max_edge - min_edge) / 2)
    return(count)  # выход из цикла, если угадали
