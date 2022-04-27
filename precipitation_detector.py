import cv2 as cv
import numpy as np

# гиперпараметры
filter_size = 3
min_area = 16
shape_coef_min = 4
max_angle = 30

alpha = 0.1
decision_thr = 3


# видеофайл для анализа
fname = '1642089573046.mp4'

# подготовка
cap = cv.VideoCapture(fname)
backSub = cv.createBackgroundSubtractorMOG2(25, 16, False)
n_frame = 0
n_cnt_avg = 0

# основной цикл по кадрам
while True:
    ret, im_src = cap.read()
    if not ret:
        break

    n_frame += 1

    # вычитание фона
    fg_mask = backSub.apply(im_src)

    # фильтрация
    kernel = np.ones((filter_size,filter_size), np.uint8)
    filtered = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)

    # поиск контуров
    cnt_all, hierarchy = cv.findContours(filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    n_cnt_all = len(cnt_all)

    # отбор по площади
    cnt_sel_by_area = []
    for cnt in cnt_all:
        if cv.contourArea(cnt, False) >= min_area:
            cnt_sel_by_area.append(cnt)
    n_cnt_sel_by_area = len(cnt_sel_by_area)

    # отбор по коэффициенту формы
    cnt_sel_by_shape = []
    for cnt in cnt_sel_by_area:
        peri = cv.arcLength(cnt, True)
        area = cv.contourArea(cnt)
        shape_coef = (peri ** 2) / area / (4 * np.pi)
        if shape_coef >= shape_coef_min:
            cnt_sel_by_shape.append(cnt)
    n_cnt_sel_by_shape = len(cnt_sel_by_shape)

    # отбор по коэффициенту формы
    cnt_sel = []
    for cnt in cnt_sel_by_shape:
        vx, vy, x, y = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01).flatten()
        vect_line = np.array([vx, vy])
        vect_y_axis = np.array([0, 1])
        dot_product = np.dot(vect_line, vect_y_axis)
        angle = np.rad2deg(np.arccos(dot_product))
        angle = min(angle, 180 - angle)
        if angle <= max_angle:
            cnt_sel.append(cnt)

    # подсчет количества контуров
    n_cnt_curr = len(cnt_sel)            
            
    # усреднение
    n_cnt_avg = n_cnt_curr * alpha + n_cnt_avg * (1 - alpha)
    n_cnt_avg_int = int(round(n_cnt_avg))

    # заключение
    is_precipitation = n_cnt_avg_int >= decision_thr
    concl_dict_ru = {True: 'есть осадки', False: 'нет осадков'}
    conclusion = concl_dict_ru[is_precipitation]

    # вывод информации в консоль
    text = f'{n_frame} {n_cnt_all} {n_cnt_sel_by_area}' + \
           f' {n_cnt_sel_by_shape} {n_cnt_curr} {n_cnt_avg_int}' + \
           f' {is_precipitation} {conclusion}'
    print(text)

    # остановка по нажатию ESC
    if cv.waitKey(1) & 0xFF == 27:
        break

#cap.close()
