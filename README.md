# Проект: Поиск экстремумов многомерных функций с помощью метода градиентного спуска

Автор: Матвей Шелякин 

## Требуемые пакеты

-numpy

-matplotlib

-random

## Введение в градиентный спуск.

Градиентный спуск - это метод оптимизации, который используется для нахождения минимума функции. Он основан на идее итеративного изменения параметров функции таким образом, чтобы уменьшать значение функции.

В общем случае, градиентный спуск может быть использован для минимизации функций любого размерности. Однако он часто используется для минимизации функций высокой размерности, таких как функции потерь в машинном обучении.

Градиентный спуск работает следующим образом:

Выбирается начальное приближение для параметров функции.
Вычисляется градиент функции в текущей точке. Градиент - это вектор, состоящий из частных производных функции по каждому из параметров. Он показывает аправление самого быстрого роста функции.
3. Параметры функции обновляются следующим образом:
parameter = parameter - d * gradient
Здесь d - это скорость обучения, которая определяет, насколько быстро мы будем двигаться в сторону антиградиента.

Шаги 2 и 3 повторяются до тех пор, пока функция не достигнет минимума или не будет достигнуто какое-то предельное число итераций.
Важно отметить, что градиентный спуск работает только для функций, у которых есть градиент, то есть функций, у которых можно вычислить частные производные. Он также работает лучше для функций, которые выпуклые и гладкие, так как в этом случае градиент указывает направление самого быстрого роста функции, и мы можем быстро добраться до минимума. Однако, градиентный спуск может работать медленно или даже застревать в локальных минимумах в случае невыпуклых или резко изгибающихся функций.

## Общее описание проекта

В рамках проекта представлен алгоритм поиска экстремумов многомерных дифференцируемых функций. Пользователь задает интересующую его функцию и ограничительную компактную область, на которой будет происходить поиск экстремальных точек. Ограничительная область в $n$-мерном пространстве представляет собой $(n-1)$-мерный шар радиусом $r$, так например для техмерного пространства где задана некоторая функция $z = f(x,y)$ ограничительной областью будет окружность в плоскости XY.

Далее берется некоторое количество $k$ случайных точек из ограничительной области в качестве начальных приближений для градиентного спуска. Решение иницировать алгоритм с разных точек обусловлено тем, что градиентный спуск способен найти только один локальный минимум (или максимум), а в ограничительной области точек максимума или минимума может быть несколько. Таким образом такой подход увеличивает вероятность найти все экстремумы на области, и чем больше $k$, тем выше эта вероятность, но и тем дольше время работы алгоритма. 

Сам же градиентный спуск реализован с постепенно убывающим шагом. Такой подход снижает время сходимости, но решает проблему зацикливания возле экстремума. Пользователь может выбрать, хочет ли он найти точки максимума или минимума функции. Если пользователь выбрал максимум, поиск идет в направлении вектора градиента, если нужен минимум, то движение в направлении антиградиента.

С некоторой прогрешностью $\varepsilon$ алгоритм представляет $k$ точек, подозрительных на экстремум.Из них удаляются точки, попавшие на границу области. Далее точки группируются по экстремумам: если координаты каких-то точек отличаются не больше чем на погрешность \varepsilon, то мы предполагаем, что эти точки - результат попытки алгоритма прийти к одному и тому же экстремуму, поэтому из этих точек мы оставляем только одну. Далее мы понимаем, что нулевой градиент могут иметь не только экстремумы, но так же некоторые точки перегиба и седловые точки, поэтому для этих точек мы проверяем достаточные условия экстремума с помощью криетрия Сильвестра - для каждой точки смотрем положительно или отрицательно определена матрица вторых производных. Если все угловые миноры этой матрицы положительны, то это точка минимума, если же миноры знакочередуются, начиная с <0, то это точка максимума, во всех остальных случаях мы полагаем, что точка не является экстремумом и удаляется из списка.Сравение значения миноров с нулем так же происходит на основе некоторой погрешности эпсилон.

После того, как точки экстремума окончательно предложены, они изображаются на графике вместе с самой функцией (только для 2D и 3D случаев)

## Документация к функциям

Пользователю предлагается настройка следующих параметров:

- $radius$ - float, радиус ограничительной области

- $center$ - list, координаты точки центра, размерность которой не должна быть меньше количества аргументов функции

- $epsilon$ - float, погрешность вычислений

- $dx$ - float, дифференциал, приращение аргумента
- $d$ - float, начальный шаг сходимости градиентного спуска
- $start\_points\_qnt$ - int, количество случайных начальных приближений
- $type$ - str, enumeration i\{'min','max'\}

Следующие функции задействованы в проекте:

**1)** **differentiable_function(x)** - дифференцируемая функция, которая будет исследована на экстремумы. Задается пользователем! Пример функции: $x[0]**2+x[1]**2$

Аргументы:
- $x$ - list, координаты точки, для которой расчитывается значение функции

Вывод: float значение функции в точке


**2)** **gradient(x, dx)** -  рассчет вектора градиента в точке для заданной дифференцируемой функции

Аргументы:
- $x$ - list, координаты точки, для которой расчитывается вектор градиента
- $dx$ - float, дифференциал, приращение аргумента

Вывод: list массив частных производных функции в точке $x$


**3)** **gradient_search(x0, dx, center, radius, extr_type, d=0.1, epsilon =0.0001)** - функция, реализующая градиентный спуск

Аргументы:
- $x0$ - list, координаты точки начального приближения
- $dx$ - float, дифференциал, приращение аргумента
- $center$ - list, координаты точки центра ограничительной области
- $radius$ - float, радиус ограничительной области
- $extr\_type$ - str, enumeration i\{'min','max'\} тип экстремума, максимум или минимум
- $epsilon$ - float, погрешность вычислений

Вывод: list координата точки, подозрительной на экстремум


**4)** **derivative_2(x, dx)** -  вычисление матрицы вторых частных производных и смешанных частных производных в точке

Аргументы:
- $x$ - list, координаты точки, для которой расчитывается гессиан
- $dx$ - float, дифференциал, приращение аргумента

Вывод: list матрица 2 частных производных функции в точке $x$


**5)** **silvestr_criterion(x0, dx=0.0001, epsilon = 0.001)** - критерий Сильвестра для проверки достаточных условий экстремума. На вход ожидаются точки, для которых необходимое условие существование экстремума (координаты вектора градиента равны нулю) выполнено!

Аргументы:
- $x0$ - list, координаты точки, подозрительной на экстремум
- $dx$ - float, дифференциал, приращение аргумента
- $epsilon$ - float, погрешность вычислений

Вывод: str сообщение, является ли точка максимумом, минимумом или не является экстремумом вовсе: 'min', 'max', 'Not an extremum'

**6)** **get_grid(grid_step)** - функция создает сетку, на которой будет построен график

Аргументы:
- $dx$ - float, шаг между точками для построения граффика по осям координат

Вывод: list массив точек для построения графика, отложенных по осям координат

**7)** **draw_chart(points, grid)** - построение графика функции и найденых экстремумов

Аргументы:
- $points$ - list массив с координатами точек экстремума для нанесения на график
- $grid$ - list массив точек для построения графика, отложенных по осям координат

Вывод: график

