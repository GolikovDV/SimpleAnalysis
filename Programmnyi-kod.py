# Перед первым запуском необходимо установить сторонние библиотеки:
#   pip install numpy pandas matplotlib
# (модуль math входит в стандартную библиотеку Python и не требует установки)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def gen_data():
    """Генерация исходных случайных данных."""
    return pd.Series(np.random.randint(-10000, 10001, size=1000))


def show_fragment(s, n=15):
    """Вывод фрагмента созданных данных для предварительного анализа."""
    print("Фрагмент созданных данных (первые элементы Series):")
    print(s.head(n).to_string())
    print(f"Количество элементов: {s.size}")
    print(f"Тип данных: {s.dtype}")
    print(f"Диапазон индексов: от {s.index.min()} до {s.index.max()}")


def clean_data(s):
    """Очистка данных от цифрового мусора.

    Проверяет каждый элемент по двум критериям:
    1) является целым числом (а не текстом, пропуском или дробным значением);
    2) входит в заданный диапазон [-10000; 10000].
    Корректные элементы попадают в очищенный набор, остальные отбраковываются.
    """
    low, high = -10000, 10000

    def is_valid(x):
        # отбраковываем пропуски, логический и нечисловой/дробный мусор
        if pd.isna(x) or isinstance(x, bool):
            return False
        if not isinstance(x, (int, np.integer)):
            return False
        # отбраковываем значения за пределами заданного диапазона
        return low <= x <= high

    valid_mask = s.apply(is_valid)
    cleaned = s[valid_mask].reset_index(drop=True)
    removed = s.size - cleaned.size
    print(f"Проверено элементов: {s.size}")
    print(f"Отбраковано (цифровой мусор): {removed}")
    print(f"Корректных элементов в очищенном наборе: {cleaned.size}")
    return cleaned


def calc_stat(s):
    """Расчет и вывод базовых числовых характеристик."""
    mn = s.min()
    dp = s.duplicated().sum()
    mx = s.max()
    su = s.sum()
    sd = s.std()

    print(f"Минимальное значение: {mn}")
    print(f"Количество повторяющихся значений: {dp}")
    print(f"Максимальное значение: {mx}")
    print(f"Сумма чисел: {su}")
    print(f"Среднеквадратическое отклонение: {sd:.2f}")


def plot_line(s):
    """Построение линейного графика исходного Series."""
    plt.figure()
    plt.plot(s)
    plt.title('Линейный график исходных данных')
    plt.show()


def plot_hist(s):
    """Математическое округление до сотен и построение гистограммы."""
    s_r = (s / 100).apply(lambda x: math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)) * 100
    plt.figure()
    plt.hist(s_r, bins=20, color='orange', edgecolor='black')
    plt.title('Гистограмма (округление до сотен)')
    plt.show()


def make_df(s):
    """Формирование DataFrame с отсортированными столбцами."""
    d = pd.DataFrame({'orig': s})
    d['asc'] = s.sort_values().reset_index(drop=True)
    d['desc'] = s.sort_values(ascending=False).reset_index(drop=True)
    return d


def plot_sorted(d):
    """Построение графиков отсортированных значений."""
    plt.figure()
    plt.plot(d['asc'], label='По возрастанию', color='green')
    plt.plot(d['desc'], label='По убыванию', color='red')
    plt.title('Отсортированные значения')
    plt.legend()
    plt.show()


def run_tests():
    """Тестирование разработанных программных модулей.

    Проверяется корректность работы каждой функции на контрольных данных.
    """
    print("Запуск тестирования программных модулей...")

    # Тест 1. Генерация данных: размер, тип и диапазон значений.
    s = gen_data()
    assert s.size == 1000, "Размер набора данных должен быть равен 1000"
    assert pd.api.types.is_integer_dtype(s), "Значения должны быть целочисленными"
    assert s.min() >= -10000 and s.max() <= 10000, "Значения вне заданного диапазона"

    # Тест 2. Очистка данных: добавляем заведомый цифровой мусор и проверяем отбраковку.
    dirty = pd.Series([5, -10000, 10000, 50000, -20000, np.nan, 3.5, "abc"], dtype=object)
    cleaned = clean_data(dirty)
    assert cleaned.tolist() == [5, -10000, 10000], "Очистка отработала некорректно"

    # Тест 3. Формирование DataFrame: наличие столбцов и корректность сортировки.
    d = make_df(s)
    assert list(d.columns) == ['orig', 'asc', 'desc'], "Неверный состав столбцов"
    assert d['asc'].is_monotonic_increasing, "Столбец asc не отсортирован по возрастанию"
    assert d['desc'].is_monotonic_decreasing, "Столбец desc не отсортирован по убыванию"

    print("Все тесты пройдены успешно.\n")


def main():
    run_tests()

    s = gen_data()
    show_fragment(s)
    s = clean_data(s)
    calc_stat(s)
    plot_line(s)
    plot_hist(s)

    d = make_df(s)
    plot_sorted(d)


if __name__ == "__main__":
    main()
