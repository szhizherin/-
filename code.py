# Импорт модулей

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

# Функции

def white_noise(sigma, T, seed):
    """Реализация независимого белого шума длины T, ε ~ N(0, sigma^2)"""
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=sigma, size=(1, T))
    return noise

def random_walk(initial_condition, noise):
    """Реализация случайного блуждания длины (T + 1) при начальном условии"""
    Y = initial_condition
    walk = np.append([Y], (Y + np.cumsum(noise)))
    return walk

def rho_statistic(series):
    """Вычисление ρ-статистики по временному ряду"""
    T = len(series) - 1
    series_diff = np.array(pd.Series(series).diff().dropna())
    series_ = np.array(series)[:-1]
    sum_upper = np.sum(series_*series_diff)
    sum_lower = np.sum(series_**2)
    rho_value = T*sum_upper/sum_lower
    print(rho_value)
    return rho_value

def t_statistic(series):
    """Вычисление t-статистики по временному ряду"""
    T = len(series) - 1
    series_diff = np.array(pd.Series(series).diff().dropna())
    series_ = np.array(series)[:-1]
    _series = np.array(series)[1:]
    term_upper = np.sum(series_*series_diff)
    rho = (rho_statistic(series)/T) + 1
    s = math.sqrt((1/(T-1))*np.sum((_series - rho*series_)**2))
    term_lower = s*math.sqrt(np.sum(series_**2))
    t_value = term_upper/term_lower
    print(t_value)
    return t_value

def calculate_statistic(statistic, N, M, sigma, Y):
    """Генерирование N случайных блужданий длины M с дисперсией белого шума
       sigma и начальным условием Y и подсчёт по ним статистики statistic"""
    noise_list = [white_noise(sigma, M-1, seed) for seed in range(N)]
    walk_list = [random_walk(Y, noise) for noise in noise_list]
    statistic_array = np.array([statistic(series) for series in walk_list])
    return statistic_array

def rho_theoretical(n, seed):
    """Приближённое вычисление значения одной теоретической реализации
       ρ-статистики с разбиением отрезка интегрирования на n равных частей"""
    sample = random_walk(0, white_noise(math.sqrt(1/n), n, seed))
    middles = (sample[:-1]+sample[1:])/2
    term_upper = (sample[-1]**2 - 1)/2
    term_lower = (np.sum(middles**2))/n
    rho_value = term_upper/term_lower
    print(rho_value)
    return rho_value  

def t_theoretical(n, seed):
    """Приближённое вычисление значения одной теоретической реализации
       t-статистики с разбиением отрезка интегрирования на n равных частей"""
    np.random.seed(seed)
    sample = random_walk(0, white_noise(math.sqrt(1/n), n, seed))
    middles = (sample[:-1]+sample[1:])/2
    term_upper = (sample[-1]**2 - 1)/2
    term_lower = math.sqrt((np.sum(middles**2))/n)
    t_value = term_upper/term_lower
    print(t_value)
    return t_value

def calculate_theoretical(statistic, N, n):
    """Генерирование N реализаций теоретического распределения
       статистики statistic, где в каждой реализации отрезок 
       интегрирования разбивается на n равных частей"""
    statistic_array = np.array([statistic(n, seed) for seed in range(N)])
    return statistic_array

def calculate_critical(statistic, N, M, I, alpha):
    """Генерирование N реализаций распределения статистики statistic,
       и вычисление критического значения для вероятности alpha"""
    array = calculate_statistic(statistic, N, M, 1, 0)
    left = np.min(array)
    right = np.max(array)
    x = (left + right)/2 
    for _ in range(I):
        x_less = [k for k in array if k < x]
        P = len(x_less)/len(array)
        if P < alpha:
            left = x
            x = (left + right)/2
        elif P > alpha:
            right = x
            x = (left + right)/2
        else:
            break
    print(x)
    return x

def p_value(z, statistic, N, M):
    """Вычисление p-значения для величины z статистики statistic"""
    array = calculate_statistic(statistic, N, M, 1, 0)
    x_less = [k for k in array if k < z]
    p = len(x_less)/len(array)
    print(p)
    return p
