"""
Вам предоставлена программа, которая находит и отображает кратчайший путь между заданными точками, используя
генетический алгоритм обучения.
Вам требуется создать новый файл - "координаты.txt" - со следующими координатами:
    (100, 100)
    (70, 130)
    (70, 140)
    (80, 150)
    (90, 150)
    (100, 140)
    (110, 150)
    (120, 150)
    (130, 140)
    (130, 130)
    При создании файла опирайтесь на старую версию.
Замените необходимые части кода, для достижения нужного результата.
Результатом вашей работы будет фигура, которая появиться после выполнения программы
"""

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import array

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tsp

# установка случайного начального значения для получения повторяемых результатов
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# создание задачи коммивояжера:
TSP_NAME = "координаты_узла"  # название проблемы
tsp = tsp.TravelingSalesmanProblem(TSP_NAME)

# Константы генетического алгоритма:
POPULATION_SIZE = 300
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 1
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1   # вероятность мутации индивидуума

toolbox = base.Toolbox()

# определяем единую цель, минимизирующую фитнес-функцию:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# создаем индивидуальный класс на основе списка целых чисел:
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

# создаем оператор, который генерирует случайно перетасованные индексы:
toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))

# оператор начального создания индивидуумов, чтобы заполнить отдельный экземпляр перетасованными индексами:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# create the population creation operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# фычисление фитнес-функции - вычисляем общее расстояние до списка городов, представленных индексами:
def tpsDistance(individual):
    return tsp.getTotalDistance(individual),  # возвращает кортеж


toolbox.register("evaluate", tpsDistance)


# генетические операторы:
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))


# запуск генетического алгоритма:
def main():

    # создаем начальную популяцию (поколение 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # подготовкуа статистических операторов:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # определение "зала славы":
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # выполнение генетического алгоритма:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # вывод информаци о лучших индивидуумов:
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    # график лучшего решения:
    plt.figure(1)
    tsp.plotData(best)

    # график статистики:
    # minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    # plt.figure(2)
    # sns.set_style("whitegrid")
    # plt.plot(minFitnessValues, color='red')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel('Generation')
    # plt.ylabel('Min / Average Fitness')
    # plt.title('Min and Average fitness over Generations')

    # показ графиков:
    plt.show()


if __name__ == "__main__":
    main()
