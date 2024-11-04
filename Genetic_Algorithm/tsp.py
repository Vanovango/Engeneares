import csv
import pickle
import os

import numpy as np

import matplotlib.pyplot as plt

class TravelingSalesmanProblem:
    def __init__(self, name):

        # инициализируем переменные:
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0

        # инициализируем данные:
        self.__initData()

    def __len__(self):
        return self.tspSize

    def __initData(self):
        # попытка чтения сериализованных данных:
        try:
            self.locations = pickle.load(open(os.path.join("tsp-data", self.name + "-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join("tsp-data", self.name + "-dist.pickle"), "rb"))
        except (OSError, IOError):
            pass

        # сериализованные данные не найдены - создаем данные:
        if not self.locations or not self.distances:
            self.__createData()

        # устанавливаем размерность задачи:
        self.tspSize = len(self.locations)

    def __createData(self):
        self.locations = []
        # открываем файл с координатами городов:
        with open(self.name + ".txt") as f:
            reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
            
            # пропуск пока не найдены следующие строки:
            for row in reader:
                if row[0] in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                    break

            # чтение файла пока не найден 'EOF':
            for row in reader:
                if row[0] != 'EOF':
                    # удаляем индекс в начале строки:
                    del row[0]

                    # преобразовываем координаты:
                    self.locations.append(np.asarray(row, dtype=np.float32))
                else:
                    break

            # устанавливаем размерность задачи:
            self.tspSize = len(self.locations)

            # выводим данные:
            print("length = {}, locations = {}".format(self.tspSize, self.locations))

            # инициализируем матрицу расстояний заполняя ее нулями:
            self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]

            # заполняем матрицу расстояний:
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    # вычисляем Евклидово расстояние между двумя координатами:
                    distance = np.linalg.norm(self.locations[j] - self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
                    print("{}, {}: location1 = {}, location2 = {} => distance = {}".format(i, j, self.locations[i], self.locations[j], distance))

            # сериализуем лместоположение и расстояния:
            if not os.path.exists("tsp-data"):
                os.makedirs("tsp-data")
            pickle.dump(self.locations, open(os.path.join("tsp-data", self.name + "-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join("tsp-data", self.name + "-dist.pickle"), "wb"))

    def getTotalDistance(self, indices):
        # рассстояние между перрвым и последним городом:
        distance = self.distances[indices[-1]][indices[0]]

        # добавлчем расстояние между двумя последовательными городами:
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]

        return distance

    def plotData(self, indices):
        # рисуем точки на графике:
        plt.scatter(*zip(*self.locations), marker='.', color='red')

        # создаем список местоположений:
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # рисуем линии между городами:
        plt.plot(*zip(*locs), linestyle='-', color='blue')

        return plt

        