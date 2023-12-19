"""
Ввод 5 срок по два вещественных числа на каждой
радиус_1 и радиус_2 на первой строке
координаты контрольных точек на оставшихся

Вывод координаты, на которые необходимо разместить манипулятор
матрица из углов phi_1 и phi_2 для каждой контрольной точки соответственно

Или сообщение от том, что нет нужной точки
"""

import numpy as np
import pygame
from scipy.optimize import fsolve


def find_circle_intersections(x1, y1, x2, y2, radius1, radius2):
    # Расстояние между центрами окружностей
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if 0 < distance <= radius1 + radius2:
        # Вычисление координат точек пересечения окружностей
        a = (radius1 ** 2 - radius2 ** 2 + distance ** 2) / (2 * distance)
        x = x1 + a * (x2 - x1) / distance
        y = y1 + a * (y2 - y1) / distance

        h = np.sqrt(radius1 ** 2 - a ** 2)
        x3 = x - h * (y2 - y1) / distance
        y3 = y + h * (x2 - x1) / distance

        x4 = x + h * (y2 - y1) / distance
        y4 = y - h * (x2 - x1) / distance

        return [x3, x4], [y3, y4]

    return [], []


def main():
    # Ввод радиусов
    radii = list(map(float, input().split()))
    radius1, radius2 = radii

    # Ввод координат точек
    unique_points = set()
    input_points = []

    for _ in range(4):
        input_point = list(map(float, input().split()))
        input_points.append(input_point)
        unique_points.add(tuple(input_point))

    input_points = np.array(input_points)
    unique_points = list(unique_points)
    points_x, points_y = zip(*unique_points)

    intersections_x = []
    intersections_y = []

    # Поиск пересечений окружностей
    for r1 in [abs(radius1 - radius2), radius1 + radius2]:
        for r2 in [abs(radius1 - radius2), radius1 + radius2]:
            for i in range(len(points_x)):
                for j in range(i + 1, len(points_y)):
                    x, y = find_circle_intersections(points_x[i], points_y[i], points_x[j], points_y[j], r1, r2)
                    intersections_x += x
                    intersections_y += y

    # Обработка случая, когда все вершины совпадают
    if len(intersections_x) == 0 and len(points_x) == 1:
        intersections_x += [points_x[0] + radius1 + radius2]
        intersections_y += [points_y[0]]

    intersections_x = np.array(intersections_x)[:, np.newaxis]
    intersections_y = np.array(intersections_y)[:, np.newaxis]

    points_x = np.array(points_x)
    points_y = np.array(points_y)

    # Расчет расстояний между точками и пересечениями
    dist_x = points_x - intersections_x
    dist_y = points_y - intersections_y
    intersections = np.sqrt(dist_x ** 2 + dist_y ** 2)

    # Определение, какие пересечения находятся на кольца
    where = np.where(np.all((abs(radius2 - radius1) <= intersections) & (intersections <= radius1 + radius2), axis=1))[
        0]

    if len(where):
        where = where[0]
    else:
        print("Точки, из которой манипулятор может дотянуться до всех контрольных точек, не существует")
        exit()

    center_x = intersections_x[where][0]
    center_y = intersections_y[where][0]

    # Коррекция координат точек
    input_points[:, 0] -= center_x
    input_points[:, 1] -= center_y

    def equations(variables, x, y, r1, r2):
        phi1, phi2 = variables
        eq1 = r1 * np.cos(phi1) + r2 * np.cos(phi1 + phi2) - x
        eq2 = r1 * np.sin(phi1) + r2 * np.sin(phi1 + phi2) - y
        return [eq1, eq2]

    angles = [[0.0, 0.0]]

    initial_guess = [0.0, 0.0]

    # Расчет углов поворота для каждой точки
    for point in input_points:
        x, y = point
        result = fsolve(equations, initial_guess, args=(y, x, radius1, radius2))
        result = result % (2 * np.pi)
        phi1, phi2 = result
        center = np.array([0., 0.])
        for alpha1 in [phi1, phi1 + np.pi]:
            for alpha2 in [phi2, phi2 + np.pi]:
                point1 = center + np.array([radius1 * np.sin(alpha1), radius1 * np.cos(alpha1)])
                point2 = point1 + np.array(
                    [radius2 * np.sin(alpha1 + alpha2), radius2 * np.cos(alpha1 + alpha2)])
                if abs(point2[0] - x) <= 0.01 and abs(point2[1] - y) <= 0.01:
                    res = [alpha1, alpha2]
                    break
        angles.append(res)

    return np.array([center_x, center_y]), input_points, np.array(radii), np.array(angles)


def visualize(points, radii, angles):
    def draw_lines(radii, center, angles):
        radius1, radius2 = radii
        point1 = center + np.array([radius1 * np.sin(angles[0]), radius1 * np.cos(angles[0])]) * k
        point2 = point1 + np.array([radius2 * np.sin(angle[0] + angles[1]), radius2 * np.cos(angle[0] + angles[1])]) * k
        pygame.draw.line(screen, white, center, point1, 5)
        pygame.draw.line(screen, black, point1, point2, 5)

    pygame.init()

    pygame.display.set_caption("Manipulator modeling")

    # Установка констант
    FPS = 60
    WHIDTH, HEIGHT = 800, 800
    radius1, radius2 = radii
    k = (WHIDTH // 2) / (radius1 + radius2)
    center = np.array([WHIDTH // 2 + 50, HEIGHT // 2 + 50])
    WHIDTH, HEIGHT = 900, 900

    black = (0, 0, 0)
    white = (255, 255, 255)

    # Установка переменных
    screen = pygame.display.set_mode((WHIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    n = 1
    rotate_time = 3 # время (с) перемещения от одной точки к следующей
    stop_time = 1 # время (с) паузы между точками
    angle = np.array([0., 0.])
    t = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_DELETE]:
                running = False

        screen.fill(black)

        # Рисование областей, границ и точек
        pygame.draw.circle(screen, (161, 231, 175), center, k * (radius1 + radius2) + 1)
        pygame.draw.circle(screen, white, center, k * (radius1 + radius2) + 4, 4)
        pygame.draw.circle(screen, (228, 113, 122), center, k * abs(radius1 - radius2))
        pygame.draw.circle(screen, black, center, k * abs(radius1 - radius2), 4)
        for i in range(len(points)):
            xy = points[i]
            pygame.draw.circle(screen, (210, 145, 50 + 50 * i), center + xy * k, 5)

        # Рисование звеньев манипулятора
        if t:
            i = int(n / (rotate_time + stop_time) / FPS) + 1
            if i < len(angles) and n % ((rotate_time + stop_time) * FPS) + 1 > stop_time * FPS:
                angle += (np.array(angles[i]) - np.array(angles[i - 1])) / rotate_time / FPS
        draw_lines(radii, center, angle)
        if i > len(angles):
            t = False

        pygame.display.flip()
        clock.tick(FPS)
        n += 1
    pygame.quit()


if __name__ == "__main__":
    center, points, radii, angles = main()
    print(center)
    print(angles[1:])
    visualize(points, radii, angles)
