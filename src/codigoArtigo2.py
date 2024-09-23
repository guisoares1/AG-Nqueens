import math
import random
import turtle

def generate_chromosome(n):
    chromosome = list(range(1, n + 1))
    random.shuffle(chromosome)
    return chromosome

def generate_population(population_size, n):
    return [generate_chromosome(n) for _ in range(population_size)]

def fitness(chromosome):
    n = len(chromosome)
    t1 = 0
    t2 = 0
    f1 = [chromosome[i] - i for i in range(n)]
    f2 = [chromosome[i] + i for i in range(n)]
    f1.sort()
    f2.sort()
    for i in range(1, n):
        if f1[i] == f1[i - 1]:
            t1 += 1
        if f2[i] == f2[i - 1]:
            t2 += 1
    fitness_value = t1 + t2
    return fitness_value

def order_one_crossover(parent1, parent2):
    size = len(parent1)
    child1 = [-1] * size
    child2 = [-1] * size

    start, end = sorted(random.sample(range(size), 2))

    # Copia o segmento dos pais para os filhos
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]

    # Preenche o restante dos genes para child1
    def fill_child(child, parent):
        size = len(parent)
        current_pos = (end + 1) % size
        parent_pos = (end + 1) % size
        while -1 in child:
            gene = parent[parent_pos % size]
            if gene not in child:
                child[current_pos % size] = gene
                current_pos = (current_pos + 1) % size
            parent_pos = (parent_pos + 1) % size
    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return child1, child2

def mutate(chromosome, mutation_rate, double_mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    if random.random() < double_mutation_rate:
        idx3, idx4 = random.sample(range(len(chromosome)), 2)
        chromosome[idx3], chromosome[idx4] = chromosome[idx4], chromosome[idx3]
    return chromosome

def genetic_algorithm(n, population_size=100, max_generations=1000, mutation_rate=0.8):
    population = generate_population(population_size, n)
    bad_repo_size = max(1, int(math.sqrt(n)))
    bad_repository = generate_population(bad_repo_size, n)  # Inicializa uma vez e não atualiza

    double_mutation_rate = 0.4

    for generation in range(max_generations):
        fitnesses = [fitness(chrom) for chrom in population]

        if 0 in fitnesses:
            solution = population[fitnesses.index(0)]
            print(f"Solução encontrada na geração {generation + 1}")
            return solution, generation + 1  # Retorna a solução e o número de gerações

        new_population = []

        # 1. Cruzamento e mutação de cromossomos selecionados aleatoriamente da população
        for _ in range(population_size // 2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child1, child2 = order_one_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, double_mutation_rate)
            child2 = mutate(child2, mutation_rate, double_mutation_rate)
            new_population.extend([child1, child2])

        # 2. Cruzamento entre cromossomos do repositório e da população
        for _ in range(bad_repo_size):
            parent1 = random.choice(bad_repository)
            parent2 = random.choice(population)
            child, _ = order_one_crossover(parent1, parent2)
            child = mutate(child, mutation_rate, double_mutation_rate)
            new_population.append(child)

        # Combina e seleciona os melhores indivíduos para a próxima geração
        combined_population = population + new_population
        combined_fitnesses = [fitness(chrom) for chrom in combined_population]
        population = [chrom for _, chrom in sorted(zip(combined_fitnesses, combined_population))][:population_size]

    print("Nenhuma solução encontrada.")
    return None, max_generations

def draw_box(t, x, y, size, fill_color):
    """
    Função auxiliar para desenhar um quadrado no tabuleiro de xadrez.
    """
    t.penup()
    t.goto(x, y)
    t.pendown()

    t.fillcolor(fill_color)
    t.begin_fill()

    for _ in range(4):
        t.forward(size)
        t.right(90)

    t.end_fill()

def draw_chess_board(board_size, solvedBoard):
    """
    Desenha o tabuleiro de xadrez e posiciona as rainhas com base em solvedBoard.
    """
    board = turtle.Turtle()
    board.speed(0)
    board.hideturtle()
    square_color = "black"  # A primeira casa é preta
    start_x = -board_size * 15  # Centraliza o tabuleiro horizontalmente
    start_y = -board_size * 15  # Centraliza o tabuleiro verticalmente
    box_size = 30  # Tamanho de cada quadrado

    for i in range(board_size):
        for j in range(board_size):
            x = start_x + j * box_size
            y = start_y + i * box_size
            if solvedBoard[i][j] != 1:
                draw_box(board, x, y, box_size, square_color)
            else:
                draw_box(board, x, y, box_size, 'green')
            square_color = 'black' if square_color == 'white' else 'white'  # Alterna a cor após cada coluna
        if board_size % 2 == 0:
            square_color = 'black' if square_color == 'white' else 'white'  # Alterna a cor após cada linha se o tabuleiro tiver tamanho par

    turtle.done()

# Parâmetros do algoritmo
n = 100
population_size = 1000
max_generations = 1000
mutation_rate = 0.8  # Conforme especificado no algoritmo

# Executar o algoritmo genético
solution, iterations = genetic_algorithm(n, population_size, max_generations, mutation_rate)

# Imprimir a solução
if solution:
    print(f"Solução encontrada em {iterations} iterações.")
    print("Solução (representação das posições das rainhas):", solution)
    # Ajustar índices para zero-based
    zero_indexed_solution = [pos - 1 for pos in solution]
    # Criar o solvedBoard
    solvedBoard = [[0]*n for _ in range(n)]
    for row, col in enumerate(zero_indexed_solution):
        solvedBoard[row][col] = 1
    # Imprimir a matriz de valores
    print("Matriz de valores (tabuleiro):")
    for row in solvedBoard:
        print(row)
    # Inverter as linhas para corresponder ao sistema de coordenadas da função draw_chess_board
    solvedBoard_for_drawing = solvedBoard[::-1]
    draw_chess_board(n, solvedBoard_for_drawing)
else:
    print("Nenhuma solução encontrada.")
