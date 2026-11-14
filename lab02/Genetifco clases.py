import random

class QueenIndividual:
    def __init__(self, size=8):
        self.size = size
        self.genes = [random.randint(1, size) for _ in range(size)]
        self.max_fitness = 28
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        horizontal_collisions = sum([self.genes.count(queen)-1 for queen in self.genes]) / 2
        diagonal_collisions = 0

        left_diagonal = [0] * (2 * self.size)
        right_diagonal = [0] * (2 * self.size)

        for i in range(self.size):
            left_diagonal[i + self.genes[i] - 1] += 1
            right_diagonal[self.size - i + self.genes[i] - 2] += 1

        for i in range(2 * self.size - 1):
            if left_diagonal[i] > 1:
                diagonal_collisions += (left_diagonal[i] - 1)
            if right_diagonal[i] > 1:
                diagonal_collisions += (right_diagonal[i] - 1)

        return int(self.max_fitness - (horizontal_collisions + diagonal_collisions))

    def mutate(self):
        pos = random.randint(0, self.size - 1)
        value = random.randint(1, self.size)
        self.genes[pos] = value
        self.fitness = self.calculate_fitness()

    def crossover(self, other):
        cut = random.randint(0, self.size - 1)
        child_genes = self.genes[:cut] + other.genes[cut:]
        child = QueenIndividual(self.size)
        child.genes = child_genes
        child.fitness = child.calculate_fitness()
        return child

    def __str__(self):
        prob = self.fitness / self.max_fitness
        return f"{self.genes}, fitness = {self.fitness}, probability = {prob:.6f}"


class GeneticAlgorithm:
    def __init__(self, population_size=10, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [QueenIndividual() for _ in range(population_size)]
        self.generation = 1

    def run(self):
        while not any(ind.fitness == 28 for ind in self.population):
            print(f"=== Generation {self.generation} ===")
            self.population = self.evolve()
            max_fit = max(ind.fitness for ind in self.population)
            print(f"Maximum fitness = {max_fit}")
            self.generation += 1

        print(f"Solved in Generation {self.generation - 1}!")
        for ind in self.population:
            if ind.fitness == 28:
                print(ind)

    def evolve(self):
        new_population = []
        probabilities = [ind.fitness / 28 for ind in self.population]

        for _ in range(self.population_size):
            parent1 = self.select_parent(probabilities)
            parent2 = self.select_parent(probabilities)
            child = parent1.crossover(parent2)

            if random.random() < self.mutation_rate:
                child.mutate()

            print(child)
            new_population.append(child)

            if child.fitness == 28:
                break

        return new_population

    def select_parent(self, probabilities):
        total = sum(probabilities)
        pick = random.uniform(0, total)
        current = 0
        for ind, prob in zip(self.population, probabilities):
            current += prob
            if current >= pick:
                return ind
        return random.choice(self.population)  # fallback


if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.run()
