"""
Evolutionary algorithm for training neural network populations.
Implements selection, crossover, mutation, and elitism.
"""

import random
from typing import List, Tuple, Optional
import numpy as np
from neural_network import NeuralNetwork, crossover
import config


class Evolution:
    """
    Manages the evolutionary process across generations.
    """

    def __init__(self, population_size: int = config.POPULATION_SIZE):
        """
        Initialize the evolution system.
        
        Args:
            population_size: number of rockets per generation
        """
        self.population_size = population_size
        self.generation = 0
        
        # Network architecture
        self.layer_sizes = [
            config.INPUT_NEURONS,
            config.HIDDEN1_NEURONS,
            config.HIDDEN2_NEURONS,
            config.OUTPUT_NEURONS
        ]
        
        # Best network across all generations
        self.best_network_ever: Optional[NeuralNetwork] = None
        self.best_fitness_ever: float = float('-inf')
        
        # Adaptive mutation state
        self.plateau_boost_active: bool = False
    
    def create_initial_population(self) -> List[NeuralNetwork]:
        """
        Create initial population with random weights.
        
        Returns:
            list of neural networks with random weights
        """
        self.generation = 0
        return [NeuralNetwork(self.layer_sizes) for _ in range(self.population_size)]
    
    def evolve(self, networks: List[NeuralNetwork], 
               fitnesses: List[float]) -> List[NeuralNetwork]:
        """
        Create next generation from current population.
        
        Algorithm:
        1. Sort by fitness (descending)
        2. Keep elite networks unchanged
        3. Create rest through selection + crossover + mutation
        
        Args:
            networks: current generation's networks
            fitnesses: corresponding fitness scores
        
        Returns:
            new generation of networks
        """
        self.generation += 1
        
        # Mutation-strength annealing: decay over generations, floor at minimum
        effective_strength = max(
            config.MUTATION_STRENGTH_MIN,
            config.MUTATION_STRENGTH * (config.MUTATION_STRENGTH_DECAY ** self.generation)
        )
        
        # Plateau-adaptive boost: temporarily increase mutation
        if self.plateau_boost_active:
            effective_strength = min(config.MUTATION_STRENGTH,
                                     effective_strength * 1.5)
        
        # Pair networks with their fitness
        population = list(zip(networks, fitnesses))
        population.sort(key=lambda x: x[1], reverse=True)
        
        # Always update best ever (unconditionally)
        if population[0][1] > self.best_fitness_ever:
            self.best_fitness_ever = population[0][1]
            self.best_network_ever = population[0][0].copy()
        
        # Number of elite networks to keep unchanged
        num_elite = max(1, int(self.population_size * config.ELITE_PERCENTAGE))
        elite = [net.copy() for net, _ in population[:num_elite]]
        
        # Hall-of-fame: always inject best-ever as first elite (untouched)
        if self.best_network_ever is not None:
            elite.insert(0, self.best_network_ever.copy())
        
        # Create new generation
        new_networks: List[NeuralNetwork] = []
        
        # Add elite (unchanged)
        new_networks.extend(elite)
        
        # Tournament selection for the rest
        tournament_size = 5
        
        while len(new_networks) < self.population_size:
            if config.ENABLE_CROSSOVER and random.random() < config.CROSSOVER_RATE:
                # Crossover between two tournament winners
                parent1 = self._tournament_select(population, tournament_size)
                parent2 = self._tournament_select(population, tournament_size)
                child = crossover(parent1, parent2)
            else:
                # Clone a tournament winner
                parent = self._tournament_select(population, tournament_size)
                child = parent.copy()
            
            # Mutate the child
            child.mutate(config.MUTATION_RATE, effective_strength)
            
            new_networks.append(child)
        
        return new_networks
    
    def _tournament_select(self, population: List[Tuple[NeuralNetwork, float]],
                           tournament_size: int) -> NeuralNetwork:
        """
        Select a parent through tournament selection.
        
        Args:
            population: list of (network, fitness) tuples
            tournament_size: number of contestants
        
        Returns:
            winning network
        """
        contestants = random.sample(population, tournament_size)
        winner = max(contestants, key=lambda x: x[1])
        return winner[0]
    
    def get_statistics(self, fitnesses: List[float], 
                       num_reached: int) -> dict:
        """
        Calculate generation statistics.
        
        Args:
            fitnesses: list of fitness values
            num_reached: number of rockets that reached target
        
        Returns:
            dictionary of statistics
        """
        fitnesses = np.array(fitnesses)
        
        return {
            "generation": self.generation,
            "population": self.population_size,
            "best_fitness": float(np.max(fitnesses)),
            "avg_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "worst_fitness": float(np.min(fitnesses)),
            "success_rate": num_reached / self.population_size * 100,
            "num_reached": num_reached,
            "best_ever": self.best_fitness_ever
        }
    
    def get_top_networks(self, networks: List[NeuralNetwork],
                         fitnesses: List[float], 
                         n: int = 3) -> List[NeuralNetwork]:
        """
        Get the top N networks from the current generation.
        
        Args:
            networks: current generation's networks
            fitnesses: corresponding fitness scores
            n: number of top networks to return
        
        Returns:
            list of top N networks
        """
        population = list(zip(networks, fitnesses))
        population.sort(key=lambda x: x[1], reverse=True)
        return [net for net, _ in population[:n]]
    
    def set_plateau_boost(self, active: bool) -> None:
        """Enable or disable plateau-adaptive mutation boost."""
        self.plateau_boost_active = active
