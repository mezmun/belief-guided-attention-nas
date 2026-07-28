from __future__ import division
import numpy as np
from scipy.stats import rankdata


class Selection(object):

    def RouletteSelection(self, _a, k):
        a = np.asarray(_a)
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(float)
        selected_index = []
        for i in range(k):
            u = np.random.rand()*sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ +=sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index
        
    def WheelSelection(self, _fitness_values, k):
        fitness_values = np.asarray(_fitness_values)
        total_fitness = np.sum(fitness_values).astype(float)
        indi_probs = [fitness/total_fitness for fitness in fitness_values]
        idx_list = np.arange(len(fitness_values))
        return list(np.random.choice(idx_list,k, p=indi_probs))
    @staticmethod
    def GetGeometricPseudoFitness(fitness_list, generation, total_generations, q_start=0.99, q_end=0.96):
        """
        Computes geometric pseudo-fitness scores based on ranked original fitness values.
        Higher fitness receives higher pseudo-fitness.
    
        Parameters:
            fitness_list (list): List of original fitness scores
            generation (int): Current generation number
            total_generations (int): Total number of generations
            q_start (float): Initial q value (default: 0.99)
            q_end (float): Final q value (default: 0.96)
    
        Returns:
            list: Pseudo-fitness values corresponding to original fitness scores
        """
        #print(f"generation = {generation}, type = {type(generation)}")
        #generation = int(generation[0]) if isinstance(generation, list) else int(generation)

        #generation = int(generation)

        #if total_generations == 0:
        #    raise ValueError("total_generations must be greater than 0")
            
        slope = (q_end - q_start) / total_generations
        q = q_start + slope * generation
        q = max(min(q, q_start), q_end)
    
        # High fitness → low rank → high pseudo-fitness
        ranks = rankdata([-f for f in fitness_list], method='ordinal')
        pseudo_fitness = np.array([q ** (r - 1) for r in ranks])
    
        # Print summary
        max_idx = np.argmax(pseudo_fitness)
        min_idx = np.argmin(pseudo_fitness)
    
        print(f"[PseudoFitness] Gen {generation}/{total_generations} | q: {round(q, 4)} | "
              f"Max: {round(pseudo_fitness[max_idx], 5)} (fit: {fitness_list[max_idx]}) | "
              f"Min: {round(pseudo_fitness[min_idx], 5)} (fit: {fitness_list[min_idx]}) | "
              f"Ratio: {round(pseudo_fitness[max_idx] / pseudo_fitness[min_idx], 2)}")
    
        return pseudo_fitness.tolist()
if __name__ == '__main__':
    s = Selection()
    a = [1, 3, 2, 1, 4, 4, 5]
    selected_index = s.RouletteSelection(a, k=20)

    new_a =[a[i] for i in selected_index]
    print(list(np.asarray(a)[selected_index]))
    print(new_a)






