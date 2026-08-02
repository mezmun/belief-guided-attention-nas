"""
number of resnet/densenet/pool deneme
                    ----add resnet/densenet/pool
                    ----remove resnet/densenet/pool
properties of resnet/dense/pool
                    ----in_channel/out_channel of resnet
                    ----amount in one resnet
                    ----in_channel/out_channel of densenet
                    ----amount in one densenet
                    ----k of densenet
                    ----pooling type

firstly, three basic operations: add, remove, alter
secondly, the particular operation is chosen based on a probability
"""
import random
import numpy as np
import copy
from utils import StatusUpdateTool, Utils
import os
from genetic.population import get_inception_params #pulling the parameters of inception for reducing the code size


class CrossoverAndMutation(object):
    def __init__(self, prob_crossover, prob_mutation, _log, individuals, _params=None):
        # Initialize the CrossoverAndMutation class with crossover and mutation probabilities, logging, individuals, and additional parameters.
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.individuals = individuals
        self.params = _params  # Storing additional parameters if needed, such as the index for SXB and polynomial mutation.
        self.log = _log
        self.offspring = []
    
    def process(self):
        """Generate a unique offspring pool with an optional oversized target."""

        gen_no = self.params['gen_no']
        target_size = int(self.params.get('target_size', len(self.individuals)))
        minimum_required_size = int(
            self.params.get('minimum_required_size', min(target_size, len(self.individuals)))
        )
        excluded_architecture_ids = set(
            self.params.get('excluded_architecture_ids', ())
        )
        if target_size < 1:
            raise ValueError('target_size must be at least 1')
        if minimum_required_size < 1:
            raise ValueError('minimum_required_size must be at least 1')

        print("gen no in process =", gen_no)
        mutation_file = f'./populations/mutation_{gen_no:02d}.txt'
        if os.path.exists(mutation_file):
            self.log.info(f'[INFO] Existing mutation file found: {mutation_file}')
            pop = Utils.load_population('mutation', gen_no)
            if bool(self.params.get('legacy_mode', False)):
                self.offspring = pop.individuals
                return self.offspring

            restored_unique = {}
            for individual in pop.individuals:
                architecture_id = individual.uuid()[0]
                if architecture_id in excluded_architecture_ids:
                    continue
                restored_unique.setdefault(architecture_id, individual)

            if len(restored_unique) >= minimum_required_size:
                self.offspring = list(restored_unique.values())[:target_size]
                return self.offspring

            self.log.warn(
                'The stored mutation pool has only %d usable unknown architectures. '
                'A new candidate pool will be generated.' % len(restored_unique)
            )

        if bool(self.params.get('legacy_mode', False)):
            crossover = Crossover(self.individuals, self.prob_crossover, self.log)
            self.offspring = crossover.do_crossover()
            Utils.save_population_after_crossover(self.individuals_to_string(), gen_no)
            mutation = Mutation(self.offspring, self.prob_mutation, self.log)
            mutation.do_mutation()
            for index, individual in enumerate(self.offspring):
                individual.id = 'indi%02d%02d' % (gen_no, index)
            Utils.save_population_after_mutation(self.individuals_to_string(), gen_no)
            return self.offspring

        unique_offspring = {}
        raw_crossover_offspring = []
        parent_count = max(1, len(self.individuals))
        required_batches = int(np.ceil(target_size / parent_count))
        max_batches = max(required_batches * 5, required_batches + 2)

        for _ in range(max_batches):
            crossover = Crossover(self.individuals, self.prob_crossover, self.log)
            batch = crossover.do_crossover()
            raw_crossover_offspring.extend(copy.deepcopy(batch))

            mutation = Mutation(batch, self.prob_mutation, self.log)
            mutation.do_mutation()

            for individual in batch:
                architecture_id = individual.uuid()[0]
                if architecture_id in excluded_architecture_ids:
                    continue
                if architecture_id not in unique_offspring:
                    unique_offspring[architecture_id] = individual
                if len(unique_offspring) >= target_size:
                    break
            if len(unique_offspring) >= target_size:
                break

        # Save the full crossover pool before storing the final mutated candidates.
        self.offspring = raw_crossover_offspring
        Utils.save_population_after_crossover(self.individuals_to_string(), gen_no)

        self.offspring = list(unique_offspring.values())[:target_size]
        if len(self.offspring) < target_size:
            self.log.warn(
                'Only %d usable unique offspring could be generated for target size %d'
                % (len(self.offspring), target_size)
            )
        if len(self.offspring) < minimum_required_size:
            raise RuntimeError(
                'The guided candidate pool contains only %d unknown unique architectures, '
                'but at least %d are required. Increase candidate_multiplier or mutation diversity.'
                % (len(self.offspring), minimum_required_size)
            )

        for index, individual in enumerate(self.offspring):
            individual.id = f'indi{gen_no:02d}{index:03d}'

        Utils.save_population_after_mutation(self.individuals_to_string(), gen_no)
        return self.offspring
    """
    def process(self):
        # This method manages the overall process of crossover and mutation.
        # First, it performs crossover on the individuals to generate offspring.
        crossover = Crossover(self.individuals, self.prob_crossover, self.log)
        offspring = crossover.do_crossover()

        # Save the offspring generated by crossover.
        self.offspring = offspring
        Utils.save_population_after_crossover(self.individuals_to_string(), self.params['gen_no'])

        # Then, it performs mutation on the offspring.
        mutation = Mutation(self.offspring, self.prob_mutation, self.log)
        mutation.do_mutation()

        # Assign new IDs to the offspring after mutation and save the population.
        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%02d%02d'%(self.params['gen_no'], i)
            print("indi_no",indi_no)
            indi.id = indi_no

        Utils.save_population_after_mutation(self.individuals_to_string(), self.params['gen_no'])
        return offspring
    """
    
    def individuals_to_string(self):
        # Convert the individuals to a string representation for logging or saving.
        # This method creates a textual representation of each individual in the offspring list.
        _str = []
        for ind in self.offspring:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)


class Crossover(object):
    def __init__(self, individuals, prob_, _log):
        # Initialize the Crossover class with the individuals, probability of crossover, and logging.
        self.individuals = individuals
        self.prob = prob_
        self.log = _log
        self.pool_limit = StatusUpdateTool.get_pool_limit()[1]  # Set the limit for the number of pooling layers.

    def _choose_one_parent(self):
        # Choose one parent using a combination of binary tournament selection and roulette wheel selection.
        u_ = random.random()
        if u_ < 0.5:
            # Binary tournament selection: Choose two random individuals and return the one with higher accuracy.
            count_ = len(self.individuals)
            idx1 = int(np.floor(np.random.random() * count_))
            idx2 = int(np.floor(np.random.random() * count_))
            while idx2 == idx1:
                idx2 = int(np.floor(np.random.random() * count_))
            if self.individuals[idx1].acc > self.individuals[idx2].acc:
                return idx1
            else:
                return idx2
        else:
            # Roulette wheel selection: Assign probabilities based on fitness and select an individual.
            v_list = []
            for indi in self.individuals:
                v_list.append(indi.acc)
            fitness_values = np.asarray(v_list)
            print
            total_fitness = np.sum(fitness_values).astype(float)
            indi_probs = [fitness / total_fitness for fitness in fitness_values]
            idx_list = np.arange(len(fitness_values))
            idx = np.random.choice(idx_list, p=indi_probs)
            return idx

    def _choose_two_diff_parents(self):
        # Choose two different parents using the _choose_one_parent method.
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            # Ensure that the two chosen parents are different.
            idx2 = self._choose_one_parent()
        assert idx1 < len(self.individuals)
        assert idx2 < len(self.individuals)
        return idx1, idx2

    def _calculate_pool_numbers(self, parent1, parent2):
        # Calculate the number of pooling layers after crossover.
        t1, t2 = 0, 0
        for unit in parent1.units:
            # Count pooling layers in parent1
            if unit.type == 2: 
                t1 += 1
        for unit in parent2.units:
            # Count pooling layers in parent2
            if unit.type == 2: 
                t2 += 1

        # Determine the crossover points in each parent.
        len1, len2 = len(parent1.units), len(parent2.units)
        pos1, pos2 = int(np.floor(np.random.random() * len1)), int(np.floor(np.random.random() * len2))
        assert pos1 < len1
        assert pos2 < len2

        # Calculate the number of pooling layers to the left and right of the crossover points.
        p1_left, p1_right, p2_left, p2_right = 0, 0, 0, 0
        for i in range(0, pos1):
            if parent1.units[i].type == 2: 
                p1_left += 1
        for i in range(pos1, len1):
            if parent1.units[i].type == 2: 
                p1_right += 1

        for i in range(0, pos2):
            if parent2.units[i].type == 2: 
                p2_left += 1
        for i in range(pos2, len2):
            if parent2.units[i].type == 2: 
                p2_right += 1

        # Calculate the new pooling layer counts after the crossover.
        new_pool_number1 = p1_left + p2_right
        new_pool_number2 = p2_left + p1_right
        return pos1, pos2, new_pool_number1, new_pool_number2


    
    def do_crossover(self):
        # Perform the crossover operation between pairs of parents to generate offspring.
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []
        for _ in range(len(self.individuals) // 2):
            # Select two different parents for crossover.
            ind1, ind2 = self._choose_two_diff_parents()
    
            parent1, parent2 = copy.deepcopy(self.individuals[ind1]), copy.deepcopy(self.individuals[ind2])
            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_new'] += 2
    
                """
                Perform the actual crossover by exchanging units between the parents.
                The exchanged units must satisfy certain conditions:
                --- The number of pooling layers should not exceed the predefined limit.
                --- If there is no change after the crossover, the original accuracy is retained, and a mutation is applied.
                """
    
                first_begin_is_pool, second_begin_is_pool = True, True
                while first_begin_is_pool is True or second_begin_is_pool is True:
                    # Calculate the positions and pooling layer counts after the crossover.
                    pos1, pos2, pool_len1, pool_len2 = self._calculate_pool_numbers(parent1, parent2)
                    try_count = 1
                    while pool_len1 > self.pool_limit or pool_len2 > self.pool_limit:
                        # If the pooling layer limit is exceeded, try new crossover positions.
                        pos1, pos2, pool_len1, pool_len2 = self._calculate_pool_numbers(parent1, parent2)
                        try_count += 1
                        self.log.warn('The %d-th try to find the position for do crossover' % (try_count))
                    self.log.info('Position %d for %s, positions %d for %s' % (pos1, parent1.id, pos2, parent2.id))
                    unit_list1, unit_list2 = [], []
                    # Split and recombine the units between the two parents at the chosen positions.
                    for i in range(0, pos1):
                        unit_list1.append(parent1.units[i])
                    for i in range(pos2, len(parent2.units)):
                        unit_list1.append(parent2.units[i])
    
                    for i in range(0, pos2):
                        unit_list2.append(parent2.units[i])
                    for i in range(pos1, len(parent1.units)):
                        unit_list2.append(parent1.units[i])
    
                    # Check if the new individuals start with a pooling layer (which is not allowed).
                    first_begin_is_pool = True if unit_list1[0].type == 2 else False
                    second_begin_is_pool = True if unit_list2[0].type == 2 else False
    
                    if first_begin_is_pool is True:
                        self.log.warn('Crossovered individual#1 starts with a pooling layer, redo...')
                    if second_begin_is_pool is True:
                        self.log.warn('Crossovered individual#2 starts with a pooling layer, redo...')
    
                # Reorder the number of each unit based on its order in the list.
                for i, unit in enumerate(unit_list1):
                    unit.number = i
                for i, unit in enumerate(unit_list2):
                    unit.number = i
    
                # Re-adjust the in_channel of the next layer after the crossover.
                last_output_from_list1 = 0
                if pos1 == 0:
                    last_output_from_list1 = StatusUpdateTool.get_input_channel()
                    j = 0
                    i = -1
                else:
                    for i in range(pos1 - 1, -1, -1):
                        if unit_list1[i].type in [1, 3, 8, 9, 10, 11, 12, 13, 15, 16]: #resnet & variants, densenet & variants
                            last_output_from_list1 = unit_list1[i].out_channel
                            break
                        elif unit_list1[i].type in [4, 5, 6, 7, 14]:  # modules that contain inception module
                            # Calculate the output channel for inception based modules.
                            inception_out = unit_list1[i].out_1x1 + unit_list1[i].out_3x3 + unit_list1[i].out_5x5 + unit_list1[i].out_1x1pool
                            print("last_output_from_list1", inception_out)
                            last_output_from_list1 = inception_out
                            break
    
                keep_out_channel = last_output_from_list1
    
                for j in range(pos1, len(unit_list1)):
                    # Update the input channel of subsequent units based on the output channel of the previous unit.
                    if unit_list1[j].type in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]: # except pool
                        self.log.info('Change the input channel of unit at %d to %d that is the output channel of unit at %d in %s' % (j, keep_out_channel, i, parent1.id))
                        unit_list1[j].in_channel = keep_out_channel
                        if unit_list1[j].type in [1, 4, 5, 6, 7, 8, 9, 10, 14, 15]: #Resnet, inception and their variants
                            break
                        elif unit_list1[j].type in [3, 11, 12, 13, 16]: #Densenet and variants
                          



                          
                          
                          # Calculate the estimated output channel for DenseNet units.
                            effective_input_channel = min(
                                unit_list1[j].in_channel,
                                unit_list1[j].max_input_channel
                            )

                            estimated_out_channel = (
                                effective_input_channel
                                + unit_list1[j].k * unit_list1[j].amount
                            )

                            # DenseNet output is unchanged, so downstream channels
                            # do not need to be recalculated.
                            if estimated_out_channel == unit_list1[j].out_channel:
                                keep_out_channel = estimated_out_channel
                                continue

                            self.log.info(
                                'Due to the above change, unit at %d '
                                'changes its output channel from %d to %d'
                                % (
                                    j,
                                    unit_list1[j].out_channel,
                                    estimated_out_channel
                                )
                            )

                            unit_list1[j].out_channel = estimated_out_channel
                            keep_out_channel = estimated_out_channel





              
                last_output_from_list2 = 0
                if pos2 == 0:
                    last_output_from_list2 = StatusUpdateTool.get_input_channel()
                    j = 0
                    i = -1
                else:
                    for i in range(pos2 - 1, -1, -1):
                        if unit_list2[i].type in [1, 3, 8, 9, 10, 11, 12, 13, 15, 16]: # resnet, densenet and their variants
                            last_output_from_list2 = unit_list2[i].out_channel
                            break
                        elif unit_list2[i].type in [4, 5, 6, 7, 14]: # inception and its variants
                            # Calculate the output channel for inception, SE-inception, CBAM-inception modules.
                            inception_out = unit_list2[i].out_1x1 + unit_list2[i].out_3x3 + unit_list2[i].out_5x5 + unit_list2[i].out_1x1pool
                            last_output_from_list2 = inception_out
                            print("last_output_from_list2", inception_out)
                            break
    
                keep_out_channel = last_output_from_list2
                for j in range(pos2, len(unit_list2)):
                    # Update the input channel of subsequent units based on the output channel of the previous unit.
                    if unit_list2[j].type in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                        self.log.info('Change the input channel of unit at %d to %d that is the output channel of unit at %d in %s' % (j, keep_out_channel, i, parent2.id))
                        unit_list2[j].in_channel = keep_out_channel
                        if unit_list2[j].type in [1, 4, 5, 6, 7, 8, 9, 10, 14, 15]: # inception, resnet and their variants
                            break
                        elif unit_list2[j].type in [3, 11, 12, 13, 16]: # densenet and its variants
                            # Calculate the estimated output channel for DenseNet units.
                            effective_input_channel = min(
                                unit_list2[j].in_channel,
                                unit_list2[j].max_input_channel
                            )

                            estimated_out_channel = (
                                effective_input_channel
                                + unit_list2[j].k * unit_list2[j].amount
                            )

                            # DenseNet output is unchanged, so downstream channels
                            # do not need to be recalculated.
                            if estimated_out_channel == unit_list2[j].out_channel:
                                keep_out_channel = estimated_out_channel
                                continue

                            self.log.info(
                                'Due to the above change, unit at %d '
                                'changes its output channel from %d to %d'
                                % (
                                    j,
                                    unit_list2[j].out_channel,
                                    estimated_out_channel
                                )
                            )

                            unit_list2[j].out_channel = estimated_out_channel
                            keep_out_channel = estimated_out_channel
    
                # Assign the newly formed units back to the parents and reset their accuracy.
                parent1.units = unit_list1
                parent2.units = unit_list2
                offspring1, offspring2 = parent1, parent2
                offspring1.reset_acc()
                offspring2.reset_acc()
                new_offspring_list.append(offspring1)
                new_offspring_list.append(offspring2)
            else:
                # If crossover does not occur, retain the original parents as offspring.
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)
    
        # Log the results of the crossover operation and return the new offspring list.
        self.log.info('CROSSOVER-%d offspring are generated, new:%d, others:%d' % (len(new_offspring_list), _stat_param['offspring_new'], _stat_param['offspring_from_parent']))
        return new_offspring_list
    

class Mutation(object):

    def __init__(self, individuals, prob_, _log):
        # Initialize the Mutation class with the individuals, probability of mutation, and logging.
        self.individuals = individuals
        self.prob = prob_
        self.log = _log



    def do_mutation(self):
        # In this part, the type of mutation to be performed is determined based on the mutation rate, and the necessary function for the mutation is called.
        # Mutation type functions: do_add_unit_mutation, do_remove_unit_mutation and do_alter_mutation
        # Perform the mutation operation on the offspring individuals. There is no mutation for the modules that have inception
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0, 'ADD': 0, 'REMOVE': 0, 'ALTER': 0, 'RESNET_OUT_CHANNEL': 0, 'RESNET_AMOUNT': 0, 'DENSENET_AMOUNT': 0, 'POOLING_TYPE': 0}

        mutation_list = StatusUpdateTool.get_mutation_probs_for_each()
        for indi in self.individuals:
            p_ = random.random()
            if p_ < self.prob:
                # If the mutation probability condition is met, perform one of the mutation operations (add, remove, alter).
                _stat_param['offspring_new'] += 1
                mutation_type = self.select_mutation_type(mutation_list)
                if mutation_type == 0:
                    _stat_param['ADD'] += 1
                    self.do_add_unit_mutation(indi)
                elif mutation_type == 1:
                    _stat_param['REMOVE'] += 1
                    self.do_remove_unit_mutation(indi)
                elif mutation_type == 2:
                    mutation_p_type, mutation_p_count = self.do_alter_mutation(indi)
                    _stat_param[mutation_p_type] = mutation_p_count + _stat_param[mutation_p_type]
                    _stat_param['ALTER'] += mutation_p_count
                    if mutation_p_count == 0:
                        # If no changes occurred, adjust the offspring statistics.
                        _stat_param['offspring_new'] -= 1
                        _stat_param['offspring_from_parent'] += 1
                else:
                    raise TypeError('Error mutation type :%d, validate range:0-4' % (mutation_type))
            else:
                # If the mutation probability condition is not met, retain the individual without changes.
                _stat_param['offspring_from_parent'] += 1
        # Log the results of the mutation operation.
        self.log.info('MUTATION-mutated individuals:%d[ADD:%2d,REMOVE:%2d,ALTER:%2d,RESNET_OUT_CHANNEL:%2d, RESNET_AMOUNT:%2d, DENSENET_AMOUNT:%2d, POOLING_TYPE:%2d, no_changes:%d' % (
        _stat_param['offspring_new'], _stat_param['ADD'], _stat_param['REMOVE'], _stat_param['ALTER'], _stat_param['RESNET_OUT_CHANNEL'], _stat_param['RESNET_AMOUNT'], _stat_param['DENSENET_AMOUNT'], _stat_param['POOLING_TYPE'], _stat_param['offspring_from_parent']))



  
    def do_add_unit_mutation(self, indi):
        # Perform the mutation operation to add a new unit to an individual.
        self.log.info('Do the ADD mutation for indi:%s' % (indi.id))
        mutation_position = int(np.floor(np.random.random() * len(indi.units)))
        self.log.info('Mutation position occurs at %d' % (mutation_position))

        # Manuel olarak oluşturulmuş liste (type numaraları)
        #type_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        type_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        
        # Rastgele seçim
        type_ = random.choice(type_list)


        """
        u_ = random.random()
        # Unit type selection including new attention modules
        if u_ < 1/13:
            type_ = 1  # Add a ResNet unit.
        elif u_ < 2/13:
            type_ = 2  # Add a Pooling unit.
        elif u_ < 3/13:
            type_ = 3  # Add a DenseNet unit.
        elif u_ < 4/13:
            type_ = 4  # Add an Inception unit.
        elif u_ < 5/13:
            type_ = 5  # Add an SE-Inception unit.
        elif u_ < 6/13:
            type_ = 6  # Add a CBAM-Inception unit.
        elif u_ < 7/13:
            type_ = 7  # Add a CA-Inception unit.
        elif u_ < 8/13:
            type_ = 8  # Add an SE-ResNet unit.
        elif u_ < 9/13:
            type_ = 9  # Add a CBAM-ResNet unit.
        elif u_ < 10/13:
            type_ = 10  # Add a CA-ResNet unit.
        elif u_ < 11/13:
            type_ = 11  # Add an SE-DenseNet unit.
        elif u_ < 12/13:
            type_ = 12  # Add a CBAM-DenseNet unit.
        else:
            type_ = 13  # Add a CA-DenseNet unit.
        """

      
        # Unit type strings for logging
        type_string_list = ['RESNET', 'POOLING', 'DENSENET', 'INCEPTION', 'SE-INCEPTION', 'CBAM-INCEPTION', 'CA-INCEPTION', 
                            'SE-RESNET', 'CBAM-RESNET', 'CA-RESNET', 'SE-DENSENET', 'CBAM-DENSENET', 'CA-DENSENET', 'ECA-INCEPTION', 'ECA-RESNET', 'ECA-DENSENET']

        #self.log.info('A %s unit would be added due to the probability of %.2f' % (type_string_list[type_ - 1], u_))
        self.log.info('A %s unit was added by random selection.' % (type_string_list[type_ - 1]))

    
        # Pooling unit check
        if type_ == 2:
            num_exist_pool_units = sum(1 for unit in indi.units if unit.type == 2)


            """
            if num_exist_pool_units > StatusUpdateTool.get_pool_limit()[1] - 1:
                # If the limit for pooling layers is exceeded, randomly choose another type.
                u_ = random.random()
                if u_ < 1/12:
                    type_ = 1  # ResNet
                elif u_ < 2/12:
                    type_ = 3  # DenseNet
                elif u_ < 3/12:
                    type_ = 4  # Inception
                elif u_ < 4/12:
                    type_ = 5  # SE-Inception
                elif u_ < 5/12:
                    type_ = 6  # CBAM-Inception
                elif u_ < 6/12:
                    type_ = 7  # CA-Inception
                elif u_ < 7/12:
                    type_ = 8  # SE-ResNet
                elif u_ < 8/12:
                    type_ = 9  # CBAM-ResNet
                elif u_ < 9/12:
                    type_ = 10  # CA-ResNet
                elif u_ < 10/12:
                    type_ = 11  # SE-DenseNet
                elif u_ < 11/12:
                    type_ = 12  # CBAM-DenseNet
                else:
                    type_ = 13  # CA-DenseNet
                self.log.info('The added unit is changed to %s because the existing number of POOLING exceeds %d, limit size:%d' % (
                    'RESNET' if type_ == 1 else 'DENSENET' if type_ == 3 else 'INCEPTION' if type_ == 4 else 
                    'SE-INCEPTION' if type_ == 5 else 'CBAM-INCEPTION' if type_ == 6 else 'CA-INCEPTION' if type_ == 7 else
                    'SE-RESNET' if type_ == 8 else 'CBAM-RESNET' if type_ == 9 else 'CA-RESNET' if type_ == 10 else
                    'SE-DENSENET' if type_ == 11 else 'CBAM-DENSENET' if type_ == 12 else 'CA-DENSENET',
                    num_exist_pool_units, StatusUpdateTool.get_pool_limit()[1]))
    
        """

            if num_exist_pool_units >= StatusUpdateTool.get_pool_limit()[1]:
                # Select a non-pooling unit when the pooling limit is already full.
                #valid_types = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                valid_types = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                type_names = {
                    1: 'RESNET',
                    3: 'DENSENET',
                    4: 'INCEPTION',
                    5: 'SE-INCEPTION',
                    6: 'CBAM-INCEPTION',
                    7: 'CA-INCEPTION',
                    8: 'SE-RESNET',
                    9: 'CBAM-RESNET',
                    10: 'CA-RESNET',
                    11: 'SE-DENSENET',
                    12: 'CBAM-DENSENET',
                    13: 'CA-DENSENET',
                    14: 'ECA-INCEPTION',
                    15: 'ECA-RESNET',
                    16: 'ECA-DENSENET',
                }

                type_ = random.choice(valid_types)
                type_name = type_names[type_]

                self.log.info(
                    'The added unit is changed to %s because the pooling limit is full. '
                    'Existing pooling units: %d, limit: %d' %
                    (type_name, num_exist_pool_units, StatusUpdateTool.get_pool_limit()[1])
                )
        
            
    


        #do the details
        if type_ == 2:
            add_unit = indi.init_a_pool(mutation_position + 1, _max_or_avg=None)
        else:
            for i in range(mutation_position, -1, -1):
                # Determine the appropriate input channel for the new unit by looking back at previous units.
                if indi.units[i].type in [1, 3, 8, 9, 10, 11, 12, 13, 15, 16]:
                #if indi.units[i].type == 1 or indi.units[i].type == 3 or indi.units[i].type == 8 or indi.units[i].type == 9 or indi.units[i].type == 10 or indi.units[i].type == 11 or indi.units[i].type == 12 or indi.units[i].type == 13:
                    _in_channel = indi.units[i].out_channel
                    break
                elif indi.units[i].type in [4, 5, 6, 7, 14]:
                #elif indi.units[i].type == 4 or indi.units[i].type == 5 or indi.units[i].type == 6 or indi.units[i].type == 7:
                    # For Inception based modules, sum the output channels from all branches.
                    inception_out = indi.units[i].out_1x1 + indi.units[i].out_3x3 + indi.units[i].out_5x5 + indi.units[i].out_1x1pool
                    _in_channel = inception_out
                    break
            if type_ == 1:
                add_unit = indi.init_a_resnet(mutation_position + 1, _amount=None, _in_channel=_in_channel, _out_channel=None)
                keep_out_channel = add_unit.out_channel
            if type_ == 3:
                add_unit = indi.init_a_densenet(mutation_position + 1, _amount=None, _k=None, _max_input_channel=None, _in_channel=_in_channel)
                keep_out_channel = add_unit.out_channel
            if type_ == 4:
                # Randomly choose an inception type and initialize the corresponding Inception module.
                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                if chosen_type == "3a":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32)
                elif chosen_type == "3b":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64)
                elif chosen_type == "4a":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64)
                elif chosen_type == "4b":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4c":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4d":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4e":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
                elif chosen_type == "5a":
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
                else:
                    add_unit = indi.init_an_inception(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128)

                # Sum the outputs of the branches to determine the output channel.
                inception_out = add_unit.out_1x1 + add_unit.out_3x3 + add_unit.out_5x5 + add_unit.out_1x1pool
                keep_out_channel = inception_out



            ################## these parts added newly###########################3
            reduction_ratio_list = [16]
            if type_ == 5:# SE-Inception unit
                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)

                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)

                if chosen_type == "3a":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32, reduction_ratio=reduction_ratio)
                elif chosen_type == "3b":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4a":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4b":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4c":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4d":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4e":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                elif chosen_type == "5a":
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                else:
                    add_unit = indi.init_an_inception_se(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)

                inception_out_channels = add_unit.out_1x1 + add_unit.out_3x3 + add_unit.out_5x5 + add_unit.out_1x1pool
                keep_out_channel = inception_out_channels 

            
            if type_ == 6:  # CBAM-Inception unit
                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)

                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
            
                if chosen_type == "3a":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32, reduction_ratio=reduction_ratio)
                elif chosen_type == "3b":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4a":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4b":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4c":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4d":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4e":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                elif chosen_type == "5a":
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                else:
                    add_unit = indi.init_an_inception_cbam(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
            
                inception_out_channels = add_unit.out_1x1 + add_unit.out_3x3 + add_unit.out_5x5 + add_unit.out_1x1pool
                keep_out_channel = inception_out_channels

            
            if type_ == 7:  # For CA-Inception module
                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)

                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
            
                if chosen_type == "3a":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32, reduction_ratio=reduction_ratio)
                elif chosen_type == "3b":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4a":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4b":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4c":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4d":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4e":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                elif chosen_type == "5a":
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                else:
                    add_unit = indi.init_an_inception_ca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
            
                inception_out_channels = add_unit.out_1x1 + add_unit.out_3x3 + add_unit.out_5x5 + add_unit.out_1x1pool
                keep_out_channel = inception_out_channels  # CA block doesn't change channel count
            
            
            if type_ == 8:  # SE-ResNet
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
              
                add_unit = indi.init_a_resnet_se(mutation_position + 1, _amount=None, _in_channel=_in_channel, _out_channel=None, reduction_ratio=reduction_ratio)
                keep_out_channel = add_unit.out_channel
            
            if type_ == 9:  # CBAM-ResNet
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
              
                add_unit = indi.init_a_resnet_cbam(mutation_position + 1, _amount=None, _in_channel=_in_channel, _out_channel=None, reduction_ratio=reduction_ratio)
                keep_out_channel = add_unit.out_channel
            
            if type_ == 10:  # CA-ResNet
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
              
                add_unit = indi.init_a_resnet_ca(mutation_position + 1, _amount=None, _in_channel=_in_channel, _out_channel=None, reduction_ratio=reduction_ratio)
                keep_out_channel = add_unit.out_channel
            
            if type_ == 11:  # SE-DenseNet
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
              
                add_unit = indi.init_a_densenet_se(mutation_position + 1, _amount=None, _k=None, _max_input_channel=None, _in_channel=_in_channel, reduction_ratio=reduction_ratio)
                keep_out_channel = add_unit.out_channel
            
            if type_ == 12:  # CBAM-DenseNet
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
              
                add_unit = indi.init_a_densenet_cbam(mutation_position + 1, _amount=None, _k=None, _max_input_channel=None, _in_channel=_in_channel, reduction_ratio=reduction_ratio)
                keep_out_channel = add_unit.out_channel
            
            if type_ == 13:  # CA-DenseNet
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
              
                add_unit = indi.init_a_densenet_ca(mutation_position + 1, _amount=None, _k=None, _max_input_channel=None, _in_channel=_in_channel, reduction_ratio=reduction_ratio)
                keep_out_channel = add_unit.out_channel

            ################### FOR ECA MODULE ######################
            
            if type_ == 14:  # ECA-Inception

                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool = get_inception_params(chosen_type)
                k_size = 3
              
                add_unit = indi.init_an_inception_eca(mutation_position + 1, in_channels=_in_channel, inception_type=chosen_type, out_1x1=out_1x1, red_3x3=red_3x3, out_3x3=out_3x3, red_5x5=red_5x5, out_5x5=out_5x5, out_1x1pool=out_1x1pool, k_size=k_size)
            
                # Ortak çıkış kanalı güncellemesi
                inception_out_channels = out_1x1 + out_3x3 + out_5x5 + out_1x1pool
                keep_out_channel = inception_out_channels
          
            if type_ == 15:  # ECA-ResNet
                
                k_size = 3
                add_unit = indi.init_a_resnet_eca(mutation_position + 1, _amount=None, _in_channel=_in_channel, _out_channel=None, k_size=k_size)
                keep_out_channel = add_unit.out_channel
  

            if type_ == 16:  # ECA-DenseNet
                k_size = 3
                add_unit = indi.init_a_densenet_eca(mutation_position + 1, _amount=None, _k=None, _max_input_channel=None, _in_channel=_in_channel, k_size=k_size)
                keep_out_channel = add_unit.out_channel

          
            # Update the input channels of subsequent units based on the new unit's output channel.
            for i in range(mutation_position + 1, len(indi.units)):
                if (indi.units[i].type in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):
                    self.log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (i, indi.units[i].in_channel, keep_out_channel))
                    indi.units[i].in_channel = keep_out_channel
                    
                    # Check conditions for break based on unit type
                    if indi.units[i].type in [1, 4, 5, 6, 7, 8, 9, 10, 14, 15]: # resnet & inception and their variants
                        break
                    elif indi.units[i].type in [3, 11, 12, 13, 16]:  # DenseNet and its variants
                        # Update DenseNet unit output channels.
                        effective_input_channel = min(
                            indi.units[i].in_channel,
                            indi.units[i].max_input_channel
                        )

                        estimated_out_channel = (
                            effective_input_channel
                            + indi.units[i].k * indi.units[i].amount
                        )

                        # DenseNet output is unchanged, so downstream channels
                        # do not need to be recalculated.
                        if estimated_out_channel == indi.units[i].out_channel:
                            keep_out_channel = estimated_out_channel
                            continue

                        self.log.info(
                            'Due to the above mutation, unit at %d '
                            'changes its output channel from %d to %d'
                            % (
                                i,
                                indi.units[i].out_channel,
                                estimated_out_channel
                            )
                        )

                        indi.units[i].out_channel = estimated_out_channel
                        keep_out_channel = estimated_out_channel
                        
            
        # Insert the new unit into the individual's unit list and adjust the subsequent unit numbers.
        new_unit_list = []
        for i in range(mutation_position + 1):
            new_unit_list.append(indi.units[i])
        new_unit_list.append(add_unit)
        for i in range(mutation_position + 1, len(indi.units)):
            unit = indi.units[i]
            unit.number += 1
            new_unit_list.append(unit)
        indi.number_id += 1
        indi.units = new_unit_list
        indi.reset_acc()


    
    def do_remove_unit_mutation(self, indi):
        # Perform the mutation operation to remove a unit from an individual.
        self.log.info('Do the REMOVE mutation for indi:%s' % (indi.id))
        
        if len(indi.units) > 1:
            mutation_position = int(np.floor(np.random.random() * (len(indi.units) - 1))) + 1  # The first unit would not be removed.
            self.log.info('Mutation position occurs at %d' % (mutation_position))
            
            if indi.units[mutation_position].type in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:  # Include all ResNet, DenseNet, Inception, and attention variants
                # Adjust input channels of subsequent units based on the unit to be removed.
                keep_out_channel = indi.units[mutation_position].in_channel
                for i in range(mutation_position + 1, len(indi.units)):
                    if indi.units[i].type in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                        self.log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (i, indi.units[i].in_channel, keep_out_channel))
                        indi.units[i].in_channel = keep_out_channel
                        if indi.units[i].type in [1, 4, 5, 6, 7, 8, 9, 10, 14, 15]:  # ResNet, Inception, SE-Inception, CBAM-Inception, CA-Inception and Resnet variants
                            break
                        elif indi.units[i].type in [3, 11, 12, 13, 16]:  # DenseNet, SE-DenseNet, CBAM-DenseNet, CA-DenseNet
                            # Update DenseNet or its attention variants unit output channels.
                            effective_input_channel = min(
                                indi.units[i].in_channel,
                                indi.units[i].max_input_channel
                            )

                            estimated_out_channel = (
                                effective_input_channel
                                + indi.units[i].k * indi.units[i].amount
                            )

                            # DenseNet output is unchanged, so downstream channels
                            # do not need to be recalculated.
                            if estimated_out_channel == indi.units[i].out_channel:
                                keep_out_channel = estimated_out_channel
                                continue

                            self.log.info(
                                'Due to the above mutation, unit at %d '
                                'changes its output channel from %d to %d'
                                % (
                                    i,
                                    indi.units[i].out_channel,
                                    estimated_out_channel
                                )
                            )

                            indi.units[i].out_channel = estimated_out_channel
                            keep_out_channel = estimated_out_channel
                    else:
                        break
    
            else:
                # Remove a pooling unit.
                self.log.info('A POOLING at %d is removed' % (mutation_position))
            
            # Adjust the unit numbers after removing the unit.
            new_unit_list = []
            for i in range(mutation_position):
                new_unit_list.append(indi.units[i])
            for i in range(mutation_position + 1, len(indi.units)):
                unit = indi.units[i]
                unit.number -= 1
                new_unit_list.append(unit)
            
            indi.number_id -= 1
            indi.units = new_unit_list
            indi.reset_acc()
    
        else:
            # If there is only one unit left, removal is not allowed.
            self.log.warn('REMOVE mutation can not be performed due to it has only one unit')
    
    
        
    
    def do_alter_mutation(self, indi):
        """
        Perform mutation operations to alter properties of a unit within an individual.
        Possible alterations include:
        - Changing the output channel of a ResNet unit
        - Modifying the number of blocks in a ResNet unit
        - Adjusting the number of blocks in a DenseNet unit
        - Changing the type of pooling in a Pooling unit
        - Alterations for attention-enhanced ResNet and DenseNet variants
        """
        
        self.log.info('Do the ALTER mutation for indi:%s' % (indi.id))
        mutation_position = int(np.floor(np.random.random() * len(indi.units)))  # Randomly select a unit to mutate
        mutation_unit = indi.units[mutation_position]
        
        # Avoid altering Inception and its variants (SE, CBAM, CA) during this operation
        while mutation_unit.type in [4, 5, 6, 7, 14]:
            mutation_position = int(np.floor(np.random.random() * len(indi.units)))
            mutation_unit = indi.units[mutation_position]
        
        mutation_unit_name = ''
        if mutation_unit.type in [1, 8, 9, 10, 15]:  # ResNet and its variants
            mutation_unit_name = 'RESNET or its variants'
        elif mutation_unit.type == 2:
            mutation_unit_name = 'POOLING'
        elif mutation_unit.type in [3, 11, 12, 13, 16]:  # DenseNet and its variants
            mutation_unit_name = 'DENSENET or its variants'
        else:
            mutation_unit_name = 'INCEPTION NOT IMPLEMENTED'
    
        self.log.info('Do the %s mutation for indi:%s at position %d' % (mutation_unit_name, indi.id, mutation_position))
    
        mutation_p_type = ''
        mutation_p_count = ''
        
        if mutation_unit.type in [1, 8, 9, 10, 15]:  # ResNet and its variants
            mutation_p_type, mutation_p_count = self.do_alter_resnet_mutation(mutation_position, indi)
        elif mutation_unit.type == 2:
            mutation_p_type, mutation_p_count = self.do_alter_pooling_mutation(mutation_position, indi)
        elif mutation_unit.type in [3, 11, 12, 13, 16]:  # DenseNet and its variants
            mutation_p_type, mutation_p_count = self.do_alter_densenet_mutation(mutation_position, indi)
    
        return mutation_p_type, mutation_p_count
    
    
    
    def do_alter_resnet_mutation(self, position, indi):
        """
        This function alters properties of a ResNet unit or its variants (SE-ResNet, CBAM-ResNet, CA-ResNet).
        Possible mutations:
        - Change the output channel of the ResNet unit or its variants.
        - Adjust the number of residual blocks in the ResNet unit or its variants.
        """
        mutation_p_type = ''
        mutation_p_count = 0
    
        u_ = random.random()
        if u_ < 0.5:
            # Change the output channel of the ResNet unit or its variants.
            mutation_p_type = 'RESNET_OUT_CHANNEL'
            channel_list = StatusUpdateTool().get_output_channel()
            index_ = int(np.floor(np.random.random() * len(channel_list)))
            if indi.units[position].out_channel != channel_list[index_]:
                self.log.info('Unit at %d changes its output channel from %d to %d' % (
                    position, indi.units[position].out_channel, channel_list[index_]))
                indi.units[position].out_channel = channel_list[index_]
    
                keep_out_channel = channel_list[index_]
                for i in range(position + 1, len(indi.units)):
                    if indi.units[i].type in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:  # Include all ResNet, DenseNet, Inception, and attention variants
                        # Adjust the input channels of subsequent units after altering the output channel.
                        self.log.info('Due to above, the unit at %d should change its input channel from %d to %d' % (
                            i, indi.units[i].in_channel, keep_out_channel))
                        indi.units[i].in_channel = keep_out_channel
                        if indi.units[i].type in [1, 4, 5, 6, 7, 8, 9, 10, 14, 15]:  # Stop at any ResNet, Inception or attention variant
                            break
                        elif indi.units[i].type in [3, 11, 12, 13, 16]:  # Include DenseNet variants
                            # Adjust the output channels for DenseNet units and their variants.
                            effective_input_channel = min(
                                indi.units[i].in_channel,
                                indi.units[i].max_input_channel
                            )

                            estimated_out_channel = (
                                effective_input_channel
                                + indi.units[i].k * indi.units[i].amount
                            )

                            # DenseNet output is unchanged, so downstream channels
                            # do not need to be recalculated.
                            if estimated_out_channel == indi.units[i].out_channel:
                                keep_out_channel = estimated_out_channel
                                continue

                            self.log.info(
                                'Due to the above mutation, unit at %d '
                                'changes its output channel from %d to %d'
                                % (
                                    i,
                                    indi.units[i].out_channel,
                                    estimated_out_channel
                                )
                            )

                            indi.units[i].out_channel = estimated_out_channel
                            keep_out_channel = estimated_out_channel
    
                mutation_p_count = 1
                indi.reset_acc()
        else:
            # Change the amount of blocks in the ResNet unit or its variants.
            mutation_p_type = 'RESNET_AMOUNT'
            min_resnet_unit, max_resnet_unit = StatusUpdateTool.get_resnet_unit_length_limit()
            amount = np.random.randint(min_resnet_unit, max_resnet_unit)
            if amount != indi.units[position].amount:
                self.log.info('Unit at %d changes its amount from %d to %d' % (
                    position, indi.units[position].amount, amount))
                indi.units[position].amount = amount
                mutation_p_count = 1
                indi.reset_acc()
        
        return mutation_p_type, mutation_p_count
    
    
    
    def do_alter_densenet_mutation(self, position, indi):
        # Alter properties of a DenseNet unit, specifically the amount of layers.
        mutation_p_type = 'DENSENET_AMOUNT'
        mutation_p_count = 0
    
        k = indi.units[position].k
        if k == 12:
            _, amount_lower_limit, amount_upper_limit = StatusUpdateTool.get_densenet_k12()
        elif k == 20:
            _, amount_lower_limit, amount_upper_limit = StatusUpdateTool.get_densenet_k20()
        elif k == 40:
            _, amount_lower_limit, amount_upper_limit = StatusUpdateTool.get_densenet_k40()
        amount = np.random.randint(amount_lower_limit, amount_upper_limit + 1)
        
        if amount != indi.units[position].amount:
            self.log.info('Unit at %d changes its amount from %d to %d' % (position, indi.units[position].amount, amount))
            
            if indi.units[position].amount < amount:
                new_out_channel = (amount - indi.units[position].amount) * k + indi.units[position].out_channel
            else:
                new_out_channel = indi.units[position].out_channel - (indi.units[position].amount - amount) * k
            
            indi.units[position].amount = amount
            self.log.info('Due to the above mutation, unit at %d changes its output channel from %d to %d' % (position, indi.units[position].out_channel, new_out_channel))
            indi.units[position].out_channel = new_out_channel
    
            keep_out_channel = new_out_channel
            for i in range(position + 1, len(indi.units)):
                # Adjust the input channels of subsequent units based on the altered DenseNet unit.
                if indi.units[i].type in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:  # Include all ResNet, DenseNet, Inception, and attention variants
                    self.log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (i, indi.units[i].in_channel, keep_out_channel))
                    indi.units[i].in_channel = keep_out_channel
                    if indi.units[i].type in [1, 4, 5, 6, 7, 8, 9, 10, 14, 15]:  # Stop at any ResNet, Inception or their variant
                        break
                    elif indi.units[i].type in [3, 11, 12, 13, 16]:  # Include DenseNet variants
                        # Adjust the output channels for DenseNet units.
                        effective_input_channel = min(
                            indi.units[i].in_channel,
                            indi.units[i].max_input_channel
                        )

                        estimated_out_channel = (
                            effective_input_channel
                            + indi.units[i].k * indi.units[i].amount
                        )

                        # DenseNet output is unchanged, so downstream channels
                        # do not need to be recalculated.
                        if estimated_out_channel == indi.units[i].out_channel:
                            keep_out_channel = estimated_out_channel
                            continue

                        self.log.info(
                            'Due to the above mutation, unit at %d '
                            'changes its output channel from %d to %d'
                            % (
                                i,
                                indi.units[i].out_channel,
                                estimated_out_channel
                            )
                        )

                        indi.units[i].out_channel = estimated_out_channel
                        keep_out_channel = estimated_out_channel
    
            mutation_p_count = 1
            indi.reset_acc()
        return mutation_p_type, mutation_p_count
    



    def do_alter_pooling_mutation(self, position, indi):
        # Alter the pooling type (max or average) of a pooling layer.
        mutation_p_type = 'POOLING_TYPE'
        mutation_p_count = 1

        if indi.units[position].max_or_avg > 0.5:
            indi.units[position].max_or_avg = 0.2  # Change to max pooling.
            self.log.info('Pool type from avg_pool (>0.5) to max_pool (<0.5)')
        else:
            indi.units[position].max_or_avg = 0.75  # Change to average pooling.
            self.log.info('Pool type from max_pool (<0.5) to avg_pool (>0.5)')
        indi.reset_acc()
        return mutation_p_type, mutation_p_count

    def select_mutation_type(self, _a):
        # Select the type of mutation to perform based on predefined probabilities.
        a = np.asarray(_a)
        k = 1
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(float)
        selected_index = []
        for i in range(k):
            u = np.random.rand() * sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ += sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])
                    break
        return selected_index[0]


# if __name__ == '__main__':
#     # m = Mutation(None, None, None)
#     # m.do_mutation()
#     m = Crossover(None, None, None)
#     m.do_crossover()
