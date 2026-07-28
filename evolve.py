from utils import StatusUpdateTool, Utils, Log
from genetic.population import Population
from genetic.evaluate import FitnessEvaluate
from genetic.crossover_and_mutation import CrossoverAndMutation
from genetic.selection_operator import Selection
from genetic.belief import BeliefConfig, BeliefManager
import numpy as np
import copy
import random

import torch
#import horovod.torch as hvd
try:
    import horovod.torch as hvd
except ImportError:
    horovod_enabled = False  # Eğer yüklü değilse Horovod'u kapat

import pickle
import sys

class FlushStdout:
    def write(self, message):
        sys.__stdout__.write(message)
        sys.__stdout__.flush()  # Çıkışı zorla ekrana yazdır
    
    def flush(self):
        sys.__stdout__.flush()

#sys.stdout = FlushStdout()  # Varsayılan stdout'u değiştir

class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        self.horovod_enabled = StatusUpdateTool.is_horovod_enabled()
        self.belief_manager = None
        self.belief_enabled = False
            
    def sync_individuals(self, individuals):
            """
            Rank 0'daki 'individuals' listesini diğer rank'lara yayar.
            Tüm rank'larda 'individuals' in aynı hale gelmesini sağlar.
            """
            # 1) Bariyer (isteğe bağlı) – tüm rank’ların bu noktada buluşmasını isterseniz
            hvd.barrier()
        
            # 2) Rank 0 tarafında individuals'ı pickle'a çevir. Diğer rank’larda sıfır byte.
            if hvd.rank() == 0:
                pickled_data = pickle.dumps(individuals, protocol=pickle.HIGHEST_PROTOCOL)
                data_size = torch.IntTensor([len(pickled_data)])
            else:
                pickled_data = None
                data_size = torch.IntTensor([0])
        
            # 3) Her rank, data_size bilgisini alarak kaç byte alacağını bilsin
            data_size = hvd.broadcast(data_size, root_rank=0)
        
            # 4) Rank != 0: gelen boyut kadar boş byte dizisi hazırla
            if hvd.rank() != 0:
                pickled_data = bytearray(data_size.item())
        
            # 5) Byte dizisini ByteTensor'a çevir
            if hvd.rank() == 0:
                pickled_tensor = torch.ByteTensor(list(pickled_data))
            else:
                pickled_tensor = torch.empty(data_size.item(), dtype=torch.uint8)
        
            # 6) Asıl veriyi (pickle edilmiş) broadcast et
            pickled_tensor = hvd.broadcast(pickled_tensor, root_rank=0)
        
            # 7) Tüm rank'lar gelen tensörü unpickle ile orijinal listeye çevirir
            np_data = pickled_tensor.cpu().numpy()
            unpickled_bytes = np_data.tobytes()
            synced_individuals = pickle.loads(unpickled_bytes)
        
            return synced_individuals
        
    def initialize_population(self):
        #print(f"Rank {hvd.rank()} in initialize_population function")
        if (not self.horovod_enabled) or (self.rank == 0):
            #global.ini dosyasına yazmayı sadece rank0 yapar
            StatusUpdateTool.begin_evolution()
        
        if self.horovod_enabled:
            hvd.barrier()
        
        pops = Population(params, 0)
        pops.initialize()


        #print(f"Population is initialized by Rank {hvd.rank()}")
        self.pops = pops
        
        if self.horovod_enabled:
            # Synchronize the most up-to-date individuals known by Rank 0 with all ranks.
            #self.pops.individuals = self.sync_individuals(self.pops.individuals)
            self.pops = self.sync_individuals(self.pops)



        
        if (not self.horovod_enabled) or (self.rank == 0):
            #baslangic popunu olusturuyor ve kaydediyor
            Utils.save_population_at_begin(str(pops), 0)

    
    def fitness_evaluate(self):
        if self.horovod_enabled:
            #print(f"Rank {hvd.rank()} in fitness_evaluate function")
            #self.pops.individuals = self.sync_individuals(self.pops.individuals)
            self.pops = self.sync_individuals(self.pops)

            #Log.info('Sync process completed at the beginning of fitness evaluation')
            hvd.barrier()  # Eşitleme sonrası senkronizasyon
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()


    def setup_belief_manager(self):
        """Create the belief manager on rank 0 only."""

        belief_config = BeliefConfig.from_ini()
        self.belief_enabled = belief_config.enabled
        if (not self.horovod_enabled) or (self.rank == 0):
            self.belief_manager = BeliefManager(
                config=belief_config,
                log=Log,
                restore_existing=StatusUpdateTool.is_evolution_running(),
            )
            Log.info('Belief manager status: %s' % self.belief_manager.describe())

    def belief_prepare_cycle(self, cycle):
        """Score offspring and keep the selected evaluation batch."""

        if (not self.horovod_enabled) or (self.rank == 0):
            if self.belief_manager is not None and self.belief_manager.is_enabled:
                cache_map = Utils.load_cache_data()
                preparation = self.belief_manager.prepare_cycle(
                    candidates=self.pops.individuals,
                    cycle=cycle,
                    cache_map=cache_map,
                )
                self.pops.individuals = copy.deepcopy(preparation.selected_individuals)

        if self.horovod_enabled:
            self.pops = self.sync_individuals(self.pops)
            hvd.barrier()

    def belief_post_evaluate(self, cycle):
        """Update belief records after true fitness becomes available."""

        if (not self.horovod_enabled) or (self.rank == 0):
            if self.belief_manager is not None and self.belief_manager.is_enabled:
                metrics = self.belief_manager.post_evaluate(
                    evaluated_individuals=self.pops.individuals,
                    cycle=cycle,
                )
                if metrics is not None:
                    Log.info('Belief cycle metrics: %s' % metrics.to_dict())

    def crossover_and_mutation(self, target_size=None, legacy_mode=True):
        #print(f"Rank {hvd.rank()} in crossover_and_mutation function")
        if (not self.horovod_enabled) or (self.rank == 0):
            cm = CrossoverAndMutation(
                self.params['genetic_prob'][0],
                self.params['genetic_prob'][1],
                Log,
                self.pops.individuals,
                _params={
                    'gen_no': self.pops.gen_no,
                    'target_size': target_size or self.params['pop_size'],
                    'legacy_mode': legacy_mode,
                },
            )
            offspring = cm.process()
            self.parent_pops = copy.deepcopy(self.pops) # this is used for elitism next
            self.pops.individuals = copy.deepcopy(offspring)
        #Log.info('offsprings are coppied as new individuals after cros and mut')
            
        if self.horovod_enabled:
            # Synchronize the offspring with other ranks
            #self.pops.individuals = self.sync_individuals(self.pops.individuals)
            self.pops = self.sync_individuals(self.pops)
            #print("self.pops.individuals =", self.pops.individuals)
            #Log.info('Sync proccess has done after crossover and mutation between ranks for individuals')
            hvd.barrier()


    def environment_selection(self):
        #  Combine Individuals:
        #    - Gather individuals (and their accuracies) from both current (self.pops)
        #      and parent (self.parent_pops) populations into lists.
        #  Log:
        #    - Record details (ID, accuracy, UUID) of all individuals in a log string (_str).
        #  Find Best Individual:
        #    - Identify the highest-accuracy individual using np.argmax.
        #  Select Next Generation:
        #    - Randomly choose a selection method (Roulette or Wheel) to pick pop_size indices.
        #    - Ensure the top-accuracy individual is included (elitism).
        #  Create New Population:
        #    - Form the next-generation population with the selected individuals 
        #      and increment the generation number.
        #  Write Logs:
        #    - Append info about the new population’s individuals to _str and save it to file.
        #  Synchronize (if Distributed):
        #    - If using Horovod, broadcast the updated individuals from Rank 0 to all ranks.
        #print(f"Rank {hvd.rank()} in environment_selection function")
        if (not self.horovod_enabled) or (self.rank == 0):
            v_list = []
            indi_list = []
            for indi in self.pops.individuals:
                indi_list.append(indi)
                v_list.append(indi.acc)
            for indi in self.parent_pops.individuals:
                indi_list.append(indi)
                v_list.append(indi.acc)
    
            _str = []
            for _, indi in enumerate(self.pops.individuals):
                _t_str = 'Indi-%s-%.5f-%s'%(indi.id, indi.acc, indi.uuid()[0])
                _str.append(_t_str)
            for _, indi in enumerate(self.parent_pops.individuals):
                _t_str = 'Pare-%s-%.5f-%s'%(indi.id, indi.acc, indi.uuid()[0])
                _str.append(_t_str)
    
    
            #add log
            # find the largest one's index
            max_index = np.argmax(v_list)
            selection = Selection()

            #re-ranking yapıyoruz
            generation = self.pops.gen_no
            #print("generation=", generation)
            #print("self.max_gen= ",self.max_gen)
            #total_gen=20
            pseudo_fitness = selection.GetGeometricPseudoFitness(v_list, self.pops.gen_no, self.max_gen)
            
            u_ = random.random()
            selected_index_list = []
            
            if u_ < 0.5:
                # Bizim pseudo-fitness'lar ile roulette selection
                selected_index_list = selection.RouletteSelection(pseudo_fitness, k=self.params['pop_size'])
            else:
                # Bizim pseudo-fitness'lar ile wheel selection
                selected_index_list = selection.WheelSelection(pseudo_fitness, k=self.params['pop_size'])

            #önceki çalışmadaki kısım. Biz re-rankingiçin yukardaki şekilde update yaptık
            """
            u_ = random.random()
            selected_index_list = []
            if u_ < 0.5:
                # print("roulette")
                selected_index_list = selection.RouletteSelection(v_list, k=self.params['pop_size'])
            else:
                # print("wheel")
                selected_index_list = selection.WheelSelection(v_list, k=self.params['pop_size'])
            """
            
            if max_index not in selected_index_list:
                first_selectd_v_list = [v_list[i] for i in selected_index_list]
                min_idx = np.argmin(first_selectd_v_list)
                selected_index_list[min_idx] = max_index
    
            next_individuals = [indi_list[i] for i in selected_index_list]
    
            """Here, the population information should be updated, such as the gene no and then to the individual id"""
            next_gen_pops = Population(self.pops.params, self.pops.gen_no+1)
            next_gen_pops.create_from_offspring(next_individuals)
            self.pops = next_gen_pops
            for _, indi in enumerate(self.pops.individuals):
                _t_str = 'new -%s-%.5f-%s'%(indi.id, indi.acc, indi.uuid()[0])
                _str.append(_t_str)
            _file = './populations/ENVI_%2d.txt'%(self.pops.gen_no)
            Utils.write_to_file('\n'.join(_str), _file)
    
            Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)
            
        if self.horovod_enabled:
            hvd.barrier()
            # Synchronize the most up-to-date individuals known by Rank 0 with all ranks.
            self.pops.individuals = self.sync_individuals(self.pops.individuals)

    def do_work(self, max_gen):
        self.max_gen = max_gen
        #print("Do work function...")
        # Horovod başlangıcı
        #self.horovod_enabled = StatusUpdateTool.is_horovod_enabled()
        if self.horovod_enabled:
            self.rank = hvd.rank()
            self.size = hvd.size()
        else:
            self.rank = 0
            self.size = 1
        self.setup_belief_manager()
        Log.info('*' * 25)
        if self.horovod_enabled:
            hvd.barrier()
        print("StatusUpdateTool.is_evolution_running = ", StatusUpdateTool.is_evolution_running())
        if StatusUpdateTool.is_evolution_running():
        #if (not self.horovod_enabled) or (self.rank == 0):
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if self.horovod_enabled:
                hvd.barrier()
            if gen_no is not None:
                Log.info('Initialize from %d-th generation' % (gen_no))
                #if (not self.horovod_enabled) or (self.rank == 0):
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
                if self.horovod_enabled:
                    hvd.barrier()
                    # Synchronize the most up-to-date individuals known by Rank 0 with all ranks.
                    #self.pops = self.sync_individuals(self.pops)
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        #else:
        #    gen_no= None # Diğer rank'lar için placeholder

        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()

   
        # Sonuç: Tüm rank'lar gen_no'yu biliyor
        Log.info(f'Rank {self.rank} knows gen_no: {gen_no}')
            
        # Popülasyonun yüklenmesi veya başlatılması bitti, senkronize ol
        if self.horovod_enabled:
            hvd.barrier()
        #if (not self.horovod_enabled) or (self.rank == 0):
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no)) 
        self.fitness_evaluate() 
        if (not self.horovod_enabled) or (self.rank == 0):
            if self.belief_manager is not None and self.belief_manager.is_enabled:
                self.belief_manager.bootstrap_population(
                    self.pops.individuals, generation=gen_no
                )
        #if (not self.horovod_enabled) or (self.rank == 0):
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))
        gen_no += 1
        if self.horovod_enabled:
            hvd.barrier()
        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            # Crossover and mutation: Eğitimle ilgili değil, sadece rank 0 yapacak
            if (not self.horovod_enabled) or (self.rank == 0):
                Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (curr_gen))
            #if (not self.horovod_enabled) or (self.rank == 0):
            target_size = self.params['pop_size']
            legacy_mode = True
            if (not self.horovod_enabled) or (self.rank == 0):
                if self.belief_manager is not None and self.belief_manager.is_enabled:
                    target_size = self.belief_manager.candidate_target_size(
                        self.params['pop_size'], curr_gen
                    )
                    legacy_mode = not self.belief_manager.guided_active(curr_gen)
            self.crossover_and_mutation(
                target_size=target_size, legacy_mode=legacy_mode
            )
            if (not self.horovod_enabled) or (self.rank == 0):
                Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % (curr_gen))

            if self.belief_enabled:
                self.belief_prepare_cycle(curr_gen)

            if (not self.horovod_enabled) or (self.rank == 0):
                Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
            if self.horovod_enabled:
                hvd.barrier()
            self.fitness_evaluate()
            self.belief_post_evaluate(curr_gen)
            if (not self.horovod_enabled) or (self.rank == 0):
                Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))       
                Log.info('EVOLVE[%d-gen]-Begin to environment selection' % (curr_gen))
            #if (not self.horovod_enabled) or (self.rank == 0):
            self.environment_selection()
            if (not self.horovod_enabled) or (self.rank == 0):
                Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (curr_gen))
            if self.horovod_enabled:
                hvd.barrier()

        if (not self.horovod_enabled) or (self.rank == 0):
            StatusUpdateTool.end_evolution()
        #asagidakini evolution un durumundan tüm rankler haberdar olsun diye koydum. 
        #yoksa rank0 harici rankler önce giderse problem
        if self.horovod_enabled:
            hvd.barrier()

# Import Horovod conditionally
if StatusUpdateTool.is_horovod_enabled():
    import horovod.torch as hvd
    

horovod_enabled = StatusUpdateTool.is_horovod_enabled()

if horovod_enabled:
    hvd.init()
    print(f"Horovod initialized. Total ranks: {hvd.size()}")
    
    # Doğru GPU'yu seç
    torch.cuda.set_device(hvd.local_rank())
    print(f"Rank {hvd.rank()} assigned to GPU {hvd.local_rank()} ({torch.cuda.get_device_name(hvd.local_rank())})")

    """
    hvd.barrier()
    """
    print("işlem başlıyor...")
    #if hvd.rank() == 0:
    print(f"HOROVOD Rank: {hvd.rank()}")

    
    if __name__ == '__main__':    
        #if (hvd.rank == 0):
        params = StatusUpdateTool.get_init_params()
        hvd.barrier()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(max_gen=20) #max_gen=20
        hvd.barrier()

    
else:
    if __name__ == '__main__':
        print("işlem başlıyor...")
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(max_gen=20) #max_gen=20




