from utils import Utils, GPUTools, StatusUpdateTool, Log
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep


import time
import os
import sys
import importlib
from multiprocessing import Process
import torch
#import horovod.torch as hvd
try:
    import horovod.torch as hvd
except ImportError:
    horovod_enabled = False  # Eğer yüklü değilse Horovod'u kapat

import pickle     
import numpy as np 

class FitnessEvaluate(object):

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log
        self.horovod_enabled = StatusUpdateTool.is_horovod_enabled()

        if self.horovod_enabled:
            
            #hvd.init()
            self.rank = hvd.rank()
            self.size = hvd.size()
        else:
            self.rank = 0
            self.size = 1

    def generate_to_python_file(self):
        # Sadece Rank 0 veya Horovod kapalı ise dosyalar oluşturulur
        if (not self.horovod_enabled) or (self.rank == 0):
            self.log.info('Begin to generate python files')
            for indi in self.individuals:
                Utils.generate_pytorch_file(indi)
            self.log.info('Finish the generation of python files')
        #else:
        #    self.log.info('Rank %d in FitnessEvaluate.generate_to_python_file function but bypass generation' % (self.rank))
            #print(f"Rank %s in FitnessEvaluate.generate_to_python_file function but bypass generation")

        # Dosyalar oluşturulduktan sonra tüm rank'lar senkronize olsun
        if self.horovod_enabled:
            hvd.barrier()

    def sync_individuals_acc(self, individuals):
        """
        Rank 0'daki 'indi.acc' değerlerini diğer rank'lara yayar.
        Tüm rank'larda aynı 'indi.acc' değerlerini oluşturur.
        """
        # 1) Bariyer (isteğe bağlı) – tüm rank’ların bu noktada buluşmasını isterseniz
        hvd.barrier()
    
        # 2) Rank 0 tarafında acc değerlerini pickle'a çevir
        if hvd.rank() == 0:
            acc_list = [indi.acc for indi in individuals]  # Sadece acc değerleri
            pickled_data = pickle.dumps(acc_list, protocol=pickle.HIGHEST_PROTOCOL)
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
    
        # 7) Tüm rank'lar gelen tensörü unpickle ile orijinal acc listesine çevirir
        np_data = pickled_tensor.cpu().numpy()
        unpickled_bytes = np_data.tobytes()
        synced_acc_list = pickle.loads(unpickled_bytes)
    
        # 8) Tüm rank’ler kendi individuals listelerinde acc değerlerini günceller
        for i, indi in enumerate(individuals):
            indi.acc = synced_acc_list[i]
    
        return individuals


    def evaluate(self):
        
        # Cache sorgusu ve acc güncellemeleri sadece rank 0
        if (not self.horovod_enabled) or (self.rank == 0):
            self.log.info('Query fitness from cache')
            _map = Utils.load_cache_data()
            _count = 0
            for indi in self.individuals:
                _key, _str = indi.uuid()
                if _key in _map:
                    _count += 1
                    _acc = _map[_key]
                    self.log.info('Hit the cache for %s, key:%s, acc:%.5f, assigned_acc:%.5f' %
                                  (indi.id, _key, float(_acc), indi.acc))
                    indi.acc = float(_acc)
            self.log.info('Total hit %d individuals for fitness' % (_count))

    
        
        #bu kısma diğer rankler için individuals bilgisi dağıtılması gerekiyor.
        #yoksa cache durumundan gelen bilgileri bilemez.

        
        # Cache okuma ve acc atamaları bitti, senkronize olun
        if self.horovod_enabled:
            hvd.barrier()



        if self.horovod_enabled:
            self.individuals = self.sync_individuals_acc(self.individuals)  # Rank 0'dan diğerlerine acc dağıtımı
            hvd.barrier()
            
        has_evaluated_offspring = False
        for indi in self.individuals:
            #print(indi)
            if indi.acc < 0:
                has_evaluated_offspring = True

                if not self.horovod_enabled:
                    print("Horovod aktif değil... GPU seçiliyor...")
                    time.sleep(60)
                    gpu_id = GPUTools.detect_availabel_gpu_id()
                    while gpu_id is None:
                        time.sleep(300)
                        gpu_id = GPUTools.detect_availabel_gpu_id()
                else:
                    gpu_id = None  # Horovod GPU yönetimi yapar

                file_name = indi.id
                #print("file_name=", file_name )
                
                if self.horovod_enabled:
                    hvd.barrier()
                    
                # Log sadece rank 0
                if (not self.horovod_enabled) or (self.rank == 0):
                    self.log.info('Begin to train %s' % (file_name))

                module_name = 'scripts.%s' % (file_name)
                #print("module_name=", module_name)
                
                #if (not self.horovod_enabled) or (self.rank == 0):
                if module_name in sys.modules.keys():
                    self.log.info('Module:%s has been loaded, deleting it' % (module_name))
                    del sys.modules[module_name]


                #print("[DEBUG] Attempting to import module:", module_name)
                _module = importlib.import_module(module_name, '.')
                #print("[DEBUG] Imported module:", _module)
                
                #_module = importlib.import_module('.', module_name)
                
                _class = getattr(_module, 'RunModel')
                cls_obj = _class()

                
                # Eğitim aşaması:
                if self.horovod_enabled:
                    #hvd.barrier()
                    #if (self.rank == 0):
                    #Log.info('Starting do_work function...')
                    # Tüm rank’lar eğitim yapar
                    
                    try:
                        #print(f"[DEBUG] Rank {hvd.rank()} calling do_work...")
                        cls_obj.do_work(file_name)
                        #print(f"[DEBUG] Rank {hvd.rank()} do_work returned!")
                    except BaseException as e:
                        print(f"[DEBUG] Rank {hvd.rank()} do_work crashed! {e}")

                    
                    #cls_obj.do_work(file_name)
                    #print(f"Rank {hvd.rank()} reached barrier after training...")
                    hvd.barrier()
                else:
                    # Tek süreç
                    p = Process(target=cls_obj.do_work, args=(file_name, str(gpu_id),))
                    p.start()
                    p.join()
            else:
                # Bu indi için fitness zaten var, sadece rank 0 log ve dosya yazma yapar
                if (not self.horovod_enabled) or (self.rank == 0):
                    file_name = indi.id
                    self.log.info('%s has inherited the fitness as %.5f, no need to evaluate' %
                                  (file_name, indi.acc))
                    with open('./populations/after_%s.txt' % (file_name[4:6]), 'a+') as f:
                        f.write('%s=%.5f\n' % (file_name, indi.acc))
                        f.flush()
                    Utils.save_fitness_to_cache([indi])#sonradan eklendi. birey bazında cache kaydı için
            #if self.horovod_enabled:
            #            hvd.barrier()
                
        # Eğer offspring değerlendirildiyse GPU kontrolleri veya bariyer koyabiliriz
        if has_evaluated_offspring:
            if not self.horovod_enabled:
                # Tek süreçte GPU'ların boşalmasını bekle
                all_finished = False
                while not all_finished:
                    time.sleep(300)
                    all_finished = GPUTools.all_gpu_available()
            else:
                # Horovod etkin, tüm rank'lar buraya gelsin
                if self.horovod_enabled:
                    hvd.barrier()

            # Değerlendirme sonuçlarını after_xx.txt'den okuma sadece rank 0
            if (not self.horovod_enabled) or (self.rank == 0):
                file_name = './populations/after_%s.txt' % (self.individuals[0].id[4:6])
                assert os.path.exists(file_name)
                fitness_map = {}
                with open(file_name, 'r') as f:
                    for line in f:
                        if line.strip():
                            key, value = line.strip().split('=')
                            fitness_map[key] = float(value)

                for indi in self.individuals:
                    if indi.acc == -1:
                        if indi.id not in fitness_map:
                            self.log.warn(
                                'The individual has been evaluated, but the records are not correct, '
                                'the fitness of %s does not exist in %s, waiting 120 seconds' %
                                (indi.id, file_name))
                            time.sleep(120)
                        indi.acc = fitness_map[indi.id]
            
            # Sonuçların okuması bitti, senkronize olun
            if self.horovod_enabled:
                hvd.barrier()
        else:
            if (not self.horovod_enabled) or (self.rank == 0):
                self.log.info('No offspring has been evaluated')

        # Cache'e kaydetme sadece rank 0
        if (not self.horovod_enabled) or (self.rank == 0):
            Utils.save_fitness_to_cache(self.individuals)
        
        # Cache kaydı bitti, senkronize ol
        if self.horovod_enabled:
            hvd.barrier()


