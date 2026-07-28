"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""
import torch
import numpy as np

#from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as tdata
import torchvision
from utils import StatusUpdateTool



# Import Horovod conditionally
if StatusUpdateTool.is_horovod_enabled():
   import horovod.torch as hvd
    


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4, # 4
                           pin_memory=False): #False
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    #cifar10

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    #cinic10
    normalize = transforms.Normalize(
        mean=[0.47889522, 0.47227842, 0.43047404],
        std=[0.24205776, 0.23828046, 0.25874835],
    )
    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset 
    #cifar10

    # train_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=train_transform,
    # )

    # valid_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=valid_transform,
    # )

    #print("Dataset yükleniyor...", flush=True)  # Yükleme mesajı

    #cinic10
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_dir + '/train', transform=train_transform,
    )

    valid_dataset = torchvision.datasets.ImageFolder(
        root=data_dir + '/valid', transform=valid_transform,
    )

    # Dataset boyutlarını kontrol et ve yazdır
    #print(f'Train Dataset boyutu: {len(train_dataset)} örnek', flush=True)
    #print(f'Validation Dataset boyutu: {len(valid_dataset)} örnek', flush=True)


       # 📌 Validation dataset'in belirli bir oranını seçmek için indeks oluştur
    num_valid = len(valid_dataset)  # Toplam validation örnek sayısı
    valid_indices = list(range(num_valid))  # Tüm indeksleri listele
    split = int(np.floor(valid_size * num_valid))  # Kullanılacak validation örnek sayısı

    np.random.seed(random_seed)  # 🔒 Validation set'in sabit olması için seed ayarla
    np.random.shuffle(valid_indices)  # Rastgele karıştır
    selected_valid_indices = valid_indices[:split]  # İlk `split` kadarını al (sabit)
                              
    # 📌 Seçili validasyon örneklerinden yeni bir dataset oluştur
    subset_valid_dataset = torch.utils.data.Subset(valid_dataset, selected_valid_indices)
                               
    #num_train = len(train_dataset)
    #indices = list(range(num_train))
    #split = int(np.floor(valid_size * num_train))

    #if shuffle:
        #np.random.seed(random_seed)
    #    np.random.shuffle(indices)

    #train_idx, valid_idx = indices[split:], indices[:split]

    # --- Horovod Enabled Check ---
    horovod_enabled = StatusUpdateTool.is_horovod_enabled()
    # --------------------------------

    if horovod_enabled:
       # 🔥 Eğitim veri kümesini worker'lara bölüştür
       train_sampler = torch.utils.data.distributed.DistributedSampler(
           train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True
       )
   
       # 🔥 Validation dataset'in sadece belirlenen oranını al ve worker'lara bölüştür
       valid_sampler = torch.utils.data.distributed.DistributedSampler(
           subset_valid_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
       )
   
       train_loader = tdata.DataLoader(
           train_dataset, batch_size=batch_size, sampler=train_sampler,  
           num_workers=num_workers, pin_memory=pin_memory,
       )
   
       valid_loader = tdata.DataLoader(
           subset_valid_dataset, batch_size=batch_size, sampler=valid_sampler,  
           num_workers=num_workers, pin_memory=pin_memory,
       )
   
    else:
       # ✅ Eğitim veri kümesini tamamen kullan, shuffle ile her epoch'ta karıştır
       train_loader = tdata.DataLoader(
           train_dataset, batch_size=batch_size, shuffle=True,  
           num_workers=num_workers, pin_memory=pin_memory,
       )
   
       # ✅ Validation dataset'in belirlenen kısmını direkt kullan (sampler olmadan)
       valid_loader = tdata.DataLoader(
           subset_valid_dataset, batch_size=batch_size, shuffle=False,  
           num_workers=num_workers, pin_memory=pin_memory,
       )

    
    
    #print("train_loader",len(train_loader), flush=True)
    #print("valid_loader",len(valid_loader), flush=True)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4, # 4
                    pin_memory=True): # False
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.47889522, 0.47227842, 0.43047404],
        std=[0.24205776, 0.23828046, 0.25874835],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # dataset = datasets.CIFAR10(
    #     root=data_dir, train=False,
    #     download=True, transform=transform,
    # )

    # data_loader = tdata.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=shuffle,
    #     num_workers=num_workers, pin_memory=pin_memory,
    # )

    dataset = torchvision.datasets.ImageFolder(
        root=data_dir + '/test', transform=transform,
    )

    # --- Horovod Enabled Check ---
    #horovod_enabled = StatusUpdateTool.is_horovod_enabled()
    # --------------------------------

    
    if horovod_enabled:
        # Use DistributedSampler
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)

        data_loader = tdata.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        data_loader = tdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
                        
    return data_loader

"""
print(StatusUpdateTool.is_horovod_enabled())
# Initialize Horovod conditionally
if StatusUpdateTool.is_horovod_enabled():
    hvd.init()
    device = torch.device('cuda', hvd.local_rank())
    # Süreç ve GPU bilgisini yazdırın
    print(f"Process {hvd.rank()} is using GPU {hvd.local_rank()} ({torch.cuda.get_device_name(device)})")
"""
"""
dataset_dir = StatusUpdateTool.is_dataset_directory_set()

# Example usage:
trainloader, validate_loader = get_train_valid_loader(
    dataset_dir,
    batch_size=128,
    augment=True,
    valid_size=0.1,
    shuffle=True,
    random_seed=2312390,
    show_sample=False,
    num_workers=1,
    pin_memory=True
)
"""
