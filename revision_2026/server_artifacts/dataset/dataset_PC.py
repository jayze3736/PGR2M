
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from pathlib import Path

def _load_stats(data_root, meta_dir=None):
    candidates=[]
    if meta_dir:
        m=Path(meta_dir)
        candidates += [
            (m/"mean.npy", m/"std.npy"),
            (m/"Mean.npy", m/"Std.npy"),
        ]
    r=Path(data_root)
    candidates += [(r/"Mean.npy", r/"Std.npy")]
    for mp,sp in candidates:
        if mp.is_file() and sp.is_file():
            return np.load(mp), np.load(sp), str(mp), str(sp)
    raise FileNotFoundError("Could not locate a valid mean/std pair.")

class PCMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4,
                 use_full_sequence=False, meta_dir=None, codes_folder_name="codes"):
        self.window_size=window_size
        self.unit_length=unit_length
        self.dataset_name=dataset_name
        self.use_full_sequence=use_full_sequence

        if dataset_name == "t2m":
            self.data_root="./dataset/HumanML3D"
            self.joints_num=22
            self.max_motion_length=196
        elif dataset_name == "kit":
            self.data_root="./dataset/KIT-ML"
            self.joints_num=21
            self.max_motion_length=196
        else:
            raise ValueError(f"Unsupported dataset_name={dataset_name}")

        if self.use_full_sequence:
            raise NotImplementedError(
                "Tokenizer reproduction currently validates the README/default fixed-window path only; "
                "--use-full-sequence is intentionally not guessed."
            )

        self.motion_dir=pjoin(self.data_root,"new_joint_vecs")
        self.text_dir=pjoin(self.data_root,"texts")
        self.code_dir=pjoin(self.data_root,codes_folder_name)
        self.mean,self.std,self.mean_path,self.std_path=_load_stats(self.data_root,meta_dir)

        split_file=pjoin(self.data_root,"train.txt")
        with cs.open(split_file,"r") as f:
            id_list=[line.strip() for line in f if line.strip()]

        self.data=[]
        self.codes=[]
        self.lengths=[]
        self.names=[]
        skipped=[]

        for name in id_list:
            try:
                motion=np.load(pjoin(self.motion_dir,name+".npy"))
                code=np.load(pjoin(self.code_dir,name+".npy")).astype(np.uint8)

                if motion.ndim != 2 or motion.shape[1] != len(self.mean):
                    skipped.append((name,"motion_shape"))
                    continue
                if code.ndim != 2:
                    skipped.append((name,"code_shape"))
                    continue
                if motion.shape[0] != code.shape[0]:
                    skipped.append((name,"length_mismatch"))
                    continue
                if not np.isfinite(motion).all():
                    skipped.append((name,"nonfinite_motion"))
                    continue
                if motion.shape[0] < self.window_size:
                    continue

                self.lengths.append(motion.shape[0]-self.window_size+1)
                self.data.append(motion)
                self.codes.append(code)
                self.names.append(name)
            except Exception as e:
                skipped.append((name,type(e).__name__))

        self.skipped=skipped
        if not self.data:
            raise RuntimeError("No usable training motions were loaded.")

    def inv_transform(self,x):
        return x*self.std+self.mean

    def compute_sampling_prob(self):
        prob=np.asarray(self.lengths,dtype=np.float64)
        prob/=prob.sum()
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        motion=self.data[item]
        code=self.codes[item]
        idx=random.randint(0,len(motion)-self.window_size)
        motion=motion[idx:idx+self.window_size]
        motion=(motion-self.mean)/self.std
        code=code[idx:idx+self.window_size:self.unit_length]
        return motion,code

def DATALoader(dataset_name,
               batch_size,
               num_workers=8,
               window_size=64,
               unit_length=4,
               use_full_sequence=False,
               meta_dir=None,
               codes_folder_name="codes"):
    ds=PCMotionDataset(
        dataset_name,
        window_size=window_size,
        unit_length=unit_length,
        use_full_sequence=use_full_sequence,
        meta_dir=meta_dir,
        codes_folder_name=codes_folder_name,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
