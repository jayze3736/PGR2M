
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)

class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length=4, codebook_size=392, use_keywords=False):
        self.pointer = 0
        self.dataset_name = dataset_name
        self.codebook_size = codebook_size
        self.unit_length = unit_length
        self.use_keywords = use_keywords

        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        if dataset_name == "t2m":
            self.data_root = "./dataset/HumanML3D"
            self.joints_num = 22
            self.fps = 20
            self.dim_pose = 263
        elif dataset_name == "kit":
            self.data_root = "./dataset/KIT-ML"
            self.joints_num = 21
            self.fps = 12.5
            self.dim_pose = 251
        else:
            raise ValueError(dataset_name)

        self.motion_dir = pjoin(self.data_root, "new_joint_vecs")
        self.text_dir = pjoin(self.data_root, "texts")
        self.code_dir = pjoin(self.data_root, "codes")
        self.keyword_dir = pjoin(self.data_root, "keyword_embeddings")

        # Pose-code sequence after x4 temporal downsampling plus END.
        self.max_motion_length = 26 if unit_length == 8 else 51

        self.mean = np.load(pjoin(self.data_root, "Mean.npy"))
        self.std = np.load(pjoin(self.data_root, "Std.npy"))

        split_file = pjoin(self.data_root, "train.txt")
        with cs.open(split_file, "r") as f:
            id_list = [line.strip() for line in f if line.strip()]

        data_dict = {}
        new_name_list = []

        min_motion_len = 40 if dataset_name == "t2m" else 24

        for name in id_list:
            try:
                motion_full = np.load(pjoin(self.motion_dir, name + ".npy"))
                pose_full = np.load(pjoin(self.code_dir, name + ".npy")).astype(np.float32)
                keywords = np.load(pjoin(self.keyword_dir, name + ".npy")).astype(np.float32)

                if len(motion_full) < min_motion_len or len(motion_full) >= 200:
                    continue

                text_data = []
                has_full = False

                with cs.open(pjoin(self.text_dir, name + ".txt"), "r") as f:
                    for line in f:
                        parts = line.strip().split("#")
                        if len(parts) < 4:
                            continue

                        caption = parts[0]
                        tokens = parts[1].split(" ")
                        f_tag = float(parts[2])
                        to_tag = float(parts[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        td = {"caption": caption, "tokens": tokens}

                        if f_tag == 0.0 and to_tag == 0.0:
                            has_full = True
                            text_data.append(td)
                        else:
                            a = int(f_tag * self.fps)
                            b = int(to_tag * self.fps)
                            n_motion = motion_full[a:b]
                            n_pose = pose_full[a:b]

                            if len(n_motion) < min_motion_len or len(n_motion) >= 200:
                                continue

                            new_name = f"{name}_{f_tag:.3f}_{to_tag:.3f}"
                            data_dict[new_name] = {
                                "motion": n_motion,
                                "pose": n_pose,
                                "text": [td],
                                "keywords": keywords,
                            }
                            new_name_list.append(new_name)

                if has_full and text_data:
                    data_dict[name] = {
                        "motion": motion_full,
                        "pose": pose_full,
                        "text": text_data,
                        "keywords": keywords,
                    }
                    new_name_list.append(name)

            except Exception:
                # Match the released loaders' permissive behavior on malformed samples.
                continue

        if not new_name_list:
            raise RuntimeError("dataset_TM_train reconstruction found zero training examples.")

        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        d = self.data_dict[self.name_list[item]]
        motion = d["motion"].copy()
        pose = d["pose"].copy()
        text_data = random.choice(d["text"])
        caption = text_data["caption"]
        keywords = d["keywords"]

        # Align to temporal downsampling factor.
        m_length = (len(motion) // self.unit_length) * self.unit_length
        if m_length < self.unit_length:
            return self.__getitem__((item + 1) % len(self))

        start = random.randint(0, len(motion) - m_length)
        motion = motion[start:start + m_length]
        pose = pose[start:start + m_length][::self.unit_length]

        # CoMo-style occasional one-token crop.
        coin = np.random.choice([False, False, True])
        if coin and len(pose) > 1:
            if np.random.choice([True, False]):
                pose = pose[:-1]
                motion = motion[:-self.unit_length]
            else:
                pose = pose[1:]
                motion = motion[self.unit_length:]

        # Keyword sampling: one embedding from each group of 3.
        if self.use_keywords:
            keyword_indices = np.arange(0, 33, 3) + np.random.randint(0, 3, size=(11,))
            keyword_data = torch.from_numpy(keywords[keyword_indices].copy())

            # Released CoMo behavior randomly masks a subset of keywords.
            proba = float(np.random.rand())
            mask = torch.bernoulli(proba * torch.ones((11,))).bool()
            keyword_data[mask] = 0
        else:
            keyword_data = torch.zeros((11, keywords.shape[-1]), dtype=torch.float32)

        motion = (motion - self.mean) / self.std
        motion_len = len(motion)

        max_frames = 200
        if motion_len < max_frames:
            motion = np.concatenate(
                [motion, np.zeros((max_frames - motion_len, motion.shape[1]), dtype=motion.dtype)],
                axis=0,
            )

        token_len = pose.shape[0]

        # Base Transformer target: pose codes + END/PAD channels.
        end_token = np.zeros((1, self.codebook_size + 2), dtype=np.float32)
        end_token[:, self.mot_end_idx] = 1.0

        pose_ext = np.concatenate(
            [pose.astype(np.float32), np.zeros((token_len, 2), dtype=np.float32)],
            axis=1,
        )

        if token_len + 1 < self.max_motion_length:
            pad_n = self.max_motion_length - 1 - token_len
            pad = np.zeros((pad_n, self.codebook_size + 2), dtype=np.float32)
            pad[:, self.mot_pad_idx] = 1.0
            tokens = np.concatenate([pose_ext, end_token, pad], axis=0)
        else:
            # Defensive crop; HumanML3D <200 frames should normally fit.
            pose_ext = pose_ext[:self.max_motion_length - 1]
            token_len = len(pose_ext)
            tokens = np.concatenate([pose_ext, end_token], axis=0)

        return (
            caption,
            tokens.astype(np.float32),
            int(token_len),
            keyword_data.float(),
            motion.astype(np.float32),
            int(motion_len),
        )

def DATALoader(dataset_name, batch_size, codebook_size, unit_length=4,
               num_workers=8, use_keywords=False):
    return torch.utils.data.DataLoader(
        Text2MotionDataset(
            dataset_name,
            unit_length=unit_length,
            codebook_size=codebook_size,
            use_keywords=use_keywords,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
