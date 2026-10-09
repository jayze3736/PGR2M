
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from pathlib import Path
from torch.utils.data._utils.collate import default_collate
import utils.paramUtil as paramUtil

def collate_fn(batch):
    batch.sort(key=lambda x:x[3],reverse=True)
    return default_collate(batch)

def _load_stats(data_root, meta_dir=None):
    candidates=[]
    if meta_dir:
        m=Path(meta_dir)
        candidates += [
            (m/"mean.npy",m/"std.npy"),
            (m/"Mean.npy",m/"Std.npy"),
        ]
    r=Path(data_root)
    candidates += [(r/"Mean.npy",r/"Std.npy")]
    for mp,sp in candidates:
        if mp.is_file() and sp.is_file():
            return np.load(mp),np.load(sp),str(mp),str(sp)
    raise FileNotFoundError("Could not locate a valid mean/std pair.")

class Text2MotionDataset(data.Dataset):
    def __init__(self,dataset_name,is_test,w_vectorizer,max_text_len=20,unit_length=4,
                 codebook_size=392,val_shuffle=True,use_keywords=False,use_word_only=False,
                 meta_dir=None,codes_folder_name="codes"):
        self.max_length=20
        self.pointer=0
        self.dataset_name=dataset_name
        self.is_test=is_test
        self.max_text_len=max_text_len
        self.unit_length=unit_length
        self.w_vectorizer=w_vectorizer
        self.codebook_size=codebook_size
        self.use_keywords=use_keywords
        self.use_word_only=use_word_only

        if use_word_only:
            raise NotImplementedError(
                "Tokenizer README/default path does not use --use-word-only; "
                "this branch is intentionally not guessed."
            )

        if dataset_name=="t2m":
            self.data_root="./dataset/HumanML3D"
            self.joints_num=22
            self.max_motion_length=196
            self.fps=20
        elif dataset_name=="kit":
            self.data_root="./dataset/KIT-ML"
            self.joints_num=21
            self.max_motion_length=196
            self.fps=12.5
        else:
            raise ValueError(f"Unsupported dataset_name={dataset_name}")

        self.motion_dir=pjoin(self.data_root,"new_joint_vecs")
        self.text_dir=pjoin(self.data_root,"texts")
        self.code_dir=pjoin(self.data_root,codes_folder_name)
        self.keyword_dir=pjoin(self.data_root,"keyword_embeddings")
        self.mean,self.std,self.mean_path,self.std_path=_load_stats(self.data_root,meta_dir)

        split_file=pjoin(self.data_root,"test.txt" if is_test else "val.txt")
        with cs.open(split_file,"r") as f:
            id_list=[line.strip() for line in f if line.strip()]

        min_motion_len=40 if dataset_name=="t2m" else 24
        data_dict={}
        new_name_list=[]
        length_list=[]
        skipped=[]

        for name in id_list:
            try:
                motion=np.load(pjoin(self.motion_dir,name+".npy"))
                code=np.load(pjoin(self.code_dir,name+".npy")).astype(np.uint8)

                if motion.shape[0] != code.shape[0]:
                    skipped.append((name,"length_mismatch"))
                    continue
                if not np.isfinite(motion).all():
                    skipped.append((name,"nonfinite_motion"))
                    continue

                if self.use_keywords:
                    keywords=np.load(pjoin(self.keyword_dir,name+".npy"))
                else:
                    keywords=np.zeros((33,512),dtype=np.float32)

                if len(motion)<min_motion_len or len(motion)>=200:
                    continue

                text_data=[]
                flag=False
                with cs.open(pjoin(self.text_dir,name+".txt")) as f:
                    for line in f:
                        line_split=line.strip().split("#")
                        if len(line_split)<4:
                            continue
                        caption=line_split[0]
                        tokens=line_split[1].split(" ")
                        f_tag=float(line_split[2]); to_tag=float(line_split[3])
                        f_tag=0.0 if np.isnan(f_tag) else f_tag
                        to_tag=0.0 if np.isnan(to_tag) else to_tag
                        td={"caption":caption,"tokens":tokens}

                        if f_tag==0.0 and to_tag==0.0:
                            flag=True
                            text_data.append(td)
                        else:
                            n_motion=motion[int(f_tag*self.fps):int(to_tag*self.fps)]
                            n_code=code[int(f_tag*self.fps):int(to_tag*self.fps)]
                            if len(n_motion)<min_motion_len or len(n_motion)>=200:
                                continue
                            new_name=random.choice("ABCDEFGHIJKLMNOPQRSTUVW")+"_"+name
                            while new_name in data_dict:
                                new_name=random.choice("ABCDEFGHIJKLMNOPQRSTUVW")+"_"+name
                            data_dict[new_name]={
                                "motion":n_motion,
                                "length":len(n_motion),
                                "text":[td],
                                "code_indices":n_code,
                                "keywords":keywords,
                            }
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag and text_data:
                    data_dict[name]={
                        "motion":motion,
                        "length":len(motion),
                        "text":text_data,
                        "code_indices":code,
                        "keywords":keywords,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                skipped.append((name,type(e).__name__))

        if not new_name_list:
            raise RuntimeError("No usable evaluation samples were loaded.")

        name_list,length_list=zip(*sorted(zip(new_name_list,length_list),key=lambda x:x[1]))
        self.length_arr=np.asarray(length_list)
        self.data_dict=data_dict
        self.name_list=name_list
        self.skipped=skipped
        self.val_shuffle=val_shuffle
        self.reset_max_len(self.max_length)

    def reset_max_len(self,length):
        assert length<=self.max_motion_length
        self.pointer=np.searchsorted(self.length_arr,length)
        self.max_length=length

    def inv_transform(self,x):
        return x*self.std+self.mean

    def forward_transform(self,x):
        return (x-self.mean)/self.std

    def __len__(self):
        return len(self.data_dict)-self.pointer

    def __getitem__(self,item):
        idx=self.pointer+item
        name=self.name_list[idx]
        d=self.data_dict[name]
        motion=d["motion"]
        m_length=d["length"]
        text_list=d["text"]
        code=d["code_indices"]
        keywords=d["keywords"]

        td=random.choice(text_list)
        caption,tokens=td["caption"],td["tokens"]

        if len(tokens)<self.max_text_len:
            tokens=["sos/OTHER"]+tokens+["eos/OTHER"]
            sent_len=len(tokens)
            tokens=tokens+["unk/OTHER"]*(self.max_text_len+2-sent_len)
        else:
            tokens=tokens[:self.max_text_len]
            tokens=["sos/OTHER"]+tokens+["eos/OTHER"]
            sent_len=len(tokens)

        pos_one_hots=[]
        word_embeddings=[]
        for token in tokens:
            word_emb,pos_oh=self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None,:])
            word_embeddings.append(word_emb[None,:])
        pos_one_hots=np.concatenate(pos_one_hots,axis=0)
        word_embeddings=np.concatenate(word_embeddings,axis=0)

        if self.unit_length<10:
            coin2=np.random.choice(["single","single","double"])
        else:
            coin2="single"
        if coin2=="double":
            m_length=(m_length//self.unit_length-1)*self.unit_length
        else:
            m_length=(m_length//self.unit_length)*self.unit_length

        start=random.randint(0,len(motion)-m_length)
        motion=motion[start:start+m_length]
        code=code[start:start+m_length].astype(np.float64)

        motion=(motion-self.mean)/self.std

        if m_length<self.max_motion_length:
            motion=np.concatenate(
                [motion,np.zeros((self.max_motion_length-m_length,motion.shape[1]))],axis=0)
            code=np.concatenate(
                [code,np.zeros((self.max_motion_length-m_length,code.shape[1]))],axis=0)

        if self.use_keywords:
            keyword_indices=np.arange(0,33,3)+np.random.randint(0,3,size=(11,))
            keyword_data=np.asarray(keywords[keyword_indices],dtype=np.float32)
        else:
            keyword_data=np.zeros((11,512),dtype=np.float32)

        compatibility_placeholder=np.zeros((1,),dtype=np.float32)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            name,
            code,
            keyword_data,
            compatibility_placeholder,
        )

def DATALoader(dataset_name,is_test,
               batch_size,w_vectorizer,
               num_workers=8,unit_length=4,
               codebook_size=392,
               val_shuffle=True,
               use_keywords=False,
               use_word_only=False,
               meta_dir=None,
               codes_folder_name="codes"):
    ds=Text2MotionDataset(
        dataset_name,is_test,w_vectorizer,
        unit_length=unit_length,
        codebook_size=codebook_size,
        val_shuffle=val_shuffle,
        use_keywords=use_keywords,
        use_word_only=use_word_only,
        meta_dir=meta_dir,
        codes_folder_name=codes_folder_name,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
