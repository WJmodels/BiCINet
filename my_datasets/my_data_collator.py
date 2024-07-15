from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
import random
import torch
from transformers import default_data_collator
from transformers.utils.dummy_pt_objects import ElectraForMaskedLM
from my_datasets.nmr_datasets import QEDDataset,molecular_weightDataset,logPDataset,SADataset
from rdkit import Chem
import random
from typing import Any, Union, List, Set
USE_RDKIT = False
import ipdb
import torch.nn.functional as F



class ConvertClass(object):
    def __init__(self, item):
        self.item = item

    def to(self, **kwargs):
        return self.item


def get_new_smiles(old_mol):
    len_mol = old_mol.GetNumAtoms()
    li = []
    for index in range(len_mol):
        try:
            new_smiles = Chem.MolToSmiles(old_mol, rootedAtAtom=index)
            li.append(new_smiles)
        except:
            pass
    if len(li) >= 1:
        return random.choice(li)
    else:
        return None


def smiles2token(tokenizer, smile_string, phase="train", block_size=512):
    smile_string_tmp = smile_string
    if phase == "train":
        old_mol = Chem.MolFromSmiles(smile_string)
        if old_mol is not None:
            try:
                smile_string_tmp = Chem.MolToSmiles(
                    old_mol, isomericSmiles=False, doRandom=True, canonical=False)
            except:
                smile_string_tmp2 = get_new_smiles(old_mol)
                if smile_string_tmp2 is not None:
                    smile_string_tmp = smile_string_tmp2

    tmp = tokenizer([smile_string_tmp], max_length=block_size)
    tmp["input_ids"] = [[187] + i[1:-1] + [188]
                        for i in tmp["input_ids"]]
    tmp["input_ids"] = tmp["input_ids"][0]
    tmp["attention_mask"] = tmp["attention_mask"][0]
    return tmp

def sub_smiles2token(tokenizer, smile_string, phase="train", block_size=512):
    smile_string_tmp = smile_string
    if phase == "train":
        old_mol = Chem.MolFromSmiles(smile_string)
        if old_mol is not None:
            smile_string_tmp2 = get_new_smiles(old_mol)
            if smile_string_tmp2 is not None:
                smile_string_tmp = smile_string_tmp2

    tmp = tokenizer([smile_string_tmp], max_length=block_size)
    tmp["input_ids"] = [i[1:-1] for i in tmp["input_ids"]]
    tmp["attention_mask"] = [i[1:-1] for i in tmp["attention_mask"]]
    tmp["input_ids"] = tmp["input_ids"][0]
    tmp["attention_mask"] = tmp["attention_mask"][0]
    return tmp


def one_of_k_encoding(x: Any, allowable_set: Union[List, Set]) -> List:
    """Converts x to one hot encoding.

    Parameters
    ----------
    x: Any,
        An element of any type
    allowable_set: Union[List, Set]
        Allowable element collection
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


class MyDataCollator_json(object):
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else None
        assert self.tokenizer is not None
        self.phase = kwargs.pop(
            "phase") if "phase" in kwargs.keys() else "val"

        self.QED_dataset = kwargs.pop(
            "QED_dataset") if "QED_dataset" in kwargs.keys() else QEDDataset()

        self.molecular_weight_dataset = kwargs.pop(
            "molecular_weight_dataset") if "molecular_weight_dataset" in kwargs.keys() else molecular_weightDataset()

        self.use_molecular_formula_prob = kwargs.pop(
            "use_molecular_formula_prob") if "use_molecular_formula_prob" in kwargs.keys() else 0

        self.logP_dataset = kwargs.pop(
            "logP_dataset") if "logP_dataset" in kwargs.keys() else logPDataset()

        self.SA_dataset = kwargs.pop(
            "SA_dataset") if "SA_dataset" in kwargs.keys() else SADataset()
                
        self.use_class_prob = kwargs.pop(
            "use_class_prob") if "use_class_prob" in kwargs.keys() else 0
        
        self.use_13C_NMR_prob = kwargs.pop(
            "use_13C_NMR_prob") if "use_13C_NMR_prob" in kwargs.keys() else 0
        
        self.use_1H_NMR_prob = kwargs.pop(
            "use_1H_NMR_prob") if "use_1H_NMR_prob" in kwargs.keys() else 0
        
        self.use_COSY_prob = kwargs.pop(
            "use_COSY_prob") if "use_COSY_prob" in kwargs.keys() else 0

        self.use_fragment_prob = kwargs.pop(
            "use_fragment_prob") if "use_fragment_prob" in kwargs.keys() else 0

        self.use_sub_smiles = kwargs.pop(
            "use_sub_smiles") if "use_sub_smiles" in kwargs.keys() else 1

        self.flag_dual = kwargs.pop(
            "flag_dual") if "flag_dual" in kwargs.keys() else False

        self.TYPE_MODEL = kwargs.pop(
            "TYPE_MODEL") if "TYPE_MODEL" in kwargs.keys() else "new"
        
        self.pred_graph = kwargs.pop("pred_graph") if "pred_graph" in kwargs.keys() else False

    def __call__(self, examples):
        input = {"input_ids": [],
                "attention_mask": []}
        output = {"input_ids": [],
                "attention_mask": []}
        for item in examples:
            import ipdb
            ipdb.set_trace()
            # print(item.keys())
            if self.phase == "train":
                rand_idx_smiles = random.randint(
                    0, len(item["smiles_input_ids"])-1)
                rand_idx_QED = random.randint(0, len(item["QED"])-1)
                rand_idx_molecular_weight = random.randint(0, len(item["molecular_weight"])-1)
                rand_idx_logP = random.randint(0, len(item["logP"])-1)
                rand_idx_SA = random.randint(0, len(item["SA"])-1)
                rand_idx_fragments = -1
                if "fragments_input_ids" in item.keys():
                    len_fragments = len(item["fragments_input_ids"])
                    if len_fragments != 0:
                        rand_idx_fragments = random.randint(0, len_fragments-1)

            else:
                rand_idx_smiles = 0
                rand_idx_QED = 0
                rand_idx_molecular_weight = 0
                rand_idx_logP = 0
                rand_idx_SA = 0
                rand_idx_fragments = -1
                if "fragments_input_ids" in item.keys():
                    len_fragments = len(item["fragments_input_ids"])
                    if len_fragments != 0:
                        rand_idx_fragments = 0

            tmp_ids, tmp_mask = [], []

            if self.phase == "train":
                QED_fn = self.QED_dataset.fill_item
                molecular_weight_fn = self.molecular_weight_dataset.fill_item
                logP_fn = self.logP_dataset.fill_item
                SA_fn = self.SA_dataset.fill_item
            else:
                QED_fn = self.QED_dataset.fill_item
                molecular_weight_fn = self.molecular_weight_dataset.fill_item
                logP_fn = self.logP_dataset.fill_item
                SA_fn = self.SA_dataset.fill_item

            QED = QED_fn(item["QED"][rand_idx_QED])
            molecular_weight = molecular_weight_fn(item["molecular_weight"][rand_idx_molecular_weight])
            logP = logP_fn(item["logP"][rand_idx_logP])
            SA = SA_fn(item["SA"][rand_idx_SA])

            tmp_ids += QED["input_ids"]
            tmp_ids += molecular_weight["input_ids"]
            tmp_ids += logP["input_ids"]
            tmp_ids += SA["input_ids"]
            tmp_mask += QED["attention_mask"]
            tmp_mask += molecular_weight["attention_mask"]
            tmp_mask += logP["attention_mask"]
            tmp_mask += SA["attention_mask"]


            if (self.phase == "val" and self.use_molecular_formula_prob > 0) or random.random() < self.use_molecular_formula_prob:
                if self.TYPE_MODEL == "new":
                    tmp_ids += item["molecular_formula_input_ids"]
                    tmp_mask += item["molecular_formula_attention_mask"]
                elif self.TYPE_MODEL == "old":
                    tmp_ids = item["molecular_formula_input_ids"] + tmp_ids
                    tmp_mask = item["molecular_formula_attention_mask"] + tmp_mask

            if (self.phase == "val" and self.use_fragment_prob > 0) or random.random() < self.use_fragment_prob:
                if rand_idx_fragments != -1:
                    tmp_ids += item["fragments_input_ids"][rand_idx_fragments]
                    tmp_mask += item["fragments_attention_mask"][rand_idx_fragments]

            input["input_ids"].append(tmp_ids)
            input["attention_mask"].append(tmp_mask)

            output["input_ids"].append(
                item["smiles_input_ids"][rand_idx_smiles])
            output["attention_mask"].append(
                item["smiles_attention_mask"][rand_idx_smiles])

        if self.flag_dual is False:
            input = self.tokenizer.pad(input, return_tensors="pt")
            output = self.tokenizer.pad(output, return_tensors="pt")
            if self.TYPE_MODEL == "old":
                input["labels"] = output["input_ids"]
            elif self.TYPE_MODEL == "new":
                input["labels"] = output["input_ids"][:, 1:]
                input["decoder_input_ids"] = output["input_ids"][:, :-1]
            return input
        else:
            input1 = {}
            input1["input_ids"] = input["input_ids"] + output["input_ids"]
            input1["attention_mask"] = input["attention_mask"] + \
                output["attention_mask"]
            output1 = {}
            output1["input_ids"] = output["input_ids"] + input["input_ids"]
            output1["attention_mask"] = output["attention_mask"] + \
                input["attention_mask"]
            input = self.tokenizer.pad(input1, return_tensors="pt")
            output = self.tokenizer.pad(output1, return_tensors="pt")
            if self.TYPE_MODEL == "old":
                input["labels"] = output["input_ids"]
            elif self.TYPE_MODEL == "new":
                input["labels"] = output["input_ids"][:, 1:]
                input["decoder_input_ids"] = output["input_ids"][:, :-1]
            return input


class MyDataCollator_json_2(object):
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else None
        assert self.tokenizer is not None
        self.phase = kwargs.pop(
            "phase") if "phase" in kwargs.keys() else "val"

        self.QED_dataset = kwargs.pop(
            "QED_dataset") if "QED_dataset" in kwargs.keys() else QEDDataset()

        self.use_QED_prob = kwargs.pop(
            "use_QED_prob") if "use_QED_prob" in kwargs.keys() else 0

        self.use_molecular_formula_prob = kwargs.pop(
            "use_molecular_formula_prob") if "use_molecular_formula_prob" in kwargs.keys() else 0

        self.molecular_weight_dataset = kwargs.pop(
            "molecular_weight_dataset") if "molecular_weight_dataset" in kwargs.keys() else molecular_weightDataset()

        self.use_molecular_weight_prob = kwargs.pop(
            "use_molecular_weight_prob") if "use_molecular_weight_prob" in kwargs.keys() else 0

        self.logP_dataset = kwargs.pop(
            "logP_dataset") if "logP_dataset" in kwargs.keys() else logPDataset()

        self.use_logP_prob = kwargs.pop(
            "use_logP_prob") if "use_logP_prob" in kwargs.keys() else 0

        self.SA_dataset = kwargs.pop(
            "SA_dataset") if "SA_dataset" in kwargs.keys() else SADataset()

        self.use_SA_prob = kwargs.pop(
            "use_SA_prob") if "use_SA_prob" in kwargs.keys() else 0
        
        self.use_class_prob = kwargs.pop(
            "use_class_prob") if "use_class_prob" in kwargs.keys() else 0
        
        self.use_13C_NMR_prob = kwargs.pop(
            "use_13C_NMR_prob") if "use_13C_NMR_prob" in kwargs.keys() else 0
        
        self.use_1H_NMR_prob = kwargs.pop(
            "use_1H_NMR_prob") if "use_1H_NMR_prob" in kwargs.keys() else 0
        
        self.use_COSY_prob = kwargs.pop(
            "use_COSY_prob") if "use_COSY_prob" in kwargs.keys() else 0

        self.use_fragment_prob = kwargs.pop(
            "use_fragment_prob") if "use_fragment_prob" in kwargs.keys() else 0

        self.use_sub_smiles = kwargs.pop(
            "use_sub_smiles") if "use_sub_smiles" in kwargs.keys() else 1

        self.flag_dual = kwargs.pop(
            "flag_dual") if "flag_dual" in kwargs.keys() else False

        self.TYPE_MODEL = kwargs.pop(
            "TYPE_MODEL") if "TYPE_MODEL" in kwargs.keys() else "new"

        self.max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512

        self.use_encoder = kwargs.pop(
            "use_encoder") if "use_encoder" in kwargs.keys() else False

        self.task_percent = kwargs.pop(
            "task_percent") if "task_percent" in kwargs.keys() else 0.5

        self.aug_smiles = kwargs.pop(
            "aug_smiles") if "aug_smiles" in kwargs.keys() else False
        
        self.aug_subsmiles = kwargs.pop(
            "aug_subsmiles") if "aug_subsmiles" in kwargs.keys() else False

        self.mode = kwargs.pop("mode") if "mode" in kwargs.keys() else "reverse"
        
        self.pred_graph = kwargs.pop("pred_graph") if "pred_graph" in kwargs.keys() else False

        if self.use_encoder or self.phase == "val":
            self.flag_dual = False
            self.task_percent = 1

    def __call__(self, examples):

        if random.random() < self.task_percent:
            return self.call_1(examples)
        else:
            return self.call_mlm(examples)

    def call_1(self, examples):
        input = {"input_ids": [],
                "attention_mask": []}
        output = {"input_ids": [],
                "attention_mask": []}
        # class_type_list = []
        # val_label = []
        edges_list = []
        # edges_indics_list = [] 
        
        if self.use_encoder:
            QED_tensor_list = []
            molecular_weight_tensor_list = []
            logP_tensor_list = []
            SA_tensor_list = []

        for item in examples:
            # import ipdb
            # ipdb.set_trace()
            # if "class_type" in item:
            #     class_type_list.append(item["class_type"])
            shuffle_list = []
            #print(item.keys())
            ## 是否使用
            if self.flag_dual:
                flag_smiles_right = True if random.random() < 0.5 else False
            else:
                flag_smiles_right = True

            # QED
            if "QED" in item.keys():
                if (self.phase == "val" and self.use_QED_prob > 0) or random.random() < self.use_QED_prob:
                    if self.phase == "train":
                        rand_idx_QED = random.randint(0, len(item["QED"])-1)
                        QED_fn = self.QED_dataset.fill_item
                    else:
                        rand_idx_QED = 0
                        QED_fn = self.QED_dataset.fill_item
                    # print("##############################",item["QED"])
                    QED = QED_fn(item["QED"][rand_idx_QED],
                                want_token=not self.use_encoder)
                    if self.use_encoder:
                        QED_tesnor, QED = QED
                        QED_tensor_list.append(QED_tesnor)
                    shuffle_list.append(QED)
                    # tmp_ids += QED["input_ids"]
                    # tmp_mask += QED["attention_mask"]

            # # molecular_formula
            # if "molecular_formula_input_ids" in item.keys():
            #     if (self.phase == "val" and self.use_molecular_formula_prob > 0) or random.random() < self.use_molecular_formula_prob:
            #         if len(item["molecular_formula_input_ids"]) > 2:
            #             tmp = {"input_ids": item["molecular_formula_input_ids"],
            #                     "attention_mask": item["molecular_formula_attention_mask"]}
            #             shuffle_list.append(tmp)
            
            # molecular_weight
            if "molecular_weight" in item.keys():
                if (self.phase == "val" and self.use_molecular_weight_prob > 0) or random.random() < self.use_molecular_weight_prob:
                    if self.phase == "train":
                        rand_idx_molecular_weight = random.randint(0, len(item["molecular_weight"])-1)
                        molecular_weight_fn = self.molecular_weight_dataset.fill_item
                    else:
                        rand_idx_molecular_weight = 0
                        molecular_weight_fn = self.molecular_weight_dataset.fill_item
                    # print("##############################",item["molecular_weight"])
                    molecular_weight = molecular_weight_fn(item["molecular_weight"][rand_idx_molecular_weight],
                                want_token=not self.use_encoder)
                    if self.use_encoder:
                        molecular_weight_tesnor, molecular_weight = molecular_weight
                        molecular_weight_tensor_list.append(molecular_weight_tesnor)
                    shuffle_list.append(molecular_weight)
                    # tmp_ids += molecular_weight["input_ids"]
                    # tmp_mask += molecular_weight["attention_mask"]

            # logP
            if "logP" in item.keys():
                if (self.phase == "val" and self.use_logP_prob > 0) or random.random() < self.use_logP_prob:
                    if self.phase == "train":
                        rand_idx_logP = random.randint(0, len(item["logP"])-1)
                        logP_fn = self.logP_dataset.fill_item
                    else:
                        rand_idx_logP = 0
                        logP_fn = self.logP_dataset.fill_item
                    # print("##############################",item["logP"])
                    logP = logP_fn(item["logP"][rand_idx_logP],
                                want_token=not self.use_encoder)
                    if self.use_encoder:
                        logP_tesnor, logP = logP
                        logP_tensor_list.append(logP_tesnor)
                    shuffle_list.append(logP)
                    # tmp_ids += logP["input_ids"]
                    # tmp_mask += logP["attention_mask"]

            # SA
            if "SA" in item.keys():
                if (self.phase == "val" and self.use_SA_prob > 0) or random.random() < self.use_SA_prob:
                    if self.phase == "train":
                        rand_idx_SA = random.randint(0, len(item["SA"])-1)
                        SA_fn = self.SA_dataset.fill_item
                    else:
                        rand_idx_SA = 0
                        SA_fn = self.SA_dataset.fill_item
                    # print("##############################",item["SA"])
                    SA = SA_fn(item["SA"][rand_idx_SA],
                                want_token=not self.use_encoder)
                    if self.use_encoder:
                        SA_tesnor, SA = SA
                        SA_tensor_list.append(SA_tesnor)
                    shuffle_list.append(SA)
                    # tmp_ids += SA["input_ids"]
                    # tmp_mask += SA["attention_mask"]


            if "smiles_input_ids" in item.keys():
                if (self.phase == "val" and self.use_sub_smiles > 0) or random.random() < self.use_sub_smiles:
                    
                    
                    if "class_input_ids" in item.keys():
                        if (self.phase == "val" and self.use_class_prob > 0) or random.random() < self.use_class_prob:
                            len_class = len(item["class_input_ids"])
                            if len_class != 0:
                                # if self.phase == "train": 
                                
                                tmp = {"input_ids": item["class_input_ids"][0],
                                    "attention_mask": item["class_attention_mask"][0]}
                                shuffle_list.append(tmp)
                    
                    
                    if self.use_fragment_prob>0:
                        # fragments_input_ids
                        if "fragments_input_ids" in item.keys() and flag_smiles_right:
                            if (self.phase == "val" and self.use_fragment_prob > 0) or random.random() < self.use_fragment_prob:
                                len_fragments = len(item["fragments_input_ids"])
                                if len_fragments != 0:
                                    rand_idx_fragments = 0
                                    tmp = {"input_ids": item["fragments_input_ids"][rand_idx_fragments],
                                            "attention_mask": item["fragments_attention_mask"][rand_idx_fragments]}
                                    shuffle_list.append(tmp)
                    
                    else:
                        if self.pred_graph is False:
                            rand_idx_smiles = random.randint(0, len(item["smiles_input_ids"])-1)
                            if (self.phase == "train" and self.aug_smiles is False) or self.phase == "val": 
                                tmp_1 = {"input_ids": item["smiles_input_ids"][rand_idx_smiles],
                                        "attention_mask": item["smiles_attention_mask"][rand_idx_smiles]}
                            else:
                                tmp_1 = smiles2token(self.tokenizer, 
                                                    item["origin_smiles"], phase=self.phase, block_size=self.max_length)
                        else:
                            if "graph_input_ids" in item:
                                tmp_1 =  {
                                    "input_ids": item["graph_input_ids"][0],
                                    "attention_mask": item["graph_attention_mask"][0]
                                }
                                edges_list.append(item["graph_edges"])
                                # edges_indics_list.append(item["graph_atom_indics"])
                            
                            else:
                                smiles = item["origin_smiles"]
                                tmp, edges, edges_indics = tokenizer_molgraph(smiles, self.tokenizer)
                                tmp_1 =  {
                                    "input_ids":tmp[0],
                                    "attention_mask":[1 for _ in range(len(tmp[0]))]
                                }
                                edges_list.append(edges)
                                # edges_indics_list.append(edges_indics)
                            
                        shuffle_list.append(tmp_1)

                                
                                # if self.phase == "train":
                                #     list_rand_idx_fragments = random.sample(
                                #         [i for i in range(len_fragments)], min(random.choice([1, 2, 3]), len_fragments))
                                #     for rand_idx_fragments in list_rand_idx_fragments:
                                #         # rand_idx_fragments = random.randint(
                                #         #     0, len_fragments-1)
                                #         tmp = {"input_ids": item["fragments_input_ids"][rand_idx_fragments],
                                #             "attention_mask": item["fragments_attention_mask"][rand_idx_fragments]}
                                #         shuffle_list.append(tmp)
                                # else:
                                #     rand_idx_fragments = 0
                                #     tmp = {"input_ids": item["fragments_input_ids"][rand_idx_fragments],
                                #             "attention_mask": item["fragments_attention_mask"][rand_idx_fragments]}
                                #     shuffle_list.append(tmp)

                                

            tmp_2 = {"input_ids": [],
                    "attention_mask": []}
            
            # molecular_formula
            if "molecular_formula_input_ids" in item.keys():
                if (self.phase == "val" and self.use_molecular_formula_prob > 0) or random.random() < self.use_molecular_formula_prob:
                    if len(item["molecular_formula_input_ids"]) > 2:
                        # import ipdb
                        # ipdb.set_trace()
                        tmp_2["input_ids"].extend(item["molecular_formula_input_ids"])
                        tmp_2["attention_mask"].extend([1 for _ in item["molecular_formula_input_ids"]])
                        # tmp = {"input_ids": item["molecular_formula_input_ids"],
                        #         "attention_mask": item["molecular_formula_attention_mask"]}
                        # shuffle_list.append(tmp)
                

            
            ## sub_smiles
            len_sub_smiles = len(item["sub_smiles_input_ids"]) if "sub_smiles_input_ids" in item else 0
            if len_sub_smiles > 0:
                tmp_2["input_ids"].append(200)
                tmp_2["attention_mask"].append(1)
                for idx_sub_smiles in range(len_sub_smiles):
                    if (self.phase == "train" and self.aug_subsmiles is False) or self.phase == "val": 
                        tmp_2["input_ids"].extend(item["sub_smiles_input_ids"][idx_sub_smiles])
                        tmp_2["attention_mask"].extend(item["sub_smiles_attention_mask"][idx_sub_smiles])
                    else:
                        sub_smile_tmp = sub_smiles2token(self.tokenizer, 
                                                        item["origin_sub_smiles"][idx_sub_smiles], phase=self.phase, block_size=self.max_length)
                        tmp_2["input_ids"].extend(sub_smile_tmp["input_ids"])
                        tmp_2["attention_mask"].extend(sub_smile_tmp["attention_mask"])

                    if idx_sub_smiles < len_sub_smiles-1:
                        tmp_2["input_ids"].append(75)
                        tmp_2["attention_mask"].append(1)
                
                tmp_2["input_ids"].append(201)
                tmp_2["attention_mask"].append(1)
            
            if flag_smiles_right:
                output["input_ids"].append(tmp_2["input_ids"])
                output["attention_mask"].append(tmp_2["attention_mask"])
            else:
                input["input_ids"].append(tmp_2["input_ids"])
                input["attention_mask"].append(tmp_2["attention_mask"])
                

            if len(shuffle_list) >= 1:
                #print(shuffle_list)

                # random.shuffle(shuffle_list)
                if flag_smiles_right:
                    tmp_ids = [i["input_ids"] for i in shuffle_list]
                    tmp_ids = [x for j in tmp_ids for x in j]
                    tmp_mask = [i["attention_mask"]for i in shuffle_list]
                    tmp_mask = [x for j in tmp_mask for x in j]
                    input["input_ids"].append(tmp_ids)
                    input["attention_mask"].append(tmp_mask)
                else:
                    #tmp_ids = shuffle_list[0]["input_ids"]
                    #tmp_mask = shuffle_list[0]["attention_mask"]
                    tmp_ids = [i["input_ids"]for i in shuffle_list]
                    tmp_ids = [x for j in tmp_ids for x in j]
                    
                    tmp_mask = [i["attention_mask"]for i in shuffle_list]
                    tmp_mask = [x for j in tmp_mask for x in j]
                    
                    output["input_ids"].append(tmp_ids)
                    output["attention_mask"].append(tmp_mask)
            else:
                if USE_RDKIT is False:
                    len_sub_smiles = len(item["origin_sub_smiles"]) if "origin_sub_smiles" in item else 0
                    #print(len_sub_smiles)
                    if len_sub_smiles > 0:
                        tmp_3 = {"input_ids": [],
                                "attention_mask": []}
                        tmp_3["input_ids"].append(200)
                        tmp_3["attention_mask"].append(1)
                        for idx_sub_smiles in range(len_sub_smiles):
                           
                            sub_smile_tmp = sub_smiles2token(
                                self.tokenizer, item["origin_sub_smiles"][idx_sub_smiles], phase=self.phase, block_size=self.max_length)
                            tmp_3["input_ids"].extend(sub_smile_tmp["input_ids"])
                            tmp_3["attention_mask"].extend(sub_smile_tmp["attention_mask"])
                            if idx_sub_smiles < len_sub_smiles-1:
                                tmp_3["input_ids"].append(75)
                                tmp_3["attention_mask"].append(1)
                        tmp_3["input_ids"].append(201)
                        tmp_3["attention_mask"].append(1)
                    if flag_smiles_right:
                        input["input_ids"].append(tmp_3["input_ids"])
                        input["attention_mask"].append(tmp_3["attention_mask"])
                    else:
                        output["input_ids"].append(tmp_3["input_ids"])
                        output["attention_mask"].append(tmp_3["attention_mask"])

                '''        
                if USE_RDKIT is False:
                    smile_tmp = smiles2token(
                    self.tokenizer, item["origin_smiles"], phase=self.phase, block_size=self.max_length)
                    #rand_idx_smilestmp = rand_idx_smiles_2
                    # rand_idx_smilestmp = rand_idx_smiles
                    if flag_smiles_right:
                        input["input_ids"].append(smile_tmp["input_ids"])
                        input["attention_mask"].append(smile_tmp["attention_mask"])
                    else:
                        output["input_ids"].append(smile_tmp["input_ids"])
                        output["attention_mask"].append(smile_tmp["attention_mask"])
                else:
                    if flag_smiles_right:
                        input["input_ids"].append(smile_tmp["input_ids"])
                        input["attention_mask"].append(smile_tmp["attention_mask"])
                    else:
                        output["input_ids"].append(smile_tmp["input_ids"])
                        output["attention_mask"].append(smile_tmp["attention_mask"])
                '''
        

        if self.mode == "forwards" or self.mode == "forward":
            input, output = output, input
        

        input["attention_mask"] = [[1 for i in j] for j in input["input_ids"]]
        output["attention_mask"] = [[1 for i in j] for j in output["input_ids"]]

        input = self.tokenizer.pad(input, return_tensors="pt")
        output = self.tokenizer.pad(output, return_tensors="pt")
        if input["input_ids"].shape[1] > self.max_length:
            input["input_ids"] = input["input_ids"][:, :self.max_length]
            input["attention_mask"] = input["attention_mask"][:, :self.max_length]
        if output["input_ids"].shape[1] > self.max_length:
            output["input_ids"] = output["input_ids"][:, :self.max_length]
            output["attention_mask"] = output["attention_mask"][:, :self.max_length]
        if self.TYPE_MODEL == "old":
            input["labels"] = output["input_ids"]
        elif self.TYPE_MODEL == "new":
            input["labels"] = output["input_ids"][:, 1:]
            input["decoder_input_ids"] = output["input_ids"][:, :-1]
        # if input["labels"].shape[1] >self.max_length or input["input_ids"].shape[1] >self.max_length:
        #     print(input["labels"].shape, input["input_ids"].shape)
        if self.use_encoder:
            QED_tensor_list = torch.stack(QED_tensor_list)
            molecular_weight_tensor_list = torch.stack(molecular_weight_tensor_list)
            logP_tensor_list = torch.stack(logP_tensor_list)
            SA_tensor_list = torch.stack(SA_tensor_list)
            input["QED"] = QED_tensor_list
            input["molecular_weight"] = molecular_weight_tensor_list
            input["logP"] = logP_tensor_list
            input["SA"] = SA_tensor_list


        # if self.phase == "val":
        #     input["val_label"] = ConvertClass(val_label)
        # if self.phase == "train" and len(class_type_list)>0:
        #     input["class_type"] = torch.tensor([int(_.split("_")[-1])-1 for _ in class_type_list])
        if len(edges_list)>0:
            max_len = max([len(edges) for edges in edges_list])
            input['edges'] = torch.stack(
                        [F.pad(torch.from_numpy(edges), (0, max_len - len(edges), 0, max_len - len(edges)), value=-100) for edges in edges_list], 
                        dim=0)
        # if len(edges_indics_list)>0:
        #     graph_indics = torch.cat([torch.tensor([_]*len(edges_indics)) for _, edges_indics in enumerate(edges_indics_list)], dim=0).reshape(-1, 1)
        #     edges_indics = torch.cat(edges_indics_list,dim=0)
        #     input['graph_indics'] = graph_indics
        #     input['edges_indics'] = edges_indics
        
        import ipdb
        ipdb.set_trace()
            
        return input
    

    def call_mlm(self, examples):
        # mlm: Masked Language Model
        input = {"input_ids": [],
                "attention_mask": []}
        # class_type_list = []

        for item in examples:
            # if "class_type" in item:
            #     class_type_list.append(item["class_type"])
            #print(item.keys())
            shuffle_list = []

            # molecular_formula
            # if "molecular_formula_input_ids" in item.keys():
            #     if (self.phase == "val" and self.use_molecular_formula_prob > 0) or random.random() < self.use_molecular_formula_prob:
            #         if len(item["molecular_formula_input_ids"]) > 2:
            #             tmp = {"input_ids": item["molecular_formula_input_ids"],
            #                    "attention_mask": item["molecular_formula_attention_mask"]}
            #             shuffle_list.append(tmp)

            len_sub_smiles = len(item["sub_smiles_input_ids"]) if ("sub_smiles_input_ids" in item) else 0
            if len_sub_smiles != 0:
                tmp_1 = {"input_ids": [],
                         "attention_mask": []}
                # if "class_input_ids" in item:
                #     tmp_1["input_ids"].extend(item["class_input_ids"][0])
                #     tmp_1["attention_mask"].extend(item["class_attention_mask"][0])

                tmp_1["input_ids"].append(200)
                tmp_1["attention_mask"].append(1)
                if self.phase == "train" or self.phase == "val":
                    for idx_sub_smiles in range(len_sub_smiles):
                        tmp_1["input_ids"].extend(item["sub_smiles_input_ids"][idx_sub_smiles])
                        tmp_1["attention_mask"].extend(item["sub_smiles_attention_mask"][idx_sub_smiles])
                        if idx_sub_smiles < len_sub_smiles-1:
                            tmp_1["input_ids"].append(75)
                            tmp_1["attention_mask"].append(1)
                    tmp_1["input_ids"].append(201)
                    tmp_1["attention_mask"].append(1)
                    shuffle_list.append(tmp_1)
                    tmp_ids = [i["input_ids"] for i in shuffle_list]
                    tmp_ids = [x for j in tmp_ids for x in j]
                    tmp_mask = [i["attention_mask"]for i in shuffle_list]
                    tmp_mask = [x for j in tmp_mask for x in j]
                    input["input_ids"].append(tmp_ids)
                    input["attention_mask"].append(tmp_mask)
            """
            if USE_RDKIT is False:
                if self.phase == "train":
                    rand_idx_smiles = random.randint(
                        0, len(item["smiles_input_ids"])-1)
                else:
                    rand_idx_smiles = 0
                tmp = {"input_ids": item["smiles_input_ids"][rand_idx_smiles],
                       "attention_mask": item["smiles_attention_mask"][rand_idx_smiles]}
            else:
                tmp = smiles2token(
                    self.tokenizer, item["origin_smiles"], phase=self.phase, block_size=self.max_length)
            shuffle_list.append(tmp)
            
            tmp_ids = [i["input_ids"]for i in shuffle_list]
            tmp_ids = [x for j in tmp_ids for x in j]
            tmp_mask = [i["attention_mask"]for i in shuffle_list]
            tmp_mask = [x for j in tmp_mask for x in j]
            input["input_ids"].append(tmp_ids)
            input["attention_mask"].append(tmp_mask)
            """
        input = self.tokenizer.pad(input, return_tensors="pt")
        if input["input_ids"].shape[1] > self.max_length:
            input["input_ids"] = input["input_ids"][:, :self.max_length]
            input["attention_mask"] = input["attention_mask"][:, :self.max_length]

        input["labels"] = input["input_ids"][:, 1:].clone()
        input["decoder_input_ids"] = input["input_ids"][:, :-1].clone()
        input["input_ids"] = my_torch_mask_tokens(input["input_ids"], self.tokenizer)
        input["decoder_input_ids"] = my_torch_mask_tokens(input["decoder_input_ids"], self.tokenizer)

        
        # if self.phase == "train" and len(class_type_list)>0:
        #     input["class_type"] = torch.tensor([int(_.split("_")[-1])-1 for _ in class_type_list])
        return input


def my_torch_mask_tokens(inputs,
                         tokenizer,
                         special_tokens_mask: Optional[Any] = None,
                         mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # labels = inputs.clone()
    input_shape = inputs.shape
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(input_shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        input_shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        input_shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), input_shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    # return inputs, labels
    return inputs

def convert_bond_type_2_int(bond):
    if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
        return 1
    elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bond.GetBondType() == rdkit.Chem.rdchem.BondType.AROMATIC:
        return 4

import rdkit
from rdkit import Chem
import random
import numpy as np
def get_graph(mol, encode_Hs=True, shuffle_nodes=True, RemoveHs=True, Kekulize=False):
    
    if RemoveHs:
        mol = Chem.RemoveHs(mol)
    if Kekulize:
        Chem.Kekulize(mol)

    chiral_atoms = []
    chiral_centers = Chem.FindMolChiralCenters(mol)
    if len(chiral_centers)>0:
        chiral_atoms = [atom_idx for (atom_idx, _) in chiral_centers]
    
    symbols = []
    index_map = {}
    
    atoms = [atom for atom in mol.GetAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    for i, atom in enumerate(atoms):
        atom_symbol = atom.GetSymbol()
       
        # atom_charge = atom.GetFormalCharge()
        
        if encode_Hs is True:
            atom_h = atom.GetTotalNumHs()
            last_atom_symbol = [atom_symbol, "%dH"%(atom_h)]
        else:
            last_atom_symbol = [atom_symbol]
        # last_atom_symbol = [atom_symbol, "%dH"%(atom_h), "charge_%d"(atom_charge)]
        symbols.append(last_atom_symbol)
        index_map[atom.GetIdx()] = i
        
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.GetBonds():
        s = index_map[bond.GetBeginAtomIdx()]
        t = index_map[bond.GetEndAtomIdx()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = convert_bond_type_2_int(bond)
        edges[t, s] = convert_bond_type_2_int(bond)
        if (len(chiral_atoms)>0) and (bond.GetBondDir() in [rdkit.Chem.rdchem.BondDir.BEGINDASH, rdkit.Chem.rdchem.BondDir.BEGINWEDGE]):
          
            if (bond.GetEndAtomIdx() in chiral_atoms) and (bond.GetBeginAtomIdx() not in chiral_atoms):
                s, t = t, s
            elif (bond.GetBeginAtomIdx() in chiral_atoms) and (bond.GetEndAtomIdx() not in chiral_atoms):
                    pass
            else:
                return {'num_atoms': -1}

            if bond.GetBondDir() == rdkit.Chem.rdchem.BondDir.BEGINDASH:
                edges[s, t] = 6
                edges[t, s] = 5
            else:
                edges[s, t] = 5
                edges[t, s] = 6
                
    graph = {
        'symbols': symbols,
        'edges': edges,
        'num_atoms': len(symbols)
    }
    return graph

def tokenizer_molgraph(smiles, tokenizer, offset=0):
    
    cano_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    mol = Chem.MolFromSmiles(cano_smiles)
    
    graph = get_graph(mol)
    tmp = [] ##List(List())
    for i in range(len(graph["symbols"])):
        for j in graph["symbols"][i]:
            tmp.append(tokenizer.convert_tokens_to_ids(j))
    tmp = [[187]+tmp+[188]]
            
    edges = graph["edges"]

    edges_indics = []
    i = 0
    j = 0
    while i < len(graph["symbols"]):
        
        
        edges_indics.extend([[j, j+_] for _ in range(len(graph["symbols"][i]))])
        j = j + len(graph["symbols"][i])
        i = i + 1
    
    return tmp, edges, torch.tensor(edges_indics)
    
