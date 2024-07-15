import torch
from torch.utils.data import Dataset
import json
import random
import ipdb
from tqdm import tqdm
from rdkit import Chem
import numpy as np
from typing import Any, Callable, Dict, List, NewType, Optional

USE_SMALL = False
USE_RDKIT = False



class MyDataset(Dataset):
    def __init__(self, 
                 args, 
                 tokenizer=None, 
                 data_dir=None, 
                 max_length=512, 
                 input_name=["sub_smiles"], 
                 output_name=["smiles"],
                 mlm_name=["smiles"],
                 phase="train",
                 aug_nmr=True,
                 return_smiles=False,
                 debug=True):

        self.args = args
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_length = max_length
        self.debug = debug
        self.phase = phase
        self.aug_nmr = aug_nmr
        self.return_smiles = return_smiles
        if self.phase == "test":
            self.return_smiles = True
        self.input_name = input_name
        self.output_name = output_name
        self.mlm_name = mlm_name
        self.get_kwargs(self.args)
        self.original_data = self.load_raw_data()
        self.data = self.process_raw_data()
        
        
        if self.phase != "train":
            self.flag_dual = False
            self.task_percent = 1
        
        self.task_percent = 0
        
        if self.aug_nmr:
            self.min_1H_NMR_index = self.tokenizer.convert_tokens_to_ids(self.min_1H_NMR)
            self.max_1H_NMR_index = self.tokenizer.convert_tokens_to_ids(self.max_1H_NMR)
            self.min_13C_NMR_index = self.tokenizer.convert_tokens_to_ids(self.min_13C_NMR)
            self.max_13C_NMR_index = self.tokenizer.convert_tokens_to_ids(self.max_13C_NMR)
        
        
    def get_kwargs(self, args):

        self.use_sub_smiles_prob = getattr(args, "use_sub_smiles_prob", 0.0)
        self.use_smiles_prob = getattr(args, "use_smiles_prob", 1.0)
        self.use_fragment_prob = getattr(args, "use_fragment_prob", 0.0)
        self.use_molecular_formula_prob = getattr(args, "use_molecular_formula_prob", 0.0)
        self.use_enzyme_prob = getattr(args, "use_enzyme_prob", 0.0)
        self.use_class_prob = getattr(args, "use_class_prob", 0.0)
        self.aug_smiles = getattr(args, "aug_smiles", False)
        if self.phase != "train":
            self.aug_smiles = False
        

        

        self.flag_dual = getattr(args, "flag_dual", False)
        self.task_percent = getattr(args, "task_percent", 0.5)
    
    def jitter(self, jitter_range: float = 2, precision: float=2):
        jitter_value = np.random.uniform(-jitter_range, +jitter_range)
        encode_jitter_value = int(jitter_value/precision)
        return encode_jitter_value
    

        
        
    def load_raw_data(self):

        original_data = []
        with open(self.data_dir,"r") as f:
            for line in f:
                original_data.append(json.loads(line))
        
        return original_data
    
    def process_raw_data(self):
        data = []
        print("=======start processing raw data======")
        for examples in tqdm(self.original_data):
            
            result = {}
            if "molecular_formula" in examples.keys():
                result["molecular_formula"] = examples["molecular_formula"]
                result["molecular_formula_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<molecular_formula>")] + \
                                                        [self.tokenizer.convert_tokens_to_ids(i) for i in examples["molecular_formula"]] + \
                                                        [self.tokenizer.convert_tokens_to_ids("</molecular_formula>")]
                result["molecular_formula_attention_mask"] = [1 for _ in range(len(result["molecular_formula_input_ids"]))]
                if self.debug:
                    assert len(result["molecular_formula_input_ids"]) == len(result["molecular_formula_input_ids"])
            
            ## List(List())
            if "fragments" in examples.keys():
                result["fragments"] = examples["fragments"]
                result["fragments_input_ids"] = []
                result["fragments_attention_mask"] = []
                for item in examples["fragments"]:
                    tmp = {}
                    tmp["fragments_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<fragment>")] + \
                                                [self.tokenizer.convert_tokens_to_ids(i) for i in item] + \
                                                [self.tokenizer.convert_tokens_to_ids("</fragment>")]
                    tmp["fragments_attention_mask"] = [1 for _ in range(len(tmp["fragments_input_ids"]))]

                    result["fragments_input_ids"].append(tmp["fragments_input_ids"])
                    result["fragments_attention_mask"].append(tmp["fragments_attention_mask"])
                    
                    if self.debug:
                        assert len(tmp["fragments_input_ids"]) == len(tmp["fragments_attention_mask"])

            ## List(List())
            if "smiles" in examples.keys():
                if self.aug_smiles is True:
                    result["smiles"] = examples["smiles"]
                else:
                    ## canonical
                    result["smiles"] = [Chem.MolToSmiles(Chem.MolFromSmiles(examples["smiles"][0]))]
                    
                result["smiles_input_ids"] = []
                result["smiles_attention_mask"] = []
                
                for item in result["smiles"]:
                    tmp = {}
                    tmp["smiles_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<SMILES>")] + \
                                                [self.tokenizer.convert_tokens_to_ids(i) for i in item] + \
                                                [self.tokenizer.convert_tokens_to_ids("</SMILES>")]
                    tmp["smiles_attention_mask"] = [1 for _ in range(len(tmp["smiles_input_ids"]))]

                    result["smiles_input_ids"].append(tmp["smiles_input_ids"])
                    result["smiles_attention_mask"].append(tmp["smiles_attention_mask"])
                    
                    if self.debug:
                        assert len(tmp["smiles_input_ids"]) == len(tmp["smiles_attention_mask"])
            
            if "sub_smiles" in examples.keys():
                result["origin_sub_smiles"] = [item for item in examples["sub_smiles"]]
                result["sub_smiles_input_ids"] = []
                result["sub_smiles_attention_mask"] = []
                tmp = {}
                for k, item in enumerate(examples["sub_smiles"]):
                    tmp["sub_smiles_input_ids"] = [self.tokenizer.convert_tokens_to_ids(i) for i in item]
                    tmp["sub_smiles_attention_mask"] = [1 for _ in range(len(tmp["sub_smiles_input_ids"]))]
                    
                    result["sub_smiles_input_ids"].extend(tmp["sub_smiles_input_ids"])
                    result["sub_smiles_attention_mask"].extend(tmp["sub_smiles_attention_mask"])
                    
                    if (k+1)!=len(examples["sub_smiles"]):
                        result["sub_smiles_input_ids"].append(self.tokenizer.convert_tokens_to_ids("."))
                        result["sub_smiles_attention_mask"].append(1)
                
                result["sub_smiles_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<MATERIALS>")] +\
                                                    result["sub_smiles_input_ids"] +\
                                                [self.tokenizer.convert_tokens_to_ids("</MATERIALS>")]
                result["sub_smiles_attention_mask"] = [1] + result["sub_smiles_attention_mask"] + [1]
                
            

                    
            if "class" in examples.keys():
                result["class"] = examples["class"]
                result["class_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<CLASS>")] + \
                                            [self.tokenizer.convert_tokens_to_ids(examples["class"])] + \
                                            [self.tokenizer.convert_tokens_to_ids("</CLASS>")]
                result["class_attention_mask"] = [1 for _ in range(len(result["class_input_ids"]))]

                if self.debug:
                    assert len(result["class_input_ids"]) == len(result["class_attention_mask"])
                    
            if "enzyme" in examples.keys():
                result["enzyme"] = examples["enzyme"]
                result["enzyme_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<enzyme>")] + \
                                                [self.tokenizer.convert_tokens_to_ids(i) for i in examples["enzyme"]] + \
                                                [self.tokenizer.convert_tokens_to_ids("</enzyme>")]
                result["enzyme_attention_mask"] = [1 for _ in range(len(result["enzyme_input_ids"]))]

                if self.debug:
                    assert len(result["enzyme_input_ids"]) == len(result["enzyme_attention_mask"])
            
        
        print("=======finished processing raw data======")
        
        return data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if random.random() < self.task_percent:
            return self.get_data(idx)
        else:
            return self.get_mlm_data(idx)
    
    
    def get_data(self, idx):
        item = self.data[idx]
        collect_dict = {}
        
        smiles = None
        #enzyme_input_ids
        if "enzyme_input_ids" in item.keys():
            if "enzyme" in self.input_name or "enzyme" in self.output_name:
                if (self.phase == "val" and self.use_enzyme_prob > 0) or random.random() < self.use_enzyme_prob:
                    tmp_dict = {"input_ids": item["enzyme_input_ids"],
                                "attention_mask": item["enzyme_attention_mask"]}
                    collect_dict["enzyme"] = tmp_dict
                    
        #class_input_ids
        if "class_input_ids" in item.keys():
            if "class" in self.input_name or "class" in self.output_name:
                if (self.phase == "val" and self.use_class_prob > 0) or random.random() < self.use_class_prob:
                    tmp_dict = {"input_ids": item["class_input_ids"],
                            "attention_mask": item["class_attention_mask"]}
                    collect_dict["class"] = tmp_dict
        
        #smiles
        #item["smiles_input_ids"]:List[List[int]]
        if "smiles_input_ids" in item.keys():
            if "smiles" in self.input_name or "smiles" in self.output_name:
                if (self.phase == "val" and self.use_smiles_prob > 0) or random.random() < self.use_smiles_prob:
                    rand_idx_smiles = random.randint(0, len(item["smiles_input_ids"])-1)
                    tmp_dict = {"input_ids": item["smiles_input_ids"][rand_idx_smiles],
                        "attention_mask": item["smiles_attention_mask"][rand_idx_smiles]}
                    
                    collect_dict["smiles"] = tmp_dict
                    
                   
                    smiles = item["smiles"][0]
        
        #fragments
        #item["fragments_input_ids"]:List[List[int]]
        if "fragments_input_ids" in item.keys():
            if "fragments" in self.input_name or "fragments" in self.output_name:
                if (self.phase == "val" and self.use_fragment_prob > 0) or random.random() < self.use_fragment_prob:
                    rand_idx_fragments = random.randint(0, len(item["fragments_input_ids"])-1)
                    tmp_dict = {"input_ids": item["fragments_input_ids"][rand_idx_fragments],
                        "attention_mask": item["fragments_attention_mask"][rand_idx_fragments]}
                    
                    collect_dict["fragments"] = tmp_dict
        
        
        # molecular_formula
        if "molecular_formula_input_ids" in item.keys():
            if "molecular_formula" in self.input_name or "molecular_formula" in self.output_name:
                if (self.phase == "val" and self.use_molecular_formula_prob > 0) or random.random() < self.use_molecular_formula_prob:
                    tmp_dict = {"input_ids": item["molecular_formula_input_ids"],
                            "attention_mask": item["molecular_formula_attention_mask"]}
                    collect_dict["molecular_formula"] = tmp_dict
        
        # sub_smiles
        if "sub_smiles_input_ids" in item.keys():
            if "sub_smiles" in self.input_name or "sub_smiles" in self.output_name:
                if (self.phase == "val" and self.use_sub_smiles_prob > 0) or random.random() < self.use_sub_smiles_prob:
                    tmp_dict = {"input_ids": item["sub_smiles_input_ids"],
                            "attention_mask": item["sub_smiles_attention_mask"]}
                    collect_dict["sub_smiles"] = tmp_dict


        
        input = {"input_ids": [],
                "attention_mask": []}
        output = {"input_ids": [],
                "attention_mask": []}
        
        for key in self.input_name:
            if key in collect_dict:
                input["input_ids"].extend(collect_dict[key]["input_ids"])
                input["attention_mask"].extend(collect_dict[key]["attention_mask"])
        
        for key in self.output_name:
            if key in collect_dict:
                output["input_ids"].extend(collect_dict[key]["input_ids"])
                output["attention_mask"].extend(collect_dict[key]["attention_mask"])
        
        input = self.tokenizer.pad(input, return_tensors="pt")
        output = self.tokenizer.pad(output, return_tensors="pt")
        
        
        """
        
        if self.flag_dual:
            flag_smiles_right = True if random.random() < 0.5 else False
        else:
            flag_smiles_right = True
        """
        
        
        if self.flag_dual and self.phase == "train":
            if random.random() < 0.5:
                input, output = output, output
        
        input["idx"] = idx
        input["smiles"] = smiles
        input["input_ids"] = input["input_ids"]
        input["input_attention_mask"] = input["attention_mask"]
        input["output_ids"] = output["input_ids"]
        input["output_attention_mask"] = output["attention_mask"]
        
        return input
    
    def get_mlm_data(self, idx):
        item = self.data[idx]
        collect_dict={}
        smiles = None
        
        #smiles
        #item["smiles_input_ids"]:List[List[int]]
        if "smiles_input_ids" in item.keys():
            if "smiles" in self.mlm_name:
                if (self.phase == "val" and self.use_smiles_prob > 0) or random.random() < self.use_smiles_prob:
                    rand_idx_smiles = random.randint(0, len(item["smiles_input_ids"])-1)
                    tmp_dict = {"input_ids": item["smiles_input_ids"][rand_idx_smiles],
                        "attention_mask": item["smiles_attention_mask"][rand_idx_smiles]}
                    
                    collect_dict["smiles"] = tmp_dict
                    
                    
                    smiles = item["smiles"][0]
        
        # sub_smiles
        if "sub_smiles_input_ids" in item.keys():
            if "sub_smiles" in self.input_name or "sub_smiles" in self.output_name:
                if (self.phase == "val" and self.use_sub_smiles_prob > 0) or random.random() < self.use_sub_smiles_prob:
                    tmp_dict = {"input_ids": item["sub_smiles_input_ids"],
                            "attention_mask": item["sub_smiles_attention_mask"]}
                    collect_dict["sub_smiles"] = tmp_dict
        
        input = {"input_ids": [],
                "attention_mask": []}
        
        for key in self.mlm_name:
            if key in collect_dict:
                input["input_ids"].extend(collect_dict[key]["input_ids"])
                input["attention_mask"].extend(collect_dict[key]["attention_mask"])

        input = self.tokenizer.pad(input, return_tensors="pt")
        
        input["idx"] = idx
        input["smiles"] = smiles
        input["output_ids"] = input["input_ids"].clone()
        input["output_attention_mask"] = input["attention_mask"][1:].clone()
        input["input_ids"] = self.my_torch_mask_tokens(input["input_ids"], self.tokenizer)
        input["input_attention_mask"] = input["attention_mask"]
        
        return input
    
    
    
    def my_torch_mask_tokens(self,
                            inputs,
                            tokenizer,
                            special_tokens_mask: Optional[Any] = None,
                            mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # labels = inputs.clone()
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
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
        indices_replaced = torch.bernoulli(torch.full(input_shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            input_shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(tokenizer), input_shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        # return inputs, labels
        if inputs.shape[0] == 1:
            inputs = inputs.reshape(-1)
        return inputs
        

    def collate_fn(self, batch):
        
        input_max_length = 0
        output_max_length = 0
        for temp in batch:
            input_max_length = max(len(temp["input_ids"]), input_max_length)
            output_max_length = max(len(temp["output_ids"]), output_max_length)
        input_max_length = min(input_max_length, self.max_length)
        output_max_length = min(output_max_length, self.max_length)
        
        idx_list = []
        smiles_list = []
        input_ids_list = []
        input_attention_mask_list = []
        output_ids_list = []
        padding_num = self.tokenizer.convert_tokens_to_ids("<pad>")
        for temp in batch:
            if "idx" in temp:
                idx_list.append(temp["idx"])
            if "smiles" in temp:
                smiles_list.append(temp["smiles"])
            temp_input_ids = temp["input_ids"][: input_max_length]
            temp_attention_mask = temp["attention_mask"][: input_max_length]
            temp_output_ids = temp["output_ids"][: output_max_length]
            if temp_input_ids.shape[-1] < input_max_length:
                temp_input_ids = torch.cat([temp_input_ids, 
                                        torch.ones(input_max_length-temp_input_ids.shape[0]).long()*padding_num])
                temp_attention_mask = torch.cat([temp_attention_mask, 
                                        torch.zeros(input_max_length-temp_attention_mask.shape[0])*0])
            
            input_ids_list.append(temp_input_ids)
            input_attention_mask_list.append(temp_attention_mask)
            
            if temp_output_ids.shape[-1] < output_max_length:
                temp_output_ids = torch.cat([temp_output_ids, 
                                            torch.ones(output_max_length-temp_output_ids.shape[0])*padding_num])

            output_ids_list.append(temp_output_ids)
            
        input = {}
        input["input_ids"] = torch.stack(input_ids_list, dim=0).long()
        input["attention_mask"] = torch.stack(input_attention_mask_list, dim=0).long()
        
        output_ids = torch.stack(output_ids_list, dim=0).long()
        input["labels"] = output_ids[:, 1:]
        input["decoder_input_ids"] = output_ids[:, :-1]
        
        return idx_list, smiles_list, input
