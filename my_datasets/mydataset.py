import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from my_datasets.my_data_collator import MyDataCollator_json, MyDataCollator_json_2
import multiprocessing as mp
import time
#import ipdb

USE_SMALL = False
USE_RDKIT = False
PRED_GRAPH = False 

def parse_folder(folder, data_files=None, key=None):
    if folder is None:
        return
    elif isinstance(folder, list):
        tmp = folder
    elif isinstance(folder, str):
        if os.path.isdir(folder):
            tmp = [os.path.join(
                folder, i) for i in os.listdir(folder) if ".json" in i]
        elif os.path.exists(folder):
            tmp = [folder]
        else:
            #ipdb.set_trace()
            raise
    else:
        raise
    if data_files is not None and key is not None:
        data_files[key] = tmp
    else:
        return tmp

def prepare_dataset(train_folder,
                    validation_folder,
                    tokenizer,
                    preprocessing_num_workers,
                    overwrite_cache,
                    per_device_train_batch_size,
                    per_device_eval_batch_size,
                    block_size=500,
                    train_percent=90,
                    load_cache=False,
                    cache_path=None,
                    cache_dir=None,
                    use_QED_prob=0,
                    use_molecular_formula_prob=0,
                    use_molecular_weight_prob=0,
                    use_logP_prob=0,
                    use_SA_prob=0,
                    use_class_prob=0,
                    use_fragment_prob=0,
                    use_sub_smiles=0.5,
                    flag_dual=False,
                    task_percent=0.5,
                    use_encoder=False,
                    aug_smiles=False,
                    aug_subsmiles=False,
                    pred_graph=False, 
                    mode="reverse",
                    ):
    
    if load_cache and os.path.exists(cache_path):
        try:
            tokenized_datasets = torch.load(cache_path)
        except:
            return prepare_dataset(train_folder,
                                   validation_folder,
                                   tokenizer,
                                   preprocessing_num_workers,
                                   overwrite_cache,
                                   per_device_train_batch_size,
                                   per_device_eval_batch_size,
                                   block_size,
                                   train_percent,
                                   load_cache=False,
                                   cache_path=None,
                                   cache_dir=cache_dir,
                                   use_QED_prob=use_QED_prob,
                                   use_molecular_formula_prob=use_molecular_formula_prob,
                                   use_molecular_weight_prob=use_molecular_weight_prob,
                                   use_logP_prob=use_logP_prob,
                                   use_SA_prob=use_SA_prob,
                                   use_class_prob=use_class_prob,
                                   use_fragment_prob=use_fragment_prob,
                                   use_sub_smiles=use_sub_smiles,
                                   flag_dual=flag_dual,
                                   task_percent=task_percent,
                                   use_encoder=use_encoder,
                                   pred_graph=pred_graph,
                                   )
    else:
        data_files = {}
        parse_folder(train_folder, data_files, "train")
        parse_folder(validation_folder, data_files, "validation")
        
        extension = data_files[list(data_files.keys())[0]][0].split(".")[-1]
        if extension == "txt":
            extension = "text"
        
        raw_datasets = load_dataset(extension,
                                    data_files=data_files,
                                    cache_dir=cache_dir
                                    )
        
        '''
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{train_percent}%:]")
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{train_percent}%]")
        '''
            
        column_names = raw_datasets[list(data_files.keys())[0]].column_names
        # text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function_json(examples):
            # import ipdb
            # ipdb.set_trace()
            result = {}
            if "QED" in examples.keys() and len(examples["QED"]) != 0:
                result["QED"] = examples["QED"]

            if "molecular_formula" in examples.keys():
                tmp = tokenizer(
                    examples["molecular_formula"], max_length=block_size)
                tmp["input_ids"] = [[181] + i[1:-1] + [182]
                                    for i in tmp["input_ids"]]
                result["molecular_formula_input_ids"] = tmp["input_ids"]
                result["molecular_formula_attention_mask"] = tmp["attention_mask"]

            if "molecular_weight" in examples.keys() and len(examples["molecular_weight"]) != 0:
                result["molecular_weight"] = examples["molecular_weight"]

            if "logP" in examples.keys() and len(examples["logP"]) != 0:
                result["logP"] = examples["logP"]

            if "SA" in examples.keys() and len(examples["SA"]) != 0:
                result["SA"] = examples["SA"]
                
            if "class" in examples.keys():
                # tmp = tokenizer(
                #     examples["class"], max_length=block_size)
                # tmp["input_ids"] = [[203] + i[1:-1] + [204]
                #                     for i in tmp["input_ids"]]
                # result["class_input_ids"] = tmp["input_ids"]
                # result["class_attention_mask"] = tmp["attention_mask"]
                
               
                result["class_type"] = examples["class"]
                tmp = tokenizer.convert_tokens_to_ids(examples["class"])
                class_input_ids = [[[202, i, 203]] for i in tmp]
                class_attention_mask = [[[1 for i in range(len(class_input_ids[j][0]))]] for j in range(len(class_input_ids))]
                result["class_input_ids"] = class_input_ids
                result["class_attention_mask"] = class_attention_mask
            
            
            if "fragments" in examples.keys():
                result["fragments_input_ids"] = []
                result["fragments_attention_mask"] = []
                for item in examples["fragments"]:
                    if len(item) == 0:
                        tmp = {"input_ids": [], "attention_mask": []}
                    else:
                        if not USE_SMALL:
                            tmp = tokenizer(item, max_length=block_size)
                        else:
                            tmp = tokenizer([item[0]], max_length=block_size)
                        tmp["input_ids"] = [[183] + i[1:-1] + [184]
                                            for i in tmp["input_ids"]]

                    result["fragments_input_ids"].append(tmp["input_ids"])
                    result["fragments_attention_mask"].append(tmp["attention_mask"])
                    
            if "sub_smiles" in examples.keys():
                if USE_RDKIT:
                    result["origin_sub_smiles"] = [item for item in examples["sub_smiles"]]
                else:
                    result["origin_sub_smiles"] = [item for item in examples["sub_smiles"]]
                    result["sub_smiles_input_ids"] = []
                    result["sub_smiles_attention_mask"] = []
                    for item in examples["sub_smiles"]:
                        if not USE_SMALL:
                            tmp = tokenizer(item, max_length=block_size)
                        else:
                            # with pretrain
                            tmp = tokenizer([item[0]], max_length=block_size)
                        tmp["input_ids"] = [i[1:-1] for i in tmp["input_ids"]]
                        tmp["attention_mask"] = [i[1:-1] for i in tmp["attention_mask"]]

                        result["sub_smiles_input_ids"].append(tmp["input_ids"])
                        result["sub_smiles_attention_mask"].append(tmp["attention_mask"])

            if "smiles" in examples.keys():
                if USE_RDKIT:
                    result["origin_smiles"] = [item[0] for item in examples["smiles"]]
                else:
                    result["origin_smiles"] = [item[0] for item in examples["smiles"]]
                    result["smiles_input_ids"] = []
                    result["smiles_attention_mask"] = []
                    
                
                    
                    for item in examples["smiles"]:
                        if not USE_SMALL:
                            tmp = tokenizer(item, max_length=block_size)
                        else:
                            # with pretrain
                            tmp = tokenizer([item[0]], max_length=block_size)
                        tmp["input_ids"] = [[187] + i[1:-1] + [188] for i in tmp["input_ids"]]

                        result["smiles_input_ids"].append(tmp["input_ids"])
                        result["smiles_attention_mask"].append(tmp["attention_mask"])
                        
                        
            
            if "13C_NMR" in examples.keys():
                result["13C_NMR"] = examples["13C_NMR"]
                tmp = [tokenizer.convert_tokens_to_ids(examples["13C_NMR"][_]) for _ in range(len(examples["13C_NMR"]))] ##List(List())
                C_NMR_input_ids = [[[tokenizer.convert_tokens_to_ids("<13C_NMR>")] + i+ [tokenizer.convert_tokens_to_ids("</13C_NMR>")]] for i in tmp] ##List(List(List()))
                C_NMR_attention_mask = [[[1 for i in range(len(C_NMR_input_ids[j][0]))]] for j in range(len(C_NMR_input_ids))]
                # ipdb.set_trace()
                # print("C_NMR",torch.tensor(C_NMR_input_ids).shape, torch.tensor(C_NMR_attention_mask).shape)
                result["13C_NMR_input_ids"] = C_NMR_input_ids
                result["13C_NMR_attention_mask"] = C_NMR_attention_mask
                # import ipdb
                # ipdb.set_trace()
            
            if "1H_NMR" in examples.keys():
                result["1H_NMR"] = examples["1H_NMR"]
                tmp = [tokenizer.convert_tokens_to_ids(examples["1H_NMR"][_]) for _ in range(len(examples["1H_NMR"]))]
                H_NMR_input_ids = [[[tokenizer.convert_tokens_to_ids("<1H_NMR>")]+ i + [tokenizer.convert_tokens_to_ids("</1H_NMR>")]] for i in tmp]
                H_NMR_attention_mask = [[[1 for i in range(len(H_NMR_input_ids[j][0]))]] for j in range(len(H_NMR_input_ids))]
                # print("H_NMR",torch.tensor(H_NMR_input_ids).shape, torch.tensor(H_NMR_attention_mask).shape)
                result["1H_NMR_input_ids"] = H_NMR_input_ids
                result["1H_NMR_attention_mask"] = H_NMR_attention_mask
                # import ipdb
                # ipdb.set_trace()
            
            if "COSY" in examples.keys():
                result["COSY"] = examples["COSY"]
                tmp = []
                for _ in range(len(examples["COSY"])):
                    tmptmp = []
                    for __ in examples["COSY"][_]:
                        tmptmp.extend(tokenizer.convert_tokens_to_ids(__))
                    tmp.append(tmptmp)
                
                COSY_input_ids = [[[tokenizer.convert_tokens_to_ids("<COSY>")] + i + [tokenizer.convert_tokens_to_ids("</COSY>")]] for i in tmp]
                COSY_attention_mask = [[[1 for i in range(len(COSY_input_ids[j][0]))]] for j in range(len(COSY_input_ids))]
                # print("COSY",torch.tensor(COSY_input_ids).shape, torch.tensor(COSY_attention_mask).shape)
                result["COSY_input_ids"] = COSY_input_ids
                result["COSY_attention_mask"] = COSY_attention_mask
                # import ipdb
                # ipdb.set_trace()
                
            return result



        if extension == "json":
            tokenize_function = tokenize_function_json
            data_collator_train = MyDataCollator_json_2(tokenizer=tokenizer,
                                                        phase="train",
                                                        use_QED_prob=use_QED_prob,
                                                        use_molecular_formula_prob=use_molecular_formula_prob,
                                                        use_molecular_weight_prob=use_molecular_weight_prob,
                                                        use_logP_prob=use_logP_prob,
                                                        use_SA_prob=use_SA_prob,
                                                        use_class_prob=use_class_prob,
                                                        use_fragment_prob=use_fragment_prob,
                                                        use_sub_smiles=use_sub_smiles,
                                                        flag_dual=flag_dual, 
                                                        max_length=block_size,
                                                        task_percent=task_percent,
                                                        use_encoder=use_encoder,
                                                        aug_smiles=aug_smiles,
                                                        aug_subsmiles=aug_subsmiles,
                                                        mode=mode, 
                                                        pred_graph=pred_graph, 
                                                        )
          
            data_collator_val = MyDataCollator_json_2(
                                                    tokenizer=tokenizer, phase="val",
                                                    use_QED_prob=use_QED_prob,
                                                    use_molecular_formula_prob=use_molecular_formula_prob,
                                                    use_molecular_weight_prob=use_molecular_weight_prob,
                                                    use_logP_prob=use_logP_prob,
                                                    use_SA_prob=use_SA_prob,
                                                    use_class_prob=use_class_prob, #class
                                                    use_fragment_prob=use_fragment_prob,
                                                    use_sub_smiles=use_sub_smiles,
                                                    max_length=block_size,
                                                    mode=mode, 
                                                    pred_graph=pred_graph, 
                                                    )

        elif extension == "text":
            # do not upgrade any more
            raise "this is not update anymore"
            # tokenize_function = tokenize_function_text
            # data_collator = MyDataCollator_text(tokenizer)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if cache_path is not None:
            torch.save(tokenized_datasets, cache_path)

    train_dataset = tokenized_datasets["train"] if train_folder is not None else None
    eval_dataset = tokenized_datasets["validation"] if validation_folder is not None else None
    # DataLoaders creation:

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator_train,
        batch_size=per_device_train_batch_size
    ) if train_folder is not None else None

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator_val,
        batch_size=per_device_eval_batch_size
    ) if validation_folder is not None else None

    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


class mp_class():
    def __init__(self, args, tokenizer, cache_size=2, num_files_used=1, pt_save_name=None) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.cache_size = cache_size+4
        self.num_files_used = num_files_used
        self.all_training_json = parse_folder(args.train_folder)
        # self.all_training_json = self.all_training_json[3655:]
        self.queue = mp.Queue(cache_size)
        self.index = 0
        self.lock = mp.Lock()
        self.process = mp.Process(target=self.__loader, args=(self.queue,
                                                              self.lock,
                                                              self.all_training_json,
                                                              num_files_used,
                                                              self.tokenizer,
                                                              self.args,))
                                  
        self.process.daemon = True
        self.process.start()
        assert pt_save_name is not None
        self.save_name = pt_save_name
    def next(self):
        while(self.queue.qsize() < 1):
                time.sleep(5)
        # self.lock.acquire()
        value = self.queue.get()
        # self.lock.release()
        if value is not None:
            print(self.all_training_json[value[0]])
        torch.save(value, self.save_name)
        return value

    @staticmethod
    def __loader(queue: mp.Queue, lock, all_training_json, num_files_used=1, tokenizer=None, args=None):
        i = 0
        while True:
            if i == len(all_training_json):
                lock.acquire()
                queue.put(None)
                lock.release()
                return
            if queue.qsize() < 3:
                (_, _, train_dataloader, _) = prepare_dataset(train_folder=all_training_json[i:i+num_files_used],
                                                              validation_folder=None,
                                                              tokenizer=tokenizer,
                                                              preprocessing_num_workers=args.preprocessing_num_workers,
                                                              overwrite_cache=args.overwrite_cache,
                                                              per_device_train_batch_size=args.per_device_train_batch_size,
                                                              per_device_eval_batch_size=args.per_device_eval_batch_size,
                                                              block_size=args.block_size,
                                                              load_cache=False,
                                                              cache_path=args.cache_path,
                                                              cache_dir=args.cache_dir,
                                                              use_fragment_prob=args.use_fragment_prob,
                                                              use_molecular_weight_prob=args.use_molecular_weight_prob,
                                                              use_logP_prob=args.use_logP_prob,
                                                              use_class_prob=args.use_class_prob,
                                                              use_SA_prob=args.use_SA_prob,
                                                              use_molecular_formula_prob=args.use_molecular_formula_prob,
                                                              use_QED_prob=args.use_QED_prob,
                                                              use_sub_smiles=args.use_sub_smiles,
                                                              flag_dual=args.flag_dual,
                                                              task_percent=args.task_percent
                                                              )
                lock.acquire()
                queue.put((i, train_dataloader))
                i += 1
                lock.release()
            else:
                time.sleep(2)


if __name__ == "__main__":

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        "../models/tokenizer-smiles-roberta-1e")
    path = "../../chem_data/data_pre/realc_t.json"
    path = "../../chem_data/Compound_1e_txt/Compound1hh.txt"
    (train_dataset,
     eval_dataset,
     train_dataloader,
     eval_dataloader) = prepare_dataset(train_folder=path,
                                        validation_folder=None,
                                        tokenizer=tokenizer,
                                        preprocessing_num_workers=4,
                                        overwrite_cache=False,
                                        per_device_train_batch_size=8,
                                        per_device_eval_batch_size=8,
                                        block_size=500,
                                        load_cache=False,
                                        cache_path=None)

    for step, batch in enumerate(train_dataloader):
        print(batch)
        raise

        

    
        
        
        
    


