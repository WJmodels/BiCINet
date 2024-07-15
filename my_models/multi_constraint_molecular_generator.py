
import os
import json
import torch
from torch import nn
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# from rdkit import Chem
# import rdkit

class MultiConstraintMolecularGenerator(nn.Module):
    def __init__(self, **kwargs):
        super(MultiConstraintMolecularGenerator, self).__init__()

        self.model_path = kwargs.pop(
            "model_path") if "model_path" in kwargs.keys() else None
        self.config_json_path = kwargs.pop(
            "config_json_path") if "config_json_path" in kwargs.keys() else None
        self.tokenizer_path = kwargs.pop(
            "tokenizer_path") if "tokenizer_path" in kwargs.keys() else None
        
        self.pred_graph = kwargs.pop(
            "pred_graph") if "pred_graph" in kwargs.keys() else False
        
        
        self.use_sim = kwargs.pop(
            "use_sim") if "use_sim" in kwargs.keys() else False
        
        self.tokenizer = None
        if self.tokenizer_path is not None:
            self.tokenizer = BartTokenizer.from_pretrained(self.tokenizer_path)

        if self.model_path is not None and os.path.exists(self.model_path):
            assert self.config_json_path is None
            self.config = BartConfig.from_pretrained(self.model_path)
            import ipdb
            ipdb.set_trace()
            if self.pred_graph:
                self.config.d_model = 256
            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_path)

        elif self.config_json_path is not None and os.path.exists(self.config_json_path):
            with open(self.config_json_path, "r") as f:
                json_dict = json.loads(f.read())
            if self.pred_graph:
                json_dict["d_model"] = 256
                json_dict["encoder_attention_heads"] = 8
                json_dict["decoder_attention_heads"] = 8
                json_dict["decoder_ffn_dim"] = 1024
                json_dict["encoder_ffn_dim"] = 1024
            
            
            if self.tokenizer is not None:
                
                if len(self.tokenizer) > json_dict["vocab_size"]:
                    json_dict["vocab_size"] = len(self.tokenizer)
            self.config = BartConfig(**json_dict)
            
            self.model = BartForConditionalGeneration(config=self.config)

        else:
            raise "ERROR: No Model Found.\n"

        
        
        if self.pred_graph:
            self.edge_predictor = GraphPredictor(decoder_dim=self.config.d_model)
        
        
        self.use_sim_layer = nn.Identity()
        if self.use_sim:
            print("use_sim")
            self.use_sim_layer = nn.Linear(self.config.d_model, 64, bias=False)
            

    def forward(self, **kwargs):
        '''
        QED_feature = kwargs.pop(
            "QED_feature") if "QED_feature" in kwargs.keys() else None
        
        if QED_feature is not None:
            input_ids = kwargs.pop(
                "input_ids") if "input_ids" in kwargs.keys() else None
            model_tmp = self.model.model.encoder
            kwargs["inputs_embeds"] = model_tmp.embed_tokens(
                input_ids) * model_tmp.embed_scale
            # print("OOOOOOOOOOOOOOO",kwargs["inputs_embeds"].shape, input_ids.shape, QED_feature.shape)
            for i in range(input_ids.shape[0]):
                kwargs["inputs_embeds"][i][input_ids[i]
                                           == 200, :] = QED_feature[i]
            kwargs["inputs_embeds"][:, 0, :] = QED_feature
        


        molecular_weight_feature = kwargs.pop(
            "molecular_weight_feature") if "molecular_weight_feature" in kwargs.keys() else None
        # if molecular_weight_feature is not None:
        #     input_ids = kwargs.pop(
        #         "input_ids") if "input_ids" in kwargs.keys() else None
        #     model_tmp = self.model.model.encoder
        #     kwargs["inputs_embeds"] = model_tmp.embed_tokens(
        #         input_ids) * model_tmp.embed_scale
        #     # print(kwargs["inputs_embeds"].shape, input_ids.shape, molecular_weight_feature.shape)
        #     for i in range(input_ids.shape[0]):
        #         kwargs["inputs_embeds"][i][input_ids[i]
        #                                    == 300, :] = molecular_weight_feature[i]
        #     kwargs["inputs_embeds"][:, 0, :] = molecular_weight_feature
        '''
        if self.pred_graph is False:
            return self.model(**kwargs)
        else:
            kwargs["output_hidden_states"] = True
            edges = kwargs.pop("edges") if "edges" in kwargs else None
            outputs = self.model(**kwargs)
            
            
            # (self.model.lm_head(outputs["decoder_hidden_states"][-1])!=outputs.logits).sum()
            hidden = outputs["decoder_hidden_states"][-1] #[B, len, d_model]
            pred_edges, edge_loss = self.edge_predictor(hidden, edges)
            outputs.loss = outputs.loss + edge_loss
            return outputs
            

    def infer(self, **kwargs):
        # subsmile——>smiles
        tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else self.tokenizer
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512
        bos_token_id = kwargs.pop(
            "bos_token_id") if "bos_token_id" in kwargs.keys() else 187

        with torch.no_grad():
            result = self.model.generate(max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         bos_token_id=bos_token_id,
                                         pad_token_id=1,
                                         eos_token_id=188,
                                         decoder_start_token_id=bos_token_id, 
                                         **kwargs)
        # print(result)
        dict_ = {"input_ids_tensor": result}
        if tokenizer is not None:
            smiles_list = []
            for _ in range(len(result)):
                try:
                    smiles = [tokenizer.decode(i) for i in result[_] if i<202] #if i<202
                    ## add filter of <SMILES> and </SMILES> by Mr Bao.
                    smiles = [i.replace("<CLASS>", "").replace("</CLASS>", "").replace("<SMILES>", "").replace("</SMILES>", "").replace("<MATERIALS>", "").replace("</MATERIALS>", "").replace("</QED>", "").replace("<QED>", "").replace("<logP>", "").replace("</logP>", "").replace("<pad>", "").replace("</s>", "").replace("</fragment>", "").replace("<fragment>", "").replace("<SA>", "").replace("</SA>", "").replace("<mask>", "") for i in smiles]
                    smiles = "".join(smiles)
                    # print("       smiles",smiles)
                    smiles_list.append(smiles)
                except Exception as e:
                    # print(e)
                    smiles_list.append(None)
            dict_["smiles"] = smiles_list
        return dict_
    
    def decode_graph(self, smiles_token, edges):
        from rdkit import Chem
        import rdkit
        
        
        mol = Chem.RWMol()
        for i in range(len(smiles_token)//2):
            atom = Chem.Atom(smiles_token[(i-1)*2])
            atom_h = int(smiles_token[(i-1)*2+1][0])
            atom.SetNumExplicitHs(atom_h)
            mol.AddAtom(atom)
        
        
        n = len(edges)
        for i in range(n):
            for j in range(i + 1, n):
                if edges[i][j] == 1: # and edges[j][i] == 1:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                elif edges[i][j] == 2: # and edges[j][i] == 2:
                    mol.AddBond(i, j, Chem.BondType.DOUBLE)
                elif edges[i][j] == 3: #and edges[j][i] == 3:
                    mol.AddBond(i, j, Chem.BondType.TRIPLE)
                elif edges[i][j] == 4:# and edges[j][i] == 4:
                    mol.AddBond(i, j, Chem.BondType.AROMATIC)
                elif edges[i][j] == 5:# and edges[j][i] == 6:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    # mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:# and edges[j][i] == 5:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    # mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
        
        return 
    
    def infer_graph(self, **kwargs):
        # subsmile——>mol_graph
        tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else self.tokenizer
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512
        kwargs["output_hidden_states"] = True
        with torch.no_grad():
            
            result = self.model.generate(max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         bos_token_id=187,
                                         pad_token_id=1,
                                         eos_token_id=188,
                                         decoder_start_token_id=187, 
                                         forced_eos_token_id=188,
                                         return_dict_in_generate=True, 
                                         output_scores=True,
                                         **kwargs)
            # ipdb> p self.model.lm_head(hidden).shape
            # torch.Size([1, 37, 4191])
            # ipdb> p self.model.lm_head(hidden).argmax(dim=-1)
            # tensor([[187,  17, 218,  17, 215,  17, 216,  17, 216,  17, 215,  29, 215,  17,
            #         215,  17, 217,  28, 215,  17, 216,  17, 215,  86, 215,  17, 216,  17,
            #         215,  29, 215,  17, 217,  17, 217,  28, 216]], device='cuda:0')
            # result["sequences"]
            # tensor([[187,  17, 218,  17, 215,  17, 216,  17, 216,  17, 215,  29, 215,  17,
            #         215,  17, 217,  28, 215,  17, 216,  17, 215,  86, 215,  17, 216,  17,
            #         215,  29, 215,  17, 217,  17, 217,  28, 216, 188]], device='cuda:0')
            temp = []
            for _ in range(len(result["decoder_hidden_states"])):
                temp.append(result["decoder_hidden_states"][_][0])
            hidden = torch.cat(temp, dim=1) #[B, len, d_model]
            hidden = hidden[:,1:,:]
            pred_edges_proba, *_ = self.edge_predictor(hidden) # [1, 7, num_atoms, num_atoms]
            pred_edges = pred_edges_proba.argmax(dim=1) # [1, num_atoms, num_atoms]
            sequence = result["sequences"]
            for _ in range(len(result["sequences"])):
                smiles_token = [tokenizer.decode(i) for i in sequence[_] if i not in [187, 188]]
                pred_edges = pred_edges[_]
                import ipdb
                ipdb.set_trace()
            
    
    def infer_2(self, **kwargs):
        # smiles-->subsmile
        tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else self.tokenizer
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512
        bos_token_id = kwargs.pop(
            "bos_token_id") if "bos_token_id" in kwargs.keys() else 200

        dict_ = {}
        with torch.no_grad():
            # result = self.model.generate(max_length=max_length,
            #                              num_beams=num_beams,
            #                              num_return_sequences=num_return_sequences,
            #                              bos_token_id=187,
            #                              pad_token_id=1,
            #                              eos_token_id=188,
            #                              decoder_start_token_id=187,
            #                              forced_bos_token_id=187,
            #                              forced_eos_token_id=188,
            #                              **kwargs)
            try:
                result = self.model.generate(max_length=max_length,
                                            num_beams=num_beams,
                                            num_return_sequences=num_return_sequences,
                                            bos_token_id=bos_token_id,
                                            pad_token_id=1,
                                            eos_token_id=201,
                                            decoder_start_token_id=bos_token_id,
                                            **kwargs)
                # print(result)
            except Exception as e:
                print(e)
                dict_["smiles"] = [None] * num_beams
                return dict_
        
        # dict_ = {"input_ids_tensor": result}
        if tokenizer is not None:
            smiles_list = []
            for _ in range(len(result)):
                try:
                    smiles = [tokenizer.decode(i) for i in result[_] if i<202] #if i<202
                    ## add filter of <SMILES> and </SMILES> by Mr Bao.
                    smiles = [i.replace("<CLASS>", "").replace("</CLASS>", "").replace("<SMILES>", "").replace("</SMILES>", "").replace("<MATERIALS>", "").replace("</MATERIALS>", "").replace("<pad>", "").replace("</s>", "") for i in smiles]
                    smiles = "".join(smiles)
                    # print("          smiles",smiles)
                    smiles_list.append(smiles)
                except Exception as e:
                    # print(e)
                    smiles_list.append(None)
            dict_["smiles"] = smiles_list
        return dict_

    def infer_smiles2nmr(self, **kwargs):
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512
        with torch.no_grad():
            result = self.model.generate(max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         bos_token_id=191,
                                         pad_token_id=1,
                                         eos_token_id=192,
                                         decoder_start_token_id=191,
                                         **kwargs)
            result[result<200] = 0
            # result[result>4190] = 0
            
        dict_ = {"input_ids_tensor": result}
        return dict_

    def load_weights(self, path, device=torch.device("cpu")):
        if path is not None:
            model_dict = torch.load(path, map_location=device)
            import collections
            new_model_dict = collections.OrderedDict()
            for k,v in model_dict.items():
                if k[:7] == "module.":
                    new_model_dict[k[7:]] = v
                else:
                    new_model_dict[k] = v
            try:
                self.load_state_dict(new_model_dict, strict=False) 
            except Exception as e:
                print("not strict load")
                
                new_model_dict["model.final_logits_bias"] = torch.cat([new_model_dict["model.final_logits_bias"], 
                                                        torch.randn([1, self.model.final_logits_bias.shape[1] - new_model_dict["model.final_logits_bias"].shape[1]])
                                                        ],dim=-1)
                
                new_model_dict["model.model.shared.weight"] = torch.cat([new_model_dict["model.model.shared.weight"], 
                                                                        torch.randn([self.model.model.shared.weight.shape[0] - new_model_dict["model.model.shared.weight"].shape[0], 768])
                                                                        ],dim=0)
                new_model_dict["model.model.encoder.embed_tokens.weight"] = torch.cat([new_model_dict["model.model.encoder.embed_tokens.weight"], 
                                                                                    torch.randn([self.model.model.encoder.embed_tokens.weight.shape[0] - new_model_dict["model.model.encoder.embed_tokens.weight"].shape[0], 768])
                                                                                    ],dim=0)
                
                new_model_dict["model.model.decoder.embed_tokens.weight"] = torch.cat([new_model_dict["model.model.decoder.embed_tokens.weight"], 
                                                                                        torch.randn([self.model.model.decoder.embed_tokens.weight.shape[0] - new_model_dict["model.model.decoder.embed_tokens.weight"].shape[0], 768])
                                                                                        ],dim=0)
                new_model_dict["model.lm_head.weight"] = torch.cat([new_model_dict["model.lm_head.weight"], 
                                                                                        torch.randn([self.model.lm_head.weight.shape[0] - new_model_dict["model.lm_head.weight"].shape[0], 768])
                                                                                        ],dim=0)
                self.load_state_dict(new_model_dict, strict=False)




class GraphPredictor(nn.Module):

    def __init__(self, decoder_dim=256, coords=False):
        super(GraphPredictor, self).__init__()
        self.coords = coords
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, 7)
        )
        self.edge_loss_fn = nn.CrossEntropyLoss() #reduction="none"
        if coords:
            self.coords_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
                nn.Linear(decoder_dim, 2)
            )

    def forward(self, hidden, edges=None, offset=2):
        if edges is None:
            atom_nums = hidden.shape[1]//offset
        else:
            b, atom_nums, atom_nums = edges.shape
        
        hidden = hidden[:, :atom_nums*2, :]
        if offset == 2:
            even_hidden = hidden[:, 0::2, :]
            odd_hidden = hidden[:, 1::2, :]
            hidden = even_hidden + odd_hidden
        elif offset == 1:
            pass
        
        b, l, dim = hidden.size()
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)
        pred_edges = self.mlp(hh).permute(0, 3, 1, 2)
        
        edge_loss = None
        if edges is not None:
            edge_loss = self.edge_loss_fn(pred_edges, edges)
            # mask = torch.ones_like(edge_loss).to(edge_loss.device)
            # mask[edges==0]=0.01
            # edge_loss = torch.mean(edge_loss*mask)
    
        return pred_edges, edge_loss
