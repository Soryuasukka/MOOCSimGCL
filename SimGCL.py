import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.loader import FileIO

# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class SimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SimGCL, self).__init__(conf, training_set, test_set)
        args = self.config['SimGCL']
        self.cl_rate = float(args['lambda'])
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        concept_cfg = args if isinstance(args, dict) else {}
        item_concept_file = concept_cfg.get('item_concept_file', '')
        prerequisite_file = concept_cfg.get('prerequisite_file', '')
        ic_adj, pre_adj, concept_num = self._build_knowledge_graph(item_concept_file, prerequisite_file)
        self.model = SimGCL_Encoder(
            self.data,
            self.emb_size,
            self.eps,
            self.n_layers,
            ic_adj=ic_adj,
            pre_adj=pre_adj,
            concept_num=concept_num
        )

    def _build_knowledge_graph(self, item_concept_file, prerequisite_file):
        if not item_concept_file:
            return None, None, 0

        ic_data = FileIO.load_item_concept(item_concept_file)
        if len(ic_data) == 0:
            return None, None, 0

        concept_map = {}
        for _, concept_id, _ in ic_data:
            if concept_id not in concept_map:
                concept_map[concept_id] = len(concept_map)

        pre_data = []
        if prerequisite_file:
            pre_data = FileIO.load_prerequisite(prerequisite_file)
            for pre_id, post_id, _ in pre_data:
                if pre_id not in concept_map:
                    concept_map[pre_id] = len(concept_map)
                if post_id not in concept_map:
                    concept_map[post_id] = len(concept_map)

        concept_num = len(concept_map)
        if concept_num == 0:
            return None, None, 0

        ic_rows, ic_cols, ic_vals = [], [], []
        for item_id, concept_id, weight in ic_data:
            if item_id in self.data.item:
                ic_rows.append(self.data.item[item_id])
                ic_cols.append(concept_map[concept_id])
                ic_vals.append(weight)

        if len(ic_rows) == 0:
            return None, None, 0

        ic_indices = torch.tensor([ic_rows, ic_cols], dtype=torch.long)
        ic_values = torch.tensor(ic_vals, dtype=torch.float32)
        ic_adj = torch.sparse_coo_tensor(
            ic_indices,
            ic_values,
            torch.Size([self.data.item_num, concept_num])
        ).coalesce()

        if len(pre_data) == 0:
            pre_adj = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.tensor([], dtype=torch.float32),
                torch.Size([concept_num, concept_num])
            ).coalesce()
        else:
            pre_rows, pre_cols, pre_vals = [], [], []
            for pre_id, post_id, weight in pre_data:
                pre_rows.append(concept_map[pre_id])
                pre_cols.append(concept_map[post_id])
                pre_vals.append(weight)
            pre_indices = torch.tensor([pre_rows, pre_cols], dtype=torch.long)
            pre_values = torch.tensor(pre_vals, dtype=torch.float32)
            pre_adj = torch.sparse_coo_tensor(
                pre_indices,
                pre_values,
                torch.Size([concept_num, concept_num])
            ).coalesce()

        return ic_adj, pre_adj, concept_num


    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()



                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, ic_adj=None, pre_adj=None, concept_num=0):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.ic_adj = ic_adj.cuda() if ic_adj is not None else None
        self.pre_adj = pre_adj.cuda() if pre_adj is not None else None
        self.use_knowledge = self.ic_adj is not None and self.pre_adj is not None and concept_num > 0
        if self.use_knowledge:
            self.concept_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(concept_num, self.emb_size)))
        else:
            self.concept_emb = None


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        concept_emb = self.concept_emb

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if self.use_knowledge:
                concept_emb = concept_emb + torch.sparse.mm(self.pre_adj, concept_emb)
                u_emb, i_emb = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num])
                i_emb = i_emb + 0.1 * torch.sparse.mm(self.ic_adj, concept_emb)
                ego_embeddings = torch.cat([u_emb, i_emb], 0)

            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
