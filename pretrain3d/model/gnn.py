from numpy.lib.shape_base import _put_along_axis_dispatcher
import torch
from torch import nn
from torch import random
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from pretrain3d.utils.features import get_atom_feature_dims, get_bond_feature_dims
from pretrain3d.model.conv import (
    MLP,
    DropoutIfTraining,
    MetaLayer,
    MLPwoLastAct,
    MLPwithselfBN,
    MLPwoLastActwithselfBN,
)
import torch.nn.functional as F
from pretrain3d.utils.torch_util import GradMultiply

_REDUCER_NAMES = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}


class GNBlock(nn.Module):
    def __init__(
        self,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 128,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 8,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        face_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        use_face: bool = True,
        dropout: float = 0.1,
        layernorm_before: bool = False,
        encoder_dropout: float = 0.0,
        use_bn: bool = False,
        global_attn: bool = False,
        node_attn: bool = False,
        face_attn: bool = False,
        pred_pos_residual: bool = False,
        pos_embedding=None,
    ):
        super().__init__()
        self.gnn_layers = nn.ModuleList()
        for i in range(num_message_passing_steps):
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLPwithselfBN(
                    latent_size * (6 if use_face else 4),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLPwithselfBN(
                    latent_size * (5 if use_face else 4),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            if i == num_message_passing_steps - 1:
                global_model = None
            else:
                global_model = MLPwithselfBN(
                    latent_size * (4 if use_face else 3),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                )
            if use_face and i < num_message_passing_steps - 1:
                face_model = MLPwithselfBN(
                    latent_size * 5,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                )
            else:
                face_model = None
            self.gnn_layers.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    face_model=face_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_edges_for_face_fn=_REDUCER_NAMES[face_reducer],
                    face_attn=face_attn,
                    node_attn=node_attn,
                    global_attn=global_attn,
                    embed_dim=latent_size,
                )
            )

        self.pos_embedding = pos_embedding
        self.pos_decoder = MLPwoLastAct(latent_size, [latent_size, 3])
        self.dropout = dropout
        self.pred_pos_residual = pred_pos_residual

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        u,
        node_batch,
        edge_batch,
        num_nodes,
        num_edges,
        face=None,
        face_batch=None,
        face_mask=None,
        face_index=None,
        num_faces=None,
        nf_node=None,
        nf_face=None,
        mode=None,
        pos=None,
    ):
        pos_predictions = []
        last_pos_pred = x.new_zeros((x.shape[0], 3)).uniform_(-1, 1)
        pos_mask_idx = self.pos_embedding.get_mask_idx(pos, mode=mode)
        for layer in self.gnn_layers:
            extended_x, extended_edge_attr = self.pos_embedding(
                pos,
                x,
                edge_attr,
                edge_index,
                last_pred=last_pos_pred,
                mask_idx=pos_mask_idx,
                mode=mode,
            )
            x_1, edge_attr_1, u_1, face_1 = layer(
                extended_x,
                edge_index,
                extended_edge_attr,
                u,
                node_batch,
                edge_batch,
                face_batch,
                face,
                face_mask,
                face_index,
                num_nodes,
                num_faces,
                num_edges,
                nf_node,
                nf_face,
                mode=mode,
            )
            x = F.dropout(x_1, p=self.dropout, training=self.training) + x
            edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
            u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            if face is not None:
                face = F.dropout(face_1, p=self.dropout, training=self.training) + face

            if self.pred_pos_residual:
                delta_pos = self.pos_decoder(x)
                last_pos_pred = self.move2origin(last_pos_pred + delta_pos, node_batch, num_nodes)
            else:
                last_pos_pred = self.pos_decoder(x)
                last_pos_pred = self.move2origin(last_pos_pred, node_batch, num_nodes)

            pos_predictions.append(last_pos_pred)

        return x, edge_attr, face, u, pos_predictions, pos_mask_idx

    def move2origin(self, pos, node_batch, num_nodes):
        pos_mean = global_mean_pool(pos, node_batch)
        return pos - torch.repeat_interleave(pos_mean, num_nodes, dim=0)


class GNNet(nn.Module):
    def __init__(
        self,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 2,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 8,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        face_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        use_face: bool = True,
        dropout: float = 0.1,
        graph_pooling: str = "sum",
        layernorm_before: bool = False,
        encoder_dropout: float = 0.0,
        pooler_dropout: float = 0.0,
        use_bn: bool = False,
        global_attn: bool = False,
        node_attn: bool = False,
        face_attn: bool = False,
        mask_prob: float = 0.15,
        pos_mask_prob: float = 0.15,
        pred_pos_residual: bool = False,
        raw_with_pos: bool = False,
        attr_predict: bool = False,
        num_tasks: int = 0,
        ap_hid_size: int = None,
        ap_mlp_layers: int = None,
        gradmultiply: float = 0.1,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.encoder_edge = MLP(
            sum(get_bond_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.node_embedding = AtomEmbeddingwithMask(
            latent_size=latent_size,
            mlp_hidden_size=mlp_hidden_size,
            mlp_layers=mlp_layers,
            use_layernorm=use_layer_norm,
            mask_prob=mask_prob,
        )
        self.global_init = nn.Parameter(torch.zeros((1, latent_size), dtype=torch.float32))
        self._init_weight()
        if use_face:
            self.encoder_face = MLP(
                latent_size * 3,
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
            )
        else:
            self.encoder_face = None

        pos_embedding = PosEmbeddingwithMask(latent_size, pos_mask_prob, raw_with_pos=raw_with_pos)
        self.gnn_layers = GNBlock(
            mlp_hidden_size=mlp_hidden_size,
            mlp_layers=mlp_layers,
            latent_size=latent_size,
            use_layer_norm=use_layer_norm,
            num_message_passing_steps=num_message_passing_steps,
            global_reducer=global_reducer,
            node_reducer=node_reducer,
            face_reducer=face_reducer,
            dropedge_rate=dropedge_rate,
            dropnode_rate=dropnode_rate,
            use_face=use_face,
            dropout=dropout,
            layernorm_before=layernorm_before,
            encoder_dropout=encoder_dropout,
            use_bn=use_bn,
            global_attn=global_attn,
            node_attn=node_attn,
            face_attn=face_attn,
            pred_pos_residual=pred_pos_residual,
            pos_embedding=pos_embedding,
        )
        if attr_predict:
            ap_hid_size = mlp_hidden_size if ap_hid_size is None else ap_hid_size
            ap_mlp_layers = mlp_layers if ap_mlp_layers is None else ap_mlp_layers
            self.attr_decoder = MLPwoLastAct(
                latent_size,
                [ap_hid_size] * ap_mlp_layers + [num_tasks],
                use_layer_norm=False,
                dropout=pooler_dropout,
                use_bn=use_bn,
            )
        else:
            self.attr_decoder = MolDecoder(
                latent_size=latent_size,
                hidden_size=mlp_hidden_size,
                mlp_layers=mlp_layers,
                use_bn=use_bn,
                pooler_dropout=pooler_dropout,
            )
        self.attr_predict = attr_predict
        self.pooling = _REDUCER_NAMES[graph_pooling]
        self.aggregate_edges_for_face_fn = _REDUCER_NAMES[face_reducer]
        self.use_face = use_face
        self.gradmultiply = gradmultiply

    def _init_weight(self):
        nn.init.normal_(self.global_init, mean=0, std=self.latent_size ** -0.5)

    def load_state_dict(self, checkpoint):
        if not self.attr_predict:
            super().load_state_dict(checkpoint)
            return
        keys_to_delete = []
        for k, v in checkpoint.items():
            if "attr_decoder" in k:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            print(f"Delete {k} from pre-trained checkpoint...")
            del checkpoint[k]

        for k, v in self.state_dict().items():
            if "attr_decoder" in k:
                print(f"Randomly init {k}...")
                checkpoint[k] = v
        super().load_state_dict(checkpoint)

    def forward(self, batch, output_no_pos=False, output_no_attr=False, mode="mask"):
        (
            x,
            edge_index,
            edge_attr,
            node_batch,
            face_mask,
            face_index,
            num_nodes,
            num_faces,
            num_edges,
            num_graphs,
            nf_node,
            nf_face,
            pos,
        ) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.ring_mask,
            batch.ring_index,
            batch.n_nodes,
            batch.num_rings,
            batch.n_edges,
            batch.num_graphs,
            batch.nf_node.view(-1),
            batch.nf_ring.view(-1),
            batch.pos,
        )
        x, attr_mask_index = self.node_embedding(x, mode=mode)
        edge_attr = one_hot_bonds(edge_attr)
        edge_attr = self.encoder_edge(edge_attr)
        edge_attr = self.node_embedding.update_edge_feat(
            edge_attr, edge_index, attr_mask_index, mode=mode
        )

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)
        u = self.global_init.expand(num_graphs, -1)

        if self.use_face:
            face_batch = torch.repeat_interleave(graph_idx, num_faces, dim=0)
            node_attributes = self.aggregate_edges_for_face_fn(
                x[nf_node], nf_face, size=num_faces.sum().item()
            )
            sent_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[0], size=num_faces.sum().item()
            )
            received_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[1], size=num_faces.sum().item()
            )
            feat = torch.cat([node_attributes, sent_attributes, received_attributes], dim=1)
            feat = torch.where(face_mask.unsqueeze(1), feat.new_zeros((feat.shape[0], 1)), feat)
            face = self.encoder_face(feat)
        else:
            face = None
            face_batch = None
            face_index = None

        x, edge_attr, face, u, pos_predictions, pos_mask_idx = self.gnn_layers(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            node_batch=node_batch,
            edge_batch=edge_batch,
            num_nodes=num_nodes,
            num_edges=num_edges,
            face=face,
            face_batch=face_batch,
            face_mask=face_mask,
            face_index=face_index,
            num_faces=num_faces,
            nf_node=nf_node,
            nf_face=nf_face,
            mode=mode,
            pos=pos,
        )
        if self.attr_predict:
            x = self.pooling(x, node_batch, size=num_graphs)
            x = GradMultiply.apply(x, self.gradmultiply)
        if mode == "mol2conf":
            return None, None, pos_predictions, None
        pred_attrs = self.attr_decoder(x, mode=mode)
        if mode == "conf2mol":
            return pred_attrs, None, None, None
        return pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx

    def compute_loss(
        self, pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx, batch, args, mode="mask"
    ):

        if mode == "mask":
            return self.compute_mask_loss(
                pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx, batch, args
            )
        elif mode == "conf2mol":
            return self.compute_conf2mol_loss(
                pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx, batch, args
            )
        elif mode == "mol2conf":
            return self.compute_mol2conf_loss(
                pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx, batch, args
            )
        else:
            raise NotImplementedError()

    def compute_mask_loss(
        self, pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx, batch, args,
    ):
        gt_x = batch.x
        gt_pos = batch.pos
        attr_mask_index = attr_mask_index.view(-1)
        pos_mask_idx = pos_mask_idx.view(-1)
        attr_loss = 0
        pos_loss = 0
        for i in range(gt_x.shape[1]):
            pred = pred_attrs[i][attr_mask_index]
            gt = gt_x[:, i][attr_mask_index]
            attr_loss = attr_loss + F.cross_entropy(pred, gt, reduction="mean")

        new_idx = GNNet.update_iso_mask(
            gt_pos,
            torch.where(pos_mask_idx.view(-1, 1), pos_predictions[-1], gt_pos),
            batch,
            pos_mask_idx,
        )
        pos_mask_idx = pos_mask_idx.index_select(0, new_idx)
        gt = gt_pos[pos_mask_idx]
        for i, pos_pred in enumerate(pos_predictions):
            pred = pos_pred.index_select(0, new_idx)[pos_mask_idx]
            pos_loss = pos_loss + (gt - pred).norm(dim=-1).mean() * (
                1 if i == len(pos_predictions) - 1 else 0.1
            )

        loss = attr_loss + pos_loss
        return loss, dict(loss=loss.item(), pos_loss=pos_loss.item(), attr_loss=attr_loss.item())

    @staticmethod
    def update_iso_mask(pos_y, pos_x, batch, pos_mask):
        with torch.no_grad():
            pre_nodes = 0
            num_nodes = batch.n_nodes
            isomorphisms = batch.isomorphisms
            new_idx_x = []
            for i in range(batch.num_graphs):
                current_isomorphisms = [
                    torch.LongTensor(iso).to(pos_x.device) for iso in isomorphisms[i]
                ]
                cur_num_nodes = num_nodes[i]
                cur_pos_mask = pos_mask[pre_nodes : pre_nodes + cur_num_nodes]

                if len(current_isomorphisms) == 1 or not torch.any(cur_pos_mask):
                    new_idx_x.append(current_isomorphisms[0] + pre_nodes)
                else:
                    pos_y_i = pos_y[pre_nodes : pre_nodes + cur_num_nodes]
                    pos_x_i = pos_x[pre_nodes : pre_nodes + cur_num_nodes]
                    pos_x_list = []
                    for iso in current_isomorphisms:
                        pos_x_list.append(torch.index_select(pos_x_i, 0, iso))

                    total_iso = len(pos_x_list)
                    pos_y_i = pos_y_i.repeat(total_iso, 1)
                    pos_x_i = torch.cat(pos_x_list, dim=0)
                    min_idx = GNNet.mask_loss_one_graph(
                        pos_y_i, pos_x_i, cur_num_nodes, total_iso, cur_pos_mask
                    )
                    new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)

                pre_nodes += cur_num_nodes

            return torch.cat(new_idx_x, dim=0)

    @staticmethod
    def mask_loss_one_graph(pos_y, pos_x, num_nodes, total_iso, pos_mask):
        with torch.no_grad():
            loss = (pos_y - pos_x).norm(dim=-1, keepdim=True).view(-1, num_nodes).mean(-1)
            return torch.argmin(loss)

    def compute_conf2mol_loss(
        self, pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx, batch, args,
    ):
        gt_x = batch.x
        attr_loss = 0
        for i in range(gt_x.shape[1]):
            pred = pred_attrs[i]
            gt = gt_x[:, i]
            attr_loss = attr_loss + F.cross_entropy(pred, gt, reduction="mean")
        return attr_loss, dict(loss=attr_loss.item(), pos_loss=0, attr_loss=attr_loss.item())

    def compute_mol2conf_loss(
        self, pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx, batch, args
    ):
        pos_loss = 0
        gt_pos = batch.pos
        new_idx = GNNet.update_iso_mol2conf(gt_pos, pos_predictions[-1], batch)
        for i, pos_pred in enumerate(pos_predictions):
            pos_loss = pos_loss + GNNet.alignment_loss(
                gt_pos, torch.index_select(pos_pred, 0, new_idx), batch
            ) * (1 if i == len(pos_predictions) - 1 else 0.1)
        return pos_loss, dict(loss=pos_loss.item(), pos_loss=pos_loss.item(), attr_loss=0)

    @staticmethod
    def update_iso_mol2conf(pos_y, pos_x, batch):
        with torch.no_grad():
            pre_nodes = 0
            num_nodes = batch.n_nodes
            isomorphisms = batch.isomorphisms
            new_idx_x = []
            for i in range(batch.num_graphs):
                current_isomorphisms = [
                    torch.LongTensor(iso).to(pos_x.device) for iso in isomorphisms[i]
                ]
                cur_num_nodes = num_nodes[i]

                if len(current_isomorphisms) == 1:
                    new_idx_x.append(current_isomorphisms[0] + pre_nodes)
                else:
                    pos_y_i = pos_y[pre_nodes : pre_nodes + cur_num_nodes]
                    pos_x_i = pos_x[pre_nodes : pre_nodes + cur_num_nodes]
                    pos_y_mean = torch.mean(pos_y_i, dim=0, keepdim=True)
                    pos_x_mean = torch.mean(pos_x_i, dim=0, keepdim=True)
                    pos_x_list = []

                    for iso in current_isomorphisms:
                        pos_x_list.append(torch.index_select(pos_x_i, 0, iso))

                    total_iso = len(pos_x_list)
                    pos_y_i = pos_y_i.repeat(total_iso, 1)
                    pos_x_i = torch.cat(pos_x_list, dim=0)
                    min_idx = GNNet.mol2conf_loss_one_graph(
                        pos_y_i,
                        pos_x_i,
                        pos_y_mean,
                        pos_x_mean,
                        num_nodes=cur_num_nodes,
                        total_iso=total_iso,
                    )
                    new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)
                pre_nodes += cur_num_nodes

            return torch.cat(new_idx_x, dim=0)

    @staticmethod
    def mol2conf_loss_one_graph(pos_y, pos_x, pos_y_mean, pos_x_mean, num_nodes, total_iso):
        with torch.no_grad():
            total_nodes = pos_y.shape[0]
            y = pos_y - pos_y_mean
            x = pos_x - pos_x_mean
            a = y + x
            b = y - x
            a = a.view(-1, 1, 3)
            b = b.view(-1, 3, 1)
            tmp0 = torch.cat(
                [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
            )
            eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
            a = a.expand(-1, 3, -1)
            tmp1 = torch.cross(eye, a, dim=-1)
            tmp1 = torch.cat([b, tmp1], dim=-1)
            tmp = torch.cat([tmp0, tmp1], dim=1)
            tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(-1, num_nodes, 16)
            tmpb = torch.mean(tmpb, dim=1).view(-1, 4, 4)
            w, v = torch.linalg.eigh(tmpb)
            min_q = v[:, :, 0]
            rotation = GNNet.quaternion_to_rotation_matrix(min_q)
            t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean.expand(total_iso, -1), rotation)
            rotation = torch.repeat_interleave(rotation, num_nodes, dim=0)
            t = torch.repeat_interleave(t, num_nodes, dim=0)
            pos_x = torch.einsum("kj,kij->ki", pos_x, rotation) + t
            loss = (pos_y - pos_x).norm(dim=-1, keepdim=True).view(-1, num_nodes,).mean(-1)
            return torch.argmin(loss)

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        q0 = quaternion[:, 0]
        q1 = quaternion[:, 1]
        q2 = quaternion[:, 2]
        q3 = quaternion[:, 3]

        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(-1, 3, 3)

    @staticmethod
    def alignment_loss(
        pos_y, pos_x, batch,
    ):
        with torch.no_grad():
            num_nodes = batch.n_nodes
            total_nodes = pos_y.shape[0]
            num_graphs = batch.num_graphs
            pos_y_mean = global_mean_pool(pos_y, batch.batch)
            pos_x_mean = global_mean_pool(pos_x, batch.batch)
            y = pos_y - torch.repeat_interleave(pos_y_mean, num_nodes, dim=0)
            x = pos_x - torch.repeat_interleave(pos_x_mean, num_nodes, dim=0)
            a = y + x
            b = y - x
            a = a.view(total_nodes, 1, 3)
            b = b.view(total_nodes, 3, 1)
            tmp0 = torch.cat(
                [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
            )
            eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
            a = a.expand(-1, 3, -1)
            tmp1 = torch.cross(eye, a, dim=-1)
            tmp1 = torch.cat([b, tmp1], dim=-1)
            tmp = torch.cat([tmp0, tmp1], dim=1)
            tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(total_nodes, -1)
            tmpb = global_mean_pool(tmpb, batch.batch).view(num_graphs, 4, 4)
            w, v = torch.linalg.eigh(tmpb)
            min_rmsd = w[:, 0]
            min_q = v[:, :, 0]
            rotation = GNNet.quaternion_to_rotation_matrix(min_q)
            t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean, rotation)
            rotation = torch.repeat_interleave(rotation, num_nodes, dim=0)
            t = torch.repeat_interleave(t, num_nodes, dim=0)
        pos_x = torch.einsum("kj,kij->ki", pos_x, rotation) + t
        loss = global_mean_pool((pos_y - pos_x).norm(dim=-1, keepdim=True), batch.batch).mean()
        return loss


class AtomEmbeddingwithMask(nn.Module):
    def __init__(self, latent_size, mlp_hidden_size, mlp_layers, use_layernorm, mask_prob):
        super().__init__()
        self.latent_size = latent_size
        self.encoder_node = MLP(
            sum(get_atom_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layernorm,
        )
        self.mask_feature = nn.Parameter(torch.zeros((1, latent_size), dtype=torch.float32))
        self.mask_edge_feature = nn.Parameter(torch.zeros((1, latent_size), dtype=torch.float32))
        self._init_weight()
        self.mask_prob = mask_prob

    def _init_weight(self):
        nn.init.normal_(self.mask_feature, mean=0, std=self.latent_size ** -0.5)
        nn.init.normal_(self.mask_edge_feature, mean=0, std=self.latent_size ** -0.5)

    def forward(self, x, mode="mask"):
        if mode == "mask":
            return self.mask(x)
        elif mode == "mol2conf":
            return self.mol2conf(x)
        elif mode == "conf2mol":
            return self.conf2mol(x)
        elif mode == "raw":
            return self.mol2conf(x)
        else:
            raise NotImplementedError()

    def mask(self, x):
        node_features = self.forward_attrs(x)
        random_variables = node_features.new_zeros((node_features.shape[0], 1)).uniform_()
        mask_idx = random_variables < self.mask_prob
        mask_feat = torch.where(mask_idx, self.mask_feature, node_features)
        return mask_feat, mask_idx

    def mol2conf(self, x):
        node_features = self.forward_attrs(x)
        return node_features, None

    def conf2mol(self, x):
        return self.mask_feature.expand(x.shape[0], -1), None

    def forward_attrs(self, x):
        x = one_hot_atoms(x)
        return self.encoder_node(x)

    def update_edge_feat(self, edge_attr, edge_index, attr_mask_index, mode="mask"):
        if mode in ["raw", "mol2conf"]:
            return edge_attr
        elif mode == "mask":
            src = edge_index[0]
            tgt = edge_index[1]
            src = attr_mask_index.index_select(0, src)
            tgt = attr_mask_index.index_select(0, tgt)
            mask = torch.logical_or(src, tgt)
            return torch.where(mask, self.mask_edge_feature, edge_attr)
        elif mode == "conf2mol":
            return self.mask_edge_feature.expand(edge_attr.shape[0], -1)
        else:
            raise NotImplementedError()


class PosEmbeddingwithMask(nn.Module):
    def __init__(self, latent_size, mask_prob, raw_with_pos=False):
        super().__init__()
        self.latent_size = latent_size
        self.pos_embedding = MLP(3, [latent_size, latent_size])
        self.dis_embedding = MLP(1, [latent_size, latent_size])
        self.mask_prob = mask_prob
        self.raw_with_pos = raw_with_pos

    def get_mask_idx(self, pos, mode="mask"):
        if mode == "mask":
            random_variables = pos.new_zeros((pos.shape[0], 1)).uniform_()
            mask_idx = random_variables < self.mask_prob
            return mask_idx
        else:
            return None

    def forward(self, pos, x, edge_attr, edge_index, last_pred=None, mask_idx=None, mode="mask"):
        if mode == "mask":
            pos = self.mask(pos, last_pred, mask_idx)
        elif mode == "mol2conf":
            pos = self.mol2conf(pos, last_pred, mask_idx)
        elif mode == "conf2mol":
            pos = self.conf2mol(pos, last_pred, mask_idx)
        elif mode == "raw":
            pos = self.raw(pos, last_pred, mask_idx)

        extended_x = x + self.pos_embedding(pos)
        row = edge_index[0]
        col = edge_index[1]
        sent_pos = pos[row]
        received_pos = pos[col]
        length = (sent_pos - received_pos).norm(dim=-1).unsqueeze(-1)
        # extended_edge_attr = torch.cat([edge_attr, length], dim=-1)
        extended_edge_attr = edge_attr + self.dis_embedding(length)
        return extended_x, extended_edge_attr

    def mask(self, pos, last_pred, mask_idx):
        pos = torch.where(mask_idx, last_pred, pos)
        return pos

    def mol2conf(self, pos, last_pred, mask_idx):
        return last_pred

    def conf2mol(self, pos, last_pred, mask_idx):
        return pos

    def raw(self, pos, last_pred, mask_idx):
        if self.raw_with_pos:
            return pos
        else:
            return last_pred


class MolDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, mlp_layers, use_bn, pooler_dropout):
        super().__init__()
        self.deocder_attrs = nn.ModuleList()
        vocab_sizes = get_atom_feature_dims()
        for size in vocab_sizes:
            self.deocder_attrs.append(
                MLPwoLastActwithselfBN(
                    latent_size,
                    [hidden_size] * mlp_layers + [size],
                    use_layer_norm=False,
                    dropout=pooler_dropout,
                    use_bn=use_bn,
                )
            )

    def forward(self, x, mode=None):
        attrs = []
        for sub_model in self.deocder_attrs:
            attrs.append(sub_model(x, mode=mode))
        return attrs


def one_hot_bonds(bonds):
    vocab_sizes = get_bond_feature_dims()
    one_hots = []
    for i in range(bonds.shape[1]):
        one_hots.append(F.one_hot(bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device))
    return torch.cat(one_hots, dim=1).float()


def one_hot_atoms(atoms):
    vocab_sizes = get_atom_feature_dims()
    one_hots = []
    for i in range(atoms.shape[1]):
        one_hots.append(F.one_hot(atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device))

    return torch.cat(one_hots, dim=1).float()

