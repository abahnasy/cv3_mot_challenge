from torch import nn

class BipartiteNeuralMessagePassingLayer(nn.Module):    
    def __init__(self, node_dim, edge_dim, dropout=0.):
        super().__init__()

        edge_in_dim  = 2*node_dim + 2*edge_dim # 2*edge_dim since we always concatenate initial edge features
        self.edge_mlp = nn.Sequential(*[nn.Linear(edge_in_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout), 
                                    nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout)])

        node_in_dim  = node_dim + edge_dim
        self.node_mlp = nn.Sequential(*[nn.Linear(node_in_dim, node_dim), nn.ReLU(), nn.Dropout(dropout),  
                                        nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Dropout(dropout)])

    def edge_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Node-to-edge updates, as descibed in slide 71, lecture 5.
        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, 2 x edge_dim) 
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_a_embeds: torch.Tensor with shape (|B|, node_dim)
            
        returns:
            updated_edge_feats = torch.Tensor with shape (|A|, |B|, edge_dim) 
        """
        
        n_nodes_a, n_nodes_b, _  = edge_embeds.shape
        
        
        nodes_a_in = nodes_a_embeds.unsqueeze(1).expand(-1, n_nodes_b, -1)
        nodes_b_in = nodes_b_embeds.unsqueeze(0).expand(n_nodes_a, -1, -1)
        edge_in = torch.cat((edge_embeds, nodes_a_in, nodes_b_in), dim=-1)
        
        
        #edge_in = ... # has shape (|A|, |B|, 2*node_dim + 2*edge_dim) 
        
        
        
        return self.edge_mlp(edge_in)

    def node_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Edge-to-node updates, as descibed in slide 75, lecture 5.

        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, edge_dim ) 
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
            
        returns:
            tuple(
                updated_nodes_a_embeds: torch.Tensor with shape (|A|, node_dim),
                updated_nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
                )
        """
        

        
        # NOTE 1: Use 'sum' as aggregation function
        
        mssgs_a = edge_embeds.sum(dim=1)
        nodes_a_in = torch.cat((nodes_a_embeds, mssgs_a), dim=-1)

        mssgs_b = edge_embeds.sum(dim=0)
        nodes_b_in = torch.cat((nodes_b_embeds, mssgs_b), dim=-1)
        
        # nodes_a_in = ... # Has shape (|A|, node_dim + edge_dim) 
        # nodes_b_in = ... # Has shape (|B|, node_dim + edge_dim) 



        nodes_a = self.node_mlp(nodes_a_in)
        nodes_b = self.node_mlp(nodes_b_in)

        return nodes_a, nodes_b

    def forward(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        edge_embeds_latent = self.edge_update(edge_embeds, nodes_a_embeds, nodes_b_embeds)
        nodes_a_latent, nodes_b_latent = self.node_update(edge_embeds_latent, nodes_a_embeds, nodes_b_embeds)

        return edge_embeds_latent, nodes_a_latent, nodes_b_latent