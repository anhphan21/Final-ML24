class EHGNN(torch.nn.Module):
    """egnn + gnn"""

    def __init__(self, args=None):
        super(EHGNN, self).__init__()
        self.args = args
        self.out_ch = 1
        self.num_node_features = args.num_node_features
        self.num_pin_features = args.num_pin_features
        self.num_edge_features = args.num_edge_features
        self.nhid = args.nhid
        self.negative_slope = 0.1
        self.dropout_ratio = args.dropout_ratio
        self.conv_layers = args.layers
        self.skip_cnt = args.skip_cnt
        self.pos_encode = args.pos_encode
        self.pos_dim = 4
        self.num_egnn = args.egnn_layers
        self.egnn_dim = args.egnn_nhid

        self.convs = nn.ModuleList(
            [
                HyperGATConv(
                    in_nch=self.num_node_features,
                    in_pch=self.num_pin_features,
                    in_ech=self.num_edge_features,
                    nhid=self.nhid,
                    out_ch=self.nhid,
                    dropout=self.dropout_ratio,
                )
            ]
        )
        for i in range(self.conv_layers - 1):
            self.convs.append(
                HyperGATConv(
                    in_nch=self.nhid,
                    in_pch=self.nhid,
                    in_ech=self.num_edge_features,
                    nhid=self.nhid,
                    out_ch=self.nhid,
                    dropout=self.dropout_ratio,
                )
            )

        self.posnet = EGNNet(
            self.num_egnn,
            self.nhid,
            self.pos_dim,
            self.egnn_dim,
            position_encoding=self.pos_encode,
            dropout=self.dropout_ratio,
            args=args,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.egnn_dim, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_ratio, inplace=True),
            nn.Linear(self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.nhid, self.out_ch),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        pin_feat, edge_weight = data.pin_offset, data.edge_weight
        batch, macro_index = data.batch, data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]
        macro_pos = data.macro_pos
        # model forward
        for i, conv in enumerate(self.convs):
            last_x, last_pin_feat = x, pin_feat
            x, pin_feat = conv(x, pin_feat, edge_index, edge_weight)
            if self.skip_cnt and i > 0:
                x, pin_feat = x + last_x, pin_feat + last_pin_feat
        #
        macro_feature = x[macro_index]
        # EGNN for position feature
        feat = self.posnet(macro_feature, macro_pos, data.macro_num)
        # mlp
        # x = torch.cat([x, gap(feat, macro_batch)], dim=-1)
        x = gap(feat, macro_batch)
        x = self.mlp(x)
        return x

    def predict(self, data, macro_pos):
        """eplicitly input macro_pos, since other info are all the same within a netlist"""
        x, edge_index = data.x, data.edge_index
        pin_feat, edge_weight = data.pin_offset, data.edge_weight
        batch, macro_index = data.batch, data.macro_index
        # add macro pos
        macro_batch = batch[macro_index]
        # model forward
        for i, conv in enumerate(self.convs):
            last_x, last_pin_feat = x, pin_feat
            x, pin_feat = conv(x, pin_feat, edge_index, edge_weight)
            if self.skip_cnt and i > 0:
                x, pin_feat = x + last_x, pin_feat + last_pin_feat
        # macro feature
        macro_feature = x[macro_index]
        # EGNN for position feature
        feat = self.posnet(macro_feature, macro_pos, data.macro_num)
        # mlp
        x = gap(feat, macro_batch)
        x = self.mlp(x)
        return x
