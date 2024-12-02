class CEHGNN(torch.nn.Module):
    """plain gnn baseline"""

    def __init__(self, args=None):
        super(CEHGNN, self).__init__()
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

        self.net = models.vgg11(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(512 * 7 * 7, self.egnn_dim),
            nn.ReLU(True),
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.egnn_dim * 2, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid),
            nn.LeakyReLU(negative_slope=self.negative_slope),
            nn.Linear(self.nhid, self.out_ch),
        )

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        pin_feat, edge_weight = data.pin_offset, data.edge_weight
        batch, macro_index = data.batch, data.macro_index
        density = data.pic
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
        x = torch.cat([gap(macro_feature, macro_batch), gap(x, batch)], dim=-1)
        # EGNN for position feature
        feat = self.posnet(macro_feature, macro_pos, data.macro_num)
        # mlp
        # x = torch.cat([x, gap(feat, macro_batch)], dim=-1)
        feat = gap(feat, macro_batch)
        # density feature
        density_feat = self.net(density)
        x = torch.cat([feat, density_feat], dim=-1)
        x = self.mlp(x)
        return x
