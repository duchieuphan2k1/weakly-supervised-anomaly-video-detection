import torch
import torch.nn as nn
import torch.nn.init as torch_init
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

def get_output_shape(model, len_feature):
    return model(torch.randn(10, 32, len_feature)).data.shape


class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        self.maxpooling = nn.MaxPool1d(3, stride=2)
        self.len_feature = len_feature
        bn = nn.BatchNorm1d
        

        self.conv_encoder_1 = nn.Sequential(
                nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(len_feature)
                # nn.dropout(0.7)
            )
        
        encoder_1_out = get_output_shape(self.maxpooling, len_feature)[2]
        
        self.conv_encoder_2 = nn.Sequential(
                nn.Conv1d(in_channels=encoder_1_out,out_channels=encoder_1_out, kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(encoder_1_out)
                # nn.dropout(0.7)
            )
        
        encoder_2_out = get_output_shape(self.maxpooling, encoder_1_out)[2]

        self.conv_encoder_3 = nn.Sequential(
                nn.Conv1d(in_channels=encoder_2_out, out_channels = encoder_2_out, kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(encoder_2_out)
                # nn.dropout(0.7)
            )
        
        encoder_3_out = get_output_shape(self.maxpooling, encoder_2_out)[2]
        
        self.residual_conv_encoder = nn.Sequential(
                nn.Conv1d(in_channels=len_feature, out_channels=encoder_2_out , kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(encoder_2_out)
                # nn.dropout(0.7)
            )

        self.conv_decoder_1 = nn.Sequential(
                nn.Conv1d(in_channels=encoder_3_out, out_channels=encoder_3_out , kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(encoder_3_out)
                # nn.dropout(0.7)
            )
        self.upsampling_1 = nn.Upsample(mode='nearest', size=encoder_2_out)
        
        self.conv_decoder_2 = nn.Sequential(
                nn.Conv1d(in_channels=encoder_2_out,out_channels=encoder_2_out, kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(encoder_2_out)
                # nn.dropout(0.7)
            )
        self.upsampling_2 = nn.Upsample(mode='nearest', size=encoder_1_out)

        self.conv_decoder_3 = nn.Sequential(
                nn.Conv1d(in_channels=encoder_1_out, out_channels=encoder_1_out, kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(encoder_1_out)
                # nn.dropout(0.7)
            )
        self.upsampling_3 = nn.Upsample(mode='nearest', size=len_feature)

        conv_decoder_4_out_size = int(len_feature*3/4)
        self.conv_decoder_4 = nn.Sequential(
                nn.Conv1d(in_channels=len_feature, out_channels=conv_decoder_4_out_size, kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(conv_decoder_4_out_size)
                # nn.dropout(0.7)
            )
        self.residual_conv_decoder = nn.Sequential(
                nn.Conv1d(in_channels=encoder_3_out, out_channels=conv_decoder_4_out_size , kernel_size=3,
                          stride=1,dilation=1, padding=1),
                nn.ReLU(),
                bn(conv_decoder_4_out_size)
                # nn.dropout(0.7)
            )

        out_size = int(len_feature/4)
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_size, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(out_size, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)
            ###print("Shape of out after permute in agregate: ", out.shape)
            residual = out

            residual_conv_encoder_out = self.residual_conv_encoder(out)

            autoencoder_out = self.conv_encoder_1(out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)

            autoencoder_out = self.maxpooling(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)

            autoencoder_out = self.conv_encoder_2(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)

            autoencoder_out = self.maxpooling(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)

            autoencoder_out = self.conv_encoder_3(autoencoder_out)

            autoencoder_out = autoencoder_out + residual_conv_encoder_out

            autoencoder_out = autoencoder_out.permute(0, 2, 1)

            autoencoder_out = self.maxpooling(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)
            residual_conv_decoder_out = self.residual_conv_decoder(autoencoder_out)

            autoencoder_out = self.conv_decoder_1(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)
            autoencoder_out = self.upsampling_1(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)
            autoencoder_out = self.conv_decoder_2(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)
            autoencoder_out = self.upsampling_2(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)
            autoencoder_out = self.conv_decoder_3(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)
            autoencoder_out = self.upsampling_3(autoencoder_out)
            autoencoder_out = autoencoder_out.permute(0, 2, 1)
            autoencoder_out = self.conv_decoder_4(autoencoder_out)
            autoencoder_out = autoencoder_out + residual_conv_decoder_out
            ###print("Shape of out d (concat out1, out2, out3) in agregate: ", out_d.shape)

            out = self.conv_4(out)
            ###print("Shape of out 4 in agregate: ", out.shape)

            out = self.non_local(out)
            ###print("Shape of out - non local in agregate: ", out.shape)

            out = torch.cat((autoencoder_out, out), dim=1)
            ###print("Shape of out after concat out_d with out 4 in agregate: ", out.shape)

            out = self.conv_5(out)   # fuse all the features together
            ###print("Shape of out5 in agregate: ", out.shape)

            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)
            ###print("Final shape of out in agregate: ", out.shape)

            return out

class AutoEncoder(nn.Module):
    def __init__(self, n_features, batch_size):
        super(AutoEncoder, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=n_features)
        
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):
        ###print("Batch size:", self.batch_size)

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        ###print("Size of inputs:", out.shape)
        bs, ncrops, t, f = out.size()

        out = out.view(-1, t, f)
        ###print("Size of inputs after reshape:", out.shape)

        ###print("Begin Aggregate")
        out = self.Aggregate(out)
        ###print("Size of out after Aggregate:", out.shape)
        ###print("End Aggregate")

        out = self.drop_out(out)

        ###print("Begin Fully connected")
        features = out
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        ###print("Size of scores after sigmoid:", scores.shape)
        scores = scores.view(bs, ncrops, -1).mean(1)
        ###print("Size of scores after reshape and mean:", scores.shape)

        scores = scores.unsqueeze(dim=2)
        ###print("Size of scores after unsqueeze:", scores.shape)
        ###print("End Fully connected")

        # Just use batch_size to select all snippets as normal
        normal_features = features[0:self.batch_size*10]
        ###print("Size of normal_features:", normal_features.shape)
        ###print("Batch size:", self.batch_size)

        normal_scores = scores[0:self.batch_size]
        ###print("Size of normal_score:", normal_scores.shape)

        # Just use batch_size to select none snippet as abnormal
        abnormal_features = features[self.batch_size*10:]
        ###print("Size of abnormal_features:", abnormal_features.shape)
        abnormal_scores = scores[self.batch_size:]
        ###print("Size of abnormal_scores:", abnormal_scores.shape)

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        ###print("Size of feat_magnitudes (norm of feature - aggreated output):", feat_magnitudes.shape)

        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        ###print("Size of feat_magnitudes after reshape and mean:", feat_magnitudes.shape)

        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        ###print("Size of nfea_magnitudes:", nfea_magnitudes.shape)

        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        ###print("Size of afea_magnitudes:", afea_magnitudes.shape)
        ###print("afea_magnitudes:", afea_magnitudes)

        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes)
        ###print("select_idx size:", select_idx.shape)
        ###print("select_idx:", select_idx)
        select_idx = self.drop_out(select_idx)
        

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        ###print("afea_magnitudes_drop size:", afea_magnitudes_drop.shape)
        ###print("afea_magnitudes_drop", afea_magnitudes_drop)
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        ###print("idx_abn:", idx_abn.shape)
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
        ###print("idx_abn_feat:", idx_abn_feat.shape)
        ###print("idx_abn_feat:", idx_abn_feat)
        
        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        ###print("abnormal_features:", abnormal_features.shape)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)
        ###print("abnormal_features after permute:", abnormal_features.shape)

        total_select_abn_feature = torch.zeros(0, device=inputs.device)
        ###print("total_select_abn_feature after permute:", total_select_abn_feature.shape)

        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        ###print("idx_abn_score:", idx_abn_score.shape)
        ###print("total_select_abn_feature:", total_select_abn_feature.shape)
        ###print(total_select_abn_feature)

        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude
        ###print("score_abnormal:", score_abnormal.shape)

        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0, device=inputs.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature
        ###print("============================================================================================================")
        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes