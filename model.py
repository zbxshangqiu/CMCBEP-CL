from torch import nn
import torch
import numpy
from tensorboardX import SummaryWriter
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet,GRUEncoder
from transformer import Block_fusion,Block_fusion0,make_mask,MHAtt,Block_gf,Block_split0,Block_split,filter,reconenc,Block_trans,Block_gff,selector,AttentionWeightedFusion,TransformerEncoder,CrossModalContrastiveLearning,EnhancedModel

class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super(MMIM,self).__init__()
        self.hp = hp
        self.add_va = hp.add_va
        #hp.d_tout = hp.d_tin
        self.AttentionWeightedFusion = AttentionWeightedFusion(
            num_predictions=5,
            input_dim=1
        )
        self.CrossModalContrastiveLearning = CrossModalContrastiveLearning(
            input_dim=64,  # 输入特征维度
            hidden_dim=64,  # 投影后的特征维度
            temperature=0.1,  # 对比损失的温度参数
        )
        self.transformer_encoder = TransformerEncoder(
            feature_dim=hp.d_tin,  # 输入特征的维度
            num_heads=hp.multi_head,  # 多头注意力机制的头数
            hidden_dim=hp.d_vh,  # 前馈神经网络的隐藏层维度
            num_layers=2,  # Transformer 编码器的层数（可选，默认为 2）
            dropout=0.1  # Dropout 概率（可选，默认为 0.1）
        )
        self.EnhancedModel = EnhancedModel(
            input_size=64,
            hidden_size=128,
            output_size=64
        )
        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )


        # Trimodal Settings
        self.fusion_prj = SubNet(
            in_size = hp.d_prjh,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )
        self.class1=nn.Linear(hp.d_prjh,hp.n_class)
        self.class2 = nn.Linear(hp.d_prjh, hp.n_class)
        self.class3 =nn.Linear(hp.d_prjh, hp.n_class)
        # Trimodal Settings
        self.fusion_prj2 = SubNet(
            in_size=4*hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fusion_prj3 = nn.Sequential(nn.Linear(3*hp.d_prjh, hp.d_prjh))
        self.fusion_prj4 = SubNet(
            in_size=2 * hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fc1=nn.Linear(768,hp.d_prjh)
        self.encoder_s1 = nn.Linear(hp.d_prjh, hp.d_prjh)
        self.encoder_s2 = nn.Linear(hp.d_prjh, hp.d_prjh)

        self.encoder_s3 = nn.Linear(hp.d_prjh, hp.d_prjh)
        #self.encoder_s = nn.Linear(hp.d_prjh, hp.d_prjh)
        #self.diffc=nn.Linear(3*hp.d_prjh, hp.d_prjh)

        self.att = MHAtt(hp)

        self.block_0=Block_split0(hp)
        self.block_1 = nn.ModuleList([Block_split(hp, i) for i in range(hp.layer_fusion)])


        self.block_2 = nn.ModuleList([Block_gf(hp, i) for i in range(hp.normlayer)])
        self.block_3 = nn.ModuleList([Block_gf(hp, i) for i in range(hp.normlayer2)])
        self.block_4=nn.ModuleList([Block_trans(hp, i) for i in range(hp.normlayer3)])
        self.filter1=filter(hp)
        self.filter2 = filter(hp)
        self.filter3 = filter(hp)
        self.recenc1=reconenc(hp)
        self.recenc2=reconenc(hp)
        self.recenc3=reconenc(hp)
        self.sig=nn.Sigmoid()
        self.selctor=selector(hp)
        self.gru1=GRUEncoder(
            in_size=hp.d_prjh,
            hidden_size=hp.d_prjh,
            out_size=hp.d_prjh,
            num_layers=4,
            dropout=0.2,
            bidirectional=True
        )
        self.gru2=GRUEncoder(
            in_size=hp.d_prjh,
            hidden_size=hp.d_prjh,
            out_size=hp.d_prjh,
            num_layers=4,
            dropout=0.2,
            bidirectional=True
        )
        #self.fumatt=nn.ModuleList([Block_trans(hp, i) for i in range(hp.layerforsemantic)])

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)
        text=self.fc1(text)
        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)
        #print(visual.size(),text.size(),acoustic.size())
        x0_mask=make_mask(visual)
        y0_mask=make_mask(text)
        z0_mask=make_mask(acoustic)
        ixy,iyz,izx=self.selctor(visual,x0_mask,text,y0_mask,acoustic,z0_mask)
        ix=ixy+izx
        iy=ixy+iyz
        iz=iyz+izx
        '''ix=ix.cpu().numpy()
        iy = iy.cpu().numpy()
        iz = iz.cpu().numpy()
        ix_m= numpy.mean(ix,axis=1)
        iy_m = numpy.mean(iy, axis=1)
        iz_m = numpy.mean(iz, axis=1)
        l1=numpy.array([ix_m,iy_m,iz_m])
        l1=numpy.argsort(l1)
        if l1==[1,0,2]:
            x=visual
            y=text
            z=acoustic
        if l1==[1,2,0]:
            x=visual
            y=acoustic
            z=text
        if l1 == [0, 1, 2]:
            x=text
            y=visual
            z=acoustic
        if l1 == [0, 2, 1]:
            x=text
            y=acoustic
            z=visual
        if l1 == [2, 1, 0]:
            x=acoustic
            y=visual
            z=text
        if l1 == [2, 0,1]:
            x=acoustic
            y=text
            z=visual'''
        ix = ix.detach().cpu().numpy()
        iy = iy.detach().cpu().numpy()
        iz = iz.detach().cpu().numpy()
        ix_m = numpy.mean(ix, axis=1)
        iy_m = numpy.mean(iy, axis=1)
        iz_m = numpy.mean(iz, axis=1)
        l1 = numpy.array([ix_m, iy_m, iz_m])
        l1_sorted_indices = numpy.argsort(l1, axis=0)
        # nt("l1_sorted_indices:", l1_sorted_indices)
        # l1=numpy.argsort(l1)
        # print("l1",l1)
        if numpy.array_equal(l1_sorted_indices[:, 0], [1, 0, 2]):
            x = visual
            y = text
            z = acoustic
        if numpy.array_equal(l1_sorted_indices[:, 0], [1, 2, 0]):
            x = visual
            y = acoustic
            z = text
        if numpy.array_equal(l1_sorted_indices[:, 0], [0, 1, 2]):
            x = text
            y = visual
            z = acoustic
        if numpy.array_equal(l1_sorted_indices[:, 0], [0, 2, 1]):
            x = text
            y = acoustic
            z = visual
        if numpy.array_equal(l1_sorted_indices[:, 0], [2, 1, 0]):
            x = acoustic
            y = visual
            z = text
        if numpy.array_equal(l1_sorted_indices[:, 0], [2, 0, 1]):
            x = acoustic
            y = text
            z = visual
        x_mask=make_mask(x)
        y_mask = make_mask(y)
        z_mask = make_mask(z)

        '''x_shared = self.encoder_s1(visual)
        #x_shared=self.encoder_s(x_shared)
        y_shared = self.encoder_s2(text)
        #y_shared = self.encoder_s(y_shared)
        z_shared = self.encoder_s3(acoustic)
        #z_shared = self.encoder_s(z_shared)
        #s_shared=(x_shared+y_shared+z_shared)/3'''
        #s_shared=self.sig(s_shared)
        y1=self.transformer_encoder(y)
        x1 = self.att(x, x, y1,None)
        #y1 = y
        z1 = self.att(z, z, y1,None)
        x=x+x1
        y=y+y1
        z=z+z1
        '''x = visual * self.sig(s_shared)
        y = self.sig(text) *s_shared
        z = acoustic * self.sig(s_shared)'''


        '''x = x_shared
        y = y_shared
        z = z_shared'''
###############abaltion x zhudao
        #x=self.gru1(x,x_mask)
        #x=self.EnhancedModel(x)
        for i,dec in enumerate(self.block_4):
            y=dec(y)
        #z=self.gru2(z,z_mask)
        #z=self.EnhancedModel(y)
        x2, y2, z2, db_loss = self.CrossModalContrastiveLearning(x, y, z)
        x_c=x2
        y_c=y2
        z_c=z2
        '''x,y1,y2,z,pre1,pre2,pre3,pre4=self.block_0(x,x_mask,y,y_mask,z,z_mask)
        for i, dec in enumerate(self.block_1):
            x_m, y_m, z_m = None, None, None


            x, y1,y2, z,pre1,pre2,pre3,pre4 = dec(x, x_m, y1,y2, y_m, z, z_m,pre1,pre2,pre3,pre4)
        y=y2'''
        #x2,y2,z2 ,db_loss= self.CrossModalContrastiveLearning(x,y,z)
        for i, dec in enumerate(self.block_2):
            x_m, y_m, z_m = None, None, None


            fusion = dec( y, y_m, x, x_m,z, z_m)
        #fusion=self.fusion_prj3(torch.cat((x,y2,z),dim=1))

        #fusion, preds = self.fusion_prj(torch.cat([x,y,z], dim=1))
        fusion, preds1 = self.fusion_prj(fusion)
        x_sin,x_dif=self.filter1(x,x,y,z,fusion)
        y_sin,y_dif=self.filter2(y,y,x,z,fusion)
        z_sin,z_dif=self.filter3(z,z,x,y,fusion)
        x_pre=self.class1(x_sin)
        y_pre=self.class2(y_sin)
        z_pre=self.class3(z_sin)

        #semantic=torch.cat((x2,y2,z2),dim=1)
        #semantic=x2+y2+z2
        #xrec=self.recenc1(x_sin,x_shared,fusion)
        #yrec=self.recenc2(y_sin,y_shared,fusion)
        #zrec=self.recenc3(z_sin,z_shared,fusion)
        #dif=torch.cat((x_sin,y_sin,z_sin), dim=1)
        #dif=self.diffc(dif)
        for i, dec in enumerate(self.block_3):
            x_m, y_m, z_m = None, None, None


            dif = dec( y_sin, y_m, x_sin, x_m,z_sin, z_m)
        #semantic = torch.cat((x_sin,y_sin,z_sin,fusion), dim=1)
        semantic = torch.cat(( dif,fusion), dim=1)
        '''for i, dec in enumerate(self.fumatt):


            semantic = dec(semantic)'''
        #f1, preds = self.fusion_prj2(semantic)
        f1, preds = self.fusion_prj4(semantic)
        final_prediction = self.AttentionWeightedFusion(preds, preds1,x_pre,y_pre,z_pre)
        return final_prediction,ixy,iyz,izx,iy,x_c,y_c,z_c,fusion,preds1,x_pre,y_pre,z_pre,x_dif,y_dif,z_dif,x,y,z,x_sin,y_sin,z_sin,db_loss
