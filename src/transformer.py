import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.fc import MLP
from layers.layer_norm import LayerNorm
from modules.encoders import  GRUEncoder

# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------
def make_mask(feature):
    mask = (torch.sum(torch.abs(feature), dim=-1) == 0)  # 形状: [batch_size, sequence_length]
    mask = mask.unsqueeze(1).unsqueeze(2)  # 形状: [batch_size, 1, 1]
    return mask

# ------------------------------
# ---------- Flattening --------
# ------------------------------

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()
        self.sig=nn.Sigmoid()

    def forward(self, x1, x2, n_moments):
        x1=self.sig(x1)
        x2=self.sig(x2)
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
class AttFlat(nn.Module):
    def __init__(self, args, flat_glimpse, merge=False):
        super(AttFlat, self).__init__()
        self.args = args
        self.merge = merge
        self.flat_glimpse = flat_glimpse
        self.mlp = MLP(
            in_size=args.d_prjh,
            mid_size=args.ff_size,
            out_size=flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        if self.merge:
            self.linear_merge = nn.Linear(
                args.d_prjh * flat_glimpse,
                args.d_prjh * 2
            )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)

            return x_atted

        return torch.stack(att_list).transpose_(0, 1)

# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.d_prjh)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.d_prjh)
        self.gate1 = nn.Sequential(
            nn.Linear(args.d_prjh, args.d_prjh),
            nn.Sigmoid())

    def forward(self, y, y_mask):
        y = self.norm1(y + self.gate1(y)*self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, args):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.d_prjh)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.d_prjh)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.d_prjh)


        '''self.FC1=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )'''
        self.FC1=nn.Linear(args.d_prjh,args.d_prjh)

        self.FC2=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )

        self.D=nn.Sequential(
            nn.Linear(args.d_prjh, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        self.gate1=nn.Sequential(
            nn.Linear(args.d_prjh,args.d_prjh),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(args.d_prjh, args.d_prjh),
            nn.Sigmoid()
        )
    def maybe_layer_norm(self, i, x):


        return self.layer_norms[i](x)

    def forward(self, x, y, x_mask, y_mask):
        x=self.FC1(x)
        y=self.FC1(y)

        x_pre=self.D(x)
        y_pre=self.D(y)
        x=x + x_pre*self.dropout1(
            self.mhatt1(v=y, k=y, q=x, mask=x_mask))
        x = self.maybe_layer_norm(1, x)

        x=x + (1-y_pre)*self.dropout2(
            self.mhatt2(v=x, k=x, q=y, mask=y_mask)
        )
        x = self.maybe_layer_norm(1, x)
        x=x + self.dropout3(
            self.ffn(x)
        )
        x = self.maybe_layer_norm(1, x)

        return x,x_pre,y_pre

class FNET(nn.Module):
    def __init__(self, args):
        super(FNET, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)
        self.ffn2 = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.d_prjh)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.d_prjh)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.d_prjh)
        self.dropout4 = nn.Dropout(args.dropout_r)
        self.norm4 = LayerNorm(args.d_prjh)
        self.layer_norms = nn.ModuleList([LayerNorm(args.d_prjh) for _ in range(2)])


        '''self.FC1=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )'''
        self.FC1=nn.Linear(args.d_prjh,args.d_prjh)

        self.FC2=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )
        self.FC3 = nn.Linear(args.d_prjh, args.d_prjh)
        self.D=nn.Sequential(
            #nn.Linear(args.d_prjh, 128),
            #nn.ReLU(),
            #nn.Linear(128,1),
            nn.Linear(args.d_prjh,1),
            nn.Sigmoid()
        )
        self.gate1=nn.Sequential(
            nn.Linear(args.d_prjh,args.d_prjh),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(args.d_prjh, args.d_prjh),
            nn.Sigmoid()
        )
    def maybe_layer_norm(self, i, x):


        return self.layer_norms[i](x)

    def forward(self, x, y, x_mask, y_mask):

        x1=self.FC1(x)
        y0=self.FC1(y)
        '''x1=x
        y0=y'''
        #x1 =self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        #y0=self.mhatt2(v=y, k=y, q=y, mask=y_mask)
        x_pre=self.D(x)
        y_pre=self.D(y)
        g1=self.gate1(x1)
        g2=self.gate1(y0)
        x0=x1 +g1*self.dropout1(
            self.mhatt1(v=y0, k=y0, q=x1, mask=y_mask))
        x0 = self.maybe_layer_norm(1, x0)

        y0=(1-g2)*y0 +self.dropout2(
            self.mhatt2(v=x1, k=x1, q=y0, mask=x_mask))
        ''' x0 = x1 + x_pre * self.dropout1(
        self.mhatt1(v=y0, k=y0, q=x1, mask=y_mask))
        x0 = self.maybe_layer_norm(1, x0)

        y0 = (1 - y_pre) * y0 + self.dropout2(
        self.mhatt2(v=x1, k=x1, q=y0, mask=x_mask))

        y =  y +x_pre * self.dropout1(
            self.mhatt1(v=y, k=y, q=x, mask=x_mask))
        y = self.maybe_layer_norm(1, y)

        x =(1 - y_pre) * x +  self.dropout2(
            self.mhatt2(v=x, k=x, q=y, mask=y_mask)))'''

        y0 = self.maybe_layer_norm(1, y0)



        '''x0=x_pre*x0 + self.dropout3(
            self.ffn(x0)
        )
        y0=y_pre*y0 + self.dropout4(
            self.ffn2(y0))'''
        x0 =  g1 *x0 + self.dropout3(
            self.ffn(x0)
        )
        y0 =  g2 *y0 + self.dropout4(
            self.ffn2(y0))
        x0 = self.maybe_layer_norm(1, x0)
        y0 = self.maybe_layer_norm(1, y0)


        return x0,y0,x_pre,y_pre

class FUNET(nn.Module):
    def __init__(self, args):
        super(FUNET, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)
        self.ffn2 = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.d_prjh)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.d_prjh)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.d_prjh)
        self.dropout4 = nn.Dropout(args.dropout_r)
        self.norm4 = LayerNorm(args.d_prjh)
        self.layer_norms = nn.ModuleList([LayerNorm(args.d_prjh) for _ in range(2)])


        '''self.FC1=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )'''
        self.FC1=nn.Linear(args.d_prjh,args.d_prjh)

        self.FC2=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )

        self.D=nn.Sequential(
            nn.Linear(args.d_prjh, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        self.gate1=nn.Sequential(
            nn.Linear(args.d_prjh,args.d_prjh),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(args.d_prjh, args.d_prjh),
            nn.Sigmoid()
        )
        self.gate3 = nn.Sequential(
            nn.Linear(args.d_prjh, args.d_prjh),
            nn.Sigmoid()
        )
    def maybe_layer_norm(self, i, x):


        return self.layer_norms[i](x)

    def forward(self, x, y, z,x_mask,y_mask,z_mask):

        x=self.FC1(x)
        y=self.FC1(y)
        z=self.FC1(z)
        g1=self.gate1(x)
        g2=self.gate1(y)
        g3=self.gate1(z)
        x=x +g2*self.dropout1(
            self.mhatt1(v=y, k=y, q=x, mask=y_mask))
        x = self.maybe_layer_norm(1, x)

        x=z +g3*self.dropout2(
            self.mhatt2(v=z, k=z, q=x, mask=z_mask)
        )
        x = self.maybe_layer_norm(1, x)

        x=x + g1*self.dropout3(
            self.ffn(x)
        )
        x = self.maybe_layer_norm(1, x)


        return x
# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.d_prjh, args.d_prjh)
        self.linear_k = nn.Linear(args.d_prjh, args.d_prjh)
        self.linear_q = nn.Linear(args.d_prjh, args.d_prjh)
        self.linear_merge = nn.Linear(args.d_prjh, args.d_prjh)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            #-1,
            self.args.multi_head,
            int(self.args.d_prjh / self.args.multi_head)
        )
        #print(q.size())

        k = self.linear_k(k).view(
            n_batches,
            #-1,
            self.args.multi_head,
            int(self.args.d_prjh / self.args.multi_head)
        )

        q = self.linear_q(q).view(
            n_batches,
            #-1,
            self.args.multi_head,
            int(self.args.d_prjh / self.args.multi_head)
        )
        #print(v.size(),q.size(),k.size())

        #print(q.size())
        atted = self.att(v, k, q, mask)
        #print(atted.size())
        atted = atted.transpose(1,2).contiguous().view(
            n_batches,
            #-1,
            self.args.d_prjh
        )
        #print(atted.size())
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        #print(query.size(),key.size())
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            #print(scores.size(),mask.size())
            #print()
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class MHAtt2(nn.Module):
    def __init__(self, args):
        super(MHAtt2, self).__init__()
        self.args = args
        self.args.dprjh=args.d_prjh
        self.hiddensize = 4 * self.args.dprjh
        self.linear_v = nn.Linear(self.hiddensize, self.hiddensize)
        self.linear_k = nn.Linear(self.hiddensize, self.hiddensize)
        self.linear_q = nn.Linear(self.hiddensize, self.hiddensize)
        self.linear_merge = nn.Linear(self.hiddensize, self.hiddensize)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            #-1,
            self.args.multi_head,
            int(self.hiddensize / self.args.multi_head)
        )
        #print(q.size())

        k = self.linear_k(k).view(
            n_batches,
            #-1,
            self.args.multi_head,
            int(self.hiddensize / self.args.multi_head)
        )

        q = self.linear_q(q).view(
            n_batches,
            #-1,
            self.args.multi_head,
            int(self.hiddensize/ self.args.multi_head)
        )
        #print(v.size(),q.size(),k.size())

        #print(q.size())
        atted = self.att(v, k, q, mask)
        #print(atted.size())
        atted = atted.transpose(1,2).contiguous().view(
            n_batches,
            #-1,
            self.hiddensize
        )
        #print(atted.size())
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        #print(query.size(),key.size())
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            #print(scores.size(),mask.size())
            #print()
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.d_prjh,
            mid_size=args.ff_size,
            out_size=args.d_prjh,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

# ---------------------------
# ---- FF + norm  -----------
# ---------------------------
class FFAndNorm(nn.Module):
    def __init__(self, args):
        super(FFAndNorm, self).__init__()

        self.ffn = FFN(args)
        self.norm1 = LayerNorm(args.d_prjh)
        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.d_prjh)

    def forward(self, x):
        x = self.norm1(x)
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x
class Block_0(nn.Module):
    def __init__(self, args):
        super(Block_0, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa2= SGA(args)
        self.sa3= SGA(args)


        self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
        self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
        self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
        self.norm_l = LayerNorm(args.d_prjh)
        self.norm_a = LayerNorm(args.d_prjh)
        self.norm_v = LayerNorm(args.d_prjh)
        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self,  x, x_mask,y, y_mask, z, z_mask):

        ax = self.sa1(x, x_mask)
        ay,pre_1,pre_2 = self.sa2(y, x, y_mask, x_mask)
        az,pre_3,pre_4 = self.sa3(z, x, z_mask, x_mask)

        x = ax + x
        y = ay + y
        z = az + z



        ax = self.att_lang(x, x_mask)
        ay = self.att_audio(y, y_mask)
        az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y + self.dropout(ay)), \
               self.norm_v(z + self.dropout(az)),pre_1,pre_2,pre_3,pre_4

class Block(nn.Module):
    def __init__(self, args, i):
        super(Block, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa2= SGA(args)
        self.sa3= SGA(args)

        self.last = (i == args.layer_fusion-1)
        if not self.last:
            #self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            #self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            #self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.d_prjh)
            self.norm_a = LayerNorm(args.d_prjh)
            self.norm_v = LayerNorm(args.d_prjh)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask, z, z_mask,pre1,pre2,pre3,pre4):

        ax = self.sa1(x, x_mask)
        ay,pre_1,pre_2 = self.sa2(y, x, y_mask, y_mask)
        az,pre_3,pre_4 = self.sa3(z, y, z_mask, z_mask)
        pre1=torch.cat((pre1,pre_1))
        pre2 = torch.cat((pre2, pre_2))
        pre3 = torch.cat((pre3, pre_3))
        pre4 = torch.cat((pre4, pre_4))
        x = ax + x
        y = ay + y
        z = az + z

        if self.last:
            return x, y, z,pre1,pre2,pre3,pre4

        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y + self.dropout(ay)), \
               self.norm_v(z + self.dropout(az)),pre1,pre2,pre3,pre4

class Block_normal(nn.Module):
    def __init__(self, args, i):
        super(Block_normal, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa2= SA(args)
        self.sa3= SA(args)

        self.last = (i == args.normlayer-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.d_prjh)
            self.norm_a = LayerNorm(args.d_prjh)
            self.norm_v = LayerNorm(args.d_prjh)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask, z, z_mask):

        ax = self.sa1(x, x_mask)
        ay = self.sa2(y,  y_mask)
        az= self.sa3(z, z_mask)

        x = ax + x
        y = ay + y
        z = az + z

        if self.last:
            return x, y, z

        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y + self.dropout(ay)), \
               self.norm_v(z + self.dropout(az))

class Block_fusion0(nn.Module):
    def __init__(self, args):
        super(Block_fusion0, self).__init__()
        self.args = args
        self.f1=FNET(args)
        self.f2=FNET(args)
        self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
        self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
        self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
        self.norm_l = LayerNorm(args.d_prjh)
        self.norm_a = LayerNorm(args.d_prjh)
        self.norm_v = LayerNorm(args.d_prjh)
        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self,  x, x_mask,y, y_mask, z, z_mask):

        ax,ay,pre_1,pre_2 = self.f1(x, y, x_mask, y_mask)
        ay1,az,pre_3,pre_4 = self.f2(y, z, y_mask, z_mask)

        x = ax + x
        y = 0.5*ay+0.5*ay1+ y
        z = az + z



        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y + self.dropout(ay)), \
               self.norm_v(z + self.dropout(az)),pre_1,pre_2,pre_3,pre_4

class Block_fusion(nn.Module):
    def __init__(self, args, i):
        super(Block_fusion, self).__init__()
        self.args = args
        self.fnet1=FNET(args)
        self.fnet2=FNET(args)


        self.last = (i == args.layer_fusion-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.d_prjh)
            self.norm_a = LayerNorm(args.d_prjh)
            self.norm_v = LayerNorm(args.d_prjh)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask, z, z_mask,pre1,pre2,pre3,pre4):


        ax,ay,pre_1,pre_2 = self.fnet1(x, y, x_mask, y_mask)
        ay1,az,pre_3,pre_4 = self.fnet2(y, z, y_mask, z_mask)
        pre1=torch.cat((pre1,pre_1))
        pre2 = torch.cat((pre2, pre_2))
        pre3 = torch.cat((pre3, pre_3))
        pre4 = torch.cat((pre4, pre_4))
        x = ax + x
        y = 0.5*ay +0.5*ay1+ y
        z = az + z

        if self.last:
            return x, y, z,pre1,pre2,pre3,pre4

        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y + self.dropout(ay)), \
               self.norm_v(z + self.dropout(az)),pre1,pre2,pre3,pre4

class Block_gf(nn.Module):
    def __init__(self, args, i):
        super(Block_gf, self).__init__()
        self.args = args
        self.fnet1=FUNET(args)
        self.fnet2=FUNET(args)


        self.last = (i == args.normlayer-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.d_prjh)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask, z, z_mask):


        x=self.fnet1(x,y,z,x_mask,y_mask,z_mask)



        if self.last:
            return x

        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x )

class Block_split0(nn.Module):
    def __init__(self, args):
        super(Block_split0, self).__init__()
        self.args = args
        self.f1=FNET(args)
        self.f2=FNET(args)
        self.layer_norms = nn.ModuleList([LayerNorm(args.d_prjh) for _ in range(2)])
        self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
        self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
        self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
        self.norm_l = LayerNorm(args.d_prjh)
        self.norm_a = LayerNorm(args.d_prjh)
        self.norm_v = LayerNorm(args.d_prjh)
        self.dropout = nn.Dropout(args.dropout_r)
    def maybe_layer_norm(self, i, x):


        return self.layer_norms[i](x)
    def forward(self,  x, x_mask,y, y_mask, z, z_mask):

        ax,ay1,pre_1,pre_2 = self.f1(x, y, x_mask, y_mask)
        #ay2,az,pre_3,pre_4 = self.f2(ay1, z, y_mask, z_mask)
        az, ay2, pre_3, pre_4 = self.f2(z, ay1, z_mask, y_mask)

        x = ax + x
        x=self.maybe_layer_norm(1,x)
        y1 = ay1+ y
        y1 = self.maybe_layer_norm(1, y1)
        y2=ay2+y

        y2 = self.maybe_layer_norm(1, y2)

        z = az + z
        z = self.maybe_layer_norm(1, z)



        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y1 + self.dropout(ay1)), \
               self.norm_a(y2 + self.dropout(ay2)), \
               self.norm_v(z + self.dropout(az)),pre_1,pre_2,pre_3,pre_4

class Block_split(nn.Module):
    def __init__(self, args, i):
        super(Block_split, self).__init__()
        self.args = args
        self.fnet1=FNET(args)
        self.fnet2=FNET(args)

        self.layer_norms = nn.ModuleList([LayerNorm(args.d_prjh) for _ in range(2)])
        self.last = (i == args.layer_fusion-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.d_prjh)
            self.norm_a = LayerNorm(args.d_prjh)
            self.norm_v = LayerNorm(args.d_prjh)
            self.dropout = nn.Dropout(args.dropout_r)
    def maybe_layer_norm(self, i, x):


        return self.layer_norms[i](x)
    def forward(self, x, x_mask, y1,y2, y_mask, z, z_mask,pre1,pre2,pre3,pre4):


        ax,ay1,pre_1,pre_2 = self.fnet1(x, y1, x_mask, y_mask)
        #ay2,az,pre_3,pre_4 = self.fnet2(ay1, z, y_mask, z_mask)
        az, ay2, pre_3, pre_4 = self.fnet2(z, ay1, y_mask, z_mask)
        pre1=torch.cat((pre1,pre_1))
        pre2 = torch.cat((pre2, pre_2))
        pre3 = torch.cat((pre3, pre_3))
        pre4 = torch.cat((pre4, pre_4))
        x = ax + x
        y1 = ay1+ y1
        y2=ay2+y2
        z = az + z
        x = self.maybe_layer_norm(1, x)
        y1 = self.maybe_layer_norm(1, y1)
        y2 = self.maybe_layer_norm(1, y2)

        z = self.maybe_layer_norm(1, z)
        if self.last:
            return x, y1,y2, z,pre1,pre2,pre3,pre4

        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_a(y1 + self.dropout(ay1)), \
               self.norm_a(y2 + self.dropout(ay2)), \
               self.norm_v(z + self.dropout(az)),pre1,pre2,pre3,pre4


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()
        self.sig1=nn.Sigmoid()


    def forward(self, input1, input2):
        input1=self.sig1(input1)
        input2=self.sig1(input2)
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class filter(nn.Module):
    def __init__(self, args):
        super(filter, self).__init__()
        self.fc=nn.Linear(args.d_prjh,args.d_prjh)
        self.fc2=nn.Linear(args.d_prjh,args.d_prjh)
        self.sig=nn.Sigmoid()
        self.att=MHAtt(args)
        self.fumatt = nn.ModuleList([Block_trans(args, i) for i in range(args.layerforsemantic)])
        self.fumatt0 = nn.ModuleList([Block_trans(args, i) for i in range(args.layerforspace)])
        self.layer_norms = nn.ModuleList([LayerNorm(args.d_prjh) for _ in range(2)])
    def forward(self,x0,x,y,z,fusion):
        #x_shift=fusion-(1/2)*(y+z)
        #x_shift = fusion -x
        F=self.sig(x)
        x_shift=F*fusion
        x_1 = self.maybe_layer_norm(1, x_shift)

        #x_1=self.fc(x_shift)
        #x_1 = self.maybe_layer_norm(1, x_1)
        xs=self.sig(x_1)
        #xs_1=self.sig(y+z)
        #xs_2 = self.sig(fusion)
        #xs_2=self.sig(y+z)
        x_d = self.fc(x0)
        #x_d=self.fc2(x0)
        x_d = self.maybe_layer_norm(1, x_d)
        '''for i, dec in enumerate(self.fumatt0):
            x_d = dec(x)
        x_d = self.maybe_layer_norm(1, x_d)'''
        #x,xk=self.gru(x_d,x,60)
        #x=xs_2*x0+xs_1*x_d
        #x=x_d+xs_1*x0
        x=x_d+xs*x0
        x=self.maybe_layer_norm(1,x)
        for i, dec in enumerate(self.fumatt):
            x = dec(x)
        x = self.maybe_layer_norm(1, x)
        #x_d=self.sig(x_d)
        return x,x_d
    def maybe_layer_norm(self, i, x):
        return self.layer_norms[i](x)

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class reconenc(nn.Module):
    def __init__(self, args):
        super(reconenc, self).__init__()
        self.enc=nn.Sequential(nn.Linear(args.d_prjh,args.d_prjh)

                               )
        self.att=MHAtt(args)

    def forward(self,x_sin,x_shared,fusion):
        x_s=self.att(x_shared,x_shared,fusion,None)
        x=x_s+x_sin
        x=self.enc(x)
        return x

class Block_trans(nn.Module):
    def __init__(self, args, i):
        super(Block_trans, self).__init__()
        self.args = args
        self.mhatt=MHAtt(args)
        self.ffn = nn.Sequential(nn.Linear(args.d_prjh,args.d_prjh),
                                 nn.ReLU(),
                                 nn.Linear(args.d_prjh,args.d_prjh))
        self.dropout1 = nn.Dropout(args.dropout_r)
        self.dropout2 = nn.Dropout(args.dropout_r)
        self.layer_norms = nn.ModuleList([LayerNorm(args.d_prjh) for _ in range(2)])
        self.last = (i == args.layerforsemantic-1)

    def maybe_layer_norm(self, i, x):


        return self.layer_norms[i](x)
    def forward(self, y):
        y = (y +  self.dropout1(
            self.mhatt(y, y, y, None)
        ))
        y=self.maybe_layer_norm(1,y)
        y = y + self.dropout2(
            self.ffn(y)
        )
        y = self.maybe_layer_norm(1, y)
        if self.last:
            return y

        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return y


class Block_gff(nn.Module):
    def __init__(self, args, i):
        super(Block_gf, self).__init__()
        self.args = args
        self.fnet1=FUNET(args)
        self.fnet2=FUNET(args)


        self.last = (i == args.normlayer-1)
        if not self.last:
            self.att_lang = AttFlat(args, args.lang_seq_len, merge=False)
            self.att_audio = AttFlat(args, args.audio_seq_len, merge=False)
            self.att_vid = AttFlat(args, args.video_seq_len, merge=False)
            self.norm_l = LayerNorm(args.d_prjh)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask, z, z_mask):


        x=self.fnet1(x,y,z,x_mask,y_mask,z_mask)



        if self.last:
            return x

        #ax = self.att_lang(x, x_mask)
        #ay = self.att_audio(y, y_mask)
        #az = self.att_vid(z, y_mask)

        return self.norm_l(x )

class x_y_net(nn.Module):
    def __init__(self, args):
        super(x_y_net, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)
        self.ffn2 = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.d_prjh)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.d_prjh)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.d_prjh)
        self.dropout4 = nn.Dropout(args.dropout_r)
        self.norm4 = LayerNorm(args.d_prjh)
        self.layer_norms = nn.ModuleList([LayerNorm(args.d_prjh) for _ in range(3)])


        '''self.FC1=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )'''
        self.FC1=nn.Linear(args.d_prjh,args.d_prjh)

        self.FC2=nn.Sequential(
            nn.Linear(args.d_prjh, 2 * args.d_prjh),
            nn.ReLU(),
            nn.Linear(2 * args.d_prjh, args.d_prjh)
        )

        self.D=nn.Sequential(
            nn.Linear(args.d_prjh, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        self.gate1=nn.Sequential(
            nn.Linear(args.d_prjh,args.d_prjh),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(args.d_prjh, args.d_prjh),
            nn.Sigmoid()
        )
        self.gate3 = nn.Sequential(
            nn.Linear(args.d_prjh, args.d_prjh),
            nn.Sigmoid())

    def maybe_layer_norm(self, i, x):


        return self.layer_norms[i](x)

    # def forward(self, x, y,x_mask,y_mask):
    #
    #     x=self.FC1(x)
    #     y=self.FC1(y)
    #     #g1=self.gate1(x)
    #     #g2=self.gate1(y)
    #     x=x +self.dropout1(
    #         self.mhatt1(v=y, k=y, q=x, mask=y_mask))
    #     x = self.maybe_layer_norm(1, x)
    #
    #
    #     x=x +self.dropout3(
    #         self.ffn(x)
    #     )
    #     x = self.maybe_layer_norm(1, x)
    #     return x
    def forward(self, x, y, x_mask, y_mask):
        # 生成 mask（假设你已经定义了 make_mask 函数）


        # 将 FC1 应用于 x 和 y
        x = self.FC1(x)
        y = self.FC1(y)

        # 应用 gate1 控制信息流动
        g1 = self.gate1(x)
        g2 = self.gate1(y)

        # 应用多头注意力（MHAtt）和残差连接，之后进行层归一化
        x = self.maybe_layer_norm(0, x)
        x = x + self.dropout1(self.mhatt1(v=y, k=y, q=x, mask=y_mask))
        x = self.maybe_layer_norm(1, x)

        # 应用前馈网络（FFN）和残差连接，之后进行层归一化
        x = x + self.dropout3(self.ffn(x))
        x = self.maybe_layer_norm(2, x)

        return x

class Block_x_y(nn.Module):
    def __init__(self, args, i):
        super(Block_x_y, self).__init__()
        self.args = args
        self.fnet1=x_y_net(args)


        self.last = (i == args.normlayer-1)
        if not self.last:
            self.norm_l = LayerNorm(args.d_prjh)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask):


        x=self.fnet1(x,y,x_mask,y_mask)

        if self.last:
            return x
        return self.norm_l(x )

class estimator(nn.Module):
    def __init__(self, args):
        super(estimator, self).__init__()
        self.args = args
        self.biGRU=GRUEncoder(
            in_size=args.d_prjh,
            hidden_size=args.d_prjh,
            out_size=args.d_prjh,
            num_layers=4,
            dropout=0.2,
            bidirectional=True
        )
        self.x_y=Block_x_y(args,1)
        self.net=nn.Linear(args.d_prjh,1)

    def forward(self, x, x_mask, y, y_mask):
        x_y=self.x_y(x, x_mask, y, y_mask)
        # x1,x2=self.biGRU(x)
        # x0=(x1+x2)/2
        i=x_y
        i=self.net(i)
        return i
# class estimator(nn.Module):
#     def __init__(self, args):
#         super(estimator, self).__init__()
#         self.args = args
#         self.EnhancedModel = EnhancedModel(
#             input_size=64,
#             hidden_size=128,
#             output_size=64
#         )
#         self.GNNDependency = GNNDependency(
#             feature_dim=64,
#             hidden_dim=128
#         )
#
#         self.x_y = Block_x_y(args, 1)
#         self.net = nn.Linear(args.d_prjh, 1)
#
#         # 添加可学习的权重
#         self.weight_attention = nn.Parameter(torch.tensor(0.5))  # 注意力权重的初始值
#         self.weight_gnn = nn.Parameter(torch.tensor(0.5))        # 图神经网络权重的初始值
#
#     def forward(self, x, x_mask, y, y_mask):
#         # 计算基于注意力机制的相关系数
#         x_y = self.x_y(x, x_mask, y, y_mask)
#
#         # 计算基于图神经网络的相关系数
#         i0 = self.GNNDependency(x, y)
#
#         # 加权融合两种相关系数
#         combined_correlation = self.weight_attention * x_y + self.weight_gnn * i0
#
#         # 通过全连接层输出最终相关系数
#         i = self.net(combined_correlation)
#
#         return i


class selector(nn.Module):
    def __init__(self, args):
        super(selector,self).__init__()
        self.args = args
        self.estimator1=estimator(args)
        self.estimator2 = estimator(args)
        self.estimator3 = estimator(args)

    def forward(self, x, x_mask, y, y_mask, z, z_mask):
        i_x_y=self.estimator1(x, x_mask, y, y_mask)
        i_y_z=self.estimator2(y,y_mask,z,z_mask)
        i_z_x=self.estimator3(z,z_mask,x,x_mask)
        return i_x_y,i_y_z,i_z_x
class AttentionWeightedFusion(nn.Module):
    def __init__(self, input_dim, num_predictions):
        super(AttentionWeightedFusion, self).__init__()
        self.attention = nn.Linear(input_dim * num_predictions, num_predictions)  # 注意力机制生成权重
        self.softmax = nn.Softmax(dim=1)  # 归一化权重

    def forward(self, preds, preds1,x_pre,y_pre,z_pre):
        # 拼接所有预测结果
        combined = torch.cat([preds, preds1,x_pre,y_pre,z_pre], dim=1)  # 形状: (batch_size, 5 * output_dim)

        # 生成权重
        weights = self.attention(combined)  # 形状: (batch_size, 5)
        weights = self.softmax(weights)  # 归一化权重

        # 加权平均
        weighted_predictions = torch.stack([preds, preds1,x_pre,y_pre,z_pre], dim=1)  # 形状: (batch_size, 5, output_dim)
        final_prediction = torch.sum(weights.unsqueeze(-1) * weighted_predictions, dim=1)  # 形状: (batch_size, output_dim)

        return final_prediction
class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(feature_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        # 如果输入是二维的 [batch_size, feature_dim]，添加 seq_len 维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 形状变为 [batch_size, seq_len=1, feature_dim]
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x)
        # 如果输入原本是二维的，移除 seq_len 维度
        if x.size(1) == 1:
            x = x.squeeze(1)  # 形状变为 [batch_size, feature_dim]
        return x
class TransformerEncoderLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        # 确保 embed_dim 与 feature_dim 一致
        self.self_attn = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(64, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
    def forward(self, x):
        #print(f"Input shape: {x.shape}")  # 打印输入形状
        #print(f"Expected embed_dim: {self.self_attn.embed_dim}")  # 打印 embed_dim
        # 如果输入是二维的 [batch_size, feature_dim]，添加 seq_len 维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 形状变为 [batch_size, seq_len=1, feature_dim]
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x)  # 输入形状: [batch_size, seq_len, feature_dim]
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)  # 层
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(ff_output)  # 残差连接
        x = self.norm2(x)  # 层归一化
        # 如果输入原本是二维的，移除 seq_len 维度
        if x.size(1) == 1:
            x = x.squeeze(1)  # 形状变为 [batch_size, feature_dim]
        return x
class CrossModalContrastiveLearning(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, temperature=0.1):
        super(CrossModalContrastiveLearning, self).__init__()
        self.temperature = temperature

        # 线性投影层，将输入特征映射到同一空间
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.video_proj = nn.Linear(input_dim, hidden_dim)
        self.audio_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, y, z):
        """
        输入:
            x: 文本模态特征, torch.Size([120, 64])
            y: 视频模态特征, torch.Size([120, 64])
            z: 音频模态特征, torch.Size([120, 64])
        输出:
            x1: 对比学习后的文本特征, torch.Size([120, 64])
            y1: 对比学习后的视频特征, torch.Size([120, 64])
            z1: 对比学习后的音频特征, torch.Size([120, 64])
        """
        # 投影特征到同一空间
        x_proj = self.text_proj(x)  # [120, 64]
        y_proj = self.video_proj(y)  # [120, 64]
        z_proj = self.audio_proj(z)  # [120, 64]

        # 归一化特征
        x_proj = F.normalize(x_proj, dim=1)
        y_proj = F.normalize(y_proj, dim=1)
        z_proj = F.normalize(z_proj, dim=1)

        # 计算对比损失
        loss_xy = self.contrastive_loss(x_proj, y_proj)  # 文本-视频对比损失
        loss_xz = self.contrastive_loss(x_proj, z_proj)  # 文本-音频对比损失
        loss_yz = self.contrastive_loss(y_proj, z_proj)  # 视频-音频对比损失

        # 总对比损失
        contrastive_loss = (loss_xy + loss_xz + loss_yz) / 3

        # 返回对比学习后的特征和损失
        return x_proj, y_proj, z_proj, contrastive_loss

    def contrastive_loss(self, feat1, feat2):
        """
        计算对比损失（InfoNCE Loss）
        输入:
            feat1: 模态1的特征, torch.Size([120, 64])
            feat2: 模态2的特征, torch.Size([120, 64])
        输出:
            loss: 对比损失值
        """
        # 计算相似度矩阵
        sim_matrix = torch.matmul(feat1, feat2.T) / self.temperature  # [120, 120]

        # 正样本对是同一实例的模态对
        labels = torch.arange(feat1.size(0)).to(feat1.device)  # [120]

        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
class EnhancedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnhancedModel, self).__init__()
        self.feature_enhancer = FeatureEnhancer(input_size, hidden_size, hidden_size)
        self.attention = SelfAttention(feature_dim=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, input_size]
        # 通过特征增强模块
        enhanced_features = self.feature_enhancer(x)  # [batch_size, hidden_size]
        # 通过自注意力机制
        attended_output = self.attention(enhanced_features)  # [batch_size, hidden_size]
        # 通过全连接层映射到输出维度
        final_output = self.fc(attended_output)  # [batch_size, output_size]
        return final_output
class FeatureEnhancer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeatureEnhancer, self).__init__()
        # 全连接层用于特征变换
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, input_size]
        x = self.relu(self.fc1(x))  # [batch_size, hidden_size]
        x = self.fc2(x)  # [batch_size, output_size]
        return x

# 定义自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, feature_dim]
        Q = self.query(x)  # [batch_size, feature_dim]
        K = self.key(x)    # [batch_size, feature_dim]
        V = self.value(x)  # [batch_size, feature_dim]
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # [batch_size, batch_size]
        attention_weights = self.softmax(attention_scores)  # [batch_size, batch_size]
        # 加权求和
        attended_output = torch.matmul(attention_weights, V)  # [batch_size, feature_dim]
        return attended_output
from torch_geometric.nn import GATConv
class GNNDependency(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(GNNDependency, self).__init__()
        # 图注意力层
        self.gat = GATConv(feature_dim, hidden_dim, heads=1)
        # 全连接层，将 GAT 输出映射到 feature_dim
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, feature1, feature2):
        batch_size = feature1.size(0)  # 获取批量大小
        feature_dim = feature1.size(1)  # 获取特征维度

        # 初始化输出张量
        i0 = torch.zeros(batch_size, feature_dim).to(feature1.device)

        # 对每个样本单独处理
        for i in range(batch_size):
            # 获取当前样本的特征
            f1 = feature1[i].unsqueeze(0)  # (1, feature_dim)
            f2 = feature2[i].unsqueeze(0)  # (1, feature_dim)

            # 构建图
            x = torch.cat([f1, f2], dim=0)  # (2, feature_dim)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(feature1.device)

            # 图卷积
            x = self.gat(x, edge_index)  # (2, hidden_dim)

            # 通过全连接层映射到 feature_dim
            x = self.fc(x)  # (2, feature_dim)

            # 取两个节点的平均值作为当前样本的输出
            i0[i] = x.mean(dim=0)  # (feature_dim,)

        return i0  # (batch_size, feature_dim)