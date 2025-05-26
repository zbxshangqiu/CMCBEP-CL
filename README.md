# CMCBEP-CL
CMCBEP-CL is a framework for multimodal sentiment analysis task.
## Introdction
We introduced a novel multimodal emotion prediction method, CMCBEP-CL, which leverages contrastive learning and cross-modal complementary balance to enhance sentiment analysis accuracy. The method comprises three core modules: a lead mode supplementation module, a cross-modal comparative analysis module, and a balanced prediction result module. The lead mode supplementation module identifies and utilizes the most representative modality to supplement feature representations of other modalities. The cross-modal comparative analysis module extracts specific and differential features through contrastive learning, improving discriminative semantic information extraction. Finally, the balanced prediction result module integrates prediction results from each stage to ensure robust sentiment analysis. Experimental results on the CMU-MOSEI, CMU-MOSI, and CH-SIMS datasets demonstrate that our proposed approach outperforms state-of-the-art methods.
## Requirements
Our code is written by Python, based on Pytorch (Version ≥ 1.4)
## Datasets
[CMU_MOSEI](https://aclanthology.org/P18-1208.pdf)/[CMU_MOSI](https://ieeexplore.ieee.org/document/7742221)/[CH-SMIS](https://aclanthology.org/2020.acl-main.343/)

The CMCBEP-CL uses feature files that are organized as follows:
``` 
{
    "train": {
        "raw_text": [],              # raw text
        "audio": [],                 # audio feature
        "vision": [],                # video feature
        "id": [],                    # [video_id$_$clip_id, ..., ...]
        "text": [],                  # bert feature
        "text_bert": [],             # word ids for bert
        "audio_lengths": [],         # audio feature lenth(over time) for every sample
        "vision_lengths": [],        # same as audio_lengths
        "annotations": [],           # strings
        "classification_labels": [], # Negative(0), Neutral(1), Positive(2). Deprecated in v_2.0
        "regression_labels": []      # Negative(<0), Neutral(0), Positive(>0)
    },
    "valid": {***},                  # same as "train"
    "test": {***},                   # same as "train"
}
```
## Results
We use the same metric set that has been consistently presented and compared before. Mean absolute error (MAE) is the average mean difference value between predicted values and truth values. Pearson correlation (Corr) measures the degree of prediction skew. Seven-class classification accuracy (Acc-7) indicates the proportion of predictions that correctly fall into the same interval of seven intervals between -3 and +3 as the corresponding truths. Binary classification accuracy (Acc-2) and F1 score are computed for positive/negative and non-negative/negative classification results. The results are listed in table. 

<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="276" colspan="5" valign="top" style="width:207.3pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MOSEI</span></p>
  </td>
  <td width="277" colspan="5" valign="top" style="width:207.5pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MOSI</span></p>
  </td> 
  <td width="277" colspan="5" valign="top" style="width:207.5pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">CH-SMIS</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="55" valign="top" style="width:41.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MAE</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">CORR</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-7</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-2</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">F1</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MAE</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">CORR</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-7</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-2</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">F1</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MAE</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Corr</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Acc-2</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">Acc-5</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">F1</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;mso-yfti-lastrow:yes">
  <td width="55" valign="top" style="width:41.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.529</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.768</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">54.20</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.28</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.19</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.710</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.803</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">46.56</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.13</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.01</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.380</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.686</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">83.68</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">46.26</span></p>
  </td>
     <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">84.98</span></p>
  </td>
 </tr>
</tbody></table>

### Visualization results

Here, we further visualize the feature distribution in the CMCBEP-CL model by generating a heatmap.
* Visualization Sample1

Visualization of feature distribution in the first round of training.

![Visualization1.pdf](https://github.com/zbxshangqiu/CMCBEP-CL/Fig4a.png)  
* Visualization Sample2

Visualization of feature distribution in the 42nd round of training.

![Visualization2.pdf](https://github.com/zbxshangqiu/CMCBEP-CL/Fig4b.png)  


* Visualization Sample3

Visualization of feature distribution in the 83rd round of training.

![Visualization3.pdf](https://github.com/zbxshangqiu/CMCBEP-CL/Fig4c.png)  

* Visualization Sample4

Overall framework diagram of the model

![Visualization4.pdf](https://github.com/zbxshangqiu/CMCBEP-CL/Fig2.png)  

* Visualization Sample5

Balance prediction result module

![Visualization5.pdf](https://github.com/zbxshangqiu/CMCBEP-CL/Fig3.png) 

## Usage
1.Clone the repository
``` 
git clone https://github.com/zbxshangqiu/CMCBEP-CL.git
```
2.Download dataset config and put the split dataset folders into $ROOT_DIR/datasets/. The folders are arranged like this:
```
├datasets         
    
    ├── MOSEI
    │   ├── mosei_data_noalign.pkl    
    │   ├── MOSEI-label
    
    ├── MOSI    
    │   ├── mosi_data_noalign.pkl    
    │   ├── MOSI-label

    ├── CH-SIMS    
    │   ├── ch-sims_data_noalign.pkl    
    │   ├── CH-SIMS-label  
 ```
 3.Train the model
  ```
cd src
python main.py
  ```
  
## Conclusion  
  
This article proposes a cross modal complementary balanced multimodal sentiment analysis technique driven by contrastive learning. CMCBEP-CL consists of three main modules, namely the dominant mode supplement module, the balanced prediction result module, and the cross mode comparative analysis module. This model utilizes multimodal supplementation, enhancement, and fusion to enhance feature differences and correlations through cross modal contrastive learning, balancing the predictive results of each stage of multimodal sentiment analysis. Three well used datasets, CMU-MOSEI, CMU-MOSI, and CH-SIMS, were used to evaluate the proposed CMCBEP-CL. Many tests and ablation studies have demonstrated the effectiveness of this algorithm.
