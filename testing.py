import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn import metrics
import numpy as np
import json
import shutil
import os

info_path = "list/frame_info.json"

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh-test.npy')
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        
        if args.error_analysis == 1:
          r_pred = np.array([np.round(x) for x in pred])

          comparison = (r_pred==gt)
          indexes = [index for index in range(len(comparison)) if comparison[index] == False]
          frame_indexes = [indexes[0]]

          for j in range(len(indexes)):
            try:
              if indexes[j+1] - indexes[j] > 1:
                frame_indexes.append(indexes[j])
                frame_indexes.append(indexes[j+1])
            except:
              pass
          frame_indexes.append(indexes[-1])
          with open(info_path, "r") as f:
            frame_info = f.read()
          frame_info = json.loads(frame_info)

          err_frame_info = []
          err_videos = []
          i=0
          while i < len(frame_indexes):
            for vid in frame_info:
              if frame_indexes[i] <= vid[list(vid.keys())[0]][1] and frame_indexes[i] >= vid[list(vid.keys())[0]][0]:
                  err_frame_info.append({list(vid.keys())[0]: [frame_indexes[i], frame_indexes[i+1]]})
                  if list(vid.keys())[0] not in err_videos:
                    err_videos.append(list(vid.keys())[0])
            i = i+2
            if os.path.isdir('error'):
              shutil.rmtree('error')
            os.mkdir('error')

            with open('error/err_videos.txt', 'w') as f:
              f.write(str(err_videos))
            with open('error/err_frame_info.txt', 'w') as f:
              f.write(str(err_frame_info))

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = metrics.auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = metrics.auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)

        return rec_auc