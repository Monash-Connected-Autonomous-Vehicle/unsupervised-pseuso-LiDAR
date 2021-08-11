import numpy as np
import matplotlib.pyplot as plt

from geometry.pose_geometry import disp_to_depth

def compute_errors(gt, pred):

    # convert to numpy
    gt   = gt.cpu().detach().numpy()
    
    pred = disp_to_depth(pred[0])
    pred = pred.cpu().detach().numpy()

    # calculate Accuracy
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    accuracy = {"silog": silog, "abs_rel": abs_rel, "log10": log10, \
                "rms": rms, "sq_rel": rms, "log_rms": log_rms, \
                "d1": d1, "d2": d2, "d3": d3}

    return accuracy
