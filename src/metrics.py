import torch

@torch.no_grad()
def calculate_metrics(y_pred, y_true, threshold=0.5, epsilon=1e-7):

    probs=torch.sigmoid(y_pred)

    y_pred=(probs>threshold).float()
    y_true=y_true.float()

    y_pred=y_pred.view(-1)
    y_true=y_true.view(-1)

    intersection = (y_pred * y_true).sum()
    total = (y_pred.sum() + y_true.sum())
    union = total - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2 * intersection + epsilon) / (total + epsilon)

    return iou.item(), dice.item()

if __name__ == "__main__":

    preds=torch.randn(256,256)
    y_true=torch.randn(256,256)

    print(calculate_metrics(preds, y_true))
