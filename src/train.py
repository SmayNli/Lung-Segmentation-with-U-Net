from src.model import UNET
from src.dataloader import create_dataloaders
import torch
from src.metrics import calculate_metrics
from src.loss_function import DiceBCELoss

def train_step(model, dataloader, loss_fn, optimizer, device):

    model.train()

    train_loss, iou_score, dice_score = 0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_preds=model(X)

        iou, dice = calculate_metrics(y_preds, y)
        iou_score+=iou
        dice_score+=dice

        loss=loss_fn(y_preds, y)
        train_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss=train_loss/len(dataloader)
    iou_score=iou_score/len(dataloader)
    dice_score=dice_score/len(dataloader)

    return train_loss, iou_score, dice_score

def val_step(model, dataloader, loss_fn, device):

    model.eval()

    with torch.inference_mode():

        val_loss, iou_score, dice_score = 0, 0, 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_preds=model(X)

            iou, dice = calculate_metrics(y_preds, y)
            iou_score+=iou
            dice_score+=dice

            loss=loss_fn(y_preds, y)
            val_loss+=loss.item()


    val_loss=val_loss/len(dataloader)
    iou_score=iou_score/len(dataloader)
    dice_score=dice_score/len(dataloader)

    return val_loss, iou_score, dice_score

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epochs=10):
    train_loss_list=[]
    train_iou_list=[]
    train_dice_list=[]

    val_loss_list=[]
    val_iou_list=[]
    val_dice_list=[]

    for epoch in range(epochs):
        train_loss, train_iou, train_dice = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        val_loss, val_iou, val_dice = val_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device)

        print(f"Epochs: {epoch+1}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train DICE: {train_dice:.4f}, Val loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val DICE: {val_dice:.4f}")
        train_loss_list.append(train_loss)
        train_iou_list.append(train_iou)
        train_dice_list.append(train_dice)

        val_loss_list.append(val_loss)
        val_iou_list.append(val_iou)
        val_dice_list.append(val_dice)

    return {"Train Loss": train_loss_list, "Train IoU": train_iou_list, "Train DICE": train_dice_list, "Val Loss": val_loss_list, "Val IoU": val_iou_list, "Val DICE": val_dice_list}

if __name__=="__main__":
    model=UNET()

    loss_fn=DiceBCELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataloader, val_dataloader = create_dataloaders(batch_size=1)

    results=train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, loss_fn=loss_fn, optimizer=optimizer, device="cpu", epochs=10)
    