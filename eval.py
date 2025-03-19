import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Funzione per valutare il modello
def evaluate_model(model, dataloader, device):
    model.eval()  # Metti il modello in modalità valutazione
    y_true = []
    y_pred = []
    
    with torch.no_grad():  # Disabilita il calcolo del gradiente
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Ottieni le classi con la massima probabilità
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy del modello: {accuracy:.2f}')
    return accuracy

# Supponiamo che tu abbia un modello pre-addestrato
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(pretrained=True)  # Modello pre-addestrato
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adatta alla tua classificazione
model.to(device)

# Valutiamo il modello
evaluate_model(model, dataloader_test, device)
