import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW,AutoConfig
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub
# loader 
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label)}

def main():
    # whether there is any available GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # your model or any pretrain model on the hugging face 
    tokenizer = AutoTokenizer.from_pretrained("Yeerchiu/mtwitter-roberta-base-model-reviewingcls")
    model = AutoModelForSequenceClassification.from_pretrained("Yeerchiu/mtwitter-roberta-base-model-reviewingcls")
    config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    # 
    model = model.to(device)
    # define the weights for each class
    weights = torch.tensor([3.0, 5.0, 1.0]).to(device)  # change weight to balance the dataset 

    # define the loss function
    criterion = nn.CrossEntropyLoss(weight=weights)

    # data
    ds =pd.read_json("put your data path(json) here")
    ds = ds.dropna()
    texts = ds["review_text"].reset_index(drop=True)
    labels = ds["label"].reset_index(drop=True)

    
    texts_train, texts_temp, labels_train, labels_temp = train_test_split(texts, labels, test_size=0.2, random_state=42)
    texts_train = texts_train.reset_index(drop=True)
    labels_train = labels_train.reset_index(drop=True)
    texts_temp = texts_temp.reset_index(drop=True)
    labels_temp = labels_temp.reset_index(drop=True)

    
    texts_val, texts_test, labels_val, labels_test = train_test_split(texts_temp, labels_temp, test_size=0.2, random_state=42)
    texts_val = texts_val.reset_index(drop=True)
    labels_val = labels_val.reset_index(drop=True)
    texts_test = texts_test.reset_index(drop=True)
    labels_test = labels_test.reset_index(drop=True)

    train_dataset = TextClassificationDataset(texts_train, labels_train, tokenizer)
    val_dataset = TextClassificationDataset(texts_val, labels_val, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    # train
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    epochs=3
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)#loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # pred
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        scheduler.step()

         
        avg_loss = total_loss / len(train_dataloader)
        report = classification_report(all_labels, all_preds, target_names=['0', '1', '2'])#0: Neg, 1:Neu, 2:Pos
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}\n{report}')

        # eval
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_report = classification_report(val_labels, val_preds, target_names=['0', '1', '2'])
        print(f'Validation Report:\n{val_report}')

        model.train()
        #save your model on local
        torch.save(model, f'twitter-roberta-base-mode_{epoch}.pth')


        huggingface_hub.login(token = "put your own hugging face access token here")#write
        #save your training result to hugging face, need to change to your own path
        model.push_to_hub("Yeerchiu/mtwitter-roberta-base-model-reviewingcls")
        tokenizer.push_to_hub("Yeerchiu/mtwitter-roberta-base-model-reviewingcls")
        config.push_to_hub("Yeerchiu/mtwitter-roberta-base-model-reviewingcls")

if __name__ == '__main__':
    main()
