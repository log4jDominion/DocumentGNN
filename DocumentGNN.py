import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from transformers import BertModel, BertTokenizer
import numpy as np

# Document Embedding using BERT
class DocumentEmbedder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)

        # Averaging across the sequence length to get a single embedding per document
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

        return embedding


# Graph Neural Network Model
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Construct Graph
def build_document_graph(doc_embeddings):
    num_docs = len(doc_embeddings)
    edge_index = []

    # Create edges based on cosine similarity between document embeddings
    sim_matrix = cosine_similarity(doc_embeddings)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            if sim_matrix[i, j] > 0.7:  # Threshold for similarity
                edge_index.append([i, j])
                edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


# Training the GNN
def train_gnn(train_data, labels, num_classes):
    model = GNN(num_node_features=train_data.x.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out[train_data.train_mask], labels[train_data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model


def test_model(model, data):
    """
    Function to test the model on the given data and return predictions with percentages.

    Args:
    - model: The trained GNN model.
    - data: The data object containing the test node features and graph structure.

    Returns:
    - A sorted list of tuples where each tuple contains the predicted class (box number) and the corresponding probability.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Perform a forward pass through the model to get the raw logits
        out = model(data)

        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(out, dim=1)

        # Get the predicted class and the associated probability for each document
        pred_classes = torch.argmax(probabilities, dim=1).cpu().numpy()  # Predicted class for each document
        pred_probs = probabilities.cpu().numpy()  # Probabilities for each class (for each document)

    predictions = []
    for i in range(len(pred_classes)):
        # Sort the probabilities and get the corresponding class labels
        sorted_probs = np.argsort(pred_probs[i])[::-1]  # Sort in descending order
        sorted_classes_with_probs = [(cls, pred_probs[i][cls] * 100) for cls in sorted_probs]  # Convert to percentage
        predictions.append(sorted_classes_with_probs)

    return predictions


def example_test():
    # Assume you have trained model and test_data available
    # model: The trained GNN model
    # data: The test data object containing node features, edge indices, etc.

    # Get sorted predictions with probabilities
    predictions = test_model(model, data)

    # Print the predictions for each document
    for doc_id, doc_predictions in enumerate(predictions):
        print(f"\nDocument {doc_id} predictions (sorted by probability):")
        for pred_class, probability in doc_predictions:
            print(f"Box {pred_class}: {probability:.2f}%")


# Example Usage
if __name__ == '__main__':
    # Example documents (In practice, you will use your own dataset)
    documents = ["Document 1 text", "Document 2 text", "Document 3 text", "Document 4 text", "Document 5 text"]
    labels = torch.tensor([0, 0, 1, 1, 2])  # Box numbers for each document

    embedder = DocumentEmbedder()
    doc_embeddings = [embedder.get_embedding(doc) for doc in documents]

    # Build document graph
    edge_index = build_document_graph(doc_embeddings)
    node_features = torch.tensor(doc_embeddings, dtype=torch.float)

    # Create data object for GNN

    data = Data(x=node_features, edge_index=edge_index)

    # Define training masks (for simplicity, all nodes are part of training data)
    data.train_mask = torch.tensor([True, True, True, True, True])

    # Train GNN
    model = train_gnn(data, labels, num_classes=3)
    example_test()

def test_model(model, data):
    """
    Function to test the model on the given data and return predictions with percentages.

    Args:
    - model: The trained GNN model.
    - data: The data object containing the test node features and graph structure.

    Returns:
    - A sorted list of tuples where each tuple contains the predicted class (box number) and the corresponding probability.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Perform a forward pass through the model to get the raw logits
        out = model(data)

        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(out, dim=1)

        # Get the predicted class and the associated probability for each document
        pred_classes = torch.argmax(probabilities, dim=1).cpu().numpy()  # Predicted class for each document
        pred_probs = probabilities.cpu().numpy()  # Probabilities for each class (for each document)

    predictions = []
    for i in range(len(pred_classes)):
        # Sort the probabilities and get the corresponding class labels
        sorted_probs = np.argsort(pred_probs[i])[::-1]  # Sort in descending order
        sorted_classes_with_probs = [(cls, pred_probs[i][cls] * 100) for cls in sorted_probs]  # Convert to percentage
        predictions.append(sorted_classes_with_probs)

    return predictions

