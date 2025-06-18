from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

############ Load zero-shot classification model and tokenizer
model_name = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)  #AutoTokenizer= model automatically selects the appropriate tokenization

# Load AG News test dataset
dataset = load_dataset("ag_news", split="test")
dataset = dataset.shuffle(seed=69)
dataset = dataset.select(range(20))  #Sample number

# Labels
labels = ["World", "Sports", "Business", "Science"]

# Few-shot examples
few_shot_examples = [
    ("The government announced new policies today.", "World"),
    ("A peace treaty was signed between the two nations.", "World"),
    ("Elections are scheduled for next month.", "World"),
    ("Diplomatic talks between countries resumed.", "World"),
    ("A major earthquake struck the city.", "World"),
    ("Floods have devastated several regions.", "World"),
    ("International sanctions were imposed.", "World"),
    ("The president met with foreign leaders.", "World"),
    ("War tensions are rising in the Middle East.", "World"),
    ("New immigration laws are being debated.", "World"),

    ("The football team won their game last night.", "Sports"),
    ("The basketball finals are scheduled for next week.", "Sports"),
    ("The tennis player won her 10th Grand Slam title.", "Sports"),
    ("A star athlete broke the world record.", "Sports"),
    ("The Olympics were postponed due to weather.", "Sports"),
    ("The baseball league announced the new season.", "Sports"),
    ("Fans celebrated the championship victory.", "Sports"),
    ("The coach announced the team lineup.", "Sports"),
    ("The marathon attracted thousands of runners.", "Sports"),
    ("The boxer defended his title in a thrilling match.", "Sports"),

    ("Stock prices rose sharply after the announcement.", "Business"),
    ("The company reported record profits this quarter.", "Business"),
    ("The CEO resigned amid controversy.", "Business"),
    ("New startups are attracting investor interest.", "Business"),
    ("The market saw a significant downturn today.", "Business"),
    ("The merger between two tech giants was finalized.", "Business"),
    ("Oil prices surged due to supply concerns.", "Business"),
    ("Unemployment rates fell last month.", "Business"),
    ("The retail sector experienced strong sales.", "Business"),
    ("New regulations affect the banking industry.", "Business"),

    ("Scientists discovered a new exoplanet.", "Science"),
    ("Researchers developed an innovative AI model.", "Science"),
    ("A breakthrough in cancer research was announced.", "Science"),
    ("The tech company unveiled a new smartphone.", "Science"),
    ("Advances in renewable energy technology continue.", "Science"),
    ("SpaceX launched its latest rocket successfully.", "Science"),
    ("New software improves data security.", "Science"),
    ("Scientists mapped the human genome in detail.", "Science"),
    ("Quantum computing made significant progress.", "Science"),
    ("Researchers study climate change effects.", "Science"),

    ("The government passed new environmental laws.", "World"),
    ("The local team won the regional championship.", "Sports"),
    ("Tech stocks are performing well on the market.", "Business"),
    ("A new vaccine shows promising results in trials.", "Science"),
    ("International aid was sent after the disaster.", "World"),
    ("The soccer league announced schedule changes.", "Sports"),
    ("Banks reported increased loan approvals.", "Business"),
    ("Innovative gadgets are showcased at the expo.", "Science"),
    ("Peace talks continue despite setbacks.", "World"),
    ("The swimmer set a new national record.", "Sports"),
]

# Prediction + Metric Storage
y_true = []
y_pred = []
input_token_counts = []
output_token_counts = []

# Loop through examples
for example in dataset:
    news_text = example["text"]
    true_label = labels[example["label"]]

    # Prompt construction
    prompt = "Classify the following news article into one of these categories: World, Sports, Business, Science.\n\n"
    for ex_text, ex_label in few_shot_examples:
        prompt += f"Text: {ex_text}\nCategory: {ex_label}\n\n"
    prompt += f"Text: {news_text}\nCategory:"

    # Token usage metrics
    input_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]

    # Model call
    result = classifier(prompt, labels, multi_label=False)
    prediction = result["labels"][0]
    output_tokens = tokenizer(prediction, return_tensors="pt")["input_ids"].shape[-1]

    # Store results
    y_true.append(true_label)
    y_pred.append(prediction)
    input_token_counts.append(input_tokens)
    output_token_counts.append(output_tokens)

    # Print
    print("\n📰 News:", news_text[:120], "...")
    print("  True Category:", true_label)
    print(" Prediction:", prediction)

# Classification Metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

print(f"\n Accuracy: {accuracy:.2f}")
print(f" Precision: {precision:.2f}")
print(f" Recall: {recall:.2f}")
print(f" F1 Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Prediction")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Token Usage Metrics
avg_input_tokens = sum(input_token_counts) / len(input_token_counts)
avg_output_tokens = sum(output_token_counts) / len(output_token_counts)
avg_total_tokens = avg_input_tokens + avg_output_tokens

print(f"\n Average Input Tokens per Inference: {avg_input_tokens:.2f}")
print(f" Average Output Tokens per Inference: {avg_output_tokens:.2f}")
print(f" Average Total Tokens per Inference: {avg_total_tokens:.2f}")
