LLM Text Classification on AG News Dataset

This project demonstrates text classification using a Large Language Model (LLM) to categorize news articles from the AG News dataset into four classes: World, Sports, Business, and Science.

Features

- Prompt engineering for effective LLM classification  
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix  
- Token usage efficiency measurement (input/output tokens)  
- Simple Python implementation  

Requirements

- Python 3.8+  
- See `requirements.txt` for all necessary packages  

Setup

1. Clone the repository:

git clone https://github.com/mehmetegeinan/llm-text-classification-agnews.git
cd llm-text-classification-agnews

2. Create and activate a virtual environment (recommended):

- On Windows:

python -m venv venv
venv\Scripts\activate

- On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install required packages:

pip install -r requirements.txt

4. Run the classification script:

python main.py

This will perform classification on a subset of the AG News dataset and output performance and token usage metrics.

Notes

- No fine-tuning of the LLM is performed; classification is based purely on prompt design.  
- The AG News dataset is used as the evaluation benchmark.  
- Token usage statistics help understand the efficiency of different prompt designs.
