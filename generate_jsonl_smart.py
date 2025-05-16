#!/usr/bin/env python3
import json
import re
import string
import nltk
import dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------
# SETUP
# ----------------------------

nltk.download('punkt', quiet=True)

# Load models
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, light
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define known real-world diary topics
topic_labels = [
    "Work", "Travel", "Family", "Friends", "Health", "Emotions", "Finance",
    "Achievements", "Challenges", "Education", "Shopping", "Events",
    "Entertainment", "Food", "Weather", "Technology", "Random thoughts"
]

# Embed the topics once
topic_embeddings = embed_model.encode(topic_labels)

# Map topic → smart question
category_to_question = {
    "Work": "What work or career experiences did I have?",
    "Travel": "What travel or trip experiences did I have?",
    "Family": "What family-related events did I experience?",
    "Friends": "What did I do with my friends?",
    "Health": "What health or fitness events did I go through?",
    "Emotions": "How was I feeling emotionally at that time?",
    "Finance": "What financial decisions or actions did I take?",
    "Achievements": "What achievement or success did I celebrate?",
    "Challenges": "What challenges or failures did I face?",
    "Education": "What educational experiences did I have?",
    "Shopping": "What shopping or buying activities did I do?",
    "Events": "What event or celebration did I participate in?",
    "Entertainment": "What entertainment activities did I enjoy?",
    "Food": "What food-related experiences did I have?",
    "Weather": "What was the weather like or what nature did I observe?",
    "Technology": "What technology-related activities did I do?",
    "Random thoughts": "What thoughts or dreams did I reflect on?"
}

# ----------------------------
# FUNCTIONS
# ----------------------------

def extract_date(text):
    """Extract date like Dec 25, 2023 from paragraph."""
    match = re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \s*\d{4}\b', text)
    if match:
        parsed_date = dateparser.parse(match.group(0))
        return parsed_date.strftime("%Y-%m-%d") if parsed_date else None
    return None

def detect_year_from_text(text):
    """Detect year like 2018 from sentence."""
    match = re.search(r'\b(19|20)\d{2}\b', text)
    if match:
        return match.group(0)
    return None

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def predict_best_topic(sentence_embedding):
    """Predict best matching topic based on semantic similarity."""
    similarities = [cosine_similarity(sentence_embedding, topic_emb) for topic_emb in topic_embeddings]
    best_idx = np.argmax(similarities)
    return topic_labels[best_idx]

def generate_super_smart_question_and_topic(sentence):
    """Generate both smart question and real topic."""
    embedding = embed_model.encode(sentence)
    best_topic = predict_best_topic(embedding)
    question = category_to_question.get(best_topic, "What happened during this time?")
    return question, best_topic

# ----------------------------
# MAIN
# ----------------------------

def diary_to_memories(input_file='diary.txt', output_file='memories.jsonl'):
    with open(input_file, 'r', encoding='utf-8') as f:
        diary_text = f.read()

    paragraphs = [p.strip() for p in diary_text.split('\n\n') if p.strip()]
    sentences, dates = [], []

    for para in paragraphs:
        dt = extract_date(para) or "Unknown"
        for sent in nltk.sent_tokenize(para):
            if len(sent.strip()) < 10:
                continue
            sentences.append(sent.strip())
            dates.append(dt)

    if not sentences:
        print("❌ No sentences found. Check your diary.txt format.")
        return

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for idx, sentence in enumerate(sentences):
            real_year = detect_year_from_text(sentence)
            memory_date = f"{real_year}-01-01" if real_year else dates[idx]

            question, predicted_topic = generate_super_smart_question_and_topic(sentence)

            memory = {
                "input": question,
                "output": sentence,
                "date": memory_date,
                "topic": predicted_topic
            }
            out_file.write(json.dumps(memory, ensure_ascii=False) + '\n')

    print(f"✅ {len(sentences)} memories generated and saved to '{output_file}'!")

if __name__ == "__main__":
    diary_to_memories()
    