import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample intents with example phrases
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening", "oi", "howdy"],
    "goodbye": ["bye", "see you", "take care", "goodbye", "talk to you later"],
    "order_status": ["where is my order", "track my package", "order status", "order not received"],
    "refund": ["i want a refund", "return my product", "money back please", "refund request"],
    "smalltalk": ["how are you", "how's it going", "who are you", "what can you do", "what's up"]
}

# Prepare training data
X_train = []
y_train = []

for intent, phrases in intents.items():
    for phrase in phrases:
        X_train.append(phrase)
        y_train.append(intent)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X_train)

# Train classifier
model = LogisticRegression()
model.fit(X_vectors, y_train)

# Response templates
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "order_status": ["Let me check your order.", "Your order is in transit.", "Checking your order status..."],
    "refund": ["Please provide your order number.", "We'll process your refund soon.", "Refund has been initiated."],
    "smalltalk": ["I'm your virtual assistant!", "I help with order queries and more.", "Here to help!"]
}

# Chatbot function
def chatbot_response(user_input):
    input_vec = vectorizer.transform([user_input.lower()])
    intent = model.predict(input_vec)[0]
    return random.choice(responses[intent])

# Chat loop
print("Lightweight Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_response(user_input))
