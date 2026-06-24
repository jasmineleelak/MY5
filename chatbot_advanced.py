import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
import pyttsx3
import speech_recognition as sr
import sys
import threading

# Initialize
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Load model and data
print("Loading model and data...")
try:
    classifier = pickle.load(open('chatbot_model.pkl', 'rb'))
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    with open('intents.json', 'r') as file:
        intents = json.load(file)
    print("✓ Model loaded successfully!\n")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("\nModel files not found!")
    print("Please run: python train_model.py")
    sys.exit(1)

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listen to microphone and convert speech to text"""
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except:
            return None

def clean_input(text):
    """Clean and tokenize user input"""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return tokens

def bag_of_words(tokens):
    """Convert tokens to bag of words"""
    bag = [1 if word in tokens else 0 for word in words]
    return bag

def get_intent(user_input):
    """Predict intent from user input"""
    tokens = clean_input(user_input)
    bag = bag_of_words(tokens)
    
    prediction = classifier.predict_proba([bag])[0]
    intent_index = prediction.argmax()
    confidence = prediction[intent_index]
    
    if confidence < 0.3:
        return None, 0
    
    return classes[intent_index], confidence

def get_response(intent):
    """Get random response for an intent"""
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm not sure how to respond to that. Can you rephrase?"

def chat(user_input):
    """Main chat function"""
    intent, confidence = get_intent(user_input)
    
    if intent is None:
        return "I didn't understand that. Can you rephrase?"
    
    response = get_response(intent)
    return response

# ============= REST API ENDPOINTS =============

@app.route('/chat', methods=['POST'])
def chat_api():
    """Chat endpoint - Text based"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = chat(user_message)
        intent, confidence = get_intent(user_message)
        
        return jsonify({
            'user_message': user_message,
            'bot_response': response,
            'intent': intent,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat-voice', methods=['POST'])
def chat_voice_api():
    """Chat endpoint - Voice based (text input, voice output)"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = chat(user_message)
        intent, confidence = get_intent(user_message)
        
        # Generate voice in background
        threading.Thread(target=speak, args=(response,)).start()
        
        return jsonify({
            'user_message': user_message,
            'bot_response': response,
            'intent': intent,
            'confidence': float(confidence),
            'voice_enabled': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/listen', methods=['GET'])
def listen_api():
    """Listen to microphone and convert to text"""
    try:
        text = listen()
        if text:
            return jsonify({'status': 'success', 'text': text})
        else:
            return jsonify({'status': 'error', 'message': 'Could not understand audio'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/speak', methods=['POST'])
def speak_api():
    """Convert text to speech"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Generate voice in background
        threading.Thread(target=speak, args=(text,)).start()
        
        return jsonify({'status': 'success', 'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'OK', 'message': 'Advanced Chatbot is running!'})

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'name': 'AI Chatbot - Advanced Version',
        'version': '2.0',
        'features': ['Text Chat', 'Voice Chat', 'Speech Recognition', 'Text-to-Speech'],
        'endpoints': {
            'chat': 'POST /chat - Text based chat',
            'chat_voice': 'POST /chat-voice - Chat with voice output',
            'listen': 'GET /listen - Convert speech to text',
            'speak': 'POST /speak - Convert text to speech',
            'health': 'GET /health - Health check'
        }
    })

# ============= INTERACTIVE MODE =============

def interactive_mode():
    """Interactive text mode"""
    print("\n" + "="*60)
    print("AI CHATBOT - Interactive Text Mode")
    print("="*60)
    print("Type 'quit' or 'exit' to close")
    print("Type 'help' for available intents")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nBot: Goodbye! Have a great day!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable intents:")
                for intent in intents['intents']:
                    print(f"  - {intent['tag']}: {', '.join(intent['patterns'])}")
                print()
                continue
            
            response = chat(user_input)
            intent, confidence = get_intent(user_input)
            print(f"Bot: {response}")
            print(f"(Intent: {intent}, Confidence: {confidence:.2%})\n")
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

def voice_mode():
    """Interactive voice mode"""
    print("\n" + "="*60)
    print("🎤 AI CHATBOT - Voice Mode")
    print("="*60)
    print("Make sure your microphone is connected!")
    print("Say 'quit' or 'exit' to close")
    print("="*60 + "\n")
    
    speak("Hello! I'm your AI Chatbot. How can I help you?")
    
    while True:
        try:
            print("🎤 Listening...")
            user_input = listen()
            
            if user_input is None:
                speak("Sorry, I didn't catch that. Can you repeat?")
                continue
            
            print(f"You: {user_input}")
            
            if user_input.lower() in ['quit', 'exit']:
                speak("Goodbye! Have a great day!")
                break
            
            if user_input.lower() == 'help':
                help_text = "Available intents: " + ", ".join([i['tag'] for i in intents['intents']])
                speak(help_text)
                continue
            
            response = chat(user_input)
            intent, confidence = get_intent(user_input)
            speak(response)
            print(f"(Intent: {intent}, Confidence: {confidence:.2%})\n")
            
        except KeyboardInterrupt:
            speak("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--api':
            print("\n" + "="*60)
            print("Starting Advanced Chatbot API Server...")
            print("="*60)
            print("API Running at: http://localhost:5000")
            print("\nEndpoints:")
            print("  POST /chat - Text chat")
            print("  POST /chat-voice - Chat with voice output")
            print("  GET /listen - Speech to text")
            print("  POST /speak - Text to speech")
            print("  GET /health - Health check")
            print("="*60 + "\n")
            app.run(debug=True, port=5000)
        elif sys.argv[1] == '--voice':
            voice_mode()
        else:
            interactive_mode()
    else:
        interactive_mode()
