from flask import Flask, render_template, url_for, Response, request, jsonify
import cv2
from keras.models import load_model
import numpy as np
import time
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

app = Flask(__name__,static_url_path='/static')

# Opens Camera
cam = cv2.VideoCapture(0)

# Loading the face detection and the emotion classification models
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('mobile_net_v2_firstmodel.h5')
max_emotion = None
max_count = 0

def predict_emotion(face_image):
    face_image = cv2.imdecode(np.frombuffer(face_image, np.uint8), cv2.IMREAD_COLOR)
    final_image = cv2.resize(face_image, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    predictions = model.predict(final_image)

    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Surprise", "Sad", "Neutral"]
    predicted_emotion = emotion_labels[np.argmax(predictions)]

    return predicted_emotion

def detection():
    face_images = []
    capture_interval = 1
    start_time = time.time()
    
    global max_count, max_emotion

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            def expand_roi(x, y, w, h, scale_w, scale_h, img_shape):
                new_x = max(int(x - w * (scale_w - 1) / 2), 0)
                new_y = max(int(y - h * (scale_h - 1) / 2), 0)
                new_w = min(int(w * scale_w), img_shape[1] - new_x)
                new_h = min(int(h * scale_h), img_shape[0] - new_y)
                return new_x, new_y, new_w, new_h

            scale_w = 1.3
            scale_h = 1.5

            new_x, new_y, new_w, new_h = expand_roi(x, y, w, h, scale_w, scale_h, frame.shape)
            roi_color = frame[new_y:new_y+new_h, new_x:new_x+new_w]

            if time.time() - start_time >= capture_interval:
                face_images.append(cv2.imencode('.png', roi_color)[1].tobytes())
                if len(face_images) > 5:
                    face_images.pop(0)
                start_time = time.time()
                
            emotion_counts = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Surprise": 0, "Sad": 0, "Neutral": 0}
            if len(face_images) >= 4:
                    for face_image in face_images:
                        predicted_emotion = predict_emotion(face_image)
                        emotion_counts[predicted_emotion] += 1

                    max_emotion = max(emotion_counts, key=emotion_counts.get)
                    max_count = emotion_counts[max_emotion]
            status = max_emotion
            if status=="Surprise":
                status="Neutral"
            cv2.putText(frame, status, (100, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))   
                    #face_images=[]

        if cv2.waitKey(1) & 0xFF == 13:
            break
        
        ret, buffer = cv2.imencode('.png', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
    cam.release()
    cv2.destroyAllWindows()

def initialize_bot(current_mood):
    api_key = 'AIzaSyDCOQih0o6bZu_uSz4lWyOTP8d3ABEQis4'
    genai.configure(api_key = api_key)
    
    if current_mood=='Happy' or current_mood=='Surprise':
        title_template = PromptTemplate(input_variables = ['topic','current_mood'], template = 'You are a Personal Finance Advisor named BOB Saarthi from Bank of Badra. Your goal is to solve problems related to finance and health and engage in deep human-like conversations. As a personal mentor for investment, you should engage in therapeutic conversations, asking follow-up questions and talking like a caring friend or counselor. You should be hearing the user more. The user is in a happy and joyful mood. Acknowledge this and compliment them accordingly. Engage like a friendly counselor and make the conversation worth holding up to. Keeping that in mind, the query of the user is: {topic}. Respond to their queries with a warm, empathetic, therapeutic, and joyful tone, considering their mood, and aim to improve their emotional well-being. Keep it engaging by adding follow-up questions, making it like a joyful human conversation. Limit your answer to 60-80 words..')
    elif current_mood=='Angry' or current_mood=='Disgust':
        title_template = PromptTemplate(input_variables = ['topic','current_mood'], template = 'You are a Personal Finance Advisor named BOB Saarthi from Bank of Badra. Your goal is to assist users with their financial inquiries and engage in deep, supportive conversations. Your aim should be to provide financial advice, asking follow-up questions and talking like a caring financial counselor. You should be hearing the user more. The user is feeling stressed about their finances. Engage like a friendly financial advisor, maintaining a warm, empathetic, and professional tone. Respond to their queries with the intent to improve their financial well-being. Keep it engaging by adding follow-up questions, making it feel like a human conversation. Keep a calming tone and try to alleviate the user s financial stress by listening and replying appropriately. Limit your answer to 60-80 words.')
    elif current_mood=='Fear' or current_mood=='Sad':
        title_template = PromptTemplate(input_variables = ['topic','current_mood'], template = 'You are a Personal Finance Advisor named BOB Saarthi from Bank of Badra. Your goal is to assist users with their financial inquiries and engage in deep, supportive conversations. Your aim should be to provide financial advice, asking follow-up questions and talking like a caring financial counselor. You should be hearing the user more. The user seems to be in a sad mood. Acknowledge this and console them accordingly. Keeping that in mind, the query of the user is: {topic}. Respond to their queries with a warm, empathetic, and supportive tone, considering their mood, and aim to improve their financial well-being. Keep it engaging by adding follow-up questions, making it feel like a human conversation. Limit your answer to 60-80 words')
    else:
        title_template = PromptTemplate(input_variables = ['topic','current_mood'], template = 'You are a Personal Finance Advisor named BOB Saarthi from Bank of Badra. Your goal is to assist users with their financial inquiries and engage in deep, supportive conversations. Your aim should be to provide financial advice, asking follow-up questions and talking like a caring financial counselor. You should be hearing the user more. Engage like a friendly financial advisor. Keeping that in mind, the query of the user is: {topic}. Respond to their queries with a warm, empathetic tone, and aim to improve their financial well-being. Keep it engaging by adding follow-up questions, making it feel like a human conversation. Limit your answer to 60-80 words..')
    

    llm = ChatGoogleGenerativeAI(model = 'gemini-pro',google_api_key = api_key, temperature=0.5)
    answer_chain = LLMChain(llm = llm, prompt = title_template, verbose = False)
    
    return answer_chain
def clean_text(text):
    # Remove asterisks
    cleaned_text = text.replace('*', '')

    # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r' +', ' ', cleaned_text)

    # Replace tab characters with spaces
    cleaned_text = cleaned_text.replace('\t', ' ')

    # Trim leading and trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text

def bot_answer(question, current_mood):
    answer_chain = initialize_bot(current_mood)
    bot_response = answer_chain.run(topic = question, current_mood = current_mood)
    bot_response = clean_text(bot_response)
    return str(bot_response)

@app.route('/')
def about():
    return render_template('together.html')

@app.route('/submit', methods=['POST'])
def submit():
    global max_emotion, max_count
    response = bot_answer('Who are you?', max_emotion)
    return render_template('together.html', emotion=max_emotion, response=response)

@app.route('/video', methods=['GET', 'POST'])
def video():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    global max_emotion
    bot_response = bot_answer(user_message, max_emotion)
    return jsonify({'bot_message': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
