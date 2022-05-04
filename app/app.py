# import required module
import multiprocessing
import os

import cv2
import pandas as pd

import bcrypt
import detectlanguage
import playsound
import speech_recognition as s_r
from cv2 import VideoCapture, namedWindow, destroyWindow, imwrite
from deep_translator import GoogleTranslator
from flask import Flask, make_response, request, jsonify, render_template, session
from flask_mongoengine import MongoEngine
from gtts import gTTS
from mongoengine import IntField
from mongoengine import StringField
from playsound import playsound
import chatbot_fonctions as chat
import json
import compare_image
import time
from werkzeug.utils import secure_filename


def text_to_speech(text, language):
    mytext = "bonjour je suis ton assitant personelle comment puis je vous aidez"


detectlanguage.configuration.api_key = "d7f3b6137274f895cc6fd059befdd9b2"

import os

last_answer = ""

dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, root_path=dir_path)

database_name = "API"
DB_URI = "mongodb+srv://shadow:IoWeSNE9oFAdvgV4@cluster0.w2fqm.mongodb.net/API?retryWrites=true&w=majority"

app.config["MONGODB_HOST"] = DB_URI
db = MongoEngine()
db.init_app(app)
p = multiprocessing.Process(target=playsound, args=("1.mp3",))


class Disscussion(db.Document):
    client_question = StringField()
    chatbot_answer = StringField()
    is_correct_answer = IntField()

    def to_json(self):
        return {
            "client_question": self.client_question,
            "chatbot_answer": self.chatbot_answer,
            "is_correct_answer": self.is_correct_answer
        }


class User(db.Document):
    email = StringField()
    password = StringField()
    role = StringField()
    img = StringField()

    def to_json(self):
        return {
            "email": self.email,
            "password": self.password,
            "img": self.img,
            "role": self.role

        }


class Book(db.Document):
    book_id = IntField()
    name = StringField()
    author = StringField()

    def to_json(self):
        return {
            "book_id": self.book_id,
            "name": self.name,
            "author": self.author}


class Chat(db.Document):
    chat_id = IntField()
    Question = StringField()
    Answer = StringField()
    Class = StringField()
    Explication = StringField()

    def to_json(self):
        return {

            "Question": self.Question,
            "Answer": self.Answer,
            "Class": self.Class,
            "Explication": self.Explication

        }


@app.route('/')
def hello_world():  # put application's code here

    return render_template('chat.html')


@app.route('/d')
def hello_worldd():  # put application's code here

    return render_template('Test.html')


@app.route('/hello2')
def hello_world2():  # put application's code here

    return render_template('index2.html')


# exemple d'ajout
@app.route('/api/dbPopulate', methods=['POST'])
def db_populate():
    chat = User(email="hamma", password="123456", role="role")
    chat.save()
    return make_response("", 201)


# petit exemple de crud
@app.route('/api/books/<book_id>', methods=['GET', 'PUT', 'DELETE'])
def api_each_book(book_id):
    if request.method == "GET":
        book_obj = Book.objects(book_id=book_id).first()
        if book_obj:
            return make_response(jsonify(book_obj.to_json()), 200)
        else:
            return make_response("", 404)


    elif request.method == "PUT":
        '''
        Sample Request Body
        {
        "book_id" : 1,
        "name" : "Game of thrones",
        "author" : "George Martin luther king"
        }

        '''
        content = request.json
        book_obj = Book.objects(book_id=book_id).first()
        book_obj.update(author=content['author'], name=content['name'])
        return make_response("", 204)

    elif request.method == "DELETE":
        book_obj = Book.objects(book_id=book_id).first()
        book_obj.delete()
        return make_response("")


def chatbot_response(msg, var):
    res = chat.get_response(msg)
    global last_answer

    try:
        if res[0].__contains__("IDU1"):
            if last_answer == "":
                return "manajemch nfaserlek akther "
            last_answer2 = last_answer
            last_answer = ""
            return last_answer2

        if res[0].__contains__("IDU2"):
            if last_answer == "":
                return "Desolé mais j'ai pas d'explication "
            last_answer2 = last_answer
            last_answer = ""
            return last_answer2
        if res[0].__contains__("IDU3"):
            if last_answer == "":
                return "I don't have an explication"
            last_answer2 = last_answer
            last_answer = ""
            return last_answer2
    except:
        print("None answer value")
    if res[1] < [[0.52]] and detectlanguage.simple_detect(msg) == "en":
        if res[1] == [[0]]:
            dis = Disscussion(client_question=msg, chatbot_answer="", is_correct_answer=0)
            dis.save()
            return chat.randomsuggestion()
        else:

            dis = Disscussion(client_question=msg, chatbot_answer="", is_correct_answer=0)
            dis.save()
            try:
                return chat.get_response2(msg)
            except:
                return chat.randomsuggestion()

    if (detectlanguage.simple_detect(msg) == "fr" or detectlanguage.simple_detect(
            msg) == "ar" or detectlanguage.simple_detect(msg) == "ko" or detectlanguage.simple_detect(msg) == "es") and \
            res[1] < [[0.52]]:

        print(detectlanguage.languages())
        translated = GoogleTranslator(source='auto', target='en').translate(msg)

        print("translation ", translated)
        res = chat.get_response(translated)
        try:
            res1 = GoogleTranslator(source='auto', target=detectlanguage.simple_detect(msg)).translate(res[2])
        except:
            res1 = ""
        print(res[1])
        res = GoogleTranslator(source='auto', target=detectlanguage.simple_detect(msg)).translate(res[0])
        print(res)
        last_answer = res1
        dis = Disscussion(client_question=msg, chatbot_answer=res, is_correct_answer=-1)
        dis.save()

        return res
    else:
        try:
            dis = Disscussion(client_question=msg, chatbot_answer=res[0], is_correct_answer=-1)
            dis.save()

        except:
            print("hi")

    try:
        last_answer = res[2]
    except:
        print("dont have explication")

    return res[0]


def chatbot_response_lang(msg, lang):
    translated = GoogleTranslator(source='auto', target='en').translate(msg)
    res = chat.get_response(translated)
    res = GoogleTranslator(source='auto', target=lang).translate(res)

    return res


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText, last_answer)


@app.route("/count_answer")
def get_stat2():
    data = pd.DataFrame.from_records(chat.collection_name.find())

    n = data["Question"].count()
    n2 = data["Answer"].count()

    return str(n) + "." + str(n2)


lang = "en"


@app.route("/correct_answer")
def get_corrects():
    n = 0
    n2 = 0
    data2 = pd.DataFrame.from_records(chat.collection_name2.find())
    for i in data2["is_correct_answer"]:
        if i == -1:
            n = n + 1
        else:
            n2 = n2 + 1

    return str(n) + "." + str(n2)


@app.route('/add_update_chat')
def add_chat2():
    question = request.args.get('question')
    answer = request.args.get('answer')
    explication = request.args.get('explication')
    diss_obj = Disscussion.objects(client_question=question,is_correct_answer=0).first()
    diss_obj.update(is_correct_answer=-1)

    new_chat = Chat(Question=question, Answer=answer, Explication=explication, Class="accounts")
    new_chat.save()


    return make_response("", 204)


@app.route("/q_for_answering")
def get_w():
    n = 0
    n2 = 0

    data2 = pd.DataFrame.from_records(chat.collection_name2.find())
    for i in chat.data2.index:
        print(data2["is_correct_answer"][i])
        if data2["is_correct_answer"][i] == 0:
            return data2["client_question"][i]

    return "No Question"


@app.route("/question_for_answering")
def get_wrong():
    n = 0
    n2 = 0

    data2 = pd.DataFrame.from_records(chat.collection_name2.find())
    for i in chat.data2.index:
        print(data2["is_correct_answer"][i])
        if data2["is_correct_answer"][i] == 0:
            return data2["client_question"][i]

    return "No Question"


@app.route("/get_arab")
def get_arab():
    try:
        import os

        os.remove("1.mp3")
    except:
        print("no file ")

    userText = ""
    print(s_r.__version__)  # just to print the version not required
    r = s_r.Recognizer()
    my_mic = s_r.Microphone(device_index=1)  # my device index is 1, you have to put your device index
    with my_mic as source:
        print("Say now!!!!")
        r.adjust_for_ambient_noise(source)  # reduce noise
        audio = r.listen(source)  # take voice input from the microphone
    userText = r.recognize_google(audio, language="ar")  # to print voice into text
    userText2 = chatbot_response(userText, last_answer)

    myobj = gTTS(text=userText2, lang="ar", slow=False)
    myobj.save("1.mp3")
    p.start()


@app.route("/get_english")
def get_english():
    try:
        import os
        os.remove("1.mp3")
    except:
        print("no file ")
    userText = ""
    print(s_r.__version__)  # just to print the version not required
    r = s_r.Recognizer()
    my_mic = s_r.Microphone(device_index=1)  # my device index is 1, you have to put your device index
    with my_mic as source:
        print("Say now!!!!")
        r.adjust_for_ambient_noise(source)  # reduce noise
        audio = r.listen(source)  # take voice input from the microphone
    userText = r.recognize_google(audio, language="en")  # to print voice into text
    userText2 = chatbot_response(userText, last_answer)

    myobj = gTTS(text=userText2, lang="en", slow=False)

    myobj.save("1.mp3")
    p.start()

    return userText


@app.route("/get_fr")
def get_fr():
    global p
    try:
        p.terminate()
        p = multiprocessing.Process(target=playsound, args=("1.mp3",))

    except:
        print("no sound going")

    try:

        import os

        os.remove("1.mp3")

    except:
        print("no file ")

    r = s_r.Recognizer()
    my_mic = s_r.Microphone(device_index=1)  # my device index is 1, you have to put your device index
    with my_mic as source:
        print("Say now!!!!")
        r.adjust_for_ambient_noise(source)  # reduce noise
        audio = r.listen(source)  # take voice input from the microphone
    userText = r.recognize_google(audio, language="fr")  # to print voice into text
    userText2 = chatbot_response(userText, last_answer)
    myobj = gTTS(text=userText2, lang="fr", slow=False, tld='com')

    myobj.save("1.mp3")
    p.start()


@app.route('/login')
def login():
    email = request.args.get('username')
    password = request.args.get('pass')
    login_user = User.objects(email=email).first()

    if login_user:

        if bcrypt.hashpw(password.encode('utf-8'), login_user['password'].encode('utf-8')) == \
                login_user[
                    'password'].encode('utf-8'):
            return "1"

    return "0"


@app.route("/register1")
def register1():
    users = db.Document()

    # existing_user = users.find_one({'name': request.form['username']})

    passs = "aaa"
    hashpass = bcrypt.hashpw(passs.encode('utf-8'), bcrypt.gensalt())
    password_hash = hashpass.decode('utf8')
    chat = User(email="baines", password=password_hash, role="employee", img="ines.jpg")
    chat.save()

    return render_template('login.html')


@app.route('/add_chat')
def add_chat():
    question = request.args.get('question')
    answer = request.args.get('answer')
    explication = request.args.get('explication')
    new_chat = Chat(Question=question, Answer=answer, Explication=explication, Class="accounts")
    new_chat.save()


@app.route('/api/v1/compare_faces', methods=['POST'])
def compare_faces():
    target = request.files['target']
    faces = request.files.getlist("faces")
    typ = type(target)
    target_filename = secure_filename(target.filename)
    response = []
    for face in faces:
        start = time.time()
        distance, result = compare_image.main(target, face)
        end = time.time()
        json_contect = {
            'result': str(result),
            'distance': round(distance, 2),
            'time_taken': round(end - start, 3),
            'target': target_filename,
            'face': secure_filename(face.filename),
        }
        response.append(json_contect)
    python2json = json.dumps(response)
    return app.response_class(python2json, content_type='application/json')


@app.route('/cam')
def camera_opening2():
    # initialize the camera
    try:
        cam = VideoCapture(0)  # 0 -> index of camera

        s, img = cam.read()
        if s:  # frame captured without any errors
            namedWindow("cam-test")
            destroyWindow("cam-test")
            imwrite("filename.jpg", img)  # save image
        result = False

        target = "filename.jpg"
        faces = chat.data_user["img"]
        print(faces)

        response = []
        for face in faces:
            face = "images/" + face
            distance, result = compare_image.main(target, face)
            if result:
                return str(result)

        return str(result)
    except:
        return str(False)


@app.route('/api/v1/compare_faces2')
def compare_faces2():
    result = False

    target = "images/hamma1.jpg"
    faces = ["images/hamma2.jpg"]
    typ = type(faces)

    response = []
    for face in faces:
        start = time.time()
        distance, result = compare_image.main(target, face)
        end = time.time()
        json_contect = {
            'result': str(result),
            'distance': round(distance, 2),
            'time_taken': round(end - start, 3),
            'target': target,
            'face': face,
        }
        response.append(json_contect)
    python2json = json.dumps(response)
    return str(result)


if __name__ == '__main__':
    app.run(port=8000, host="0.0.0.0")
