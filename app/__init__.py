from flask import Flask, render_template
from app.routers.chat_router import chat_router 

app = Flask(__name__,template_folder='../templates')

@app.route('/')
def index():
    return render_template("index.html")


app.register_blueprint(chat_router) 