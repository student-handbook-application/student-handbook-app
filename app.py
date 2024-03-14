from flask import Flask, url_for, render_template, request
from chatbot import chatbot_api
import os
from dotenv import load_dotenv

load_dotenv()
SECRET_KEY = os.environ.get("KEY")
DB_NAME = os.environ.get("DB_NAME")


app = Flask(__name__)

"""config app"""
app.config['SECRET_KEY']  = SECRET_KEY
"""khởi tạo base.html"""
@app.route('/')
def home():
    return render_template("base.html")


"""nhận message từ base html và trả về kết quả đã qua xử lý"""
app.register_blueprint(chatbot_api.bp)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=8000) 