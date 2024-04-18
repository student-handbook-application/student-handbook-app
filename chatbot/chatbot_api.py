import functools

from flask import (
    Blueprint, render_template, request, json, jsonify
)
from .chatbot import Chatbot
from .model import *
from chatbot.auguments import load_auguments

bp = Blueprint('chatbot', __name__)


#ktra model dc load len hay chua, neu chua se load len, neu ton tai thi se hoi lai thoi
#hoac xep cung model(model luon mo) chay trc trang web
llm = None
@bp.before_app_first_request
def load_llm():
    global llm
    google_api_key ,hf_embedding_model, model_id, url_database, api_key_database = load_auguments()
    model = Model(model_id,google_api_key,0.2)
    llm = model.load_model()

@bp.route("/get", methods=['POST'])  # Đảm bảo route chỉ chấp nhận phương thức POST
def get_bot_response():
    
    try:
        
        # Lấy dữ liệu JSON từ yêu cầu POST
        data = request.json
        
        # Kiểm tra xem dữ liệu có tồn tại và có chứa key 'messages' không
        if data  and 'messages' in data:
            messages = data['messages']
            # Trả về nội dung của tin nhắn đầu tiên
            first_message_content = messages[0]['content'] # vd ng dung nhap hello thi no se lay chu hello \
            processed_data = {"processed_message": Chatbot(first_message_content,llm)}
            
            # Trả về kết quả xử lý dưới dạng JSON
            return jsonify(processed_data)
        else:
            # Trả về thông báo lỗi nếu dữ liệu không hợp lệ
            return jsonify({"error": "Invalid request format"}), 400
    except Exception as e:
        # Trả về thông báo lỗi nếu có bất kỳ lỗi nào xảy ra trong quá trình xử lý
        return jsonify({"error": str(e)}), 500