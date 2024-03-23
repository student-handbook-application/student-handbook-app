import functools

from flask import (
    Blueprint, render_template, request, json, jsonify
)
from .model import Chatbot


bp = Blueprint('chatbot', __name__)


#ktra model dc load len hay chua, neu chua se load len, neu ton tai thi se hoi lai thoi
#hoac xep cung model(model luon mo) chay trc trang web


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
            processed_data = {"processed_message": Chatbot(first_message_content)}
            
            # Trả về kết quả xử lý dưới dạng JSON
            return jsonify(processed_data)
        else:
            # Trả về thông báo lỗi nếu dữ liệu không hợp lệ
            return jsonify({"error": "Invalid request format"}), 400
    except Exception as e:
        # Trả về thông báo lỗi nếu có bất kỳ lỗi nào xảy ra trong quá trình xử lý
        return jsonify({"error": str(e)}), 500