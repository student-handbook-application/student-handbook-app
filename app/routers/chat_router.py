import functools

from flask import (
    Blueprint, render_template, request, json, jsonify
)
from app.services.test import chat_response

chat_router = Blueprint('chat_router', __name__)

@chat_router.route("/get", methods=['POST'])  # Đảm bảo route chỉ chấp nhận phương thức POST
def get_bot_response():
    try:
        # Lấy dữ liệu JSON từ yêu cầu POST
        data = request.json

        print(data)
        # Kiểm tra xem dữ liệu có tồn tại và có chứa key 'messages' không
        if data  and 'messages' in data:
            print("da chay")
            messages = data['messages']
            print(messages)
            # Trả về nội dung của tin nhắn đầu tiên
            print(messages[0]['content'])
            first_message_content = messages[0]['content'] # vd ng dung nhap hello thi no se lay chu hello \
            print(first_message_content)
            processed_data = {"processed_message": chat_response(first_message_content)}
            
            # Trả về kết quả xử lý dưới dạng JSON
            return jsonify(processed_data)
        else:
            # Trả về thông báo lỗi nếu dữ liệu không hợp lệ
            return jsonify({"error": "Invalid request format"}), 400
    except Exception as e:
        # Trả về thông báo lỗi nếu có bất kỳ lỗi nào xảy ra trong quá trình xử lý
        return jsonify({"error": str(e)}), 500