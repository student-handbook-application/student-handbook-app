from flask import (
    Blueprint, render_template, request, json, jsonify
)
from app.configs.llm_models import llm 
from app.modules.chatbot.models.chatbot import Chatbot
view = Blueprint('chatbot', __name__)


@view.route("/get", methods=['POST'])  # Đảm bảo route chỉ chấp nhận phương thức POST
def get_bot_response():
    
    try:
        
        # Lấy dữ liệu JSON từ yêu cầu POST
        data = request.json
        
        # Kiểm tra xem dữ liệu có tồn tại và có chứa key 'messages' không
        if data  and 'messages' in data:
            messages = data['messages']
            print(messages) 
            # Trả về nội dung của tin nhắn đầu tiên
            first_message_content = messages[0]['content'] # vd ng dung nhap hello thi no se lay chu hello \
            print(first_message_content)

            processed_data = {"processed_message": Chatbot(first_message_content,llm)}
            print(processed_data)
            
            # Trả về kết quả xử lý dưới dạng JSON
            return jsonify(processed_data)
        else:
            # Trả về thông báo lỗi nếu dữ liệu không hợp lệ
            return jsonify({"error": "Invalid request format"}), 400
    except Exception as e:
        # Trả về thông báo lỗi nếu có bất kỳ lỗi nào xảy ra trong quá trình xử lý
        return jsonify({"error": str(e)}), 500