import os
from flask import Flask
from app.modules import view
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv("KEY")
    app.register_blueprint(view)

    return app

