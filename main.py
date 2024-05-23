from flask import Flask, url_for, render_template, request
from website import create_app
app=create_app()

@app.route('/')
def home():
    return render_template("base.html")


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=8000) 