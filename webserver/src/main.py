from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    print(f"Request from {request.remote_addr}")
    return jsonify({"command": "f", "id": "1"})
