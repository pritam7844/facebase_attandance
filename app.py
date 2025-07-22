from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    print("GET / called")
    return jsonify({'message': 'Server is running ✅'}), 200

@app.route('/record', methods=['POST'])
def record_attendance():
    print("POST /record called")
    data = request.json
    print("Received data:", data)

    if not data or 'name' not in data:
        return jsonify({'error': 'Missing \"name\" in request body'}), 400

    name = data['name']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return jsonify({
        'message': f'Attendance received for {name}',
        'timestamp': timestamp,
        'status': '✅ success (not stored)'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
