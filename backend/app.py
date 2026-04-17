from flask import Flask, jsonify

app = Flask(__name__)

customers = [
    {
        "id": 1,
        "name": "Rahul",
        "plan": "Premium",
        "charging_sessions": 40,
        "failed_sessions": 5,
        "usage_hours": 120
    }
]

@app.route('/customers')
def get_customers():
    return jsonify(customers)

if __name__ == "__main__":
    app.run(debug=True)