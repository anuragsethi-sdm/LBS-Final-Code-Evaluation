    from flask import Flask, jsonify, request

    app = Flask(__name__)

    # Sample data (e.g., a list of users)
    users = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]

    # Define an API endpoint to get all users
    @app.route('/users', methods=['GET'])
    def get_users():
        return jsonify(users)

    # Define an API endpoint to get a specific user by ID
    @app.route('/users/<int:user_id>', methods=['GET'])
    def get_user(user_id):
        user = next((u for u in users if u['id'] == user_id), None)
        if user:
            return jsonify(user)
        return jsonify({"message": "User not found"}), 404

    # Run the Flask app
    if __name__ == '__main__':
        app.run(debug=True)