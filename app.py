from flask import Flask, redirect, render_template, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# ----------------- MODELS ----------------- #
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=True)
    password = db.Column(db.String(255), nullable=False)

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=True)
    ipc_matches = db.Column(db.Text, nullable=True)  # Store matched IPC sections as JSON
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('complaints', lazy=True))

# Create database tables
with app.app_context():
    db.create_all()

# ----------------- LOAD MODELS & DATA ----------------- #
# Load SBERT Model and IPC Data
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
data = pd.read_csv('ipc_sections.csv')
data['Description'] = data['Description'].apply(lambda x: re.sub(r'\W+', ' ', x.lower()))
data['SBERT_Embedding'] = data['Description'].apply(lambda desc: sbert_model.encode([desc])[0])

# Load Summarization Model
tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-led-base-16384")
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-led-base-16384")

# ----------------- FUNCTIONS ----------------- #
def match_ipc_sections(text, top_n=5):
    if not text.strip():
        return []
    
    text_cleaned = re.sub(r'\W+', ' ', text.lower())
    text_vec = sbert_model.encode([text_cleaned])[0]
    distances = cosine_similarity([text_vec], data['SBERT_Embedding'].tolist())[0]
    closest_indices = np.argsort(distances)[-top_n:][::-1]
    matching_sections = data.iloc[closest_indices][['Section', 'Description']]
    
    return matching_sections.to_dict(orient='records')

def summarize_text(input_text):
    if not input_text.strip():
        return "Error: No input text provided."
    
    try:
        inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=6144)
        summary_ids = model.generate(inputs.input_ids, num_beams=4, no_repeat_ngram_size=3, length_penalty=3, min_length=100, max_length=500)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------- ROUTES ----------------- #

# User Registration
@app.route('/register_user', methods=['POST'])
def register_user():
    data = request.get_json()
    full_name = data.get('full_name')
    username = data.get('username')
    email = data.get('email')
    phone = data.get('phone')
    password = data.get('password')

    if not all([full_name, username, email, password]):
        return jsonify({'error': 'Please provide all required fields'}), 400
    
    # Check if user already exists
    user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
    if user_exists:
        return jsonify({'error': 'Username or email already exists'}), 400

    # Hash password before storing
    hashed_password = generate_password_hash(password)

    # Save user
    new_user = User(full_name=full_name, username=username, email=email, phone=phone, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'User registered successfully'}), 201

# User Login
@app.route('/login_user', methods=['POST'])
def login_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Please provide both username and password'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid username or password'}), 401

    return jsonify({'message': 'Login successful'}), 200

# Reset Password
@app.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json()
    username = data.get('username')
    new_password = data.get('new_password')
    
    user = User.query.filter_by(username=username).first()
    if user:
        hashed_password = generate_password_hash(new_password)
        user.password = hashed_password
        db.session.commit()
        return jsonify({'message': 'Password reset successfully'}), 200
    else:
        return jsonify({'error': 'User not found'}), 404

# Verify User
@app.route('/verify_user', methods=['POST'])
def verify_user():
    data = request.get_json()
    username = data.get('username')

    user = User.query.filter_by(username=username).first()
    if user:
        return jsonify({'message': 'User verified'}), 200
    else:
        return jsonify({'error': 'User not found'}), 404

# Match IPC Sections
@app.route('/match_ipc', methods=['POST'])
def match_ipc():
    data = request.json
    input_text = data.get('text', '')
    if not input_text:
        return jsonify({"message": "No text provided"}), 400
    matches = match_ipc_sections(input_text)
    return jsonify({"matches": matches}), 200

# Summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    input_text = data.get('text', '')
    if not input_text:
        return jsonify({"message": "No text provided"}), 400
    summary = summarize_text(input_text)
    return jsonify({"summary": summary}), 200

# Register Complaint
@app.route('/register_complaint', methods=['POST'])
def register_complaint():
    data = request.get_json()
    user_id = data.get('user_id')
    description = data.get('description')

    if not user_id or not description.strip():
        return jsonify({'error': 'User ID and complaint description are required'}), 400

    # Step 1: Summarize the complaint
    summary = summarize_text(description)

    # Step 2: Match IPC sections
    ipc_matches = match_ipc_sections(summary)

    # Extract only IPC section numbers
    ipc_section_numbers = [match["Section"] for match in ipc_matches]

    # Step 3: Save complaint in the database
    complaint = Complaint(user_id=user_id, description=description, summary=summary, ipc_matches=json.dumps(ipc_section_numbers))

    db.session.add(complaint)
    db.session.commit()

    return jsonify({'message': 'Complaint registered successfully', 'summary': summary, 'ipc_matches': ipc_section_numbers}), 201
@app.route('/complaints', methods=['GET'])
def get_complaints():
    complaints = Complaint.query.all()
    complaint_list = []

    for complaint in complaints:
        ipc_matches = json.loads(complaint.ipc_matches) if complaint.ipc_matches else []

        complaint_list.append({
            'id': complaint.id,
            'user_id': complaint.user_id,
            'description': complaint.description,
            'summary': complaint.summary,
            'ipc_matches': ipc_matches,  # Only IPC section numbers
            'created_at': complaint.created_at
        })

    return jsonify({'complaints': complaint_list}), 200

@app.route("/")
def default_route():
    return redirect(url_for("home")) 
    
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/complaint")
def complaint():
    return render_template("complaint.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/feedback")
def feedback():   
    return render_template("feedback.html")

@app.route("/contact")
def contact():   
    return render_template("contact.html") 

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
