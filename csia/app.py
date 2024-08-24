from flask import Flask, render_template, url_for, request, redirect, flash, session, jsonify, request, make_response, Response, send_file
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, SubmitField, SelectField, BooleanField, TextAreaField, IntegerField, FloatField
from wtforms.validators import InputRequired, Length, ValidationError, NumberRange, DataRequired, Optional
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import certifi
import requests
from datetime import date
import datetime
import html2text
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_JUSTIFY
import io
from sqlalchemy.sql.expression import func
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import numpy as np
from scipy import stats
from webforms import RegistrationForm, LoginForm, InformationalPageForm, QuestionForm, GradeSubmissionForm, RevisionGoalForm, UpdateStudentForm

# Appropiate Library Imports Above


matplotlib.use('Agg')  # Set the backend to 'Agg'

os.environ['SSL_CERT_FILE'] = certifi.where()

nltk.download('vader_lexicon') # Download the VADER model to perform sentiment analysis.

# App config
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
app.config['SECRET_KEY'] = "secret_key"
db = SQLAlchemy(app)

# Handling for file uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quizzes')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Important Functions

def check_remember_token(token):
    user = User.query.filter_by(access_token=token).first()
    return user if user else None

def auto_login_user():
    remember_token = request.cookies.get('remember_token')
    return check_remember_token(remember_token) if remember_token else None

def is_admin():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    return user and user.account_type == 'Admin'

def jaro_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0

    max_distance = max(len(s1), len(s2)) // 2 - 1
    matches = 0  
    hash_s1 = [0] * len(s1)  
    hash_s2 = [0] * len(s2)  

    for i in range(len(s1)):  
        for j in range(max(0, i - max_distance), min(len(s2), i + max_distance + 1)):  
            if s1[i] == s2[j] and not hash_s2[j]:  
                hash_s1[i] = 1  
                hash_s2[j] = 1  
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    point = 0
    for i in range(len(s1)):
        if hash_s1[i]:
            while not hash_s2[point]:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1
    t /= 2

    return (matches / len(s1) + matches / len(s2) + (matches - t) / matches) / 3.0


def get_inspirational_quote():
    response = requests.get("https://zenquotes.io/api/random")
    if response.status_code == 200:
        quote_data = response.json()
        return quote_data[0]["q"] + " - " + quote_data[0]["a"]
    else:
        return "No quote available at the moment."
    


def create_pdf(questions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()

    # Custom styles for IB-like format
    styles.add(ParagraphStyle(name='QuestionTitle', fontSize=14, leading=16, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name='QuestionBody', fontSize=12, leading=14, spaceAfter=12, alignment=TA_JUSTIFY))

    content = []
    html_converter = html2text.HTML2Text()
    html_converter.ignore_links = True  # Optionally ignore converting links

    for i, question in enumerate(questions, start=1):
        # Add question title
        question_title = f'Question {i} ({question.marks} marks):'
        content.append(Paragraph(question_title, styles['QuestionTitle']))

        # Convert HTML to text and add question body
        question_body = html_converter.handle(question.content)
        content.append(Paragraph(question_body, styles['QuestionBody']))

        # Add a spacer or page break as needed
        content.append(Spacer(1, 20))

    doc.build(content)
    buffer.seek(0)
    return buffer

# Database Tables: Automatically created whenever program starts

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(256), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(256), unique=True, nullable=False)
    account_type = db.Column(db.String(50), nullable=False)
    access_token = db.Column(db.String(256))
    revision_goals = db.relationship('RevisionGoals', backref='student', lazy=True, cascade='delete')

    def __repr__(self):
        return '<User %r>' % self.username

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method="sha256")

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class InformationalPages(db.Model):
    page_id = db.Column(db.Integer, primary_key=True)
    page_title = db.Column(db.String(256), nullable=False)
    page_content = db.Column(db.Text, nullable=False)
    topic = db.Column(db.String(2), nullable=False)


    def __repr__(self):
        return f"<InformationalPage {self.page_title}>"

class Question(db.Model):
    question_id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(256))
    marks = db.Column(db.Integer, nullable=False)
    content = db.Column(db.Text, nullable=False)
    difficulty_rating = db.Column(db.Float, nullable=False)
    paper = db.Column(db.String(2))
    topic = db.Column(db.String(2))
    teachers_notes = db.Column(db.Text)
    mark_scheme = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Question {self.slug}>'


class QuizPerformance(db.Model):
    quiz_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    topics = db.Column(db.String(256), nullable=True)
    difficulty = db.Column(db.Float, nullable=True)
    percent_score = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<QuizPerformance {self.quiz_id}>'



class RevisionGoals(db.Model):
    rev_id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    revision_goal = db.Column(db.String(30), nullable=False)
    revision_status = db.Column(db.Boolean, nullable=False, default=False)

    def __repr__(self):
        return f'<RevisionGoals {self.revision_goal}>'


class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # Additional fields like date_uploaded, score, etc.

    def __repr__(self):
        return f'<Quiz {self.filename}>'


# Routes: Very Important!


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            # Session-based login
            session['user_id'] = user.id  

            resp = make_response(redirect(url_for('dashboard')))  # Redirect to dashboard

            # "Remember Me" functionality
            if form.remember.data:
                remember_token = os.urandom(16).hex()  # Generate a secure token
                user.access_token = remember_token
                db.session.commit()
                resp.set_cookie('remember_token', remember_token, max_age=30 * 24 * 3600)  # Expires in 30 days

            flash('Logged in successfully!', 'success')
            return resp
        else:
            flash('Invalid username or password!', 'danger')
    return render_template("login.html", form=form)
    
@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Clear the session

    resp = make_response(redirect(url_for('login')))
    remember_token = request.cookies.get('remember_token')
    if remember_token:
        user = check_remember_token(remember_token)
        if user:
            user.access_token = None
            db.session.commit()
        resp.delete_cookie('remember_token')

    flash('Logout successful', 'info')
    return resp

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Checks for cookie 
    remember_token = request.cookies.get('remember_token')
    if remember_token:
        user = check_remember_token(remember_token)
        if user:
            return redirect(url_for('dashboard'))  # Redirect to dashboard

    # The form for registration.
    form = RegistrationForm()
    if form.validate_on_submit():
        new_user = User(username=form.username.data, email=form.email.data, account_type=form.account_type.data)
        new_user.set_password(form.password.data)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template("register.html", form=form)


@app.route('/')
def home():
    remember_token = request.cookies.get('remember_token')
    if remember_token:
        user = check_remember_token(remember_token)
        if user:
            return redirect(url_for('dashboard'))  # Redirect to dashboard
    user = auto_login_user()
    return render_template("home.html", user=user)

@app.route('/questions', defaults={'page': 1})
@app.route('/questions/page/<int:page>')
def questions(page):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    difficulty = request.args.get('difficulty')
    topic = request.args.get('topic')
    paper = request.args.get('paper')

    search_query = request.args.get('search')
    per_page = 7
    is_search = False

    # Start with a base query
    query = Question.query

    # Apply filters based on the presence of query parameters
    if difficulty:
        query = query.filter(Question.difficulty_rating == difficulty)
    if topic:
        query = query.filter(Question.topic == topic)
    if paper:
        query = query.filter(Question.paper == paper)

    if search_query:

        scored_questions = [(q, jaro_similarity(search_query.lower(), q.slug.lower())) for q in query.all()]
        filtered_scored_questions = [(q, score) for q, score in scored_questions if score > 0.5]
        filtered_scored_questions.sort(key=lambda x: x[1], reverse=True)
        total = len(filtered_scored_questions)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_questions = filtered_scored_questions[start:end]
        total_pages = (total + per_page - 1) // per_page
        is_search = True
        return render_template('questions.html', questions=paginated_questions, user=user, page=page, per_page=per_page, total=total, total_pages=total_pages, is_search=is_search)
    else:
        # Apply regular pagination
        paginated_results = query.paginate(page=page, per_page=per_page, error_out=False)
        paginated_questions = [(q, None) for q in paginated_results.items]
        return render_template('questions.html', questions=paginated_questions, paginated_results=paginated_results, user=user, page=page, per_page=per_page, is_search=False)



    

@app.route('/question_details/<int:question_id>')
def question_details(question_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    question = Question.query.get_or_404(question_id)
    return render_template('question_details.html', question=question, user=user)


@app.route('/add_question', methods=['GET', 'POST'])
def add_question():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    form = QuestionForm()
    if form.validate_on_submit():
        question = Question(
            slug=form.slug.data,
            marks=form.marks.data,
            content=form.content.data,
            difficulty_rating=form.difficulty_rating.data,
            paper=form.paper.data,
            topic=form.topic.data,
            teachers_notes=form.teachers_notes.data,
            mark_scheme=form.mark_scheme.data  # Include mark_scheme data
        )
        db.session.add(question)
        db.session.commit()
        flash('Question added successfully!', 'success')
        return redirect(url_for('questions'))
    return render_template('add_question.html', form=form, user=user)

@app.route('/edit_question/<int:question_id>', methods=['GET', 'POST'])
def edit_question(question_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    
    question = Question.query.get_or_404(question_id)
    if not (user and user.account_type == 'Admin'):
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('home'))

    form = QuestionForm(obj=question)
    if form.validate_on_submit():
        question.slug = form.slug.data
        question.marks = form.marks.data
        question.content = form.content.data
        question.difficulty_rating = form.difficulty_rating.data
        question.paper = form.paper.data
        question.topic = form.topic.data
        question.teachers_notes = form.teachers_notes.data
        question.mark_scheme = form.mark_scheme.data  # Update mark_scheme
        db.session.commit()
        flash('Question updated successfully!', 'success')
        return redirect(url_for('questions'))

    return render_template('edit_question.html', form=form, user=user)


@app.route('/api/calculate_difficulty', methods=['POST'])
def calculate_difficulty():
    data = request.json
    notes = data.get('notes', '')

    # Sentiment analysis from VADER nltk
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(notes)['compound']
    difficulty_rating = (1 - sentiment_score) / 2  # Normalize to 0-1

    return jsonify({'difficulty_rating': difficulty_rating})


@app.route('/delete_question/<int:question_id>', methods=['POST'])
def delete_question(question_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    question = Question.query.get_or_404(question_id)
    if not (user and user.account_type == 'Admin'):
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('home'))

    db.session.delete(question)
    db.session.commit()
    flash('Question deleted successfully!', 'success')
    return redirect(url_for('questions'))





@app.route('/new_page', methods=['GET', 'POST'])
def new_page():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        return redirect(url_for('login'))
    
    if not is_admin():  
        return redirect(url_for('home'))
    
    form = InformationalPageForm()
    if form.validate_on_submit():
        page = InformationalPages(
            page_title=form.page_title.data,
            page_content=form.page_content.data,
            topic=form.topic.data
        )
        db.session.add(page)
        db.session.commit()
        flash('Informational page created!', 'success')
    return render_template('create_page.html', form=form, user=user)
    

@app.route('/information-pages')
def information_pages():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    pages = InformationalPages.query.all()  # Get all informational pages
    return render_template('information_pages.html', pages=pages, user=user)

@app.route('/information-page/<int:page_id>')
def information_page(page_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    page = InformationalPages.query.get_or_404(page_id)
    return render_template('information_page.html', page=page,user=user)

@app.route('/edit_page/<int:page_id>', methods=['GET', 'POST'])
def edit_page(page_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    page = InformationalPages.query.get_or_404(page_id)
    if not is_admin():  
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('home'))

    form = InformationalPageForm(obj=page)
    if form.validate_on_submit():
        page.page_title = form.page_title.data
        page.page_content = form.page_content.data
        page.topic = form.topic.data
        db.session.commit()
        flash('The page has been updated!', 'success')
        return redirect(url_for('information_page', page_id=page.page_id))

    return render_template('edit_page.html', form=form, user=user)

@app.route('/delete_page/<int:page_id>', methods=['POST'])
def delete_page(page_id):
    page = InformationalPages.query.get_or_404(page_id)
    if not is_admin():  
        return redirect(url_for('home'))

    db.session.delete(page)
    db.session.commit()
    flash('The page has been deleted!', 'success')
    return redirect(url_for('information_pages'))

@app.route('/generate_quiz', methods=['GET', 'POST'])
def generate_quiz():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    total_time_seconds = None  # Initialize total time

    if request.method == 'POST':
        if 'file' in request.files:  # Check if it's a file upload
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                if 'user_id' in session:
                    user_id = session['user_id']
                    new_quiz = Quiz(filename=filename, student_id=user_id)
                    db.session.add(new_quiz)
                    db.session.commit()
                flash('Quiz submitted successfully!')
                return redirect(url_for('dashboard'))
                
        else:  # Handle quiz generation
            topic = request.form.get('topic')
            num_questions = int(request.form.get('num_questions', 10))
            questions = Question.query.filter_by(topic=topic).limit(num_questions).all()

            # Calculate total time
            total_time_seconds = sum([int(question.difficulty_rating * 81 * question.marks) for question in questions])

            pdf_buffer = create_pdf(questions)
            pdf_buffer.seek(0)
            response = make_response(send_file(pdf_buffer, as_attachment=True, download_name='quiz.pdf', mimetype='application/pdf'))
            response.set_cookie('total_time_seconds', str(total_time_seconds))  # Set total time in cookie
            return response

    # GET request - show the topic selection form
    return render_template('generate_quiz.html', user=user, total_time_seconds=total_time_seconds)

@app.route('/view_quizzes/<int:student_id>', methods=['GET', 'POST'])
def view_quizzes(student_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    if not is_admin():
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    student = User.query.get_or_404(student_id)
    quizzes = Quiz.query.filter_by(student_id=student_id).all()

    # Create a form for each quiz
    grade_forms = {quiz.id: GradeSubmissionForm(prefix=str(quiz.id)) for quiz in quizzes}

    # Check if any form has been submitted
    for quiz in quizzes:
        form = grade_forms[quiz.id]
        if form.validate_on_submit() and str(quiz.id) in request.form:
            # Logic to save the grade
            grade = form.grade.data

        # Update or create quiz performance record
            quiz_performance = QuizPerformance.query.filter_by(quiz_id=quiz.id, student_id=student_id).first()
            if not quiz_performance:
                quiz_performance = QuizPerformance(quiz_id=quiz.id, student_id=student_id)
            quiz_performance.percent_score = grade
            db.session.add(quiz_performance)
            db.session.commit()
            flash(f'Grade for Quiz {quiz.id} submitted successfully!', 'success')
            return redirect(url_for('view_quizzes', student_id=student_id))

    return render_template('view_quizzes.html', student=student, quizzes=quizzes, grade_forms=grade_forms, user=user)

    

@app.route('/download_quiz/<int:quiz_id>')
def download_quiz(quiz_id):
    # Fetch the quiz from the database
    quiz = Quiz.query.get_or_404(quiz_id)

    # Define the path to the quiz file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], quiz.filename)

    # Send the file for download
    return send_file(file_path, as_attachment=True)

@app.route('/submit_grade', methods=['GET', 'POST'])
def submit_grade():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    if not is_admin():
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    form = GradeSubmissionForm()
    if form.validate_on_submit():
        # Logic to save the grade
        quiz_id = form.quiz_id.data
        student_id = form.student_id.data
        grade = form.grade.data

        # Fetch the quiz performance record or create a new one
        quiz_performance = QuizPerformance.query.filter_by(quiz_id=quiz_id, student_id=student_id).first()
        if not quiz_performance:
            quiz_performance = QuizPerformance(quiz_id=quiz_id, student_id=student_id)

        quiz_performance.percent_score = grade
        db.session.add(quiz_performance)
        db.session.commit()

        flash('Grade submitted successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('submit_grade.html', form=form, user=user)

# Error Pages
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500

@app.route('/dashboard')
def dashboard():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        return redirect(url_for('login'))

    if is_admin():
        # Fetch all student accounts
        students = User.query.filter_by(account_type='Student').all()
    else:
        students = None

    inspirational_quote = get_inspirational_quote()
    today_goal = None
    if user and user.account_type == 'Student':
        today_goal = RevisionGoals.query.filter_by(
            student_id=user.id, 
            date=datetime.date.today(),
            revision_status=False  # Only fetch goals that are not completed
        ).order_by(RevisionGoals.date.desc()).first()

    return render_template("dashboard.html", user=user, students=students, quote=inspirational_quote, today_goal=today_goal)


@app.route('/performance_graph/<int:student_id>')
def performance_graph(student_id):
    # Fetch recent quiz performances for the student
    performances = QuizPerformance.query.filter_by(student_id=student_id).order_by(QuizPerformance.quiz_id.desc()).limit(10).all()

    # Prepare data for the graph
    scores = [performance.percent_score for performance in performances]
    quizzes = [performance.quiz_id for performance in performances]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(quizzes, scores)

    # Create a line of best fit
    def predict(x):
        return slope * x + intercept

    fit_line = predict(np.array(quizzes))

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.scatter(quizzes, scores, marker='o', label='Actual Scores')  
    plt.plot(quizzes, fit_line, color='red', label='Predicted Trend')
    plt.title('Recent Quiz Performance with Predictive Trend')
    plt.xlabel('Quiz ID')
    plt.ylabel('Percent Score')
    plt.grid(True)
    plt.xticks(quizzes)
    plt.yticks(range(0, 101, 10))
    plt.legend()

    # Save plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Return the buffer content as a response
    return send_file(buf, mimetype='image/png', as_attachment=False)

@app.route('/student_performance/<int:student_id>')
def student_performance(student_id):
    # Ensure the user is an admin
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user or not is_admin():
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    # Fetch quiz performance data for the specified student
    performances = QuizPerformance.query.filter_by(student_id=student_id).all()

    # Extract data for graph
    quizzes = [performance.quiz_id for performance in performances]
    scores = [performance.percent_score for performance in performances]

    slope, intercept, r_value, p_value, std_err = stats.linregress(quizzes, scores)

    # Create a line of best fit
    def predict(x):
        return slope * x + intercept

    fit_line = predict(np.array(quizzes))

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.scatter(quizzes, scores, marker='o', label='Actual Scores')  # Use scatter instead of plot
    plt.plot(quizzes, fit_line, color='red', label='Predicted Trend')
    plt.title('Recent Quiz Performance with Predictive Trend')
    plt.xlabel('Quiz ID')
    plt.ylabel('Percent Score')
    plt.grid(True)
    plt.xticks(quizzes)
    plt.yticks(range(0, 101, 10))
    plt.legend()

    # Save plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Return the buffer content as a response
    return send_file(buf, mimetype='image/png', as_attachment=False)
    

@app.route('/add_revision_goal', methods=['GET', 'POST'])
def add_revision_goal():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    else:
        return redirect(url_for('login'))

    if user and user.account_type == 'Student':
        form = RevisionGoalForm()
        if form.validate_on_submit():
            new_goal = RevisionGoals(
                student_id=user.id,
                date=datetime.date.today(),
                revision_goal=form.revision_goal.data,
                revision_status=False
            )
            db.session.add(new_goal)
            db.session.commit()
            flash('Revision goal added successfully.')
            print('works!')
            return redirect(url_for('dashboard', user=user))
        return render_template('add_revision_goal.html', form=form, user=user)
    else:
        flash('Unauthorized access.', 'danger')
        print('doesnT WORK')
        return redirect(url_for('dashboard', user=user))

@app.route('/mark_goal_complete/<int:goal_id>', methods=['POST'])
def mark_goal_complete(goal_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])

    if not user or user.account_type != 'Student':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))

    goal = RevisionGoals.query.get_or_404(goal_id)
    if goal.student_id != user.id:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('dashboard'))

    goal.revision_status = True
    db.session.commit()
    flash('Goal marked as complete.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/update_student/<int:student_id>', methods=['GET', 'POST'])
def update_student(student_id):
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        return redirect(url_for('login'))
    
    if not is_admin():  
        return redirect(url_for('home'))
   

    student = User.query.get_or_404(student_id)
    form = UpdateStudentForm(obj=student)

    if form.validate_on_submit():
        student.username = form.username.data
        student.email = form.email.data
        db.session.commit()
        flash('Student account updated successfully!', 'success')
        return redirect(url_for('dashboard', user=user))

    return render_template('update_student.html', form=form, student=student, user=user)
    return render_template('update_student.html', form=form, student=student, user=user)

@app.route('/delete_student/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    # Ensure the user is an admin
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    elif 'remember_token' in request.cookies:
        user = check_remember_token(request.cookies.get('remember_token'))

    if not user:
        return redirect(url_for('login'))
    if not is_admin():
        return redirect(url_for('dashboard'))
    
    

    student = User.query.get_or_404(student_id)
    db.session.delete(student)
    db.session.commit()
    flash('Student account deleted successfully!', 'success')
    return redirect(url_for('dashboard'))

# Creates database ONLY IF no database.db exists. Starts app.

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


