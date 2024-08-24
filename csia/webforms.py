from wtforms import StringField, PasswordField, SubmitField, SelectField, BooleanField, TextAreaField, IntegerField, FloatField
from wtforms.validators import InputRequired, Length, ValidationError, NumberRange, DataRequired, Optional
from flask_wtf import FlaskForm

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=8, max=256)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[InputRequired(), Length(min=8)])
    email = StringField('Email', validators=[InputRequired(), Length(max=256)])
    account_type = SelectField('Account Type', choices=[('Admin', 'Admin'), ('Student', 'Student')])
    submit = SubmitField('Register')
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

    def validate_confirm_password(form, field):
        if field.data != form.password.data:
            raise ValidationError("Passwords must match.")

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=256)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8)])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class InformationalPageForm(FlaskForm):
    page_title = StringField('Title', validators=[InputRequired(), Length(max=256)])
    page_content = TextAreaField('Content', validators=[InputRequired(), Length(max=500000)])
    topic = StringField('Topic', validators=[InputRequired(), Length(max=2)])
    submit = SubmitField('Submit')

class QuestionForm(FlaskForm):
    slug = StringField('Slug', validators=[InputRequired(), Length(max=256)])
    marks = IntegerField('Marks', validators=[InputRequired()])
    content = TextAreaField('Content', validators=[InputRequired(), Length(max=5000)])
    difficulty_rating = FloatField('Difficulty Rating', validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    paper = StringField('Paper', validators=[Length(max=2)]) 
    topic = StringField('Topic', validators=[Length(max=2)])
    teachers_notes = TextAreaField('Teacher\'s Notes', validators=[Length(max=5000)])
    submit = SubmitField('Submit Question')
    mark_scheme = TextAreaField('Mark Scheme', validators=[InputRequired(), Length(max=5000)])

class GradeSubmissionForm(FlaskForm):
    grade = FloatField('Grade (%)', validators=[DataRequired(), NumberRange(min=0, max=100)])
    submit = SubmitField('Submit Grade')

class RevisionGoalForm(FlaskForm):
    revision_goal = StringField('Goal', validators=[InputRequired(), Length(max=30)])
    submit = SubmitField('Add Goal')

class UpdateStudentForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=8, max=256)])
    email = StringField('Email', validators=[InputRequired(), Length(max=256)])
    submit = SubmitField('Update')