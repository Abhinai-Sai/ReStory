"""Web routes blueprint."""
from flask import Blueprint, render_template

web_bp = Blueprint('web', __name__)


@web_bp.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')
