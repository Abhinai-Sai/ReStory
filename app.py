"""Flask application for AI-Based Art Restoration."""
import os
import logging
from flask import Flask, jsonify, request
from dotenv import load_dotenv

from config import Config
from routes.web import web_bp
from routes.api import api_bp
from models.model_manager import ModelManager

# Load environment variables
load_dotenv()


def create_app(config_class=Config):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configure logging
    log_file = app.config.get('LOG_FILE', os.path.join(Config.BASE_DIR, 'logs', 'app.log'))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Create required directories
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
    os.makedirs(app.config.get('OUTPUT_FOLDER', 'outputs'), exist_ok=True)
    os.makedirs(app.config.get('MODEL_DIR', 'weights'), exist_ok=True)

    # Initialize ModelManager (lazy + async background preloading)
    app.model_manager = ModelManager(config=app.config)
    app.model_manager.preload_async()
    logger.info(f"App initialized — device: {app.model_manager.get_device()}")

    # Register blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)

    # --- Error handlers ---
    @app.errorhandler(400)
    def bad_request(error):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Bad Request', 'code': 400}), 400
        return 'Bad Request', 400

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Not Found', 'code': 404}), 404
        return 'Not Found', 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'File too large (max 20MB)', 'code': 413}), 413
        return 'File too large (max 20MB)', 413

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f'Server Error: {error}')
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Internal Server Error', 'code': 500}), 500
        return 'Internal Server Error', 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

