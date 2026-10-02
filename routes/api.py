"""API routes for art restoration."""
import os
import time
import uuid
import logging
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

from pipeline.image_utils import allowed_file, validate_image_file
from pipeline.restoration_pipeline import RestorationPipeline

api_bp = Blueprint('api', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)


@api_bp.route('/restore', methods=['POST'])
def restore_image():
    """Accept image upload and run the restoration pipeline."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided', 'code': 400}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected', 'code': 400}), 400

        # Validate file
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify({'success': False, 'error': error_msg, 'code': 400}), 400

        # Save uploaded file to disk
        filename = secure_filename(file.filename)
        input_id = str(uuid.uuid4())
        input_ext = os.path.splitext(filename)[1] or '.png'
        input_filename = f"{input_id}{input_ext}"
        input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)

        # Optional outscale option (default 4x, allow 2x for fast mode)
        outscale = 4
        try:
            val = int(request.form.get('outscale', 4))
            if val in (2, 4):
                outscale = val
        except (ValueError, TypeError):
            outscale = 4

        # Run restoration pipeline
        start_time = time.time()
        pipeline = RestorationPipeline(current_app.model_manager)
        result = pipeline.restore(
            input_path=input_path,
            output_dir=current_app.config['OUTPUT_FOLDER'],
            outscale=outscale
        )
        processing_time = round(time.time() - start_time, 2)

        # Clean up uploaded input file
        try:
            os.remove(input_path)
        except Exception as e:
            logger.warning(f"Failed to remove input file {input_path}: {e}")

        if not result.get('success', False):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Processing failed'),
                'code': 500
            }), 500

        return jsonify({
            'success': True,
            'output_id': result['output_id'],
            'message': 'Image restored successfully',
            'details': {
                'has_faces': result.get('details', {}).get('has_faces', False),
                'device': str(current_app.model_manager.get_device()),
                'processing_time': processing_time
            }
        })

    except Exception as e:
        logger.error(f"Error in /api/restore: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal Server Error', 'code': 500}), 500


@api_bp.route('/result/<output_id>', methods=['GET'])
def get_result(output_id):
    """Return the restored image for display."""
    try:
        if '..' in output_id or '/' in output_id or '\\' in output_id:
            return jsonify({'success': False, 'error': 'Invalid output ID', 'code': 400}), 400

        output_dir = current_app.config['OUTPUT_FOLDER']
        # Try common extensions
        mimes = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'webp': 'image/webp'}
        for ext in ('png', 'jpg', 'jpeg', 'webp'):
            path = os.path.join(output_dir, f"{output_id}.{ext}")
            if os.path.exists(path):
                return send_file(path, mimetype=mimes.get(ext, 'image/png'))

        return jsonify({'success': False, 'error': 'Result not found', 'code': 404}), 404
    except Exception as e:
        logger.error(f"Error serving result {output_id}: {e}")
        return jsonify({'success': False, 'error': 'Internal Server Error', 'code': 500}), 500


@api_bp.route('/download/<output_id>', methods=['GET'])
def download_result(output_id):
    """Return the restored image as a downloadable attachment."""
    try:
        if '..' in output_id or '/' in output_id or '\\' in output_id:
            return jsonify({'success': False, 'error': 'Invalid output ID', 'code': 400}), 400

        output_dir = current_app.config['OUTPUT_FOLDER']
        for ext in ('png', 'jpg', 'webp'):
            path = os.path.join(output_dir, f"{output_id}.{ext}")
            if os.path.exists(path):
                return send_file(
                    path,
                    as_attachment=True,
                    download_name=f"restored_{output_id}.{ext}"
                )

        return jsonify({'success': False, 'error': 'Result not found', 'code': 404}), 404
    except Exception as e:
        logger.error(f"Error serving download {output_id}: {e}")
        return jsonify({'success': False, 'error': 'Internal Server Error', 'code': 500}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Return application health status."""
    try:
        mm = current_app.model_manager
        return jsonify({
            'status': 'healthy',
            'device': str(mm.get_device()),
            'models_loaded': bool(mm.models),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'code': 500
        }), 500
