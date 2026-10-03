"""API routes for art restoration."""
import os
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

from pipeline.image_utils import allowed_file, validate_image_file
from pipeline.restoration_pipeline import RestorationPipeline

api_bp = Blueprint('api', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)

# Single concurrent restoration lock for 512 MB Render Free Tier protection
restoration_lock = threading.Lock()


@api_bp.route('/restore', methods=['POST'])
def restore_image():
    """Accept image upload and run the restoration pipeline."""
    if not restoration_lock.acquire(blocking=False):
        logger.warning("Concurrent restoration request rejected (429 Server Busy)")
        return jsonify({
            'success': False,
            'error': 'Server busy',
            'message': 'Another restoration is currently in progress. Please try again in a few seconds.',
            'code': 429
        }), 429

    input_path = None
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided', 'code': 400}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected', 'code': 400}), 400

        # Validate file
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify({'success': False, 'error': error_msg or 'Invalid image file', 'code': 400}), 400

        # Save uploaded file to disk safely
        filename = secure_filename(file.filename)
        input_id = str(uuid.uuid4())
        input_ext = os.path.splitext(filename)[1] or '.png'
        input_filename = f"{input_id}{input_ext}"
        input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_path)

        # Parse optional settings (default 2x scale, face restoration disabled for Render Free Tier baseline)
        outscale = 2
        try:
            val = int(request.form.get('outscale', 2))
            if val in (2, 4):
                outscale = val
        except (ValueError, TypeError):
            outscale = 2

        has_explicit_enable = 'enable_faces' in request.form or 'face_restoration' in request.form or 'enable_gfpgan' in request.form
        if has_explicit_enable:
            enable_faces = (
                request.form.get('enable_faces', 'false').lower() in ('true', '1', 'yes') or
                request.form.get('face_restoration', 'false').lower() in ('true', '1', 'yes') or
                request.form.get('enable_gfpgan', 'false').lower() in ('true', '1', 'yes')
            )
        else:
            enable_faces = (outscale == 4)

        # Run restoration pipeline
        start_time = time.time()
        pipeline = RestorationPipeline(current_app.model_manager)
        result = pipeline.restore(
            input_path=input_path,
            output_dir=current_app.config['OUTPUT_FOLDER'],
            outscale=outscale,
            enable_faces=enable_faces
        )
        processing_time = round(time.time() - start_time, 2)

        # Periodically clean up old outputs (>30 mins) to preserve container disk space on Render
        try:
            now = time.time()
            output_dir = current_app.config['OUTPUT_FOLDER']
            for fname in os.listdir(output_dir):
                if fname == '.gitkeep':
                    continue
                fpath = os.path.join(output_dir, fname)
                if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 1800:
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        except Exception as err:
            logger.warning(f"Output cleanup warning: {err}")

        if not result.get('success', False):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Restoration failed'),
                'message': 'The image could not be processed on the available server resources.',
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
        return jsonify({
            'success': False,
            'error': 'Restoration failed',
            'message': 'An unexpected error occurred while processing the image.',
            'code': 500
        }), 500

    finally:
        # Clean up temporary uploaded file
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception as e:
                logger.warning(f"Failed to remove input file {input_path}: {e}")

        # Always release concurrency lock
        restoration_lock.release()


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
    """Return application health status quickly without running AI inference."""
    try:
        mm = current_app.model_manager
        model_dir = current_app.config.get('MODEL_DIR', 'weights')
        weights_exist = os.path.exists(os.path.join(model_dir, 'RealESRGAN_x4plus.pth'))
        return jsonify({
            'status': 'healthy',
            'device': str(mm.get_device()),
            'weights_available': weights_exist,
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
