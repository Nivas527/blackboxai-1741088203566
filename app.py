from flask import Flask, render_template, request, jsonify, Response, flash, redirect, url_for
import cv2
import numpy as np
from face_recognizer import FaceRecognizer
from attendance_db import AttendanceDB
import base64
import logging
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize face recognizer and database
face_recognizer = FaceRecognizer()
db = AttendanceDB()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add template filter for datetime conversion
@app.template_filter('to_datetime')
def to_datetime(value):
    if isinstance(value, str):
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    return value

def init_system():
    """Initialize the face recognition system."""
    if os.path.exists("data/known_faces"):
        face_recognizer.load_encodings()

# Initialize the system on startup
init_system()

@app.route('/')
def index():
    """Render the dashboard page."""
    return render_template('index.html', now=datetime.now())

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    """Handle employee enrollment."""
    if request.method == 'POST':
        try:
            employee_id = request.form.get('employee_id')
            name = request.form.get('name')
            image_data = request.form.get('image_data')

            if not all([employee_id, name, image_data]):
                return jsonify({'success': False, 'message': 'Missing required fields'})

            # Decode base64 image
            image_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Save face image
            if face_recognizer.save_training_image(img, employee_id):
                # Add employee to database
                if db.add_employee(employee_id, name):
                    return jsonify({'success': True, 'message': 'Employee enrolled successfully'})
                else:
                    return jsonify({'success': False, 'message': 'Employee ID already exists'})
            else:
                return jsonify({'success': False, 'message': 'No face detected in the image'})

        except Exception as e:
            logger.error(f"Error during enrollment: {str(e)}")
            return jsonify({'success': False, 'message': 'Error during enrollment'})

    return render_template('enroll.html', now=datetime.now())

@app.route('/mark-attendance', methods=['GET', 'POST'])
def mark_attendance():
    """Handle attendance marking."""
    if request.method == 'POST':
        try:
            image_data = request.form.get('image_data')
            
            if not image_data:
                return jsonify({'success': False, 'message': 'No image data received'})

            # Decode base64 image
            image_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Recognize face
            employee_id, confidence = face_recognizer.recognize(img)

            if employee_id:
                # Log attendance
                result = db.log_attendance(employee_id)
                if result['status'] != 'completed':
                    employee = db.get_employee(employee_id)
                    return jsonify({
                        'success': True,
                        'message': f'Attendance marked for {employee[1]}',
                        'employee_name': employee[1],
                        'attendance_type': result['status'],
                        'timestamp': result['time'].strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Attendance already completed for today'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Face not recognized or confidence too low'
                })

        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            return jsonify({'success': False, 'message': 'Error marking attendance'})

    return render_template('attendance.html', now=datetime.now())

@app.route('/view-attendance')
def view_attendance():
    """Display attendance records."""
    date_str = request.args.get('date')
    
    try:
        if date_str:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            records = db.get_attendance_records(date)
        else:
            records = db.get_attendance_records()
            
        return render_template('view_attendance.html', 
                             records=records, 
                             selected_date=date_str, 
                             now=datetime.now())
    
    except Exception as e:
        logger.error(f"Error retrieving attendance records: {str(e)}")
        flash('Error retrieving attendance records', 'error')
        return render_template('view_attendance.html', 
                             records=[], 
                             selected_date=date_str, 
                             now=datetime.now())

@app.route('/manage-users')
def manage_users():
    """Display registered users."""
    try:
        employees = db.get_all_employees()
        return render_template('manage_users.html', 
                             employees=employees,
                             now=datetime.now())
    except Exception as e:
        logger.error(f"Error retrieving employees: {str(e)}")
        flash('Error retrieving employees', 'error')
        return render_template('manage_users.html', 
                             employees=[],
                             now=datetime.now())

@app.route('/delete-employee/<employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    """Delete an employee and their records."""
    try:
        if db.delete_employee(employee_id):
            return jsonify({
                'success': True,
                'message': 'Employee deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Error deleting employee'
            })
    except Exception as e:
        logger.error(f"Error deleting employee: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error deleting employee'
        })

@app.route('/employees')
def get_employees():
    """Get list of all employees."""
    try:
        employees = db.get_all_employees()
        return jsonify({'success': True, 'employees': employees})
    except Exception as e:
        logger.error(f"Error retrieving employees: {str(e)}")
        return jsonify({'success': False, 'message': 'Error retrieving employees'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
