import sqlite3
import os
from datetime import datetime
import shutil

class AttendanceDB:
    def __init__(self, db_file="attendance.db"):
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        """Initialize the database and create required tables if they don't exist."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Enable foreign key support
        cursor.execute("PRAGMA foreign_keys = ON")

        # Drop existing tables to recreate with proper foreign key constraints
        cursor.execute("DROP TABLE IF EXISTS attendance")
        cursor.execute("DROP TABLE IF EXISTS employees")

        # Create employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_encoding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create attendance table with check_in and check_out times
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                check_in TIMESTAMP,
                check_out TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
                ON DELETE CASCADE
            )
        ''')

        conn.commit()
        conn.close()

    def add_employee(self, employee_id, name, face_encoding=None):
        """Add a new employee to the database."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO employees (employee_id, name, face_encoding) VALUES (?, ?, ?)",
                (employee_id, name, face_encoding)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def delete_employee(self, employee_id):
        """Delete an employee and their attendance records."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Enable foreign key support
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # First delete attendance records
            cursor.execute("DELETE FROM attendance WHERE employee_id=?", (employee_id,))
            
            # Then delete employee
            cursor.execute("DELETE FROM employees WHERE employee_id=?", (employee_id,))
            
            # Delete face data directory
            face_data_path = os.path.join("data/known_faces", str(employee_id))
            if os.path.exists(face_data_path):
                shutil.rmtree(face_data_path)
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting employee: {str(e)}")
            return False
        finally:
            conn.close()

    def get_employee(self, employee_id):
        """Retrieve employee details by ID."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees WHERE employee_id=?", (employee_id,))
        employee = cursor.fetchone()
        conn.close()
        return employee

    def log_attendance(self, employee_id):
        """Log attendance for an employee with check-in and check-out times."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Get current date's attendance record
            today = datetime.now().date()
            cursor.execute("""
                SELECT id, check_in, check_out 
                FROM attendance 
                WHERE employee_id=? AND date(check_in)=date(?)
                ORDER BY check_in DESC LIMIT 1
            """, (employee_id, today))
            
            record = cursor.fetchone()
            current_time = datetime.now()
            
            if record is None:
                # No record today - create new check-in
                cursor.execute(
                    "INSERT INTO attendance (employee_id, check_in) VALUES (?, ?)",
                    (employee_id, current_time)
                )
                conn.commit()
                return {'status': 'check_in', 'time': current_time}
            
            elif record[2] is None:
                # Has check-in but no check-out - update with check-out
                cursor.execute(
                    "UPDATE attendance SET check_out=? WHERE id=?",
                    (current_time, record[0])
                )
                conn.commit()
                return {'status': 'check_out', 'time': current_time}
            
            else:
                # Create a new check-in record for the same day
                cursor.execute(
                    "INSERT INTO attendance (employee_id, check_in) VALUES (?, ?)",
                    (employee_id, current_time)
                )
                conn.commit()
                return {'status': 'check_in', 'time': current_time}
            
        finally:
            conn.close()

    def get_attendance_records(self, date=None):
        """Retrieve attendance records with employee details."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        query = """
            SELECT a.id, e.employee_id, e.name, 
                   datetime(a.check_in) as check_in, 
                   datetime(a.check_out) as check_out
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
        """
        
        if date:
            query += " WHERE date(a.check_in) = date(?)"
            cursor.execute(query, (date,))
        else:
            cursor.execute(query)
            
        records = cursor.fetchall()
        conn.close()
        return records

    def get_all_employees(self):
        """Retrieve all employees."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees ORDER BY created_at DESC")
        employees = cursor.fetchall()
        conn.close()
        return employees
