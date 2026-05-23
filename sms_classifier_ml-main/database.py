import sqlite3
import hashlib
from datetime import datetime
import pandas as pd

class DatabaseManager:
    def __init__(self, db_name="sms_classifier.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables and schema updates."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Users table with role support
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Classifications table with language and URL metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                language TEXT,
                urls TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Feedback table for wrong-prediction reports
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT NOT NULL,
                prediction TEXT,
                comment TEXT,
                status TEXT NOT NULL DEFAULT 'Pending',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Activity log for user actions and system events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity_type TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        self.ensure_columns()
        self.ensure_admin_account()

    def ensure_columns(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(users)")
        users_columns = [row[1] for row in cursor.fetchall()]
        if 'role' not in users_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")

        cursor.execute("PRAGMA table_info(classifications)")
        classifications_columns = [row[1] for row in cursor.fetchall()]
        if 'language' not in classifications_columns:
            cursor.execute("ALTER TABLE classifications ADD COLUMN language TEXT")
        if 'urls' not in classifications_columns:
            cursor.execute("ALTER TABLE classifications ADD COLUMN urls TEXT")

        conn.commit()
        conn.close()

    def ensure_admin_account(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        admin_count = cursor.fetchone()[0]
        if admin_count == 0:
            password_hash = self.hash_password('Admin@123')
            try:
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    ('admin', 'admin@example.com', password_hash, 'admin')
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # If the default admin username or email is taken, create a fallback admin.
                cursor.execute("SELECT COUNT(*) FROM users")
                total_users = cursor.fetchone()[0]
                fallback_username = f'admin{total_users + 1}'
                fallback_email = f'admin{total_users + 1}@example.com'
                cursor.execute(
                    "INSERT OR IGNORE INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    (fallback_username, fallback_email, password_hash, 'admin')
                )
                conn.commit()

        conn.close()
    
    def hash_password(self, password):
        """Hash password for secure storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username, email, password, role='user'):
        """Register a new user"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (username, email, password_hash, role)
            )
            
            conn.commit()
            conn.close()
            return True, "Registration successful!"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists!"
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, username, password):
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute(
            "SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            if user[2] == 'blocked':
                return False, "This account has been blocked."
            return True, {"id": user[0], "username": user[1], "role": user[2]}
        return False, "Invalid username or password!"
    
    def save_classification(self, user_id, message, prediction, confidence=None, language=None, urls=None):
        """Save classification result"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        urls_text = None
        if isinstance(urls, list):
            urls_text = ','.join(urls)
        elif isinstance(urls, str):
            urls_text = urls

        cursor.execute(
            "INSERT INTO classifications (user_id, message, prediction, confidence, language, urls) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, message, prediction, confidence, language, urls_text)
        )
        
        conn.commit()
        conn.close()
        self.log_activity(user_id, 'classification', f'Predicted {prediction} (confidence={confidence}, language={language})')

    def save_feedback(self, user_id, message, prediction, comment=None, status='Pending'):
        """Save user feedback reports."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO feedback (user_id, message, prediction, comment, status) VALUES (?, ?, ?, ?, ?)",
            (user_id, message, prediction, comment, status)
        )
        conn.commit()
        conn.close()
        self.log_activity(user_id, 'feedback', f'Feedback submitted: {comment}')

    def get_feedback(self, status=None):
        """Get feedback reports."""
        conn = sqlite3.connect(self.db_name)
        query = "SELECT f.id, u.username, f.message, f.prediction, f.comment, f.status, f.timestamp FROM feedback f LEFT JOIN users u ON f.user_id = u.id"
        params = ()
        if status:
            query += " WHERE f.status = ?"
            params = (status,)
        query += " ORDER BY f.timestamp DESC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_user_feedback(self, user_id, status=None):
        """Get feedback reports for a specific user."""
        conn = sqlite3.connect(self.db_name)
        query = "SELECT id, message, prediction, comment, status, timestamp FROM feedback WHERE user_id = ?"
        params = (user_id,)
        if status:
            query += " AND status = ?"
            params = (user_id, status)
        query += " ORDER BY timestamp DESC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def mark_feedback_reviewed(self, feedback_id):
        """Mark feedback as reviewed."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE feedback SET status = 'Reviewed' WHERE id = ?",
            (feedback_id,)
        )
        conn.commit()
        conn.close()
        self.log_activity(None, 'feedback_review', f'Feedback id {feedback_id} marked reviewed')

    def log_activity(self, user_id, activity_type, details=None):
        """Log user actions and system events."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO activity_logs (user_id, activity_type, details) VALUES (?, ?, ?)",
            (user_id, activity_type, details)
        )
        conn.commit()
        conn.close()

    def get_activity_logs(self, limit=100):
        """Get recent activity logs."""
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT a.id, u.username, a.activity_type, a.details, a.timestamp
            FROM activity_logs a
            LEFT JOIN users u ON a.user_id = u.id
            ORDER BY a.timestamp DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    
    def get_user_stats(self, user_id):
        """Get user's classification statistics"""
        conn = sqlite3.connect(self.db_name)
        
        # Total classifications
        total_query = "SELECT COUNT(*) FROM classifications WHERE user_id = ?"
        total = pd.read_sql_query(total_query, conn, params=(user_id,)).iloc[0, 0]
        
        # Spam vs Ham counts
        spam_ham_query = """
            SELECT prediction, COUNT(*) as count 
            FROM classifications 
            WHERE user_id = ? 
            GROUP BY prediction
        """
        spam_ham_df = pd.read_sql_query(spam_ham_query, conn, params=(user_id,))
        
        # Recent activity
        recent_query = """
            SELECT message, prediction, timestamp 
            FROM classifications 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        """
        recent_df = pd.read_sql_query(recent_query, conn, params=(user_id,))
        
        # Daily activity for line chart
        daily_query = """
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM classifications 
            WHERE user_id = ?
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """
        daily_df = pd.read_sql_query(daily_query, conn, params=(user_id,))
        
        conn.close()
        
        return {
            'total_classifications': total,
            'spam_ham_counts': spam_ham_df,
            'recent_activity': recent_df,
            'daily_activity': daily_df
        }
    
    def get_all_user_history(self, user_id):
        """Get complete classification history for a user"""
        conn = sqlite3.connect(self.db_name)

        query = """
            SELECT message, prediction, timestamp
            FROM classifications
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()

        return df

    def get_total_users(self):
        """Get total number of registered users"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def get_total_classifications(self):
        """Get total number of classifications across all users"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM classifications")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def get_all_users(self):
        """Get all registered users"""
        conn = sqlite3.connect(self.db_name)
        df = pd.read_sql_query(
            "SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC",
            conn
        )
        conn.close()
        return df

    def delete_user(self, user_id):
        """Delete a user and their classifications."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM classifications WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

    def block_user(self, user_id):
        """Block a user from logging in."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET role = 'blocked' WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

    def get_recent_classifications(self, limit=50):
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT c.id, u.username, c.message, c.prediction, c.confidence, c.language, c.urls, c.timestamp
            FROM classifications c
            LEFT JOIN users u ON c.user_id = u.id
            ORDER BY c.timestamp DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df

    def get_all_classifications(self):
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT c.id, u.username, c.message, c.prediction, c.confidence, c.language, c.urls, c.timestamp
            FROM classifications c
            LEFT JOIN users u ON c.user_id = u.id
            ORDER BY c.timestamp DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def get_system_stats(self):
        """Get system-wide statistics"""
        conn = sqlite3.connect(self.db_name)

        # Total classifications
        total_query = "SELECT COUNT(*) FROM classifications"
        total = pd.read_sql_query(total_query, conn).iloc[0, 0]

        # Spam vs Ham counts across all users
        spam_ham_query = """
            SELECT prediction, COUNT(*) as count
            FROM classifications
            GROUP BY prediction
        """
        spam_ham_df = pd.read_sql_query(spam_ham_query, conn)

        # Daily activity across all users
        daily_query = """
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM classifications
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """
        daily_df = pd.read_sql_query(daily_query, conn)

        conn.close()

        return {
            'total_classifications': total,
            'spam_ham_counts': spam_ham_df,
            'daily_activity': daily_df
        }