import sqlite3
import streamlit as st

def init_db():
    conn = sqlite3.connect('plant_disease.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        phone TEXT,
        show_notifications BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_path TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        rating INTEGER NOT NULL,
        review TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  phone TEXT,
                  show_notifications BOOLEAN DEFAULT TRUE,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  image_path TEXT NOT NULL,
                  prediction TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Create notifications table (with title column)
    c.execute('''CREATE TABLE IF NOT EXISTS notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  content TEXT NOT NULL,
                  is_active BOOLEAN DEFAULT TRUE,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create reviews table
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  rating INTEGER NOT NULL,
                  review TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    # Create indexes
    c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_user 
              ON predictions(user_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_reviews_user 
              ON reviews(user_id)''')
    
    # Insert sample notification if none exists
    c.execute('''INSERT INTO notifications (title, content, is_active)
                 SELECT 'Welcome!', 'Thank you for using Agrodoc!', 1
                 WHERE NOT EXISTS (SELECT 1 FROM notifications)''')
    
    conn.commit()
    conn.close()

def migrate_db():
    """Handles database migrations safely"""
    conn = sqlite3.connect('plant_disease.db')
    c = conn.cursor()
    
    try:
        # First check if notifications table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'")
        table_exists = c.fetchone() is not None
        
        if table_exists:
            # Check if title column exists
            c.execute("PRAGMA table_info(notifications)")
            columns = [column[1] for column in c.fetchall()]
            if 'title' not in columns:
                c.execute('''ALTER TABLE notifications ADD COLUMN title TEXT''')
                # Set default value for existing notifications
                c.execute('''UPDATE notifications SET title = 'Important Update' 
                           WHERE title IS NULL''')
                conn.commit()
                st.success("Database migration completed successfully")
    except Exception as e:
        st.error(f"Migration error: {str(e)}")
    finally:
        conn.close()

# Initialize database and run migrations in correct order
init_db()  # Create tables first
migrate_db()  # Then run migrations