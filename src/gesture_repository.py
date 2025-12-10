"""
Gesture repository for persistent storage using SQLite.
"""
import sqlite3
import numpy as np
from typing import Optional, List, Callable
from datetime import datetime
from pathlib import Path
import json
import logging
import shutil

from src.models import Gesture, Action, ActionType

logger = logging.getLogger(__name__)


class GestureRepository:
    """
    Repository for storing and retrieving gestures using SQLite backend.
    
    Attributes:
        db_path: Path to the SQLite database file
    """
    
    def __init__(self, db_path: str = "gestures.db"):
        """
        Initialize the gesture repository.
        
        Automatically detects and recovers from database corruption.
        
        Args:
            db_path: Path to the SQLite database file
        
        Requirements: 9.5
        """
        self.db_path = db_path
        self._recovery_callback: Optional[Callable[[str], None]] = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """
        Create the gestures table if it doesn't exist.
        
        Detects database corruption and recovers by creating a new database.
        
        Requirements: 9.5
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try to create the table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gestures (
                    name TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    action_type TEXT NOT NULL,
                    action_data TEXT NOT NULL,
                    execution_mode TEXT DEFAULT 'trigger_once',
                    created_at TEXT NOT NULL
                )
            """)
            
            # Check if execution_mode column exists (for migration)
            cursor.execute("PRAGMA table_info(gestures)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'execution_mode' not in columns:
                logger.info("Migrating database: adding execution_mode column")
                cursor.execute("""
                    ALTER TABLE gestures 
                    ADD COLUMN execution_mode TEXT DEFAULT 'trigger_once'
                """)
                logger.info("Database migration completed")
            
            # Verify database integrity
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result[0] != "ok":
                logger.error(f"Database integrity check failed: {result[0]}")
                conn.close()
                self._recover_from_corruption()
                return
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except sqlite3.DatabaseError as e:
            logger.error(f"Database error during initialization: {e}")
            # Close connection if it was opened
            try:
                conn.close()
            except:
                pass
            self._recover_from_corruption()
            
        except Exception as e:
            logger.exception(f"Unexpected error during database initialization: {e}")
            # Close connection if it was opened
            try:
                conn.close()
            except:
                pass
            self._recover_from_corruption()
    
    def _recover_from_corruption(self) -> None:
        """
        Recover from database corruption by backing up and recreating.
        
        Creates a backup of the corrupted database and initializes
        a new empty database.
        
        Requirements: 9.5
        """
        logger.warning("Attempting to recover from database corruption...")
        
        try:
            # Create backup of corrupted database if it exists
            db_path_obj = Path(self.db_path)
            if db_path_obj.exists():
                backup_path = db_path_obj.with_suffix(f".corrupted.{int(datetime.now().timestamp())}.db")
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Corrupted database backed up to: {backup_path}")
                
                # Remove corrupted database
                db_path_obj.unlink()
                logger.info("Corrupted database removed")
            
            # Create new empty database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE gestures (
                    name TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    action_type TEXT NOT NULL,
                    action_data TEXT NOT NULL,
                    execution_mode TEXT DEFAULT 'trigger_once',
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("New database created successfully")
            
            # Notify user of recovery
            recovery_message = (
                "Database corruption detected and recovered. "
                "A new empty database has been created. "
                "Previous gestures have been lost but backed up."
            )
            self._notify_recovery(recovery_message)
            
        except Exception as e:
            logger.exception(f"Failed to recover from database corruption: {e}")
            error_message = f"Critical error: Unable to recover database. Error: {e}"
            self._notify_recovery(error_message)
    
    def set_recovery_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set a callback function for recovery notifications.
        
        Args:
            callback: Function that takes a recovery message string
        """
        self._recovery_callback = callback
    
    def _notify_recovery(self, message: str) -> None:
        """
        Notify about recovery via callback if set.
        
        Args:
            message: Recovery message to send
        """
        logger.warning(message)
        if self._recovery_callback:
            self._recovery_callback(message)
    
    def save_gesture(self, gesture: Gesture) -> None:
        """
        Save a gesture to the database.
        
        Args:
            gesture: The gesture object to save
            
        Raises:
            sqlite3.IntegrityError: If a gesture with the same name already exists
            sqlite3.DatabaseError: If database is corrupted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize embedding as bytes
            embedding_bytes = gesture.embedding.tobytes()
            
            # Serialize action data as JSON
            action_data_json = json.dumps(gesture.action.data)
            
            cursor.execute("""
                INSERT INTO gestures (name, embedding, action_type, action_data, execution_mode, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                gesture.name,
                embedding_bytes,
                gesture.action.type.value,
                action_data_json,
                gesture.action.execution_mode.value,
                gesture.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Gesture '{gesture.name}' saved successfully")
            
        except sqlite3.DatabaseError as e:
            logger.error(f"Database error while saving gesture: {e}")
            try:
                conn.close()
            except:
                pass
            raise
    
    def get_gesture(self, name: str) -> Optional[Gesture]:
        """
        Retrieve a gesture by name.
        
        Args:
            name: The name of the gesture to retrieve
            
        Returns:
            Optional[Gesture]: The gesture object if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, embedding, action_type, action_data, execution_mode, created_at
                FROM gestures
                WHERE name = ?
            """, (name,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            return self._row_to_gesture(row)
            
        except sqlite3.DatabaseError as e:
            logger.error(f"Database error while retrieving gesture '{name}': {e}")
            try:
                conn.close()
            except:
                pass
            return None
    
    def get_all_gestures(self) -> List[Gesture]:
        """
        Retrieve all stored gestures.
        
        Returns:
            List[Gesture]: List of all gesture objects, empty list on error
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, embedding, action_type, action_data, execution_mode, created_at
                FROM gestures
                ORDER BY created_at DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_gesture(row) for row in rows]
            
        except sqlite3.DatabaseError as e:
            logger.error(f"Database error while retrieving all gestures: {e}")
            try:
                conn.close()
            except:
                pass
            return []
    
    def delete_gesture(self, name: str) -> bool:
        """
        Delete a gesture by name.
        
        Args:
            name: The name of the gesture to delete
            
        Returns:
            bool: True if the gesture was deleted, False if it didn't exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM gestures
            WHERE name = ?
        """, (name,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return rows_affected > 0
    
    def update_gesture(self, name: str, gesture: Gesture) -> bool:
        """
        Update an existing gesture.
        
        Args:
            name: The name of the gesture to update
            gesture: The new gesture data
            
        Returns:
            bool: True if the gesture was updated, False if it didn't exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize embedding as bytes
        embedding_bytes = gesture.embedding.tobytes()
        
        # Serialize action data as JSON
        action_data_json = json.dumps(gesture.action.data)
        
        cursor.execute("""
            UPDATE gestures
            SET name = ?,
                embedding = ?,
                action_type = ?,
                action_data = ?,
                execution_mode = ?,
                created_at = ?
            WHERE name = ?
        """, (
            gesture.name,
            embedding_bytes,
            gesture.action.type.value,
            action_data_json,
            gesture.action.execution_mode.value,
            gesture.created_at.isoformat(),
            name
        ))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return rows_affected > 0
    
    def _row_to_gesture(self, row: tuple) -> Gesture:
        """
        Convert a database row to a Gesture object.
        
        Args:
            row: Tuple containing (name, embedding, action_type, action_data, execution_mode, created_at)
            
        Returns:
            Gesture: The reconstructed gesture object
        """
        from src.models import ExecutionMode
        
        name, embedding_bytes, action_type, action_data_json, execution_mode_str, created_at_str = row
        
        # Deserialize embedding from bytes
        embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
        
        # Deserialize action data from JSON
        action_data = json.loads(action_data_json)
        
        # Parse execution mode (default to TRIGGER_ONCE for backward compatibility)
        try:
            execution_mode = ExecutionMode(execution_mode_str) if execution_mode_str else ExecutionMode.TRIGGER_ONCE
        except (ValueError, AttributeError):
            execution_mode = ExecutionMode.TRIGGER_ONCE
        
        # Create action object
        action = Action(
            type=ActionType(action_type),
            data=action_data,
            execution_mode=execution_mode
        )
        
        # Parse datetime
        created_at = datetime.fromisoformat(created_at_str)
        
        return Gesture(
            name=name,
            embedding=embedding,
            action=action,
            created_at=created_at
        )
