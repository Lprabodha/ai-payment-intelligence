"""
Database connection and initialization
"""
from pymongo import MongoClient
from config.settings import settings

class DatabaseManager:
    """Manages MongoDB connection and database operations"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(
                settings.MONGO_URI,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000
            )
            self.db = self.client[settings.DATABASE_NAME]
            
            # Test connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB")
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name):
        """Get a specific collection"""
        return self.db[collection_name]
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

# Global database instance
db_manager = DatabaseManager()
db = db_manager.db
