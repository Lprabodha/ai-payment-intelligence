"""
Database indexes management
"""
from database.connection import db

def ensure_database_indexes():
    """Ensure required database indexes exist"""
    try:
        required_indexes = {
            'transactions': [
                ('email', 1),
                ('created_at', -1),
                ('transaction_id', 1),
                ('ip_address', 1),
                ('fingerprint', 1)
            ],
            'fraud_results': [
                ('transaction_id', 1),
                ('created_at', -1)
            ],
            'chargeback_predictions': [
                ('transaction_id', 1),
                ('created_at', -1)
            ],
            'routing_predictions': [
                ('transaction_id', 1),
                ('created_at', -1)
            ],
            'revenue_predictions': [
                ('subscription_id', 1),
                ('created_at', -1)
            ],
            'processed_webhook_events': [
                ('event_id', 1),
                ('created_at', -1)
            ]
        }
        
        for collection_name, indexes in required_indexes.items():
            collection = db[collection_name]
            for index_spec, direction in indexes:
                # Check if index already exists
                existing_indexes = collection.list_indexes()
                index_name = f"{index_spec}_{direction}"
                index_exists = any(idx['name'] == index_name for idx in existing_indexes)
                
                if not index_exists:
                    # Create index with specific name to avoid conflicts
                    index_options = {
                        'background': True,
                        'name': index_name
                    }
                    
                    # Make transaction_id unique if it's the primary key
                    if index_spec == 'transaction_id' and collection_name == 'transactions':
                        index_options['unique'] = True
                    
                    collection.create_index([(index_spec, direction)], **index_options)
                    print(f"Created index {index_name} on {collection_name}")
                else:
                    print(f"Index {index_name} already exists on {collection_name}")
        
        print("Database indexes ensured")
        
    except Exception as e:
        print(f"Error creating database indexes: {e}")
        raise
