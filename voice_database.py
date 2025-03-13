import numpy as np
import os
from datetime import datetime
from numpy.linalg import norm

class VoiceDatabase:
    """
    Manages a database of voice imprints for multiple users in a single file.
    """
    
    def __init__(self, database_file="voice_database.npy"):
        """
        Initialize the voice database manager.
        
        Args:
            database_file: Path to the database file
        """
        self.database_file = database_file
        self.database = self._load_database()
    
    def _load_database(self):
        """
        Load the voice database or create a new one if it doesn't exist.
        """
        if os.path.exists(self.database_file):
            try:
                return np.load(self.database_file, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading database: {e}")
                return self._create_new_database()
        else:
            return self._create_new_database()
    
    def _create_new_database(self):
        """
        Create a new empty database structure.
        """
        return {
            "users": {},
            "version": "1.1",
            "created": datetime.now(),
            "last_updated": datetime.now(),
            "settings": {
                "global_threshold": 0.85,  # Higher default threshold for better accuracy
                "second_best_margin": 0.1,  # Required margin between best and second best match
                "min_samples": 5,          # Recommended minimum samples for enrollment
                "enable_adaptive_threshold": True
            }
        }
    
    def save_database(self):
        """
        Save the current database to disk.
        """
        try:
            # Update last modified timestamp
            self.database["last_updated"] = datetime.now()
            
            # Save to file
            np.save(self.database_file, self.database)
            print(f"Database saved to {self.database_file}")
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_user(self, user_id, embedding, model_type="resemblyzer", 
                 sample_count=1, combination_method="single", threshold_offset=0.0):
        """
        Add or update a user in the database.
        
        Args:
            user_id: Unique identifier for the user
            embedding: Voice embedding vector
            model_type: Model used to generate the embedding
            sample_count: Number of samples used to create this embedding
            combination_method: Method used to combine multiple samples
            threshold_offset: Custom threshold adjustment for this user
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create user entry if it doesn't exist
            if "users" not in self.database:
                self.database["users"] = {}
                
            # Calculate adaptive threshold based on sample count
            adaptive_threshold = 0.0
            if self.database.get("settings", {}).get("enable_adaptive_threshold", False):
                min_samples = self.database.get("settings", {}).get("min_samples", 5)
                # If fewer samples than recommended, increase threshold requirements
                if sample_count < min_samples:
                    adaptive_threshold = 0.02 * (min_samples - sample_count)
                    print(f"Warning: Using fewer than recommended samples ({sample_count}/{min_samples}).")
                    print(f"Automatically increasing verification threshold by {adaptive_threshold:.2f}.")
            
            # Add/update user data
            self.database["users"][user_id] = {
                "embedding": embedding,
                "model_type": model_type,
                "created": datetime.now(),
                "last_updated": datetime.now(),
                "sample_count": sample_count,
                "combination_method": combination_method,
                "threshold_offset": threshold_offset + adaptive_threshold,
                "enrollment_quality": self._calculate_enrollment_quality(embedding, sample_count)
            }
            
            # Save the updated database
            return self.save_database()
        except Exception as e:
            print(f"Error adding user to database: {e}")
            return False
    
    def _calculate_enrollment_quality(self, embedding, sample_count):
        """
        Calculate enrollment quality score based on samples and embedding properties.
        
        Returns:
            Quality score between 0-1
        """
        # Basic quality score based on number of samples
        sample_quality = min(1.0, sample_count / 10.0)  # 10+ samples considered optimal
        
        # Check embedding norm (should be normalized but just in case)
        embedding_norm = np.linalg.norm(embedding)
        norm_quality = 1.0 if 0.99 <= embedding_norm <= 1.01 else 0.7
        
        # Combine factors (can be extended with more metrics)
        return (sample_quality * 0.7) + (norm_quality * 0.3)
    
    def update_settings(self, settings_dict):
        """
        Update database settings.
        
        Args:
            settings_dict: Dictionary of settings to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if "settings" not in self.database:
                self.database["settings"] = {}
                
            self.database["settings"].update(settings_dict)
            return self.save_database()
        except Exception as e:
            print(f"Error updating settings: {e}")
            return False
    
    def get_settings(self):
        """
        Get current database settings.
        
        Returns:
            Dictionary of settings
        """
        return self.database.get("settings", {})
    
    def get_user(self, user_id):
        """
        Get a user's data from the database.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            User data dictionary or None if not found
        """
        try:
            return self.database["users"].get(user_id)
        except Exception as e:
            print(f"Error retrieving user {user_id}: {e}")
            return None
    
    def get_all_users(self):
        """
        Get all users in the database.
        
        Returns:
            Dictionary of user_id -> user data
        """
        try:
            return self.database.get("users", {})
        except Exception as e:
            print(f"Error retrieving all users: {e}")
            return {}
    
    def remove_user(self, user_id):
        """
        Remove a user from the database.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if user_id in self.database.get("users", {}):
                del self.database["users"][user_id]
                return self.save_database()
            else:
                print(f"User {user_id} not found in database")
                return False
        except Exception as e:
            print(f"Error removing user {user_id}: {e}")
            return False
    
    def get_user_count(self):
        """
        Get the number of users in the database.
        
        Returns:
            Number of users
        """
        try:
            return len(self.database.get("users", {}))
        except Exception as e:
            print(f"Error counting users: {e}")
            return 0
    
    def _calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1
        """
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    
    def _perform_score_normalization(self, similarities, threshold):
        """
        Normalize scores and apply verification criteria.
        
        Args:
            similarities: Dictionary of user_id -> similarity score
            threshold: Base threshold for verification
            
        Returns:
            Dictionary of user_id -> (normalized score, is_match)
        """
        # Convert to list and sort
        scores = [(user_id, score) for user_id, score in similarities.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # If we have enough scores, we can apply cohort normalization
        if len(scores) > 1:
            best_user_id, best_score = scores[0]
            
            # Check for second-best score margin (if there's more than one user)
            if len(scores) > 1:
                second_best_score = scores[1][1]
                margin = best_score - second_best_score
                margin_threshold = self.database.get("settings", {}).get("second_best_margin", 0.1)
                
                # If the margin is too small, confidence is lower
                if margin < margin_threshold and best_score < 0.95:
                    print(f"Warning: Small margin ({margin:.3f}) between best and second-best match")
                    # Reduce score based on margin
                    best_score -= (margin_threshold - margin)
        
        # Apply results with adjusted scores
        results = {}
        for user_id, score in similarities.items():
            user_data = self.database["users"].get(user_id, {})
            user_threshold = threshold + user_data.get("threshold_offset", 0.0)
            is_match = score > user_threshold
            results[user_id] = (score, is_match)
            
        return results
    
    def identify_speaker(self, test_embedding, threshold=None):
        """
        Identify the speaker by comparing with all enrolled users.
        Uses advanced scoring and verification criteria.
        
        Args:
            test_embedding: Voice embedding to identify
            threshold: Optional similarity threshold (uses settings default if None)
            
        Returns:
            Tuple of (best_match_user_id, similarity, all_results)
            where all_results is a dict of user_id -> (similarity, is_match)
        """
        if threshold is None:
            threshold = self.database.get("settings", {}).get("global_threshold", 0.75)
            
        if len(self.database.get("users", {})) == 0:
            return None, 0.0, {}
            
        # Calculate similarities for all users
        similarities = {}
        for user_id, user_data in self.database.get("users", {}).items():
            try:
                user_embedding = user_data["embedding"]
                similarity = self._calculate_similarity(test_embedding, user_embedding)
                similarities[user_id] = similarity
            except Exception as e:
                print(f"Error comparing with user {user_id}: {e}")
                similarities[user_id] = -1.0
        
        # Apply normalization and verification criteria
        results = self._perform_score_normalization(similarities, threshold)
        
        # Find best match
        best_match = None
        best_similarity = -1
        
        for user_id, (similarity, is_match) in results.items():
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user_id
        
        return best_match, best_similarity, results