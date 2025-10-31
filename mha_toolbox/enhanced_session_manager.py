"""
Enhanced Session Manager
========================

Provides session management capabilities including session context retrieval
for the "Continue Session" feature.
"""

import os
import json
from pathlib import Path
from datetime import datetime


class EnhancedSessionManager:
    """Manages experiment sessions and their metadata"""
    
    def __init__(self):
        self.base_dir = Path("results/detailed_storage")
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_session_context(self, session_id):
        """
        Retrieve context information for a specific session.
        
        This enables the "Continue Session" workflow by providing:
        - Dataset name used in the session
        - List of algorithms already run
        - Session metadata
        
        Args:
            session_id: The session identifier to look up
            
        Returns:
            dict: Session context containing:
                - dataset_name: Name of the dataset
                - algorithms: List of algorithm names already run
                - session_id: The session ID
                - created_at: Session creation timestamp
                - last_modified: Last modification timestamp
                - total_algorithms: Count of algorithms in session
                
            None if session not found
        """
        try:
            # Scan results directories to find this session
            # Structure: results/detailed_storage/{dataset_name}/{session_id}/
            
            if not self.base_dir.exists():
                return None
            
            # Search through all dataset directories
            for dataset_dir in self.base_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                session_dir = dataset_dir / session_id
                
                if session_dir.exists() and session_dir.is_dir():
                    # Found the session directory
                    algorithms = []
                    
                    # Scan for algorithm results (typically .npz files or metadata files)
                    for item in session_dir.iterdir():
                        if item.is_file():
                            # Check for metadata files
                            if item.name.endswith('_metadata.json'):
                                # Extract algorithm name from filename
                                # Format: {algorithm}_metadata.json
                                alg_name = item.name.replace('_metadata.json', '')
                                
                                # Load metadata to get more details
                                try:
                                    with open(item, 'r') as f:
                                        metadata = json.load(f)
                                        algorithms.append({
                                            'name': metadata.get('algorithm_name', alg_name),
                                            'best_fitness': metadata.get('best_fitness', None),
                                            'total_iterations': metadata.get('total_iterations', None)
                                        })
                                except:
                                    algorithms.append({
                                        'name': alg_name,
                                        'best_fitness': None,
                                        'total_iterations': None
                                    })
                    
                    # Get session timestamps
                    session_stat = session_dir.stat()
                    created_at = datetime.fromtimestamp(session_stat.st_ctime).isoformat()
                    last_modified = datetime.fromtimestamp(session_stat.st_mtime).isoformat()
                    
                    return {
                        'dataset_name': dataset_name,
                        'algorithms': [alg['name'] if isinstance(alg, dict) else alg for alg in algorithms],
                        'algorithm_details': algorithms,
                        'session_id': session_id,
                        'created_at': created_at,
                        'last_modified': last_modified,
                        'total_algorithms': len(algorithms),
                        'session_path': str(session_dir)
                    }
            
            # Session not found
            return None
            
        except Exception as e:
            print(f"Error retrieving session context: {e}")
            return None
    
    def list_all_sessions(self):
        """
        List all available sessions across all datasets.
        
        Returns:
            list: List of session dictionaries with basic info
        """
        sessions = []
        
        try:
            if not self.base_dir.exists():
                return sessions
            
            for dataset_dir in self.base_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                
                for session_dir in dataset_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    
                    session_id = session_dir.name
                    
                    # Count algorithms in this session
                    algorithm_count = sum(1 for f in session_dir.iterdir() 
                                        if f.is_file() and f.name.endswith('_metadata.json'))
                    
                    session_stat = session_dir.stat()
                    
                    sessions.append({
                        'dataset_name': dataset_name,
                        'session_id': session_id,
                        'algorithm_count': algorithm_count,
                        'created_at': datetime.fromtimestamp(session_stat.st_ctime).isoformat(),
                        'last_modified': datetime.fromtimestamp(session_stat.st_mtime).isoformat()
                    })
            
            # Sort by last modified (most recent first)
            sessions.sort(key=lambda x: x['last_modified'], reverse=True)
            
            return sessions
            
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
    
    def get_latest_session(self):
        """
        Get the most recently modified session.
        
        Returns:
            dict: Latest session info or None
        """
        sessions = self.list_all_sessions()
        
        if sessions:
            latest = sessions[0]
            # Get full context for the latest session
            return self.get_session_context(latest['session_id'])
        
        return None
