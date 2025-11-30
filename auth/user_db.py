import json
import os
from typing import Dict, Optional
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

class UserDatabase:
    def __init__(self, users_file: str = "data/users.json"):
        self.users_file = users_file
        self.users = self._load_users()
    
    def _load_users(self) -> Dict[str, dict]:
        if not os.path.exists(self.users_file):
            return {}
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Пароли уже хэшированы
            return data
        except (json.JSONDecodeError, Exception):
            return {}
    
    def _save_users(self):
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)
    
    def get_user(self, username: str) -> Optional[dict]:
        return self.users.get(username)
    
    def create_user(self, username: str, password: str) -> bool:
        if username in self.users:
            return False
        hashed = pwd_context.hash(password)
        self.users[username] = {
            "username": username,
            "hashed_password": hashed
        }
        self._save_users()
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        return pwd_context.verify(password, user["hashed_password"])
