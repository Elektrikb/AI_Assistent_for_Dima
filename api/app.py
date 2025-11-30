# File: api/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Depends, Form
from fastapi.security import OAuth2PasswordRequestForm
import os
from security import create_access_token, get_current_user, TokenData
from auth.user_db import UserDatabase
from .schemas import QuestionRequest, RecommendationResponse, SessionStatsResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationAPI:
    def __init__(self, article_db, session_manager, env, agent, response_generator):
        self.article_db = article_db
        self.session_manager = session_manager
        self.env = env
        self.agent = agent
        self.response_generator = response_generator
        self.user_db = UserDatabase()
        if not self.user_db.get_user("admin"):
            self.user_db.create_user("admin", "admin")
        self.app = FastAPI(title="RL Recommendation System API")
        self.setup_middleware()
        self.setup_routes()
        self.setup_static_files()

    def setup_middleware(self):
        """Настройка CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_static_files(self):
        """Настройка статических файлов для фронтенда"""
        # Создаем директорию frontend если не существует
        os.makedirs("frontend", exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="frontend"), name="static")
    
    def setup_routes(self):
        @self.app.post("/register")
        async def register(form_data: OAuth2PasswordRequestForm = Depends()):
            success = self.user_db.create_user(form_data.username, form_data.password)
            if not success:
                raise HTTPException(status_code=400, detail="Username already exists")
            return {"message": "User created successfully"}

        @self.app.post("/login")
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            if not self.user_db.authenticate_user(form_data.username, form_data.password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            token = create_access_token(data={"sub": form_data.username})
            return {"access_token": token, "token_type": "bearer"}

        @self.app.get("/users")
        async def get_all_users(current_user: TokenData = Depends(get_current_user)):
            """Получение списка всех пользователей (только имена)"""
            usernames = list(self.user_db.users.keys())
            return {"users": usernames, "total": len(usernames)}

        @self.app.get("/")
        async def root():
            return {"message": "RL Recommendation System API", "status": "running"}
        
        @self.app.post("/ask", response_model=RecommendationResponse)
        async def ask_question(
            request: QuestionRequest,
            current_user: TokenData = Depends(get_current_user)
        ):
            """Основной endpoint для вопросов"""
            try:
                user_id = current_user.user_id
                # Создаем или получаем сессию
                if user_id not in self.session_manager.sessions:
                    self.session_manager.create_session(user_id)
                
                # Получаем состояние на основе вопроса
                state = self.env.reset(request.question)
                
                # Агент выбирает статью
                article_id = self.agent.select_action(state, training=False)
                recommended_article = self.article_db.get_article(article_id)
                
                if not recommended_article:
                    raise HTTPException(status_code=404, detail="Article not found")
                
                # Вычисляем reward (в продакшене это делал бы пользователь)
                reward = self.env._calculate_reward(recommended_article)
                
                # Сохраняем взаимодействие
                self.session_manager.add_interaction(
                    user_id, request.question, recommended_article, reward
                )
                
                # Генерируем ответ
                response_data = self.response_generator.generate_answer(
                    request.question, recommended_article
                )
                
                return RecommendationResponse(
                    answer=response_data["answer"],
                    recommended_article=response_data["recommended_article"],
                    suggested_actions=response_data["suggested_actions"],
                    session_id=user_id,
                    confidence=response_data["recommended_article"]["confidence"]
                )
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/session/{user_id}", response_model=SessionStatsResponse)
        async def get_session_stats(
            user_id: str,
            current_user: TokenData = Depends(get_current_user)
        ):
            if current_user.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access forbidden")

            """Получение статистики сессии"""
            stats = self.session_manager.get_session_stats(user_id)
            if not stats:
                raise HTTPException(status_code=404, detail="Session not found")
            return SessionStatsResponse(**stats)
        
        @self.app.get("/articles")
        async def get_articles():
            """Получение всех статей"""
            articles = self.article_db.get_all_articles()
            return {"articles": articles}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "articles_count": len(self.article_db.get_all_articles()),
                "sessions_count": len(self.session_manager.sessions)
            }
        @self.app.get("/chat")
        async def chat_interface():
            """Serve the chat interface"""
            return FileResponse("frontend/chat.html")
        
        @self.app.get("/")
        async def root():
            return {
                "message": "RL Recommendation System API", 
                "status": "running",
                "endpoints": {
                    "api_docs": "/docs",
                    "chat_interface": "/chat",
                    "ask_question": "/ask",
                    "health_check": "/health"
                }
            }