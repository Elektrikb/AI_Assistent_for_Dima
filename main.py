# File: main.py
import uvicorn
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auth.user_db import UserDatabase
from config.settings import Config
from database.article_db import ArticleDatabase
from database.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_system():
    """Инициализация системы"""
    logger.info("Initializing RL Recommendation System...")
    
    try:
        config = Config()
        logger.info("Configuration loaded")
        
        # Инициализация базы данных с поддержкой Excel
        article_db = ArticleDatabase(config.ARTICLES_PATH, config.EXCEL_PATH)
        articles = article_db.get_all_articles()
        logger.info(f"Loaded {len(articles)} articles")

        # Инициализация базы пользователей
        user_db = UserDatabase()
        if not user_db.get_user("admin"):
            user_db.create_user("admin", "admin")  # демо-пользователь
        logger.info("User database initialized")
        
        if not articles:
            logger.error("No articles available. System cannot start.")
            return None
        
        # Инициализация менеджера сессий
        session_manager = SessionManager(config.SESSIONS_PATH)
        logger.info(f"Loaded {len(session_manager.sessions)} existing sessions")
        
        # Инициализация остальных компонентов
        from models.state_encoder import StateEncoder
        from rl_environment.env import RecommendationEnv
        from agents.dqn_agent import DQNAgent
        from models.response_generator import ResponseGenerator
        from api.app import RecommendationAPI
        
        # Инициализация кодировщика состояний
        state_encoder = StateEncoder(article_db)
        state_dim = state_encoder.get_state_dimension()
        logger.info(f"State encoder initialized with dimension: {state_dim}")
        
        # Инициализация RL среды
        env = RecommendationEnv(article_db, state_encoder, config.environment)
        action_dim = env.get_action_space_size()
        logger.info(f"RL Environment initialized with {action_dim} actions")
        
        # Обновление конфигурации с реальными размерами
        config.model.action_dim = action_dim
        config.model.state_dim = state_dim
        
        # Инициализация RL агента
        agent = DQNAgent(state_dim, action_dim, config.model)
        logger.info("DQN Agent initialized")
        
        # ПРЕДВАРИТЕЛЬНОЕ ОБУЧЕНИЕ (если используется)
        try:
            from training.pretrain import Pretrainer
            logger.info("Starting pretraining...")
            pretrainer = Pretrainer(env, agent, article_db)
            pretrainer.pretrain_with_supervised(episodes=500)  # Уменьшил для скорости
            eval_results = pretrainer.evaluate_pretraining()
            logger.info(f"Pretraining completed! Accuracy: {eval_results['accuracy']:.3f}")
        except Exception as e:
            logger.warning(f"Pretraining skipped: {e}")
        
        # Инициализация генератора ответов
        response_generator = ResponseGenerator(article_db)
        logger.info("Response generator initialized")
        
        return {
            'config': config,
            'article_db': article_db,
            'user_db': user_db,
            'session_manager': session_manager,
            'state_encoder': state_encoder,
            'env': env,
            'agent': agent,
            'response_generator': response_generator,
            'api_class': RecommendationAPI
        }
        
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    components = initialize_system()
    
    if components is None:
        logger.error("System initialization failed. Exiting.")
        return
    
    try:
        # Создание API
        api = components['api_class'](
            article_db=components['article_db'],
            session_manager=components['session_manager'],
            env=components['env'],
            agent=components['agent'],
            response_generator=components['response_generator']
        )
        
        logger.info("Starting FastAPI server...")
        logger.info(f"API will be available at: http://{components['config'].api.host}:{components['config'].api.port}")
        logger.info(f"API documentation: http://{components['config'].api.host}:{components['config'].api.port}/docs")
        logger.info(f"Chat UI: http://{components['config'].api.host}:{components['config'].api.port}/chat")
        
        uvicorn.run(
            api.app,
            host=components['config'].api.host,
            port=components['config'].api.port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()