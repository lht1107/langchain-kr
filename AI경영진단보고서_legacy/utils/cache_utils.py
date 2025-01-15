import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import redis.asyncio as redis
from fastapi import HTTPException

from core.config import settings
from utils.logger import get_logger


from core.cache import cache_manager
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# async def get_redis_client():
#     """Redis 클라이언트를 가져오는 함수"""
#     from core.dependencies import redis_client  # 필요할 때만 import
#     return redis_client

# def generate_cache_key(company_name: str, access_time: datetime) -> str:
#     """캐시 키 생성 함수"""
#     return f"{company_name}:{access_time.strftime('%Y-%m-%d')}"
def generate_cache_key(company_name: str, access_time: datetime) -> str:
    """캐시 키 생성 함수"""
    return f"{company_name}:{access_time.strftime(settings.TIMESTAMP)}"


# async def get_cached_data(key: str, redis: Optional[redis.Redis] = None) -> Optional[Dict]:
#     """캐시 데이터를 조회하는 함수"""
#     if not redis:
#         redis = await get_redis_client()

#     if not redis:
#         logger.warning("[Cache] Redis client not available")
#         return None

#     try:
#         data = await redis.get(key)
#         if data:
#             logger.info(f"[Cache] Cache hit - Key: {key}")
#             return json.loads(data)

#         logger.info(
#             f"[Cache] Cache miss - Initializing new analysis for key: {key}")
#         return None
#     except redis.RedisError as e:
#         logger.error(f"[Cache] Redis error occurred: {str(e)}")
#         return None
async def get_cached_data(key: str) -> Optional[Dict]:
    """캐시 데이터를 조회하는 함수"""
    try:
        cached_data = await cache_manager.get(key)
        if cached_data:
            logger.info(f"[Cache] Cache hit - Key: {key}")
            if 'analysis_status' in cached_data:
                logger.info(
                    f"[Cache] Current Status - "
                    f"Strength: {cached_data['analysis_status']['strength']}, "
                    f"Weakness: {cached_data['analysis_status']['weakness']}, "
                    f"Insight: {cached_data['analysis_status']['insight']}"
                )
        else:
            logger.info(f"[Cache] Cache miss - Key: {key}")
        return cached_data
    except Exception as e:
        logger.error(f"[Cache] Error getting cached data: {str(e)}")
        return None


# def serialize_datetime(obj):
#     """Timestamp 객체를 문자열로 직렬화하는 함수"""
#     if isinstance(obj, (datetime, pd.Timestamp)):
#         return obj.strftime('%Y-%m-%d %H:%M:%S')
#     raise TypeError(f"Type {type(obj)} not serializable")


# async def set_cached_data(
#     key: str,
#     data: Dict,
#     analysis_type: str,
#     expire_time: int = settings.CACHE_EXPIRE_TIME
# ) -> None:
#     """캐시 데이터를 저장하는 함수"""
#     redis_client = await get_redis_client()

#     if not redis_client:
#         logger.warning("[Cache] Redis client not available")
#         return

#     try:
#         # 기존 캐시 데이터 조회 및 기본 구조 생성
#         cached_data = await get_cached_data(key)
#         if not cached_data:
#             cached_data = {
#                 'company_data': None,
#                 'analysis_status': {
#                     'strength': False,  # 강점 분석 수행 여부
#                     'weakness': False,  # 약점 분석 수행 여부
#                     'insight': False  # 통찰 분석 수행 여부
#                 },
#                 'strength': {
#                     'indicator': None,  # 강점 지표 항목
#                     'detailed_analysis': None,  # 상세 분석 결과
#                     'summary': None  # 요약 분석 결과
#                 },
#                 'weakness': {
#                     'indicator': None,  # 약점 지표 항목
#                     'detailed_analysis': None,  # 상세 분석 결과
#                     'summary': None  # 요약 분석 결과
#                 },
#                 'insight': {
#                     'summary': None  # 통찰 분석 결과
#                 }
#             }
#         else:
#             logger.info(f"[Cache] Updating existing cache - Key: {key}")

#         # company_data가 없는 경우에만 업데이트
#         if not cached_data['company_data'] and data.get('company_data'):
#             cached_data['company_data'] = data['company_data']

#         # 데이터 구조 검증
#         if not validate_cache_data(cached_data):
#             raise ValueError("Invalid cache data structure")

#         # 분석 타입별 결과 업데이트
#         match analysis_type:
#             case "strength":
        # cached_data.update({
        #     'strength': {
        #         'indicator': data['strength']['indicator'],
        #         'detailed_analysis': data['strength']['detailed_analysis'],
        #         'summary': data['strength']['summary']
        #     },
        #     'analysis_status': {**cached_data['analysis_status'], 'strength': True}
        # })
#             case "weakness":
#                 cached_data.update({
#                     'weakness': {
#                         'indicator': data['weakness']['indicator'],
#                         'detailed_analysis': data['weakness']['detailed_analysis'],
#                         'summary': data['weakness']['summary']
#                     },
#                     'analysis_status': {**cached_data['analysis_status'], 'weakness': True}
#                 })
#             case "insight":
#                 cached_data.update({
#                     'insight': {
#                         'summary': data['insight']['summary']
#                     },
#                     'analysis_status': {
#                         **cached_data['analysis_status'],
#                         'strength': True,
#                         'weakness': True,
#                         'insight': True
#                     }
#                 })

#         await redis_client.set(
#             key,
#             json.dumps(cached_data, default=serialize_datetime),
#             ex=expire_time
#         )
#         logger.info(f"[Cache] Data stored - Key: {key}")

#     except ValueError as e:
#         logger.error(f"[Cache] Validation error: {str(e)}")
#         raise HTTPException(
#             status_code=400,
#             detail=f"Cache validation failed: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"[Cache] Failed to process data: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Cache operation failed: {str(e)}"
#         )

async def get_cached_data(key: str) -> Optional[Dict]:
    """캐시 데이터를 조회하는 함수"""
    try:
        cached_data = await cache_manager.get(key)
        if cached_data:
            logger.info(f"[Cache] Cache hit - Key: {key}")
            if 'analysis_status' in cached_data:
                logger.info(
                    f"[Cache] Current Status - "
                    f"Strength: {cached_data['analysis_status']['strength']}, "
                    f"Weakness: {cached_data['analysis_status']['weakness']}, "
                    f"Insight: {cached_data['analysis_status']['insight']}"
                )
        else:
            logger.info(f"[Cache] Cache miss - Key: {key}")
        return cached_data
    except Exception as e:
        logger.error(f"[Cache] Error getting cached data: {str(e)}")
        return None


async def set_cached_data(key: str, data: Dict, analysis_type: str) -> None:
    """캐시 데이터를 저장하는 함수"""
    try:
        # 기존 캐시 데이터 조회
        cached_data = await get_cached_data(key)
        if not cached_data:
            logger.info(f"[Cache] Creating new cache entry - Key: {key}")
            cached_data = {
                'company_data': None,
                'analysis_status': {
                    'strength': False,
                    'weakness': False,
                    'insight': False
                },
                'strength': {
                    'indicator': None,
                    'detailed_analysis': None,
                    'summary': None
                },
                'weakness': {
                    'indicator': None,
                    'detailed_analysis': None,
                    'summary': None
                },
                'insight': {
                    'summary': None
                }
            }
        else:
            logger.info(f"[Cache] Updating existing cache - Key: {key}")

        # 데이터 업데이트
        if not cached_data['company_data'] and data.get('company_data'):
            cached_data['company_data'] = data['company_data']

        # 분석 타입별 결과 업데이트
        match analysis_type:
            case "strength":
                cached_data.update({
                    'strength': {
                        'indicator': data['strength']['indicator'],
                        'detailed_analysis': data['strength']['detailed_analysis'],
                        'summary': data['strength']['summary']
                    },
                    'analysis_status': {**cached_data['analysis_status'], 'strength': True}
                })
            case "weakness":
                cached_data.update({
                    'weakness': {
                        'indicator': data['weakness']['indicator'],
                        'detailed_analysis': data['weakness']['detailed_analysis'],
                        'summary': data['weakness']['summary']
                    },
                    'analysis_status': {**cached_data['analysis_status'], 'weakness': True}
                })
            case "insight":
                cached_data.update({
                    'insight': {
                        'summary': data['insight']['summary']
                    },
                    'analysis_status': {
                        **cached_data['analysis_status'],
                        'strength': True,
                        'weakness': True,
                        'insight': True
                    }
                })

        await cache_manager.set(key, cached_data)
        logger.info(
            f"[Cache] Updated Status - Key: {key}, "
            f"Type: {analysis_type}, "
            f"Status: {cached_data['analysis_status']}"
        )
    except Exception as e:
        logger.error(f"[Cache] Failed to set cached data: {str(e)}")
        raise


def validate_cache_data(data: Dict) -> bool:
    """
    캐시 데이터 유효성 검증 함수

    Args:
        data: 검증할 캐시 데이터 딕셔너리

    Returns:
        bool: 유효성 검증 결과
    """
    try:
        # 기본 타입 검증
        if not isinstance(data, dict):
            logger.warning("[Validation] Cache data is not a dictionary")
            return False

        # 필수 최상위 필드 검증
        required_fields = {
            'company_data',
            'analysis_status',
            'strength',
            'weakness',
            'insight'
        }
        if not all(field in data for field in required_fields):
            logger.warning("[Validation] Missing required top-level fields")
            return False

        # analysis_status 구조 검증
        required_status = {'strength', 'weakness', 'insight'}
        if not all(status in data['analysis_status'] for status in required_status):
            logger.warning("[Validation] Invalid analysis_status structure")
            return False

        # strength와 weakness 구조 검증
        required_analysis_fields = {
            'indicator', 'detailed_analysis', 'summary'}
        for analysis_type in ['strength', 'weakness']:
            if not isinstance(data[analysis_type], dict):
                logger.warning(
                    f"[Validation] Invalid {analysis_type} structure")
                return False
            if not all(field in data[analysis_type] for field in required_analysis_fields):
                logger.warning(
                    f"[Validation] Missing required fields in {analysis_type}")
                return False

        # insight 구조 검증
        if not isinstance(data['insight'], dict) or 'summary' not in data['insight']:
            logger.warning("[Validation] Invalid insight structure")
            return False

        return True

    except Exception as e:
        logger.error(f"[Validation] Error during cache validation: {str(e)}")
        return False
