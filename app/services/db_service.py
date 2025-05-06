import json
from typing import List, Dict, Any, Optional
import asyncpg
import os
from datetime import datetime


class DatabaseService:
    def __init__(self):
        self.pool = None
        self.config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", 5432),
            "database": os.getenv("DB_NAME", "address_ner_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
        }

    async def init_pool(self):
        """데이터베이스 연결 풀 초기화"""
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(**self.config)
                print("PostgreSQL 연결 풀 생성 완료")
                await self._init_db()
            except Exception as e:
                print(f"데이터베이스 연결 오류: {e}")
                # SQLite 대체 옵션을 제공할 수도 있음
                raise

    async def close_pool(self):
        """데이터베이스 연결 풀 종료"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            print("데이터베이스 연결 풀 종료")

    async def _init_db(self):
        """데이터베이스 초기화"""
        async with self.pool.acquire() as conn:
            # 학습 데이터 테이블
            await conn.execute(
                """
            CREATE TABLE IF NOT EXISTS training_data (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                address TEXT NOT NULL,
                is_valid BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # 모델 버전 테이블
            await conn.execute(
                """
            CREATE TABLE IF NOT EXISTS model_versions (
                id SERIAL PRIMARY KEY,
                version TEXT NOT NULL,
                metrics JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            print("데이터베이스 테이블 초기화 완료")

    async def add_training_data(self, text: str, address: str, is_valid: bool):
        """학습 데이터 추가"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO training_data (text, address, is_valid) 
                VALUES ($1, $2, $3)
                """,
                text,
                address,
                is_valid,
            )
            return True

    async def get_training_data(self, limit=1000):
        """학습 데이터 조회"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM training_data 
                ORDER BY created_at DESC 
                LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]

    async def add_model_version(self, version: str, metrics: Dict[str, float]):
        """모델 버전 정보 추가"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_versions (version, metrics) 
                VALUES ($1, $2)
                """,
                version,
                json.dumps(metrics),
            )
            return True

    async def get_model_versions(self, limit=10):
        """모델 버전 정보 조회"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM model_versions 
                ORDER BY created_at DESC 
                LIMIT $1
                """,
                limit,
            )

            versions = []
            for row in rows:
                row_dict = dict(row)
                row_dict["metrics"] = json.loads(row_dict["metrics"])
                versions.append(row_dict)

            return versions

    async def get_latest_model_version(self):
        """최신 모델 버전 정보 조회"""
        versions = await self.get_model_versions(limit=1)
        return versions[0] if versions else None

    async def bulk_add_training_data(self, data_list: List[Dict[str, Any]]):
        """다수의 학습 데이터 일괄 추가"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for data in data_list:
                    await conn.execute(
                        """
                        INSERT INTO training_data (text, address, is_valid) 
                        VALUES ($1, $2, $3)
                        """,
                        data["text"],
                        data["address"],
                        data["is_valid"],
                    )
            return True


# 전역 데이터베이스 서비스 인스턴스
db_service = DatabaseService()
