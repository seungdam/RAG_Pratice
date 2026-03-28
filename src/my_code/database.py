from __future__ import annotations

# rag.py에서 MetadataRepository를 분리한 파일
# 기존 로직 100% 유지 + try_build_metadata_repo 팩토리 함수 추가

from typing import Optional, Sequence

from langchain_core.documents import Document

if __package__:
    from . import structure
else:
    import structure


class MetadataRepository:  # Postgres Manager
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        try:
            import psycopg2
        except ImportError as exc:
            raise ImportError("Install `psycopg2-binary` to use Postgres.") from exc
        return psycopg2.connect(self.dsn)

    def initialize_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            chunk_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            section TEXT NOT NULL,
            content TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts
        ON rag_chunks USING GIN (to_tsvector('simple', content));
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(ddl)
            conn.commit()

    def upsert_documents(self, docs: Sequence[Document]) -> None:
        if not docs:
            return

        sql = """
        INSERT INTO rag_chunks (chunk_id, source, page, section, content)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id)
        DO UPDATE SET
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            section = EXCLUDED.section,
            content = EXCLUDED.content;
        """
        rows = [
            (
                str(doc.metadata.get("chunk_id")),
                str(doc.metadata.get("source", "unknown")),
                int(doc.metadata.get("page", -1)),
                str(doc.metadata.get("section", "unknown")),
                doc.page_content,
            )
            for doc in docs
        ]
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, rows)
            conn.commit()

    def keyword_search(self, query: str, k: int = 6) -> list[structure.RetrievalResult]:
        sql = """
        SELECT
            chunk_id,
            source,
            page,
            section,
            content,
            ts_rank_cd(
                to_tsvector('simple', content),
                plainto_tsquery('simple', %s)
            ) AS score
        FROM rag_chunks
        WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s;
        """
        results: list[structure.RetrievalResult] = []
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (query, query, k))
                for chunk_id, source, page, section, content, score in cursor.fetchall():
                    doc = Document(
                        page_content=content,
                        metadata={
                            "chunk_id": chunk_id,
                            "source":   source,
                            "page":     page,
                            "section":  section,
                        },
                    )
                    results.append(
                        structure.RetrievalResult(
                            document=doc,
                            score=float(score),
                            retriever="sparse",
                        )
                    )
        return results


def try_build_metadata_repo(dsn: str | None) -> Optional[MetadataRepository]:
    """
    DSN이 없거나 DB 연결에 실패해도 None을 반환하는 안전한 팩토리.
    None 반환 시 HybridRetriever가 자동으로 Dense 전용 모드로 동작합니다.

    기존 코드에서는 MetadataRepository(dsn)을 직접 호출했는데,
    DSN이 없거나 DB가 꺼져 있으면 즉시 에러가 났습니다.
    이 함수를 쓰면 DB 없어도 main.py가 그냥 실행됩니다.
    """
    if not dsn:
        print("DB DSN 없음 → Dense 전용 모드로 실행합니다.")
        return None
    try:
        repo = MetadataRepository(dsn)
        repo._connect().close()  # 연결 가능 여부만 확인
        print("PostgreSQL 연결 성공")
        return repo
    except Exception as e:
        print(f"PostgreSQL 연결 실패 ({e}) → Dense 전용 모드로 실행합니다.")
        return None
