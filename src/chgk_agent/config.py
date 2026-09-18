"""Конфигурация приложения на pydantic-settings."""

from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

DEFAULT_USER_AGENT = "chgk-agent/0.1 (+https://localhost/chgk-agent)"
TEST_DATABASE_SUFFIX = "_test"


def derive_test_dsn(dsn: str) -> str:
    """Вывести адрес тестовой базы из адреса рабочей.

    Имя базы получает суффикс `_test`, остальные части адреса — сервер,
    порт, пользователь — сохраняются. Пароль возвращается как есть: строка
    попадает только в настройки тестового окружения.
    """

    url = make_url(dsn)
    if not url.database:
        raise ValueError("в DSN не указано имя базы")
    return url.set(database=f"{url.database}{TEST_DATABASE_SUFFIX}").render_as_string(
        hide_password=False
    )


class DatabaseSettings(BaseSettings):
    """Параметры подключения к PostgreSQL с pgvector."""

    dsn: str = "postgresql+asyncpg://prokofyev@localhost:5432/chgk_agent"
    test_dsn: str | None = None
    """Адрес базы для интеграционных тестов.

    По умолчанию выводится из `dsn` заменой имени базы на `<имя>_test`,
    чтобы тесты не могли писать в рабочую базу.
    """

    echo: bool = False

    @model_validator(mode="after")
    def _fill_test_dsn(self) -> "DatabaseSettings":
        """Подставить выведенный адрес тестовой базы, если он не задан."""

        if self.test_dsn is None:
            self.test_dsn = derive_test_dsn(self.dsn)
        return self


class GigaChatSettings(BaseSettings):
    """Параметры доступа к GigaChat."""

    credentials: SecretStr = SecretStr("")
    scope: str = "GIGACHAT_API_PERS"
    model: str = "GigaChat"
    embedding_model: str = "Embeddings"
    embedding_dim: int = Field(default=1024, ge=1)
    verify_ssl: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_concurrency: int = Field(default=4, ge=1)
    max_retries: int = Field(default=3, ge=0)


class ExternalSourceSettings(BaseSettings):
    """Параметры внешнего источника вопросов."""

    enabled: bool = True
    base_url: str = "https://gotquestions.online"
    user_agent: str = Field(default=DEFAULT_USER_AGENT, min_length=1)
    max_query_chars: int = Field(default=50, ge=1, le=50)
    page_limit: int = Field(default=20, ge=1)
    max_pages: int = Field(default=1, ge=1)
    min_interval_seconds: float = Field(default=1.0, ge=0)
    timeout_seconds: float = Field(default=15.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_reset_seconds: float = Field(default=60.0, gt=0)
    respect_robots: bool = False
    """Соблюдать ли `Disallow: /search` из robots.txt.

    У gotquestions.online страница поиска закрыта правилами сайта. Владелец
    сервиса явно принял решение работать с ней, ограничившись rate limiting,
    поэтому по умолчанию правило не применяется. Флаг оставлен, чтобы
    включить строгий режим без релиза.
    """


class SearchSettings(BaseSettings):
    """Параметры поиска."""

    top_k: int = Field(default=20, ge=1)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_query_chars: int = Field(default=3, ge=1)
    max_query_chars: int = Field(default=2000, ge=1)
    generate_answer: bool = True
    use_lexical: bool = True
    semantic_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    lexical_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    external_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    local_timeout_seconds: float = Field(default=10.0, gt=0)
    external_timeout_seconds: float = Field(default=15.0, gt=0)


class ObservabilitySettings(BaseSettings):
    """Параметры наблюдаемости."""

    log_level: str = "INFO"
    json_logs: bool = False


class Settings(BaseSettings):
    """Корневая конфигурация приложения."""

    model_config = SettingsConfigDict(
        env_prefix="CHGK_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    gigachat: GigaChatSettings = Field(default_factory=GigaChatSettings)
    external: ExternalSourceSettings = Field(default_factory=ExternalSourceSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Вернуть текущие настройки приложения."""

    return Settings()
