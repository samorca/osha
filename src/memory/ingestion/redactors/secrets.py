from typing import Any


SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
}


def redact_sensitive_data(
    record: dict[str, Any],
) -> dict[str, Any]:

    def redact(value: Any) -> Any:

        if isinstance(value, dict):
            result = {}

            for key, item in value.items():

                normalized_key = str(key).lower().replace("-", "_")
                if (
                    normalized_key in SENSITIVE_KEYS
                    or normalized_key.endswith("_secret")
                    or normalized_key.endswith("_password")
                    or normalized_key.endswith("_token")
                    or normalized_key.endswith("_api_key")
                ):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = redact(item)

            return result

        if isinstance(value, list):
            return [redact(item) for item in value]

        return value

    return redact(record)