from abc import ABC, abstractmethod


class EncryptionProvider(ABC):
    @abstractmethod
    def encrypt(self, plaintext: bytes) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def decrypt(self, ciphertext: bytes) -> bytes:
        raise NotImplementedError


class CryptographyUnavailableError(RuntimeError):
    pass


try:
    from cryptography.fernet import Fernet

    class FernetEncryptionProvider(EncryptionProvider):
        def __init__(self, key: bytes | None = None) -> None:
            self._fernet = Fernet(key or Fernet.generate_key())

        def encrypt(self, plaintext: bytes) -> bytes:
            if not isinstance(plaintext, bytes):
                raise TypeError("plaintext must be bytes")
            return self._fernet.encrypt(plaintext)

        def decrypt(self, ciphertext: bytes) -> bytes:
            if not isinstance(ciphertext, bytes):
                raise TypeError("ciphertext must be bytes")
            return self._fernet.decrypt(ciphertext)
except ImportError:
    class FernetEncryptionProvider(EncryptionProvider):
        def __init__(self, key: bytes | None = None) -> None:
            raise CryptographyUnavailableError(
                "Install 'cryptography' to use encryption at rest"
            )

        def encrypt(self, plaintext: bytes) -> bytes:
            raise CryptographyUnavailableError(
                "Install 'cryptography' to use encryption at rest"
            )

        def decrypt(self, ciphertext: bytes) -> bytes:
            raise CryptographyUnavailableError(
                "Install 'cryptography' to use encryption at rest"
            )