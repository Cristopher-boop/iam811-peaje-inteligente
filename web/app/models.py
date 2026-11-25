"""Modelos relacionales del sistema de peaje."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Usuario(Base):
    __tablename__ = "usuarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email_google: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    nombre: Mapped[str] = mapped_column(String(150), nullable=False)
    rol: Mapped[str] = mapped_column(String(30), nullable=False, default="operador")  # admin | operador
    activo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Vehiculo(Base):
    __tablename__ = "vehiculos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    placa: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    propietario: Mapped[str] = mapped_column(String(150), nullable=False)
    categoria: Mapped[str] = mapped_column(String(30), nullable=False)  # moto, auto, minibus, camion, bus
    ejes: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    saldo: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    activo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Caseta(Base):
    __tablename__ = "casetas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    nombre: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    ubicacion: Mapped[str] = mapped_column(String(200), nullable=False)
    activa: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Tarifa(Base):
    __tablename__ = "tarifas"
    __table_args__ = (UniqueConstraint("categoria", "ejes", name="uq_tarifa_categoria_ejes"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    categoria: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    ejes: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    monto: Mapped[float] = mapped_column(Float, nullable=False)


class Evidencia(Base):
    __tablename__ = "evidencias"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    contenido: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(60), nullable=False, default="image/jpeg")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Paso(Base):
    __tablename__ = "pasos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    placa_leida: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    placa_normalizada: Mapped[str] = mapped_column(String(20), nullable=False, default="", index=True)
    confianza_ocr: Mapped[float | None] = mapped_column(Float, nullable=True)
    categoria_detectada: Mapped[str | None] = mapped_column(String(30), nullable=True)
    ejes_detectados: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monto: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    caseta_id: Mapped[int] = mapped_column(ForeignKey("casetas.id"), nullable=False)
    usuario_id: Mapped[int | None] = mapped_column(ForeignKey("usuarios.id"), nullable=True)
    imagen_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    estado: Mapped[str] = mapped_column(String(40), nullable=False)
    latencia_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    caseta: Mapped[Caseta] = relationship("Caseta")
    usuario: Mapped[Usuario | None] = relationship("Usuario")


class Alerta(Base):
    __tablename__ = "alertas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    paso_id: Mapped[int] = mapped_column(ForeignKey("pasos.id"), nullable=False)
    tipo: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    mensaje: Mapped[str] = mapped_column(Text, nullable=False, default="")
    atendida: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    paso: Mapped[Paso] = relationship("Paso")
