"""Esquemas Pydantic para la API REST v1."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GoogleAuthRequest(BaseModel):
    id_token: str | None = Field(default=None, description="ID Token emitido por Google Sign-In")
    token: str | None = Field(default=None, description="Alias alternativo para id_token")


class PlacaConsultarRequest(BaseModel):
    placa: str = Field(..., min_length=2, max_length=20, examples=["2481-KLM"])


class AgenteChatRequest(BaseModel):
    mensaje: str = Field(default="", examples=["¿Cuál es el estado de la placa 2481-KLM?"])
    imagen_base64: str | None = Field(default=None, description="Imagen codificada en base64 (opcional)")


class VehiculoCreate(BaseModel):
    placa: str = Field(..., min_length=4, max_length=20)
    propietario: str = Field(..., min_length=2, max_length=150)
    categoria: str = Field(..., pattern="^(moto|auto|minibus|camion|bus)$")
    ejes: int = Field(default=2, ge=2, le=10)
    saldo: float = Field(default=50.0, ge=0.0)
    activo: bool = True


class VehiculoUpdate(BaseModel):
    propietario: str | None = None
    categoria: str | None = Field(default=None, pattern="^(moto|auto|minibus|camion|bus)$")
    ejes: int | None = Field(default=None, ge=2, le=10)
    saldo: float | None = Field(default=None, ge=0.0)
    activo: bool | None = None


class RecargaRequest(BaseModel):
    monto: float = Field(..., gt=0.0, le=50000.0)


class TarifaUpsert(BaseModel):
    categoria: str = Field(..., pattern="^(moto|auto|minibus|camion|bus)$")
    ejes: int = Field(..., ge=2, le=10)
    monto: float = Field(..., gt=0.0)


class CasetaCreate(BaseModel):
    nombre: str = Field(..., min_length=2, max_length=100)
    ubicacion: str = Field(..., min_length=2, max_length=200)
    activa: bool = True


class UsuarioCreate(BaseModel):
    email_google: str = Field(..., min_length=5, max_length=255)
    nombre: str = Field(..., min_length=2, max_length=150)
    rol: str = Field(default="operador", pattern="^(admin|operador)$")
    activo: bool = True


class UsuarioToggle(BaseModel):
    rol: str | None = Field(default=None, pattern="^(admin|operador)$")
    activo: bool | None = None
