"""Inicializacion idempotente de datos semilla (docente admin, casetas, tarifas y >=20 vehiculos)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Alerta, Caseta, Paso, Tarifa, Usuario, Vehiculo

VEHICULOS_SEED = [
    ("2481KLM", "Carlos Mamani Quispe", "auto", 2, 120.0, True),
    ("4120XYZ", "Transporte Andino SRL", "camion", 3, 250.0, True),
    ("1895ABC", "Sindicato 1ro de Mayo", "minibus", 2, 85.0, True),
    ("3502DEF", "Flota Bolivar", "bus", 3, 300.0, True),
    ("0942MOT", "Luis Fernando Rojas", "moto", 2, 40.0, True),
    ("5118GHI", "Ana Maria Flores", "auto", 2, 0.0, True),       # Sin saldo para pruebas de alerta
    ("6274JKL", "Constructora Altiplano", "camion", 4, 420.0, True),
    ("7390MNO", "Pedro Pablo Vargas", "auto", 2, 65.0, True),
    ("8411PQR", "Trans Copacabana", "bus", 2, 190.0, True),
    ("9520STU", "Cooperativa Murillo", "minibus", 2, 55.0, True),
    ("1104VWX", "Diego Alejandro Torrez", "moto", 2, 30.0, True),
    ("2236YZA", "Logistica Sur SA", "camion", 5, 500.0, True),
    ("3347BCD", "Valeria Mendez", "auto", 2, 95.0, True),
    ("4458EFG", "Transportes Illimani", "camion", 2, 160.0, True),
    ("5569HIJ", "Sindicato Eduardo Avaroa", "minibus", 2, 70.0, True),
    ("6670KMN", "Flota El Dorado", "bus", 3, 280.0, True),
    ("7781OPQ", "Jorge Luis Choque", "moto", 2, 25.0, True),
    ("8892RST", "Distribuidora Tunari", "camion", 3, 18.0, True),  # Saldo bajo para camion 3 ejes (25 Bs)
    ("9903UVW", "Sofia Condori", "auto", 2, 110.0, True),
    ("1014XAB", "Expreso Cochabamba", "bus", 2, 210.0, True),
    ("2125CDE", "Minibuses El Alto", "minibus", 2, 60.0, True),
    ("3236FGH", "Carga Pesada Oruro", "camion", 6, 600.0, True),
    ("4347IJK", "Camila Navarro", "auto", 2, 45.0, False),          # Vehiculo inactivo
    ("5458LMN", "Mototaxi Express", "moto", 2, 50.0, True),
]

TARIFAS_SEED = [
    ("moto", 2, 3.0),
    ("auto", 2, 5.0),
    ("minibus", 2, 8.0),
    ("bus", 2, 15.0),
    ("bus", 3, 20.0),
    ("camion", 2, 18.0),
    ("camion", 3, 25.0),
    ("camion", 4, 35.0),
    ("camion", 5, 45.0),
    ("camion", 6, 55.0),
]

CASETAS_SEED = [
    ("Caseta Central", "Autopista La Paz - El Alto, Carril 1", True),
    ("Caseta Norte", "Carretera a Copacabana, Km 12", True),
    ("Caseta Sur", "Salida Carretera a Oruro, Km 8", True),
]


def seed_database(db: Session) -> None:
    settings = get_settings()

    # 1. Usuarios administradores obligatorios
    nombres_conocidos = {
        "doc.alvaro.pinto.ma@unifranz.edu.bo": "Alvaro Pinto Mamani",
        "cristopherrodriguez1510@gmail.com": "Cristopher Diego Rodriguez Ayala",
    }
    for email in settings.admin_emails:
        user = db.scalar(select(Usuario).where(Usuario.email_google == email))
        nombre_default = nombres_conocidos.get(email) or email.split("@")[0].replace(".", " ").title()
        if user is None:
            db.add(Usuario(email_google=email, nombre=nombre_default, rol="admin", activo=True))
        else:
            user.rol = "admin"
            user.activo = True
            if email in nombres_conocidos:
                user.nombre = nombres_conocidos[email]

    # 2. Casetas
    for nombre, ubicacion, activa in CASETAS_SEED:
        if db.scalar(select(Caseta).where(Caseta.nombre == nombre)) is None:
            db.add(Caseta(nombre=nombre, ubicacion=ubicacion, activa=activa))

    # 3. Tarifas
    for cat, ejes, monto in TARIFAS_SEED:
        if db.scalar(select(Tarifa).where(Tarifa.categoria == cat, Tarifa.ejes == ejes)) is None:
            db.add(Tarifa(categoria=cat, ejes=ejes, monto=monto))

    # 4. Vehiculos (>= 20)
    for placa, prop, cat, ejes, saldo, activo in VEHICULOS_SEED:
        if db.scalar(select(Vehiculo).where(Vehiculo.placa == placa)) is None:
            db.add(Vehiculo(placa=placa, propietario=prop, categoria=cat, ejes=ejes, saldo=saldo, activo=activo))

    db.commit()

    # 5. Pasos y alerta de ejemplo iniciales (solo si la tabla pasos esta vacia)
    if db.scalar(select(Paso.id).limit(1)) is None:
        caseta = db.scalar(select(Caseta).order_by(Caseta.id))
        admin = db.scalar(select(Usuario).where(Usuario.rol == "admin").order_by(Usuario.id))
        if caseta is not None:
            ahora = datetime.now(timezone.utc)
            pasos_iniciales = [
                Paso(
                    placa_leida="2481-KLM",
                    placa_normalizada="2481KLM",
                    confianza_ocr=0.94,
                    categoria_detectada="auto",
                    ejes_detectados=2,
                    monto=5.0,
                    caseta_id=caseta.id,
                    usuario_id=admin.id if admin else None,
                    estado="AUTORIZADO",
                    latencia_ms=92.5,
                    created_at=ahora - timedelta(minutes=35),
                ),
                Paso(
                    placa_leida="4120-XYZ",
                    placa_normalizada="4120XYZ",
                    confianza_ocr=0.91,
                    categoria_detectada="camion",
                    ejes_detectados=3,
                    monto=25.0,
                    caseta_id=caseta.id,
                    usuario_id=admin.id if admin else None,
                    estado="AUTORIZADO",
                    latencia_ms=108.2,
                    created_at=ahora - timedelta(minutes=20),
                ),
                Paso(
                    placa_leida="8888-EVD",
                    placa_normalizada="8888EVD",
                    confianza_ocr=0.89,
                    categoria_detectada="auto",
                    ejes_detectados=2,
                    monto=5.0,
                    caseta_id=caseta.id,
                    usuario_id=admin.id if admin else None,
                    estado="NO_REGISTRADA",
                    latencia_ms=97.0,
                    created_at=ahora - timedelta(minutes=10),
                ),
            ]
            db.add_all(pasos_iniciales)
            db.flush()
            db.add(
                Alerta(
                    paso_id=pasos_iniciales[2].id,
                    tipo="PLACA_NO_REGISTRADA",
                    mensaje="Placa 8888EVD detectada sin registro en el sistema (posible evasion).",
                    atendida=False,
                    created_at=ahora - timedelta(minutes=10),
                )
            )
            db.commit()
