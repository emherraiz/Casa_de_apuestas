import numpy as np
import tensorflow as tf

from main.map import ApuestaSchema
from main.repositories.repositorioapuesta import ApuestaRepositorio
from main.repositories.repositoriocuota import CuotaRepositorio
from abc import ABC

import os

from main.services.avg_goals import AVG_GOALS
from main.services.perc_victories import PERC_VICT
from main.services.teams_code import TEAMS_CODE
from main.services.teams_map import TEAMS_MAPS
from main.services.matches_played import MATCHES_PLAYED

apuesta_schema = ApuestaSchema()
apuesta_repositorio = ApuestaRepositorio()
cuota_repositorio = CuotaRepositorio()


class ApuestaService:

    def agregar_apuesta(self, apuesta, local, visitante):
        cuota = cuota_repositorio.find_by_partido(apuesta)
        probabilidad = self.set_cuota(cuota, local, visitante)
        apuesta.ganancia = round(apuesta.monto * probabilidad, 2)
        return apuesta_repositorio.create(apuesta)

    def set_cuota(self, cuota, local, visitante):
        if local:
            cuota_local = CuotaLocal()
            probabilidad = cuota_local.calcular_cuota(cuota)
            return probabilidad
        if visitante:
            cuota_visitante = CuotaVisitante()
            probabilidad = cuota_visitante.calcular_cuota(cuota)
            return probabilidad
        cuota_empate = CuotaEmpate()
        probabilidad = cuota_empate.calcular_cuota(cuota)
        return probabilidad

    def obtener_apuesta_por_id(self, id):
        return apuesta_repositorio.find_one(id)

    def obtener_apuestas_ganadas(self):
        return apuesta_repositorio.find_wins()

    def obtener_apuestas(self):
        return apuesta_repositorio.find_all()


class CuotaStrategy(ABC):
    def calcular_cuota(self, cuota):
        """Calcular probabilidad"""

        model = tf.keras.models.load_model('main/services/my_model.h5')

        equipo_local = TEAMS_MAPS.get(cuota.partido.equipo_local.nombre, cuota.partido.equipo_local.nombre)
        equipo_visitante = TEAMS_MAPS.get(cuota.partido.equipo_visitante.nombre, cuota.partido.equipo_visitante.nombre)
        datos = np.array([
            TEAMS_CODE.get(equipo_local, 0),
            TEAMS_CODE.get(equipo_visitante, 0),
            PERC_VICT.get(equipo_local, 0),
            PERC_VICT.get(equipo_visitante, 0),
            AVG_GOALS.get(equipo_local, 0),
            AVG_GOALS.get(equipo_visitante, 0),
            MATCHES_PLAYED.get(equipo_local, 0),
            MATCHES_PLAYED.get(equipo_visitante, 0),
        ])

        cuota_visitante, cuota_local, cuota_empate = 1 / model.predict(datos.reshape(1, 8))[0]

        return cuota_visitante, cuota_local, cuota_empate

    def calcular_couta_final(self, couta):
        """
        Reduce la couta en un margen de 1.25 para que sea redituable
        """
        couta = couta / 1.25
        return couta


class CuotaLocal(CuotaStrategy):
    def calcular_cuota(self, cuota):
        cuota_visitante, cuota_local, cuota_empate = CuotaStrategy.calcular_cuota(self, cuota)
        return self.calcular_couta_final(cuota_local)


class CuotaVisitante(CuotaStrategy):
    def calcular_cuota(self, cuota):
        cuota_visitante, cuota_local, cuota_empate = CuotaStrategy.calcular_cuota(self, cuota)
        return self.calcular_couta_final(cuota_visitante)


class CuotaEmpate(CuotaStrategy):
    def calcular_cuota(self, cuota):
        cuota_visitante, cuota_local, cuota_empate = CuotaStrategy.calcular_cuota(self, cuota)
        return self.calcular_couta_final(cuota_empate)
