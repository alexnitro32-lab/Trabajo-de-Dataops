# DIAGNOSTICO DESCRIPTIVO DEL RITMO DE INGRESO
#
# Antes de modelar hay que saber cada cuanto vuelve la gente. Son TRES medidas
# distintas, y son distintas de verdad: no se pueden promediar juntas porque
# describen tres momentos del ciclo de vida del cliente.
#
#   1. El vehiculo NUEVO que acabamos de entregar: cuanto tarda en aparecer por
#      primera vez, y con cuanto kilometraje llega.
#   2. Ese mismo vehiculo, ya con historia: cada cuanto vuelve de ahi en adelante.
#   3. El vehiculo EXTERNO (no lo vendimos nosotros) que ya volvio una segunda
#      vez: cada cuanto vuelve.
#
# Este modulo NO entrena nada. Solo mide, y deja las cifras en un diccionario
# para que train.py las imprima y las guarde junto con las metricas del modelo.

import pandas as pd
from src import config


def _resumen(serie: pd.Series) -> dict:
    """Estadisticos de una distribucion muy asimetrica.

    Se reporta la MEDIANA antes que la media a proposito: los intervalos entre
    visitas tienen una cola larguisima (hay retornos de mas de 2.000 dias), y en
    esa forma la media queda arrastrada por unos pocos casos extremos. La mediana
    describe al cliente del medio, que es de quien habla el negocio.
    """
    serie = serie.dropna()
    if serie.empty:
        return {"n": 0}
    return {
        "n": int(serie.size),
        "p10": float(serie.quantile(0.10)),
        "p25": float(serie.quantile(0.25)),
        "mediana": float(serie.median()),
        "media": float(serie.mean()),
        "p75": float(serie.quantile(0.75)),
        "p90": float(serie.quantile(0.90)),
    }


def diagnosticar_ritmos(df_con_historia: pd.DataFrame) -> dict:
    """Calcula las tres medidas de ritmo sobre las visitas ya enriquecidas."""
    df = df_con_historia
    col_vin = config.COLUMNA_GRUPO
    vendidos = df[df["Es_Vehiculo_Vendido"] == 1]
    externos = df[df["Es_Vehiculo_Vendido"] == 0]

    # --- MEDIDA 1: de la entrega al primer ingreso real ---------------------
    # La visita 1 de un vehiculo vendido por nosotros es el ALISTAMIENTO (la
    # entrega). El primer ingreso de verdad es la visita 2, y su
    # Dias_Desde_Visita_Ant es justamente el tiempo que tardo en aparecer.
    primer_ingreso = vendidos[vendidos["Num_Visita"] == 2]
    total_vendidos = int(vendidos[col_vin].nunique())
    volvieron = int(primer_ingreso[col_vin].nunique())

    dias_1 = primer_ingreso["Dias_Desde_Visita_Ant"]
    km_1 = primer_ingreso["Km_Desde_Visita_Ant"]

    medida_1 = {
        "descripcion": "Vehiculo nuevo: de la entrega al primer ingreso a taller",
        "vehiculos_entregados": total_vendidos,
        "volvieron_alguna_vez": volvieron,
        "nunca_volvieron": total_vendidos - volvieron,
        "tasa_de_regreso_pct": round(100 * volvieron / total_vendidos, 1) if total_vendidos else 0.0,
        "dias": _resumen(dias_1),
        "km_recorridos": _resumen(km_1),
        # Cual de las dos mitades de la regla se cumple primero en la practica.
        "llegan_antes_de_365_dias_pct": round(100 * (dias_1 <= config.REGLA_MANTENIMIENTO_DIAS).mean(), 1),
        "llegan_antes_de_10000_km_pct": round(100 * (km_1 <= config.REGLA_MANTENIMIENTO_KMS).mean(), 1),
        "manda_el_kilometraje": int(
            (km_1 / config.REGLA_MANTENIMIENTO_KMS > dias_1 / config.REGLA_MANTENIMIENTO_DIAS).sum()
        ),
        "manda_el_calendario": int(
            (km_1 / config.REGLA_MANTENIMIENTO_KMS <= dias_1 / config.REGLA_MANTENIMIENTO_DIAS).sum()
        ),
    }

    # --- MEDIDAS 2 y 3: ritmo de quien ya tiene historia --------------------
    def _ritmo(sub: pd.DataFrame, descripcion: str) -> dict:
        con_target = sub[(sub["Num_Visita"] >= 2) & sub[config.TARGET].notna() & (sub[config.TARGET] >= 0)]
        return {
            "descripcion": descripcion,
            "vehiculos": int(con_target[col_vin].nunique()),
            "intervalos_observados": int(len(con_target)),
            "dias": _resumen(con_target[config.TARGET]),
            "km_recorridos": _resumen(con_target["Km_Desde_Visita_Ant"]),
            "vuelve_dentro_de_365_dias_pct": round(
                100 * (con_target[config.TARGET] <= config.REGLA_MANTENIMIENTO_DIAS).mean(), 1
            ) if len(con_target) else 0.0,
        }

    medida_2 = _ritmo(vendidos, "Vendido por nosotros, ya con historia: cada cuanto vuelve")
    medida_3 = _ritmo(externos, "Externo que volvio una 2a vez: cada cuanto vuelve")

    # --- Cuantos vehiculos de cada origen son irrecuperables para el modelo --
    cobertura = {}
    for valor, nombre in ((1, "vendidos_por_nosotros"), (0, "externos")):
        sub = df[df["Es_Vehiculo_Vendido"] == valor]
        total = int(sub[col_vin].nunique())
        if total == 0:
            continue
        una_sola = int((sub.groupby(col_vin).size() == 1).sum())
        cobertura[nombre] = {
            "vehiculos": total,
            "con_una_sola_visita": una_sola,
            "con_una_sola_visita_pct": round(100 * una_sola / total, 1),
            "predecibles": total - una_sola,
        }

    return {
        "medida_1_primer_ingreso": medida_1,
        "medida_2_vendidos_con_historia": medida_2,
        "medida_3_externos_con_historia": medida_3,
        "cobertura_por_origen": cobertura,
        "demora_en_taller_dias": _resumen(df["Dias_En_Taller"]),
    }


def imprimir_diagnostico(d: dict) -> None:
    """Deja el diagnostico legible en consola, con la lectura de negocio."""
    m1, m2, m3 = d["medida_1_primer_ingreso"], d["medida_2_vendidos_con_historia"], d["medida_3_externos_con_historia"]

    print("   MEDIDA 1 - Del alistamiento al primer ingreso (vehiculos que vendimos)")
    print(f"      Entregados: {m1['vehiculos_entregados']}  |  volvieron alguna vez: {m1['volvieron_alguna_vez']} "
          f"({m1['tasa_de_regreso_pct']}%)  |  nunca volvieron: {m1['nunca_volvieron']}")
    print(f"      Dias hasta el primer ingreso : mediana {m1['dias']['mediana']:.0f}  "
          f"(p25 {m1['dias']['p25']:.0f} - p75 {m1['dias']['p75']:.0f})")
    print(f"      Km recorridos hasta entonces : mediana {m1['km_recorridos']['mediana']:.0f}  "
          f"(p25 {m1['km_recorridos']['p25']:.0f} - p75 {m1['km_recorridos']['p75']:.0f})")
    print(f"      Llegan antes de 365 dias: {m1['llegan_antes_de_365_dias_pct']}%  |  "
          f"antes de 10.000 km: {m1['llegan_antes_de_10000_km_pct']}%")
    print(f"      Quien manda primero -> kilometraje: {m1['manda_el_kilometraje']}  "
          f"calendario: {m1['manda_el_calendario']}")

    for etiqueta, m in (("MEDIDA 2 - Vendidos por nosotros, ya con historia", m2),
                        ("MEDIDA 3 - Externos que volvieron una 2a vez", m3)):
        print(f"\n   {etiqueta}")
        print(f"      Vehiculos: {m['vehiculos']}  |  intervalos observados: {m['intervalos_observados']}")
        print(f"      Dias entre ingresos : mediana {m['dias']['mediana']:.0f}  "
              f"(p25 {m['dias']['p25']:.0f} - p75 {m['dias']['p75']:.0f})  media {m['dias']['media']:.0f}")
        print(f"      Km entre ingresos   : mediana {m['km_recorridos']['mediana']:.0f}")
        print(f"      Vuelve dentro de 365 dias: {m['vuelve_dentro_de_365_dias_pct']}%")

    print("\n   COBERTURA - cuantos vehiculos son predecibles (necesitan 2+ ingresos)")
    for nombre, c in d["cobertura_por_origen"].items():
        print(f"      {nombre:<22} {c['vehiculos']:>5} vehiculos  |  "
              f"{c['con_una_sola_visita']:>5} con una sola visita ({c['con_una_sola_visita_pct']}%)  |  "
              f"predecibles: {c['predecibles']}")

    dem = d["demora_en_taller_dias"]
    print(f"\n   DEMORA EN TALLER (F. Cierre - F. Entrada): mediana {dem['mediana']:.0f} dias  "
          f"| p75 {dem['p75']:.0f}  | p90 {dem['p90']:.0f}")
