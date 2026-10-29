# lab2_utils.py
import math
import numpy as np

def normaliza_angulo(a):
    """Leva o ângulo para o intervalo (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))

def atualiza_odometria(x, y, theta, dphi_e, dphi_d, r, L):
    """Integra um passo de odometria. Retorna a nova pose (x, y, theta)."""
    d_e = r * dphi_e
    d_d = r * dphi_d
    ds = (d_d + d_e) / 2
    dtheta = (d_d - d_e) / L
    x += ds * math.cos(theta + dtheta / 2)
    y += ds * math.sin(theta + dtheta / 2)
    theta = normaliza_angulo(theta + dtheta)
    return x, y, theta

def scan_para_mundo(ranges, angle_min, angle_increment, pose,
                    range_min=0.12, range_max=3.5, passo=1):
    """Converte um scan em pontos (N x 2) no referencial global.
    - descarte leituras inválidas (inf, nan, fora de [range_min, range_max])
    - use apenas 1 a cada 'passo' feixes
    - pose = (x, y, theta) do robô no mundo
    """
    indices = np.arange(len(ranges))[::passo]
    distancias = np.asarray(ranges, dtype=float)[::passo]
    validos = (np.isfinite(distancias)
               & (distancias >= range_min)
               & (distancias <= range_max))
    distancias = distancias[validos]
    angulos = angle_min + indices[validos] * angle_increment
    x, y, theta = pose
    return np.column_stack((x + distancias * np.cos(theta + angulos),
                            y + distancias * np.sin(theta + angulos)))
