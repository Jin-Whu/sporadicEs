#!/usr/bin/env python
# coding:utf-8

import os
import requests
import astropy.time as Time
import numpy as np
import pandas as pd
from ftplib import FTP
from collections import namedtuple

PI = np.pi
D2R = PI / 180.
R2D = 180. / PI
AS2R = D2R / 3600.

RE_WGS84 = 6378137.  # earth semimajor axis (WGS84) (m)
FE_WGS84 = 1. / 298.257223563  # earth flattening (WGS84)

J2K = Time.Time('2000-01-01 12:00:00', scale='tt')
GPST0 = Time.Time('1980-01-06 00:00:00', scale='utc')

EOP = namedtuple('EOP', ['XP', 'YP', 'UT1_UTC', 'LOD'])  # rad, rad, sec, sec


class Geo(object):
    def __init__(self):
        self.__tutc = GPST0
        self.__eci2ecft = np.matrix('0 0 0; 0 0 0; 0 0 0')

        # read eop
        eoppath = os.path.join(os.path.dirname(__file__), 'finals2000A.all')
        if not os.path.exists(eoppath):
            self.downloadeop(eoppath)
        self.__eoptable = self.readeop(eoppath)

        # read nutation table
        nuttab = os.path.join(os.path.dirname(__file__), 'nut_IAU1980.dat')
        if not os.path.exists(nuttab):
            self.downloadnut(nuttab)
        self.__nuttab = self.readnutation(nuttab)

    def eci2geo(self, eci, sec, flag=False):
        """Convert eci position to geodetic posiiton.

        Args:
            eci: eci position {x, y, z} (m), type:np.array.
            sec: gps seconds (gpsweek * 7 * 86400 + time of week).
            flag: geo position lal lon unit. True: degree; False: radian

        Returns:
            geo: geodetic position {lat, lon, h} (rad|deg, m), type:np.ndarray.
        """
        return self.ecef2geo(self.eci2ecef(eci, sec), flag)

    def eci2ecef(self, eci, sec):
        """Convert eci position to ecef position.

        Args:
            eci: eci position {x, y, z} (m), type:np.ndarray.
            sec: gps seconds (gpsweek * 7 * 86400 + time of week).

        Returns:
            ecef: ecef position {x, y, z} (m), type:np.array.
        """
        tutc = GPST0 + Time.TimeDelta(sec, format='sec')
        if (tutc - self.__tutc).value < 0.01:
            ecef = self.__eci2ecft * np.array([eci]).T
            ecef = np.squeeze(np.asarray(ecef))
            return ecef
        self.__tutc = tutc
        t = (tutc - J2K).value / 36525.0
        P = self.precession(t)
        N = self.nutation(t)
        R = self.gastr(tutc)
        W = self.poler(tutc)
        U = W * R * N * P
        ecef = U.T * np.array([eci]).T
        ecef = np.squeeze(np.asarray(ecef))
        self.__eci2ecft = U.T
        return ecef

    def ecef2geo(self, ecef, flag=False):
        """Convert ecef position to geodetic position.

        Args:
           ecef: ecef position {x, y, z} (m), type:np.array.
           flag: geo position lal lon unit. True: degree; False: radian

        Returns:
           geo: geodetic position {lat, lon, h} (rad|deg, m), type:np.array.
        """
        e2 = FE_WGS84 * (2. - FE_WGS84)
        r2 = np.dot(ecef[:2], ecef[:2])
        z = ecef[2]
        zk = 0.
        v = RE_WGS84
        sinp = 0.
        while abs(z - zk) >= 1E-4:
            zk = z
            sinp = z / np.sqrt(r2 + z * z)
            v = RE_WGS84 / np.sqrt(1. - e2 * sinp * sinp)
            z = ecef[2] + v * e2 * sinp
        pos = np.array([0., 0., 0.])
        if r2 > 1E-12:
            pos[0] = np.arctan(z / np.sqrt(r2))
            pos[1] = np.arctan2(ecef[1], ecef[2])
        else:
            pos[0] = 90 if ecef[2] > 0 else -90
            pos[1] = 0
        pos[2] = np.sqrt(r2 + z * z) - v
        if flag:
            pos[0] *= R2D
            pos[1] *= R2D
        return pos

    def precession(self, t):
        """Precession matrix.

        Args:
            t: julian epoch from J2000.

        Returns:
            R: precession matrix.
        """
        # iau 1976 precession
        ze = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * AS2R
        th = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * AS2R
        z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * AS2R
        R1 = self.rz(-z)
        R2 = self.ry(th)
        R3 = self.rz(-ze)
        R = R1 * R2 * R3
        return R

    def nutation(self, t):
        """nutation matrix.

        Args:
           t: julian epoch from J2000.

        Returns:
            R: nutation matrix.
        """
        # iau 1980 nutation
        f = self.astargs(t)
        t2 = t * t
        t3 = t2 * t
        eps = (84381.448 - 46.8150 * t - 0.00059 * t2 + 0.001813 * t3) * AS2R
        dpsi = 0.
        deps = 0.
        for i in range(106):
            ang = 0.
            for j in range(5):
                ang += self.__nuttab.iloc[i, j] * f[j]
            dpsi += (self.__nuttab.iloc[i, 6] + self.__nuttab.iloc[i, 7] * t
                     ) * np.sin(ang)
            deps += (self.__nuttab.iloc[i, 8] + self.__nuttab.iloc[i, 9] * t
                     ) * np.cos(ang)
        dpsi *= 1E-4 * AS2R
        deps *= 1E-4 * AS2R
        R1 = self.rx(-eps - deps)
        R2 = self.rz(-dpsi)
        R3 = self.rx(eps)
        R = R1 * R2 * R3
        return R

    def gastr(self, t):
        """GAST rotation matrix.

        Apparent gcrs to apparent ecef at t.

        Args:
            t: utc time.

        Returns:
            R: gast rotation matrix.
        """
        eop = self.geteop(self.__eoptable, t.tt.mjd)
        t.delta_ut1_utc = eop.UT1_UTC
        gast = t.sidereal_time('apparent', 'greenwich').value * np.pi / 12
        R = self.rz(gast)
        return R

    def poler(self, t):
        """pole rotation matrix.

        Apparent ecef to ITRS.

        Args:
            t: utc time.

        Returns:
            R: pole rotation matrix.
        """
        eop = self.geteop(self.__eoptable, t.tt.mjd)
        R1 = self.ry(-eop.XP)
        R2 = self.rx(-eop.YP)
        R = R1 * R2
        return R

    def rx(self, theta):
        """X-axis coordinate rotation matrix.

        Args:
            theta: X-axis rotation angle (rad).

        Returns:
            matrix: X-axis rotation matrix, type:np.matrix.
        """
        matrix = np.matrix([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
        return matrix

    def ry(self, theta):
        """Y-axis coordinate rotation matrix.

        Args:
            theta: Y-axis rotation angle (rad).

        Returns:
            matrix: Y-axis rotation matrix, type:np.matrix.
        """
        matrix = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                            [-np.sin(theta), 0,
                             np.cos(theta)]])
        return matrix

    def rz(self, theta):
        """Z-axis coordinates rotation matrix.

        Args:
            theta: Z-axis rotaion angle (rad).

        Returns:
            matrix: Z-axis rotation matrix, type:np.matrix.
        """
        matrix = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        return matrix

    def astargs(self, t):
        """Get astronomical arguments.

        Args:
            t: Julian epoch.

        Returns:
            f: astronomical arguments {1, 1, F, D, OMG} (rad), type:np.ndarray.
        """
        fc = np.matrix(
            [[134.96340251, 1717915923.2178, 31.8792, 0.051635, -0.00024470], [
                357.52910918, 129596581.0481, -0.5532, 0.000136, -0.00001149
            ], [93.27209062, 1739527262.8478, -12.7512, -0.001037, 0.00000417],
             [297.85019547, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
             [125.04455501, -6962890.2665, 7.4722, 0.007702, -0.00005939]])
        f = np.zeros(5)
        tt = np.zeros(4)
        tt[0] = t
        for i in range(1, 4):
            tt[i] = tt[i - 1] * t
        for i in range(5):
            f[i] = fc.item(i, 0) * 3600
            for j in range(4):
                f[i] += fc.item(i, j + 1) * tt[j]
            f[i] = np.fmod(f[i] * AS2R, 2 * PI)
        return f

    def downloadeop(self, eoppath):
        """Download EOP finals2000A

        Args:
            eoppath: EOP filepath.

        Returns:
            None.
        """
        f = open(eoppath, 'w')
        ftp = FTP('cddis.gsfc.nasa.gov')
        ftp.login()
        ftp.cwd('/pub/products/iers')
        ftp.retrlines('RETR finals2000A.all',
                      lambda line: f.write('%s\n' % line))
        ftp.quit()
        f.close()

    def readeop(self, eoppath):
        """Read EOP.

        Args:
            eoppath: EOP filepath.

        Returns:
            eoptable: EOP-dict(t, EOP).
        """
        eoptable = dict()
        with open(eoppath) as f:
            for line in f:
                try:
                    jd = float(line[7:15])
                    px = float(line[17:27]) * AS2R
                    py = float(line[36:46]) * AS2R
                    ut1_utc = float(line[58:68])
                    lod = float(line[78:86]) * 1E-3
                    eop = EOP(px, py, ut1_utc, lod)
                    t = Time.Time(jd, format='mjd', scale='tt')
                    eoptable[t.value] = eop
                except ValueError:
                    pass
        return eoptable

    def geteop(self, eoptab, t):
        """Get eop at mjd(tt).

        Args:
            eoptab: eop table.
            t: mjd(tt).

        Returns:
           eop: eop, type:EOP.
        """
        mjd = Time.Time(int(t), format='mjd', scale='tt')
        eop = eoptab[mjd.value]
        return eop

    def downloadnut(self, nuttab):
        """Download nutation table.

        Args:
            nuttab: nutation table file path.

        Returns:
            None.
        """
        r = requests.get(
            'http://hpiers.obspm.fr/eop-pc/models/nutations/nut_IAU1980.dat')
        with open(nuttab, 'w') as f:
            f.write(r.content)

    def readnutation(self, nuttab):
        """Read nutation table.

        Args:
            nuttab: nutation tabel file path.

        Returns:
            table: nutation table, type:pandas.DataFrame.
        """
        table = pd.read_table(
            nuttab, skiprows=3, header=None, delim_whitespace=True)
        return table
