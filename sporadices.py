#!/usr/bin/env python
# coding:utf-8

import argparse
import os
import numpy as np
from netCDF4 import Dataset
import geo
if os.name != 'nt':
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa
import matplotlib.ticker as ticker  # noqa


KF1 = 1575.42
KF2 = 1227.60


def smooth(array, WSZ=5):
    """Smooth 1-D array using moving avarage.

    Args:
        array: 1-D array.
        WSZ: smoothing window size, which must be odd number

    Returns:
        A smoothed 1-D array.

    Raises:
        ValueError: raise when wsz is even number
    """
    out0 = np.convolve(array, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(array[:WSZ - 1])[::2] / r
    stop = (np.cumsum(array[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def caltanpoint(leopos, gpspos):
    """Calculate tangent point between leo and gps.

    Args:
        leopos: leo position {x, y, z} (m).
        gpspos: gps position {x, y, z} (m).

    Returns:
        tanpoint: tangent point {x, y, z} (m).
        isocc: bool flag for occultation.
    """
    leov = np.array(leopos)
    gpsv = np.array(gpspos)
    leo_gps = leov - gpsv
    leo_gps_unit = leo_gps / np.linalg.norm(leo_gps)
    tanpoint = gpsv - np.dot(leo_gps_unit, gpsv) * leo_gps_unit
    isocc = (np.dot(np.cross(leov, tanpoint), np.cross(leov, gpsv)) >= 0
             and np.dot(np.cross(gpsv, tanpoint), np.cross(gpsv, leov)) >= 0)
    return tanpoint, isocc


def sporadices(filepath, outdir):
    """Analysis sporadicEs from netCDF4 file.

    Args:
        filepath: filepath (filetype:netCDF4)
        outdir: result store path

    Returns:
        None
    """
    geotrans = geo.Geo()
    try:
        data = Dataset(filepath)
    except IOError:
        return
    data.set_auto_mask(False)
    starttime = data.startTime
    time = data.variables['time'][:]
    exl1 = data.variables['exL1'][:] * 1e-3
    exl2 = data.variables['exL2'][:] * 1e-3
    snr = data.variables['caL1Snr'][:] * 0.1
    leo_x = data.variables['xLeo'][:] * 1E3
    leo_y = data.variables['yLeo'][:] * 1E3
    leo_z = data.variables['zLeo'][:] * 1E3
    gps_x = data.variables['xGps'][:] * 1E3
    gps_y = data.variables['yGps'][:] * 1E3
    gps_z = data.variables['zGps'][:] * 1E3
    leo_pos = zip(leo_x, leo_y, leo_z)
    gps_pos = zip(gps_x, gps_y, gps_z)
    height = list()
    lat = list()
    lon = list()
    for leop, gpsp, t in zip(leo_pos, gps_pos, time):
        tanpoint, isocc = caltanpoint(leop, gpsp)
        t = starttime + t
        geopos = geotrans.eci2geo(tanpoint, t, True)
        height.append(geopos[2] / 1E3)
        lat.append(geopos[0])
        lon.append(geopos[1])

    height = np.array(height)
    lat = np.array(lat)
    lon = np.array(lon)

    exl1pert = (exl1 - smooth(exl1, 51)) * 1e5
    exl1pert = exl1pert - smooth(exl1pert, 51)
    exl2pert = (exl2 - smooth(exl2, 51)) * 1e5
    exl2pert = exl2pert - smooth(exl2pert, 51)
    snr0 = smooth(snr, 101)
    snrpert = (snr - snr0) / snr0

    # sporadicEs location
    es_index = np.argmax(abs(snrpert) > 0.5)  # sporadicEs
    if es_index == 0:
        return
    if height[es_index] < 90 or height[es_index] > 120:
        return
    es_lon = lon[es_index]
    es_lat = lat[es_index]

    # plot
    plt.style.use('classic')
    plt.figure(figsize=(12, 10))
    xloglocator = ticker.LogLocator()

    plt.subplot(2, 3, 1)
    plt.plot(snr, height)
    plt.ylim([0, 140])
    plt.xlabel('SNR')
    plt.ylabel('Height/km')
    plt.xticks(np.arange(0, 1800, 300))

    ax = plt.subplot(2, 3, 2)
    plt.plot(exl1, height)
    plt.ylim([0, 140])
    plt.xlim([1e-5, 1])
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(xloglocator)
    plt.xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    plt.xlabel('L1 Excess Phase/km')
    plt.ylabel('Height/km')

    ax = plt.subplot(2, 3, 3)
    plt.plot(exl2, height)
    plt.plot()
    plt.ylim([0, 140])
    plt.xlim([1e-5, 1])
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(xloglocator)
    plt.xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    plt.xlabel('L2 Excess Phase/km')
    plt.ylabel('Height/km')

    plt.subplot(2, 3, 4)
    plt.plot(snrpert, height)
    plt.ylim([0, 140])
    plt.xlim([-1, 1])
    plt.xlabel('SNR/SNR0 Pert')
    plt.ylabel('Height/km')

    plt.subplot(2, 3, 5)
    plt.plot(exl1pert, height)
    plt.ylim([0, 140])
    plt.xlim([-10, 10])
    plt.xlabel('L1 phase Pert/cm')
    plt.ylabel('Height(km)')

    plt.subplot(2, 3, 6)
    plt.plot(exl2pert, height)
    plt.ylim([0, 140])
    plt.xlim([-10, 10])
    plt.xlabel('L2 phase Pert/cm')
    plt.ylabel('Height/km')

    plt.tight_layout()
    # plt.show()

    figpath = os.path.join(outdir, '%s.png' % data.fileStamp)
    plt.savefig(figpath, bbox_inches='tight')
    plt.clf()
    plt.close()

    # correlation
    heights = [(75, 103), (103, 107), (107, 120)]
    plt.figure(figsize=(12, 10))

    slope = KF1 * KF1 / (KF2 * KF2)
    x = np.linspace(-10, 10, 100)
    y = x * slope
    for i in range(3):
        ind = np.logical_and(height > heights[i][0], height < heights[i][1])
        plt.subplot(3, 2, 2 * i + 1)
        plt.scatter(exl1pert[ind], snrpert[ind])
        plt.ylim([-1, 1])
        plt.xlim([-10, 10])
        plt.xlabel('L1  Ph Pert/cm')
        plt.ylabel('SNR/SNR0 Pert')
        plt.title('%d-%dkm' % (heights[i][0], heights[i][1]))

        plt.subplot(3, 2, 2 * i + 2)
        plt.scatter(exl1pert[ind], exl2pert[ind])
        plt.plot(x, y)
        plt.ylim([-10, 10])
        plt.xlim([-10, 10])
        plt.xlabel('L1 Ph Pert/cm')
        plt.ylabel('L2 Ph Pert/cm')
        plt.title('%d-%dkm' % (heights[i][0], heights[i][1]))

    plt.tight_layout()
    figpath = os.path.join(outdir, '%s-CORR.png' % data.fileStamp)
    plt.savefig(figpath, bbox_inches='tight')
    plt.clf()
    plt.close()


def process(ins, out, flag):
    """Process.

    Args:
        ins: inputs, including filepath or directory(flag is true)
        out: output directory.
        flag: recursively process file in directory.

    Returns:
        None.
    """
    if not os.path.isdir(out):
        print('out not exists or is not directory')
        return
    if (flag):
        if not os.path.isdir(ins):
            print('input should be diretory when -r flag setted')
            return
        for base, dirs, files in os.walk(ins):
            for f in files:
                filepath = os.path.join(base, f)
                sporadices(filepath, out)
    else:
        if not os.path.isfile(ins):
            print('input not exists or is not filepath')
            return
        sporadices(ins, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sporadicEs', description='save snr perturbations')
    parser.add_argument(
        'input',
        type=str,
        help='atmphs filepath or '
        'directory contains atmphs files')
    parser.add_argument('output', type=str, help='output directory')
    parser.add_argument(
        '-r', action='store_true', help='recursively process '
        'atmphs file')
    args = parser.parse_args()
    process(args.input, args.output, args.r)
