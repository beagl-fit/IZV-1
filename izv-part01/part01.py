#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: xnovak2r

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""


from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    """
    Function for calculating integral using the trapezoidal rule instead of build in np function.

    :param x: sorted vector x (np.array)
    :param y: sorted vector y (np.array)
    :returns: calculated integral (float)
    """
    x2 = np.copy(x[1:])     # make copy of array x without the first value
    x2 -= x[:-1]            # subtract x without the last value from x2

    y2 = np.copy(y[:-1])
    y2 += y[1:]
    y2 /= 2

    x2 *= y2                # multiply across all elements and return the sum
    return float(np.sum(x2))


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Function which generates graph of function f(x) = x^2 * a, for three values of 'a', where 'x' ∈ <-3.0; 3.0>.

    :param a: list of 3 floats used as param 'a' in function f(x)
    :param show_figure: shows graph (default False)
    :param save_path: where to save graph (default None)
    """
    x = np.arange(-3, 3.1, 0.1)     # prepare x, a and calculate the functions
    a2 = np.array(a)
    y = np.power(np.array([np.copy(x), np.copy(x), np.copy(x)]), 2) * a2.reshape((a2.size, 1))

    plt.figure(figsize=(9, 5))

    plt.plot(x, y[0, :], label='$y_{1.0}(x)$')
    plt.fill_between(x, y[0, :], alpha=0.1)
    plt.annotate("$\int f_{1.0}(x)\,dx$", (3, y[0, -1]))

    plt.plot(x, y[1, :], label='$y_{2.0}(x)$')
    plt.fill_between(x, y[1, :], alpha=0.1)
    plt.annotate("$\int f_{2.0}(x)\,dx$", (3, y[1, -1]))

    plt.plot(x, y[2, :], label='$y_{-2.0}(x)$')
    plt.fill_between(x, y[2, :], alpha=0.1)
    plt.annotate("$\int f_{-2.0}(x)\,dx$", (3, y[2, -1]))

    plt.xlim([-3, 3.7])
    plt.ylim([-20, 20])
    plt.xlabel('x')
    plt.ylabel('f$_a$(x)')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    plt.tight_layout()
    plt.savefig(save_path)

    if show_figure:
        plt.show()
    plt.close()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Function generates 3 graphs (1 file) of functions:
     f1 = 0.5 * sin(1/50 * pi * t), f2 = 0.25 * sin(pi * t), f3 = f1 + f2, for 't' ∈ <0; 100>.
    Graph f3 will be green for values f3 > f1 and red for f3 <= f1.

    :param show_figure: shows graphs (default False)
    :param save_path: where to save graphs (default None)
    """
    fig, axes = plt.subplots(ncols=1, nrows=3,
                             constrained_layout=True,
                             figsize=(6, 8)
                             )

    plt.setp(axes, xlim=[0, 100], ylim=[-0.8, 0.8], xlabel='t')
    ax1, ax2, ax3 = axes
    plt.setp(ax1, ylabel='$f_1(x)$')
    plt.setp(ax2, ylabel='$f_2(x)$')
    plt.setp(ax3, ylabel='$f_1(x) + f_2(x)$')

    t = np.linspace(0, 100, 5000)

    f1 = 0.5 * np.sin(1 / 50 * np.pi * t)
    ax1.plot(t, f1)

    f2 = 0.25 * np.sin(np.pi * t)
    ax2.plot(t, f2)

    f3 = f1 + f2
    ax3.plot(t, f3, color='g')
    mask = np.ma.masked_greater(f3, f1)
    ax3.plot(t, mask, color='r')

    fig.tight_layout()
    fig.savefig(save_path)
    if show_figure:
        plt.show()

    plt.close(fig)


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    """
    Function for parsing data from online source using BeautifulSoup library.

    :param url: url to website with data table (default 'https://ehw.fit.vutbr.cz/izv/temp.html')
    :returns: list of dictionaries with values from data table
    """
    resp = requests.get(url)
    if not 200 <= resp.status_code < 300:
        print('ERROR: HTTP request')
        return
    temp_list = []
    soup = BeautifulSoup(resp.text, features='xml')
    for row in soup.find_all('tr', class_='ro1'):
        year_month = row.find_all('td')
        temp_array = []
        for temp in year_month[2:]:
            try:
                temp_array.append(np.float_(temp.p.text.replace(',', '.')))
            except Exception:
                pass

        temperatures = {'year': int(year_month[0].p.text),
                        'month': int(year_month[1].p.text),
                        'temp': np.array(temp_array)}
        temp_list.append(temperatures)

    return temp_list


def get_avg_temp(data, year=None, month=None) -> float:
    """
    Function for calculating average temperature.

    :param data: data for calculation average temperature
    :param year: calculate average only for specific year (default None)
    :param month: calculate average only for specific month (default None)
    :returns: average temperature (float)
    """
    temps = np.array([])
    for x in data:
        if year and x.get('year') != year:
            continue
        if month and x.get('month') != month:
            continue

        temps = np.concatenate((temps, x.get('temp')))

    return np.average(temps)
