from math import log2
from matplotlib import pyplot as plt


def split_to_ngramms(t, n):
    ngramms = []
    for i in range(len(t) - n):
        ngramms.append(t[i:i + n])
    return ngramms


def entropy(fd):
    H = 0
    for k, v in fd.items():
        p = v / len(fd.keys())
        H += p * log2(p)
    return -H


def count_frequency(ngramms):
    uniq_ngramms = list(set(ngramms))
    freq_dict = {}
    for ung in uniq_ngramms:
        freq_dict[ung] = ngramms.count(ung)
    return freq_dict


with open('brigantina.txt', 'rt', encoding='utf-8') as f:
    data = f.read()

text = ''
for let in data:
    if let.isalpha():
        text += let.lower()

xs = []
ys = []
for k in range(1, 6):
    ngrs = split_to_ngramms(text, k)
    freqs = count_frequency(ngrs)
    print(freqs)
    h = entropy(freqs)
    print(f'H_{k} = {h}')
    print(f'H_{k} / {k} = {h / k}')
    xs.append(k)
    ys.append(h / k)

plt.plot(xs, ys)
plt.xlabel('k')
plt.ylabel('H / k')
plt.show()