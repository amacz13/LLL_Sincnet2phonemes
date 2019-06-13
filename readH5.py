import h5py
import matplotlib.pyplot as plt
import numpy as np

dset_a = dict()
dset_b = dict()

def show_1(var):
    # to use only if both a and b file are loaded #
    y_a = dset_a[var]
    y_b = dset_b[var]
    x = np.arange(y_a.size)
    plt.plot(x, y_a, label=var+' a')
    plt.plot(x, y_b, label=var+' b')
    plt.legend()
    plt.show()

def read_fb(filename):
    h5 = h5py.File(filename+'.h5', 'r')
    dset = h5['swb'][filename]
    fb = np.transpose(dset['fb'])
    plt.plot(np.arange(0, fb.shape[1]), fb[0], label='fb '+filename)
    plt.legend()
    plt.show()

def read_vad(filename):
    print("Trying to read vad values from swb/"+filename.split('/')[8].split('.')[0])
    h5 = h5py.File(filename, 'r')
    dset = h5['swb'][filename.split('/')[8].split('.')[0]]
    vad = np.transpose(dset['vad'])
    return vad


def read_all(filename_a, filename_b):
    h5_a = h5py.File(filename_a+'.h5', 'r')
    h5_b = h5py.File(filename_b+'.h5', 'r')

    dset_a = h5_a['swb']['sw02001_a']
    dset_b = h5_b['swb']['sw02001_b']

    for key in dset_b.keys():
        print(key)

    cep_a = np.transpose(dset_a['cep'])
    energy_a = np.transpose(dset_a['energy'])
    fb_a = np.transpose(dset_a['fb'])
    cep_b = np.transpose(dset_b['cep'])
    energy_b = np.transpose(dset_b['energy'])
    fb_b = np.transpose(dset_b['fb'])

    plt.plot(np.arange(0, cep_a.shape[1]), cep_a[0], label='CEP a')
    plt.plot(np.arange(0, cep_b.shape[1]), cep_b[0], label='CEP b')
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, energy_a.size), energy_a, label='Energy a')
    plt.plot(np.arange(0, energy_b.size), energy_b, label='Energy b')
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, fb_a.shape[1]), fb_a[0], label='FB a')
    plt.plot(np.arange(0, fb_b.shape[1]), fb_b[0], label='FB b')
    plt.legend()
    plt.sho


if __name__ == "__main__":

    read_all("sw02001_a", "sw02001_b")
    # read_fb('new_h5')
