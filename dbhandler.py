from firebase import firebase


FILE = 'C:/Users/AA/PycharmProjects/Mostofa/pd_speech_features.csv'
KEYS = 'C:/Users/AA/PycharmProjects/Mostofa/keys.txt'
APP_DATABASE_URL = 'https://ecg-1-e1a65.firebaseio.com/'


def upload_file(file_location, directory):
    fp = open(file_location)
    fp_keys = open(KEYS, "a")
    dat_file = ""
    for d in fp.readlines():
        dat_file += d
    ref = firebase.FirebaseApplication( APP_DATABASE_URL, None)
    data =  { 'file_name' : 'data.dat',
              'data' : dat_file
              }
    result = ref.post("/"+directory+"/", data)
    fp_keys.write(data['file_name'] + " " + result['name']+'\n')
    #print(result)


def delete_file(directory, file_key):
    ref = firebase.FirebaseApplication(APP_DATABASE_URL , None)
    result = ref.delete(directory , file_key)
    fp_keys = open(KEYS, "r")
    all_key_pairs = fp_keys.readlines()
    idx=0
    for line in all_key_pairs:
        if line.split()[1] == file_key:
            break
        idx+=1
    all_key_pairs.pop(idx)
    fp_keys = open(KEYS, "w")
    fp_keys.writelines(all_key_pairs)


def get_data(directory, file_key):
    ref = firebase.FirebaseApplication(APP_DATABASE_URL , None)
    result = ref.get('/'+directory+'/', file_key)
    print(result)


def update_data(directory, file_key, value):
    ref = firebase.FirebaseApplication(APP_DATABASE_URL, None)
    # ref.put('/python-sample-ed7f7/Students/-LAgstkF0DT5l0IRucvm', 'Percentage', 79)
    ref.put('/'+directory+'/'+ file_key, 'data', value)


upload_file(FILE, 'ecg')
# get_data('ecg', '-LFDKPUHpfC3GN769Rtl')
# delete_file('ecg', '-LFDPye1u_dPm112liGs')
# update_data('ecg', '-LFDPye1u_dPm112liGs', 'foobar')
