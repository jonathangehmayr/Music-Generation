import mido
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import glob
import os
import cv2
from skimage.transform import resize
from copy import deepcopy

# adapted from https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
def msg2dict(msg):
    '''
    Conversion of a Mido message to a dictionary
    '''
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]


def switch_note(last_state, note, velocity, on_=True):
    '''
    Assigning of the power of a note (velocity) as cell value
    '''
    result = [0] * 128 if last_state is None else last_state.copy()
    if on_:
        result[note] = velocity
    return result


def get_new_state(new_msg, last_state):
    '''
    Calling of switch_note and msg2dict, to get the new state (on or off) of a
    note
    '''
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

def track2seq(track):
    '''
    Iterating through a mido object track by inferring the timing information
    of notes and prolonging a note until its state changes
    '''
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*128)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result


def mid2arr(mid):
    '''
    Converting a midi object into a numpy array
    '''
    all_arrs = []
    mid_list=get_midi_lists(mid)
    # iterating through all channels that contain messaged with note_on properties
    for idx,list_track in enumerate(mid_list):
        if 'note_on' in list_track:
            arr = track2seq(mid.tracks[idx])
            all_arrs.append(arr)
            
    # make all nested list the same length
    max_len = max([len(arr) for arr in all_arrs])
    for i in range(len(all_arrs)):
        if len(all_arrs[i]) < max_len:
            all_arrs[i] += [[0] * 128] * (max_len - len(all_arrs[i]))
    all_arrs = np.array(all_arrs)
    
    # max at axis 0 aggregates the individual tracks to a whole one
    all_arrs = all_arrs.max(axis=0)
    return all_arrs


def arr2mid(arr, filename):
    '''
    Converting a numpy array back into a midi file
    '''
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    
    
    # get the timing information of notes in the numpy array
    new_ary = np.concatenate([np.array([[0] * 128]), np.array(arr),np.array([[0] * 128])], axis=0)
    changes = new_ary[1:] - new_ary[:-1]
    
    
    # map array to midi messages
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=int(n), velocity=int(v), time=int(new_time)))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=int(n), velocity=int(0), time=int(new_time)))
                first_ = False
            last_time = 1
            
    mid_new.tracks.append(track)
    mid_new.save(filename)
    return mid_new



def plot_track(track, filename=None):
    '''
    Plotting of numpy array containing a track
    '''
    arr=track.transpose()
    plt.figure()
    plt.imshow(arr,aspect='auto',origin='lower')
    if filename:
        plt.savefig('{}.pdf'.format(filename))
    

def get_midi_lists(mid): 
    '''
    helper function to infer if a channel contains an instrument
    '''
    mid_list=[]
    for track in mid.tracks:
        list_track=list(track)
        list_track=[list(element.dict().values()) for element in list_track]
        list_track = [item for sublist in list_track for item in sublist]
        mid_list.append(list_track)
    return mid_list

def get_readable_midi(mid): 
    '''
    get_readable_midi conversts a midi object into a better readable fromat
    for exploring the data
    '''
    mid_list=[]
    for track in mid.tracks:
        list_track=list(track)
        list_track=[element.dict() for element in list_track]
        mid_list.append(list_track)
    return mid_list

def midi_file_print(mid):
    '''
    Printing the heading messages of a midi file
    '''
    for track in mid.tracks:
        for m in track[:20]:
            print(m)
        print()
        print()


def arr2pq(arr,filename,size):
    '''
    Saving an array in parquet format
    '''
    df=pd.DataFrame(arr)
    df.columns=[str(col) for col in df.columns]
    df.to_parquet('data/classic_midi_parquet/{}/{}.parquet.gzip'.format(size,filename), compression='gzip')


def pq2arr(size,reshape_size):
    '''
    Reading an stored in parquet format storing tracks
    '''
    arr_tracks=pd.read_parquet('data/classic_midi_parquet/{}/arr_tracks.parquet.gzip'.format(size)).values
    arr_track_lengths=pd.read_parquet('data/classic_midi_parquet/{}/track_lengths.parquet.gzip'.format(size)).values
    arr_tracks=arr_tracks.reshape(-1,reshape_size[0],reshape_size[1])
    arr_track_lengths=arr_track_lengths.reshape(-1)
    return arr_tracks, arr_track_lengths


def load_midi_files():
    '''
    Load the midi files into a list of midi objects
    '''
    list_midis=[]
    rootdir = 'data/classic_midi/'
    for file in os.listdir(rootdir):
        folder_artists = os.path.join(rootdir, file)
        for midi_path in os.listdir(folder_artists):
            midi_path=os.path.join(folder_artists, midi_path)
            midi_file = mido.MidiFile(midi_path, clip=True)
            list_midis.append(midi_file)
            print(midi_path)
    return list_midis


def scale_track(arr):
    '''
    Scaling the numpy arrays into a range betwen 0 and 1
    '''
    return np.vectorize(lambda x: np.float64(round(x / 127, 3)))(arr)


def inverse_scale_track(arr):
    '''
    Scaling back numpy array to be in a range of 0 to 127
    '''
    return (127*(arr - np.min(arr))/np.ptp(arr)).astype(int)


def produce_square_track_arrays():
    '''
    Producing of datasets with arrays representing songs with the sized 
    (256,256), (512,512) and (1024,1024)
    '''
    sizes=[256,512,1024]
    list_midis=load_midi_files()
    for size in sizes:
        list_arr=[]
        list_track_len=[]
        down_size=(size,size)
        for midi in list_midis:
            arr = mid2arr(midi)
            arr_down = resize(arr, down_size,preserve_range=True, order=0)
            len_track=arr.shape[0]
            list_arr.append(arr_down)
            list_track_len.append(len_track)
            
        arr_tracks=np.stack(list_arr, axis=0) 
        arr_tracks=np.apply_along_axis(scale_track, 0, arr_tracks)
        np.random.shuffle(arr_tracks)
        arr_2d=arr_tracks.reshape(-1, size)
        arr2pq(arr_2d, 'arr_tracks',size)
        arr2pq(list_track_len,'track_lengths',size)

def produce_track_arrays():
    '''
    Producing of a dataset with arrays representing songs with the size of (1024,128)
    '''
    
    list_midis=load_midi_files()
    list_arr=[]
    list_track_len=[]
    down_size=(1024,128)
    for midi in list_midis:
        arr = mid2arr(midi)
        arr_down = resize(arr, down_size,preserve_range=True, order=0)
        len_track=arr.shape[0]
        list_arr.append(arr_down)
        list_track_len.append(len_track)
        
    arr_tracks=np.stack(list_arr, axis=0)
    arr_tracks=np.apply_along_axis(scale_track, 0, arr_tracks)
    np.random.shuffle(arr_tracks)
    arr_2d=arr_tracks.reshape(-1, 128)
    arr2pq(arr_2d, 'arr_tracks','1024x128')
    arr2pq(list_track_len,'track_lengths','1024x128')




if __name__=='__main__':
    
    # Producing the datasets
    produce_square_track_arrays()
    produce_track_arrays()
    
    # Plotting of a smaple track    
    arr_tracks, arr_track_lengths=pq2arr('1024x128',(1024,128))
    plot_track(arr_tracks[0,:,:])
    