import os,pickle,yaml
import numpy as np

def quick_load_pickle(folder, filename):
    path = os.path.join(folder,filename)
    return pickle.load(open(path,'rb'))

def quick_dump_pickle(folder, filename, variable):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder,filename)
    pickle.dump(variable, open(path,'wb'))

def quick_load_yaml(folder, filename):
    path = os.path.join(folder,filename)
    return yaml.load(open(path,'r'), Loader=yaml.FullLoader)

def quick_dump_yaml(folder, filename, variable):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder,filename)
    yaml.dump(variable,open(path,'w'),default_flow_style=False)

filename2frameNum = lambda s: int(s.split('.jpg')[0].split('-')[-1])


def rescale_list(input_list, size, lower_frame, upper_frame,do_assert_filenames, random_frame_sampling):
    """
    For example, [0,1,2,3,4,5,6,7,8] to [0,3,6]
    """
    cut_input_list = input_list[lower_frame:upper_frame+1]
    #print(cut_input_list)
    assert len(cut_input_list) == upper_frame - lower_frame + 1
    if do_assert_filenames:
        assert filename2frameNum(cut_input_list[0]) == lower_frame+1
        assert filename2frameNum(cut_input_list[-1]) == upper_frame+1
    skip = (upper_frame - lower_frame) / (size - 1)
    if not random_frame_sampling:
        output = [cut_input_list[int(i*skip+0.5)] for i in range(size)]
    else:
        i = 0
        output = []
        for i in range(size):
            base_idx = int(i*skip+0.5)
            if i == size-1:
                idx = base_idx
            else:
                next_idx = int((i+1)*skip+0.5)
                idx =  np.random.randint(base_idx, next_idx)
            output.append(cut_input_list[idx])
    assert len(output) == size
    return output

