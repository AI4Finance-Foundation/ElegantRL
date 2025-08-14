import copy
def calc_result_file_name(file: str, add_tail: str= ''):
    new_file = copy.deepcopy(file)
    if 'data' in new_file:
        new_file = new_file.replace('data', 'result')
    # if file[0: 2] == '..':
    #     new_file = new_file.split('.txt')[0]
    #     new_file = new_file.split('/')[0] + '/' + new_file.split('/')[1] + '/' + new_file.split('/')[-1]
    # else:
    #     new_file = new_file.split('.')[0]
    #     new_file = new_file.split('/')[0] + '/' + new_file.split('/')[-1]
    new_file = new_file.split('result')[0] + 'result/' + new_file.split('/')[-1]
    if add_tail is not None:
        new_file = new_file.replace('.txt', '') + add_tail + '.txt'
    return new_file
def calc_txt_files_with_prefix():
    pass

plot_fig_over_durations = None