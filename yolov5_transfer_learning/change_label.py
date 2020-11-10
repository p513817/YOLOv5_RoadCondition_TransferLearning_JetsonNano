#%%
import os
import shutil

path = './test'

label_file = os.listdir(path)

print(len(label_file))
print(label_file)
# %%

for idx, f in enumerate(label_file):

    if f == 'classes.txt': continue

    trg_file = os.path.join(path, f)
    print("[{:02}/{:02}] {} ... ".format(idx+1, len(label_file), trg_file), end=' ')

    txt = open(trg_file,'r+')

    pre_content = txt.readlines()[0]
    ls_content  = pre_content.split(' ')
    new_content = '0'
    len_content = len(ls_content)

    for i in range(1, len_content):
        new_content+=f' {ls_content[i]}'
        if i == len_content:
            new_content+='\n'
    txt.close()

    txt_write = open(trg_file,'w')
    txt_write.write(new_content)
    txt_write.close()
    print('finish')

os.remove(os.path.join(path ,'classes.txt'))
print("Removed classes.txt")
# %%
