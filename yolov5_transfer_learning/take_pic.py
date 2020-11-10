#%%
import cv2
import numpy as np

"""  save data  """
def save(trg_path, idx, frame):
        global label
        save_path = os.path.join(trg_path, rf'{idx}_{label[idx]}.jpg')
        print(save_path)
        cv2.imwrite(save_path, frame)
        label[idx] = label[idx]+1

"""  拍照蒐集資料  """
def take_pic():

    ############## set target path ##############
    trg_path = 'data'
    if os.path.exists(trg_path)==False: os.mkdir(trg_path)

    ############## get camera & set size ##############
    #w_size, h_size = 512, 512
    global label
    label = np.zeros([10], dtype=int)
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_size)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_size)

    ############## set parameters ##############
    
    print(f'label len : {len(label)}')


    ############## open camera % save data ##############
    while(True):

        ret, frame = cap.read()
        overlay = frame.copy()
        ############## show info ##############
        text =  '{}{}{}{}{}'.format(f'Label\n0: {label[0]}\n1: {label[1]}\n', 
            f'2: {label[2]}\n3: {label[3]}\n' ,
            f'4: {label[4]}\n5: {label[5]}\n' ,
            f'6: {label[6]}\n7: {label[7]}\n' ,
            f'8: {label[8]}\n9: {label[9]}\n' )
        for i, txt in enumerate(text.split('\n')):
            cv2.putText(overlay, txt,(20,25*i+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1 )

        cv2.imshow('Create_Your_Own_Datasets', overlay)

        key = cv2.waitKey(1)
        
        if key== ord('q'): break
        
        for i in range(10):
            if key == ord(f'{i}'):
                save(trg_path, i, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    take_pic()
# # %%

# %%
