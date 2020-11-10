import vlc
import cv2
import os
import pafy
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time

""" Required: pafy、youtube-dl、python-vlc、moviepy """
""" sudo apt install ffmpeg"""

### 取得terminal視窗大小
def get_cmd_size():
    import shutil
    col, line = shutil.get_terminal_size()
    return col, line

### 打印分隔用
def print_div(text=None):
    col,line = get_cmd_size()
    col = col-1
    if text == None:
        print('-'*col)
    else:
        print('-'*col)
        print(text)
        print('-'*col)

### 計時、進度條
def clock(sec, sign = '#'):
    col, line = get_cmd_size()
    col_ = col - 42
    bar_bg = ' '*(col_)

    print_div()

    for i in range(sec+1):
        
        bar_idx = (col_//sec*(i+1))
        bar = ''
        for t in range(col_): bar += sign if t <= bar_idx else bar_bg[t]
        
        percent = int(100/sec*(i))
        end = '\n' if i==sec else '\r'
        print('Download Stream Video [{:02}/{:02}s] [{}] ({:02}%)'.format(i, sec, bar, percent), end=end)
        time.sleep(1)

### 擷取特定秒數並儲存
def cut_video(name, sec, save_name):
    print_div()
    print('Cutting Video Used Moviepy\n')
    ffmpeg_extract_subclip(name, 0, sec, targetname=save_name)
    print_div(f'save file {save_name}')

### 主程式
def capture_video(opt):

    f_name = 'org.mp4'      # 下載影片名稱
    o_name = opt.output     # 裁剪影片名稱     
    sec = opt.second        # 欲保留秒數
    video = pafy.new(opt.url)   # 取得Youtube影片
    
    r_list = video.allstreams   # 取得串流來源列表

    print_div()
    for i,j in enumerate(r_list): print( '[ {} ] {} {}'.format(i,j.title,j))
    idx = input('\nChoose the resolution : ')
    
    if idx:
        ### 選擇串流來源
        trg = r_list[int(idx)]
        print_div('您選擇的解析度是: %s'%(trg))
        
        ### 下載串流
        vlcInstance = vlc.Instance()
        player = vlcInstance.media_player_new()    # 創建一個新的MediaPlayer實例
        media = vlcInstance.media_new(trg.url)     # 創建一個新的Media實例
        media.get_mrl()
        media.add_option(f"sout=file/ts:{f_name}") # 將媒體直接儲存 
        player.set_media(media)                    # 設置media_player將使用的媒體
        player.play()                              # 播放媒體
        time.sleep(1)                              # 等待資訊跑完

        ### 進度條、擷取影片
        clock(sec)                                 # 播放 sec 秒 同時運行 進度條
        cut_video(f_name, sec, o_name)             # 裁切影片，因為停n秒長度不一定是n

        ### 關閉資源
        player.stop()                              # 關閉撥放器以及釋放媒體
        media.release()
    
    else:
        print('Please Try Again!')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', help='youtube url')
    parser.add_argument('-o', '--output', type=str, default='sample.mp4' , help='save file path\name')
    parser.add_argument('-s', '--second',type=int, default=10 , help='video length')
    opt = parser.parse_args()
    capture_video(opt)
 # %%
