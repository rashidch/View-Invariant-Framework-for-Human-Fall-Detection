from moviepy.editor import *

clip = (VideoFileClip("/home/rasho/Falling-Person-Detection-based-On-AlphaPose/outputs/dnntiny/1.avi").subclip((0.0),(30))
        .resize(30))
clip.write_gif("/home/rasho/Falling-Person-Detection-based-On-AlphaPose/outputs/dnntiny/1.gif")