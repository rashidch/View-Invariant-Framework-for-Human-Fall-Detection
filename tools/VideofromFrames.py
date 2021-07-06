
import ffmpeg

input = 'outputs/camtest/B/%03d.jpg'
output = 'outputs/camtest/B/B.mp4'
ffmpeg -framerate 20 -i input output

ffmpeg -framerate 20 -i outputs/camtest/C/frame%03d.jpg outputs/camtest/C/C.mp4

ffmpeg -framerate 10 -i outputs/camtest/C/frame%05d.png outputs/camtest/C/C.mp4

ffmpeg -framerate 10 -i outputs/camtest/C/frame%05d.png outputs/camtest/C/C.avi