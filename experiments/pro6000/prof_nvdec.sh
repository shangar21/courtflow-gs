V=/home/ubuntu/datasets/basketball/cameras/view_000.mp4
ffmpeg -version | head -1
for WH in 1920:1080 3840:2160; do
  W=${WH%:*}; H=${WH#*:}
  for mode in cpu nvdec nvdec_noscale; do
    case $mode in
      cpu) cmd="ffmpeg -v error -i $V -frames:v 200 -vf scale=$W:$H:flags=lanczos -f rawvideo -pix_fmt rgb24 -";;
      nvdec) cmd="ffmpeg -v error -hwaccel cuda -hwaccel_output_format cuda -c:v hevc_cuvid -i $V -frames:v 200 -vf scale_cuda=$W:$H:interp_algo=lanczos,hwdownload,format=nv12 -f rawvideo -pix_fmt rgb24 -";;
      nvdec_noscale) [ $W = 3840 ] || continue; cmd="ffmpeg -v error -hwaccel cuda -hwaccel_output_format cuda -c:v hevc_cuvid -i $V -frames:v 200 -vf hwdownload,format=nv12 -f rawvideo -pix_fmt rgb24 -";;
    esac
    s=$(date +%s.%N); bytes=$($cmd 2>/tmp/nverr | wc -c); e=$(date +%s.%N)
    echo "$W $mode: $(echo "($e-$s)*1000/200" | bc -l | cut -c1-5) ms/frame, frames=$((bytes/(W*H*3))) $(head -c 200 /tmp/nverr)"
  done
done
