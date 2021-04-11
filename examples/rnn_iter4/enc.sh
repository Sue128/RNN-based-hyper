for i in {01..24..1}; do
  echo Encoding /media/lzou/file/SURIGE/tensorflow_compression/examples/Kodak/kodim$i.png
  mkdir -p test/codes
  python3 rnn.py compress /media/lzou/file/SURIGE/tensorflow_compression/examples/Kodak/kodim$i.png test/codes/compressed$i.bin
done
