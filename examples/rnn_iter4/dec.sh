for i in {01..24..1}; do
  echo Decoding test/codes/compressed$i.bin
  mkdir -p test/decoded
  python3 rnn.py decompress test/codes/compressed$i.bin test/decoded/compressed$i.png
done
